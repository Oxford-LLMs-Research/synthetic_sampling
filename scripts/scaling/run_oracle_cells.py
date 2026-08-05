#!/usr/bin/env python
r"""Run the features_project oracle on our (target, country) cells.

This is a thin wrapper. survey_features.oracle.compute_oracle already does the
whole job for one cell: country subsetting, column cleaning, feature-pool
construction with the semantic near-duplicate filter, the within-country
missingness and variation filters, conditional-leakage detection, rare-class
filtering, a stratified holdout, the AutoGluon fit and permutation importance.
Nothing here reimplements any of that.

The wrapper adds exactly one thing, and it is the one thing their pipeline has
no reason to provide: a block of respondents held out before the oracle ever
sees the cell, reserved for the language model. Without it the informative arm
would be ordered by a ranking chosen partly from noise in the very people it is
then tested on. So each cell is split

    20%  reserved   removed before compute_oracle is called; the LLM's pool
    80%  passed to compute_oracle, which splits it 80/20 internally into
         its own fit and scoring halves

giving roughly 64 / 16 / 20 overall.

An earlier version of this script substituted a single XGBoost model for
AutoGluon to avoid the dependency. That produced a model worse than predicting
the modal answer in 19 of 126 cells, twelve of them Asian Barometer, because
nothing in a fixed-hyperparameter booster checks whether the fit is helping.
AutoGluon selects against its own validation split and its ensemble includes a
baseline, so it cannot lose to a constant. The substitution is gone.

    python .../run_oracle_cells.py --workers 4
    python .../run_oracle_cells.py --cells one.csv --out tmp/ --workers 1
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
FEATURES_SRC = Path(r"C:\Users\murrn\cursor\features_project\src")
OUT = REPO.parent / "analysis" / "feature_importance"
SCALE = REPO / "outputs" / "scaling_experiment"
RESERVED_FRAC = 0.20
SEED = 42


def run_chunk(survey: str, cells: list[dict], time_limit: int, num_gpus: int,
              tag: str) -> tuple[list[dict], list[dict], list[dict]]:
    sys.path.insert(0, str(FEATURES_SRC))
    sys.path.insert(0, str(REPO / "src"))
    from survey_features import oracle as ora
    from survey_features import surveys as sv
    from synthetic_sampling.config import DataPaths
    from synthetic_sampling.config.surveys import get_survey_config
    from synthetic_sampling.loaders.survey_loader import SurveyLoader

    ora._prewarm_autogluon_imports()

    paths = DataPaths.from_yaml(str(REPO / "configs" / "local.yaml"))
    cfg = get_survey_config(survey)
    data, meta = SurveyLoader(paths, verbose=False).load_survey(survey)
    country_col = cfg.country_col
    id_col = cfg.respondent_id_col
    cmap = sv.build_country_code_map(meta, country_col, data)
    admin = sv.build_admin_cols(meta, country_col)
    flat = sv.flatten_metadata(meta)
    sim = ora.load_similarity_model(ora.SIMILARITY_THRESHOLD)

    # AutoGluon writes model artefacts to disk; give each worker its own root
    # so concurrent cells cannot collide on the same path.
    tmp_root = Path(os.environ.get("TEMP", ".")) / f"ag_{tag}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rows, reserved, diag = [], [], []
    for cell in cells:
        target, country = cell["target_code"], str(cell["country"])
        t0 = time.time()
        try:
            code = cmap.get(country, country)
            if code not in set(data[country_col].dropna().unique()):
                for cand in (country, float(country) if country.replace(".", "").isdigit() else None):
                    if cand is not None and cand in set(data[country_col].dropna().unique()):
                        code = cand
                        break

            in_cell = (data[country_col] == code) & data[target].notna()
            ids = data.loc[in_cell, id_col].astype(str).unique()
            if len(ids) < 200:
                diag.append({**cell, "status": f"only {len(ids)} respondents"})
                continue
            rng = np.random.default_rng(SEED)
            n_res = max(1, int(RESERVED_FRAC * len(ids)))
            held = set(rng.permutation(np.array(sorted(ids)))[:n_res])
            visible = data[~data[id_col].astype(str).isin(held)]

            odf, pool = ora.compute_oracle(
                data=visible, metadata=meta, target_var=target,
                country_code=code, country_col=country_col, admin_cols=admin,
                metadata_flat=flat, similarity_model=sim,
                tmp_root=tmp_root, n_repeats=5, random_state=SEED,
                autogluon_time_limit=time_limit, num_gpus=num_gpus,
                ag_verbosity=0)

            odf = odf.sort_values("importance_mean", ascending=False)
            for rank, (_, r) in enumerate(odf.iterrows()):
                rows.append({
                    "survey": survey, "target_code": target, "country": country,
                    "feature": r["feature_variable"], "rank": rank,
                    "importance_mean": float(r["importance_mean"]),
                    "importance_std": float(r["importance_std"]),
                })
            for rid in sorted(held):
                reserved.append({"survey": survey, "target_code": target,
                                 "country": country, "respondent_id": rid})
            base = float(odf["majority_baseline"].iloc[0])
            diag.append({
                **cell, "status": "ok",
                "n_visible": int(len(ids) - n_res), "n_reserved": n_res,
                "n_features": int(len(pool)),
                "majority_baseline": round(base, 4),
                "top_feature": odf["feature_variable"].iloc[0],
                "top_importance": round(float(odf["importance_mean"].iloc[0]), 5),
                "seconds": round(time.time() - t0, 1),
            })
        except Exception as e:                                    # noqa: BLE001
            diag.append({**cell, "status": f"ERROR {type(e).__name__}: {e}",
                         "seconds": round(time.time() - t0, 1)})
    return rows, reserved, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=SCALE / "oracle_cells.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--time-limit", type=int, default=0,
                    help="AutoGluon seconds per cell; 0 uses their default")
    ap.add_argument("--num-gpus", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    global OUT
    cells = pd.read_csv(args.cells)
    if args.out:
        OUT = args.out
    elif len(cells) < 100:
        raise SystemExit(
            f"refusing to write a {len(cells)}-cell run to the default output "
            f"directory, which may hold a full run. Pass --out for subsets.")

    groups = {}
    for s, g in cells.groupby("survey"):
        recs = g.to_dict("records")
        for i in range(0, len(recs), args.chunk):
            groups[f"{s}#{i // args.chunk}"] = recs[i:i + args.chunk]
    print(f"{len(cells)} cells in {len(groups)} chunks, {args.workers} workers",
          flush=True)
    print(f"  writing to {OUT}", flush=True)

    rows, reserved, diag = [], [], []
    OUT.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_chunk, k.split("#")[0], c, args.time_limit,
                          args.num_gpus, k.replace("#", "_")): k
                for k, c in groups.items()}
        for fut in as_completed(futs):
            r, res, d = fut.result()
            rows.extend(r); reserved.extend(res); diag.extend(d)
            done += len(d)
            ok = sum(1 for x in d if x.get("status") == "ok")
            print(f"  {futs[fut]:<24}{ok}/{len(d)} ok  [{done}/{len(cells)}]  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "feature_importance.csv", index=False)
            pd.DataFrame(reserved).to_csv(OUT / "reserved_respondents.csv",
                                          index=False)
            pd.DataFrame(diag).to_csv(OUT / "cell_diagnostics.csv", index=False)

    dd = pd.DataFrame(diag)
    ok = dd[dd["status"] == "ok"]
    print(f"\n{len(ok)} of {len(dd)} cells in {time.time() - t0:.0f}s")
    if len(ok):
        print(f"median seconds per cell: {ok['seconds'].median():.0f}")
    bad = dd[dd["status"] != "ok"]
    if len(bad):
        print(f"\n{len(bad)} not fitted:")
        print(bad["status"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
