"""XGB-CEILING-FULL: the properly-fit same-features ceiling (13 Aug
deliberation; commissioned after XGB-CEILING's anchor was demoted to a
floor).

The anchor's ~50-row cells starve XGBoost into a marginal predictor; the
selection-era diagnostics show large-n fits lift +0.098 raw over the mode.
This run fits XGBoost per target on the FULL survey microdata, restricted
to the same informative-feature pool the C-series substrate used (the
union, per target, of the k=24 features its anchor pairs carry — feature
sets vary per respondent by availability, so the union is the faithful
"same features" matrix, exactly xgb_cell's semantics), and evaluates
out-of-fold on the exact anchor instances, under two regimes:

  grouped   GroupKFold on country over the full sample — the honest
            cross-country generalisation standard (matches the floor fit)
  within    per country: 5-fold shuffled KFold inside the country — the
            friendliest defensible setting (matches the selection-era
            diagnostics' design)

Together they bracket the same-features ceiling around the model's
matched 0.3560.

Mapping (validated end-to-end before any fit; --check runs only this):
ladder question text -> metadata code (ambiguous texts resolved by data
coverage), respondent matched on the survey's composite id, target codes
-> label text via metadata values. HARD GATE: for every anchor pair, the
microdata-derived target label must equal the ladder's ground_truth, or
the script exits 1. Feature values are raw microdata codes
(integer-encoded, native NaN); missing-value sentinel codes remain as
categories — noted, tolerable for a tree fit.

Outputs (WORK/analysis/xgb_ceiling/):
  xgb_full_anchor_dump.jsonl    per anchor instance x regime: oof dist,
                                pred, p_true, fold
  xgb_full_anchor_summary.csv   per regime: anchor-slice acc/norm_acc +
                                the floor and model rows for context
  xgb_full_per_target.csv       per (target, regime): n_train, anchor n,
                                acc, plus full-sample OOF acc (grouped)
  xgb_full_dissenter_split.csv  modal/dissenter split on anchor rows

Usage:
  python scripts/xgb_ceiling/run_xgb_full.py --check     # mapping only
  python scripts/xgb_ceiling/run_xgb_full.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
OUT_DIR = WORK / "analysis" / "xgb_ceiling"
LADDER = WORK / "outputs_recovered" / "ladder_readout_set.jsonl"
B3_TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
RAW_DATA = WORK / "data"

SEED = 42
EPS = 1e-12

sys.path.insert(0, str(REPO / "src"))


def flatten_meta(meta: dict) -> tuple[dict, dict]:
    """question text -> [codes]; code -> {value code -> label}."""
    text2codes: dict[str, list] = defaultdict(list)
    values: dict[str, dict] = {}
    for block in meta.values():
        if not isinstance(block, dict):
            continue
        for code, info in block.items():
            if isinstance(info, dict) and "question" in info:
                text2codes[info["question"].strip()].append(code)
                values[code] = info.get("values") or {}
    return text2codes, values


def label_of(values: dict, code, vmap_cache: dict) -> str | None:
    """Microdata cell value -> label text via the metadata values map.

    Some sources (Asian Barometer CSV) store TEXT LABELS directly; a
    string value is already the label and passes through.
    """
    if isinstance(code, str):
        return code.strip() or None
    if pd.isna(code):
        return None
    key = code
    if isinstance(code, float) and code.is_integer():
        key = int(code)
    return vmap_cache.get(str(key))


def load_anchor() -> dict:
    """Per (survey, target): anchor pairs + feature-text union + options."""
    b3 = {json.loads(l)["example_id"] for l in B3_TASKS.open(encoding="utf-8")}
    per_target: dict[tuple, dict] = {}
    with LADDER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            eid = r["example_id"]
            if not eid.endswith("_ladder_informative_k024") or eid not in b3:
                continue
            key = (r["survey"], r["target_code"])
            t = per_target.setdefault(key, {
                "pairs": [], "qtexts": set(),
                "options": list(dict.fromkeys(r["option_sets"]["original"])),
            })
            t["pairs"].append({
                "example_id": eid, "resp_id": str(r["id"]),
                "gt_text": r["ground_truth"], "country": str(r["country"]),
            })
            t["qtexts"].update(q.strip() for q in r["questions"])
    return per_target


def fit_oof(X: np.ndarray, y: np.ndarray, splits) -> tuple[np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier

    M = int(y.max()) + 1
    oof = np.full((len(y), M), np.nan)
    fold_of = np.full(len(y), -1)
    for f, (tr, te) in enumerate(splits):
        classes = np.unique(y[tr])
        if len(classes) < 2:
            continue
        remap = {c: i for i, c in enumerate(classes)}
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", n_jobs=8, random_state=SEED, verbosity=0)
        clf.fit(X[tr], np.array([remap[c] for c in y[tr]]))
        proba = clf.predict_proba(X[te])
        oof[te] = 0.0
        for local, c in enumerate(classes):
            oof[te, c] = proba[:, local]
        fold_of[te] = f
    return oof, fold_of


def norm_acc_paper(df: pd.DataFrame) -> float:
    M_by_t = df.groupby("cluster")["gt_text"].nunique()
    d = df.assign(M=df["cluster"].map(M_by_t))
    d = d[d["M"] >= 2]
    na = ((d["pred_text"] == d["gt_text"]).astype(float) - 1 / d["M"]) \
        / (1 - 1 / d["M"])
    return float(na.groupby(d["cluster"].to_numpy()).mean().mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="validate the mapping end-to-end, fit nothing")
    ap.add_argument("--targets", default=None,
                    help="comma list of target_codes to restrict to (test)")
    args = ap.parse_args(argv)

    from sklearn.model_selection import GroupKFold, KFold
    from synthetic_sampling.surveys import DataPaths
    from synthetic_sampling.surveys.loaders import SurveyLoader
    from synthetic_sampling.surveys.registry import get_survey_config

    anchor = load_anchor()
    if args.targets:
        keep = set(args.targets.split(","))
        anchor = {k: v for k, v in anchor.items() if k[1] in keep}
    print(f"{len(anchor)} targets, "
          f"{sum(len(t['pairs']) for t in anchor.values())} anchor pairs")

    paths = DataPaths.default_bundled(RAW_DATA, "./outputs")
    loader = SurveyLoader(paths, verbose=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_fail: list[str] = []
    anchor_rows: list[dict] = []   # dump rows, both regimes
    per_target_rows: list[dict] = []

    for survey in sorted({s for s, _ in anchor}):
        df, meta = loader.load_survey(survey)
        cfg = get_survey_config(survey)
        text2codes, values = flatten_meta(meta)
        ids = df[cfg.respondent_id_col].astype(str)
        dup = ids.duplicated(keep=False)
        if dup.any():
            # one sentinel id value per survey shares many rows (checked
            # 13 Aug: no anchor id is affected); ambiguous rows leave the
            # pool rather than matching arbitrarily
            print(f"  {survey}: dropping {int(dup.sum())} rows with "
                  f"non-unique respondent ids")
            df = df.loc[~dup.to_numpy()].reset_index(drop=True)
            ids = df[cfg.respondent_id_col].astype(str)
        id_pos = {v: i for i, v in enumerate(ids)}
        countries = df[cfg.country_col].astype(str).to_numpy()

        for (s, target), t in sorted(anchor.items()):
            if s != survey:
                continue
            cluster = f"{s}|{target}"
            opts = t["options"]
            vmap = {str(k): v for k, v in (values.get(target) or {}).items()}

            # --- target labels for every respondent
            y_label = df[target].map(
                lambda c: label_of(values, c, vmap)) if target in df.columns \
                else None
            if y_label is None:
                gate_fail.append(f"{cluster}: target column missing")
                continue
            keep = y_label.isin(opts).to_numpy()

            # --- HARD GATE: anchor rows reproduce the ladder ground truth
            missing_ids = [p["resp_id"] for p in t["pairs"]
                           if p["resp_id"] not in id_pos]
            if missing_ids:
                gate_fail.append(
                    f"{cluster}: {len(missing_ids)} anchor ids not in "
                    f"microdata, e.g. {missing_ids[:3]}")
                continue
            mismatch = 0
            for p in t["pairs"]:
                got = y_label.iloc[id_pos[p["resp_id"]]]
                if got != p["gt_text"].strip():
                    mismatch += 1
            if mismatch:
                gate_fail.append(
                    f"{cluster}: {mismatch}/{len(t['pairs'])} anchor ground "
                    f"truths do not reproduce from microdata")
                continue

            # --- feature columns: text -> code, coverage-resolved
            feat_codes = []
            for q in sorted(t["qtexts"]):
                cands = [c for c in text2codes.get(q, []) if c in df.columns
                         and c != target]
                if not cands:
                    continue
                if len(cands) > 1:
                    cands.sort(key=lambda c: df[c].notna().sum(),
                               reverse=True)
                feat_codes.append(cands[0])
            if args.check:
                print(f"  ok {cluster}: {len(t['pairs'])} anchors matched, "
                      f"{len(feat_codes)}/{len(t['qtexts'])} features mapped, "
                      f"{int(keep.sum())} usable respondents")
                continue
            if len(feat_codes) < 10:
                gate_fail.append(f"{cluster}: only {len(feat_codes)} "
                                 f"features mapped")
                continue

            # --- matrix over usable respondents
            sub_idx = np.flatnonzero(keep)
            X = np.full((len(sub_idx), len(feat_codes)), np.nan)
            for j, code in enumerate(feat_codes):
                # raw values, NOT to_numeric: text-label sources (Asian
                # Barometer CSV) must encode as categories, not coerce
                # to NaN
                arr = df[code].iloc[sub_idx].to_numpy()
                vals: dict = {}
                enc = np.full(len(sub_idx), np.nan)
                for i, v in enumerate(arr):
                    if not (isinstance(v, float) and np.isnan(v)) \
                            and v is not None and v == v:
                        enc[i] = vals.setdefault(v, len(vals))
                X[:, j] = enc
            y = np.array([opts.index(l) for l in y_label.iloc[sub_idx]])
            grp = countries[sub_idx]
            row_of = {ids.iloc[i]: k for k, i in enumerate(sub_idx)}
            anchor_pos = [row_of[p["resp_id"]] for p in t["pairs"]]

            # --- regime A: grouped
            n_splits = min(5, len(np.unique(grp)))
            regimes = {}
            if n_splits >= 2 and len(np.unique(y)) >= 2:
                oof, fold = fit_oof(
                    X, y, GroupKFold(n_splits=n_splits).split(X, y, grp))
                regimes["grouped"] = (oof, fold, np.arange(len(y)))
            # --- regime B: within-country, only countries carrying anchors
            oof_w = np.full((len(y), len(opts)), np.nan)
            fold_w = np.full(len(y), -1)
            done_rows = []
            for c in sorted({grp[i] for i in anchor_pos}):
                rows = np.flatnonzero(grp == c)
                if len(rows) < 50 or len(np.unique(y[rows])) < 2:
                    continue
                kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
                oof_c, fold_c = fit_oof(X[rows], y[rows], kf.split(rows))
                oof_w[rows] = oof_c
                fold_w[rows] = fold_c
                done_rows.extend(rows.tolist())
            if done_rows:
                regimes["within"] = (oof_w, fold_w, np.array(done_rows))

            for regime, (oof, fold, valid) in regimes.items():
                valid_set = set(valid.tolist())
                # full-sample OOF accuracy (fitted rows with predictions)
                mask = np.zeros(len(y), bool)
                mask[list(valid_set)] = True
                mask &= ~np.isnan(oof).any(axis=1) & (fold >= 0)
                full_acc = float((oof[mask].argmax(1) == y[mask]).mean()) \
                    if mask.any() else float("nan")
                per_target_rows.append({
                    "cluster": cluster, "regime": regime,
                    "n_train_pool": int(mask.sum()),
                    "n_countries": len(np.unique(grp)),
                    "full_oof_acc": full_acc,
                })
                for p, k in zip(t["pairs"], anchor_pos):
                    if k not in valid_set or fold[k] < 0 \
                            or np.isnan(oof[k]).any():
                        continue
                    pvec = oof[k]
                    anchor_rows.append({
                        "example_id": p["example_id"], "regime": regime,
                        "cluster": cluster, "resp_id": p["resp_id"],
                        "opts": opts, "p": [float(x) for x in pvec],
                        "p_true": float(pvec[opts.index(p["gt_text"])]),
                        "max_p": float(pvec.max()),
                        "pred_text": opts[int(pvec.argmax())],
                        "gt_text": p["gt_text"], "fold": int(fold[k]),
                    })
            print(f"  fit {cluster}: pool {int(keep.sum())}, "
                  f"{len(feat_codes)} features, regimes {list(regimes)}")

    if gate_fail:
        for g in gate_fail:
            print("GATE FAIL", g)
        return 1
    if args.check:
        print("mapping check PASSED for all targets")
        return 0

    d = pd.DataFrame(anchor_rows)
    with open(OUT_DIR / "xgb_full_anchor_dump.jsonl", "w",
              encoding="utf-8", newline="\n") as fh:
        for rec in anchor_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pd.DataFrame(per_target_rows).to_csv(
        OUT_DIR / "xgb_full_per_target.csv", index=False)

    summary, split_rows = [], []
    for regime, g in d.groupby("regime"):
        g = g.copy()
        summary.append({
            "regime": regime, "n": len(g),
            "n_targets": g["cluster"].nunique(),
            "acc": float((g["pred_text"] == g["gt_text"]).mean()),
            "norm_acc": norm_acc_paper(g),
            "mean_max_p": float(g["max_p"].mean()),
        })
        for cluster, gc in g.groupby("cluster"):
            modal = gc["gt_text"].mode().iloc[0]
            is_modal = gc["gt_text"] == modal
            split_rows.append({
                "regime": regime, "target": cluster, "n": len(gc),
                "modal_share": float(is_modal.mean()),
                "acc_modal": float((gc.loc[is_modal, "pred_text"]
                                    == gc.loc[is_modal, "gt_text"]).mean()),
                "acc_dissenter": (
                    float((gc.loc[~is_modal, "pred_text"]
                           == gc.loc[~is_modal, "gt_text"]).mean())
                    if (~is_modal).any() else float("nan")),
            })
    # context rows: the floor and the model, from their pinned CSVs
    floor = pd.read_csv(OUT_DIR / "xgb_anchor_k24.csv")
    floor = floor[floor["target"] == "POOLED"].iloc[0]
    matched = pd.read_csv(OUT_DIR / "xgb_anchor_matched_model.csv").iloc[0]
    summary.append({"regime": "floor_small_cell(xgb_anchor_k24)",
                    "n": int(floor["n"]), "n_targets": int(floor["n_targets"]),
                    "acc": float(floor["acc"]),
                    "norm_acc": float(floor["norm_acc"]),
                    "mean_max_p": float(floor["mean_max_p"])})
    summary.append({"regime": "model_qwen3-32b(matched)",
                    "n": int(matched["n"]),
                    "n_targets": int(matched["n_targets"]),
                    "acc": float(matched["acc"]),
                    "norm_acc": float(matched["norm_acc"]),
                    "mean_max_p": float("nan")})
    pd.DataFrame(summary).to_csv(
        OUT_DIR / "xgb_full_anchor_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        OUT_DIR / "xgb_full_dissenter_split.csv", index=False)

    for r in summary:
        print(f"{r['regime']:<32} n={r['n']:<5} "
              f"acc={r['acc']:.4f} norm={r['norm_acc']:.4f}")
    sd = pd.DataFrame(split_rows)
    for regime, g in sd.groupby("regime"):
        print(f"dissenter [{regime}]: modal {g['acc_modal'].mean():.4f} vs "
              f"dissenter {g['acc_dissenter'].mean():.4f} "
              f"({len(g)} targets)")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
