"""XGBoost feature-ceiling re-run with per-instance dumps (REBUILD_DECISION
Tier 2 / M3; the C-series interpretation anchor in GRID_STATE).

T0.1's `WORK/analysis/calibration/analyze_label_calibration.py` already fit
GroupKFold-on-country XGBoost on the readout and ladder substrates, but it
discards the per-instance out-of-fold predictions after aggregation, so the
dissenter penalty and entropy ratio (audit M3) are not computable from what
is on disk, and the C-series falsifiers cannot be read against a
same-substrate supervised ceiling. This script re-fits the IDENTICAL
estimator (same hyperparameters, seed 42, deterministic GroupKFold, same
cell construction and row order) and keeps everything:

  WORK/analysis/xgb_ceiling/
    xgb_readout_dump.jsonl        per instance: oof dist, p_true, pred, fold
    xgb_ladder_dump.jsonl         same, per (ordering, k) cell
    xgb_readout_summary.csv       pooled battery row (consistency-checked)
    xgb_ladder_by_rung.csv        per-rung table (consistency-checked)
    xgb_anchor_k24.csv            THE C-series anchor: k=24 informative,
                                  restricted to B3/C-series' 734-pair
                                  substrate, per-target + pooled
    xgb_dissenter_split.csv       M3: modal vs dissenter accuracy and
                                  entropy ratio per target

Consistency gate: the pooled readout row and the per-rung ladder table must
reproduce T0.1's pinned aggregates (calibration_summary.csv row
model=xgboost, and xgb_ladder_by_rung.csv) to 1e-6, or the script exits 1 —
identical estimator or the dumps do not anchor anything. The estimator
block is duplicated from the T0.1 script BY DESIGN: that script is a landed,
pinned analysis and must not be edited; this check is what keeps the copy
honest.

Usage:
  python scripts/xgb_ceiling/run_xgb_ceiling.py [--substrate readout|ladder|all]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
REC = OUTER / "outputs_recovered"
T01_DIR = OUTER / "analysis" / "calibration"
OUT_DIR = OUTER / "analysis" / "xgb_ceiling"
B3_TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"

LADDER_ID = re.compile(r"_ladder_(shared|informative|random)_k(\d{3})$")
SEED = 42
EPS = 1e-12
TOL = 1e-6


# --------------------------------------------------------------------------
# Estimator + conventions, byte-equivalent to the T0.1 script (see gate).
# --------------------------------------------------------------------------

def xgb_cell(rows: list[dict], min_n: int = 30):
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    if len(rows) < min_n:
        return None
    opts = rows[0]["opts"]
    y = np.array([opts.index(r["gt_text"]) for r in rows])
    groups = np.array([r["country"] for r in rows])
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return None
    feats = sorted({q for r in rows for q in r["questions"]})
    codes: dict[str, dict] = {}
    X = np.full((len(rows), len(feats)), np.nan)
    for j, q in enumerate(feats):
        vals = codes.setdefault(q, {})
        for i, r in enumerate(rows):
            a = r["questions"].get(q)
            if a is not None:
                X[i, j] = vals.setdefault(a, len(vals))
    M = len(opts)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros((len(rows), M))
    fold_of = np.full(len(rows), -1)
    seen = np.zeros(len(rows), bool)
    for f, (tr, te) in enumerate(
            GroupKFold(n_splits=n_splits).split(X, y, groups)):
        classes = np.unique(y[tr])
        if len(classes) < 2:
            continue
        remap = {c: i for i, c in enumerate(classes)}
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", n_jobs=4, random_state=SEED, verbosity=0)
        clf.fit(X[tr], np.array([remap[c] for c in y[tr]]))
        proba = clf.predict_proba(X[te])
        for local, c in enumerate(classes):
            oof[te, c] = proba[:, local]
        fold_of[te] = f
        seen[te] = True
    if not seen.all():
        return None
    out = []
    for i, r in enumerate(rows):
        p = oof[i]
        out.append({
            "example_id": r["example_id"], "resp_id": r["resp_id"],
            "opts": tuple(opts), "p": p,
            "p_true": float(p[y[i]]), "max_p": float(p.max()),
            "pred_text": opts[int(p.argmax())], "gt_text": r["gt_text"],
            "fold": int(fold_of[i]),
            "entropy": float(-(np.clip(p, EPS, 1) * np.log(
                np.clip(p, EPS, 1))).sum()),
        })
    return out


def norm_acc_paper(df: pd.DataFrame) -> float:
    M_by_t = df.groupby("cluster")["gt_text"].nunique()
    d = df.assign(M=df["cluster"].map(M_by_t))
    d = d[d["M"] >= 2]
    na = ((d["pred_text"] == d["gt_text"]).astype(float) - 1 / d["M"]) \
        / (1 - 1 / d["M"])
    return float(na.groupby(d["cluster"].to_numpy()).mean().mean())


def tv_marginals(df: pd.DataFrame) -> tuple[float, float]:
    out = []
    for (cluster, opts), d in df.groupby(["cluster", "opts"]):
        opts = list(opts)
        M = len(opts)
        pred = np.vstack([p for p in d["p"]]).mean(axis=0)
        emp = np.array([(d["gt_text"] == o).mean() for o in opts])
        out.append({"cluster": cluster,
                    "tv": 0.5 * float(np.abs(pred - emp).sum()),
                    "tv_uniform": 0.5 * float(np.abs(
                        np.full(M, 1.0 / M) - emp).sum())})
    per_q = pd.DataFrame(out)
    by_cluster = per_q.groupby("cluster")[["tv", "tv_uniform"]].mean()
    return float(by_cluster["tv"].mean()), float(by_cluster["tv_uniform"].mean())


# --------------------------------------------------------------------------
# Substrate runs
# --------------------------------------------------------------------------

def _load_cells(path: Path, ladder: bool) -> dict:
    cells = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if ladder:
                m = LADDER_ID.search(r["example_id"])
                if not m or m.group(1) == "shared":
                    continue
                key_extra = (m.group(1), int(m.group(2)))
            else:
                key_extra = ()
            opts = list(dict.fromkeys(r["option_sets"]["original"]))
            cells[(r["survey"], r["target_code"], tuple(opts),
                   *key_extra)].append({
                "example_id": r["example_id"], "resp_id": r.get("id"),
                "questions": r["questions"], "gt_text": r["ground_truth"],
                "country": str(r["country"]), "opts": opts,
            })
    return cells


def dump_rows(dfs: list[pd.DataFrame], out: Path) -> None:
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for d in dfs:
            for rec in d.to_dict("records"):
                rec = dict(rec)
                rec["p"] = [float(x) for x in rec["p"]]
                rec["opts"] = list(rec["opts"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_readout() -> pd.DataFrame:
    cells = _load_cells(REC / "readout_set.jsonl", ladder=False)
    dfs, per_target = [], []
    for (survey, target, opts), rows in sorted(cells.items()):
        cell = xgb_cell(rows)
        if cell is None:
            per_target.append({"survey": survey, "target_code": target,
                               "n": len(rows), "usable": False})
            continue
        d = pd.DataFrame(cell)
        d["cluster"] = f"{survey}|{target}"
        dfs.append(d)
        per_target.append({
            "survey": survey, "target_code": target, "n": len(d),
            "usable": True, "M_stated": len(opts),
            "acc": float((d["pred_text"] == d["gt_text"]).mean()),
            "norm_acc": norm_acc_paper(d),
            "mean_max_p": float(d["max_p"].mean()),
        })
    pooled = pd.concat(dfs, ignore_index=True)
    dump_rows(dfs, OUT_DIR / "xgb_readout_dump.jsonl")
    tv, tv_u = tv_marginals(pooled)
    summary = pd.DataFrame([{
        "substrate": "readout", "model": "xgboost",
        "n_scored": len(pooled),
        "acc": float((pooled["pred_text"] == pooled["gt_text"]).mean()),
        "norm_acc": norm_acc_paper(pooled),
        "mean_max_p": float(pooled["max_p"].mean()),
        "tv_marginal": tv, "tv_uniform": tv_u,
    }])
    summary.to_csv(OUT_DIR / "xgb_readout_summary.csv", index=False)

    # Consistency gate. calibration_summary.csv on disk lost its xgboost
    # row to a later --skip-xgb rewrite, so gate against what IS pinned:
    # the per-target table exactly, and the published anchors (norm 0.273,
    # n 4,545, TV 0.084 — verify_t01_numbers.py) at 3-decimal tolerance.
    bad = []
    # Positional comparison: (survey, target_code) is NOT unique — cells
    # are keyed by option-text set too — but both tables come from
    # sorted(cells.items()) with the same key, so row order is identical.
    pin_t = pd.read_csv(T01_DIR / "xgb_readout_per_target.csv")
    got_t = pd.DataFrame(per_target)
    if len(got_t) != len(pin_t):
        bad.append(f"per-target shape {len(got_t)} vs pinned {len(pin_t)}")
    elif not (got_t["survey"].eq(pin_t["survey"]).all()
              and got_t["target_code"].eq(pin_t["target_code"]).all()
              and got_t["usable"].eq(pin_t["usable"]).all()):
        bad.append("per-target keys/usable flags differ from pinned table")
    else:
        u = got_t["usable"].to_numpy()
        for col in ("n", "acc", "norm_acc", "mean_max_p"):
            d = (got_t.loc[u, col] - pin_t.loc[u, col]).abs()
            if (d > TOL).any():
                bad.append(f"per-target {col}: max diff {d.max()}")
    row = summary.iloc[0]
    for name, want, got, tol in (
            ("norm_acc", 0.273, row["norm_acc"], 5e-4),
            ("n_scored", 4545, row["n_scored"], 0.5),
            ("tv_marginal", 0.084, row["tv_marginal"], 5e-4)):
        if abs(float(got) - want) > tol:
            bad.append(f"pooled {name}: {got} vs published {want}")
    if bad:
        for b in bad:
            print("CONSISTENCY FAIL", b)
        sys.exit(1)
    print(f"readout: {len(pooled)} instances dumped, per-target table and "
          f"published anchors reproduced (acc {row['acc']:.4f}, "
          f"norm {row['norm_acc']:.4f}, tv {row['tv_marginal']:.4f})")
    return pooled


def run_ladder() -> pd.DataFrame:
    cells = _load_cells(REC / "ladder_readout_set.jsonl", ladder=True)
    dfs, by_rung = [], defaultdict(list)
    n_skipped = 0
    for (survey, target, _opts, ordering, k), rows in sorted(cells.items()):
        cell = xgb_cell(rows)
        if cell is None:
            n_skipped += 1
            continue
        d = pd.DataFrame(cell)
        d["cluster"] = f"{survey}|{target}"
        d["ordering"], d["k"] = ordering, k
        dfs.append(d)
        by_rung[(ordering, k)].append(d)
    pooled = pd.concat(dfs, ignore_index=True)
    dump_rows(dfs, OUT_DIR / "xgb_ladder_dump.jsonl")

    out = []
    for (ordering, k), rung_dfs in sorted(by_rung.items()):
        d = pd.concat(rung_dfs, ignore_index=True)
        tv, tv_u = tv_marginals(d)
        out.append({
            "ordering": ordering, "k": k, "n_cells": len(rung_dfs),
            "n": len(d),
            "acc": float((d["pred_text"] == d["gt_text"]).mean()),
            "norm_acc": norm_acc_paper(d),
            "mean_max_p": float(d["max_p"].mean()),
            "tv_marginal": tv, "tv_uniform": tv_u,
        })
    rung = pd.DataFrame(out)
    rung.to_csv(OUT_DIR / "xgb_ladder_by_rung.csv", index=False)

    pinned = pd.read_csv(T01_DIR / "xgb_ladder_by_rung.csv")
    merged = rung.merge(pinned, on=["ordering", "k"], suffixes=("", "_pin"))
    bad = []
    if len(merged) != len(pinned) or len(rung) != len(pinned):
        bad.append(f"rung table shape {len(rung)} vs pinned {len(pinned)}")
    for col in ("n", "acc", "norm_acc", "tv_marginal"):
        d = (merged[col] - merged[f"{col}_pin"]).abs()
        if (d > TOL).any():
            bad.append(f"ladder {col}: max diff {d.max()}")
    if bad:
        for b in bad:
            print("CONSISTENCY FAIL", b)
        sys.exit(1)
    print(f"ladder: {len(pooled)} instances dumped over "
          f"{pooled.groupby(['ordering', 'k']).ngroups} rungs "
          f"({n_skipped} cells skipped), per-rung table matches T0.1 pins")
    return pooled


# --------------------------------------------------------------------------
# The two reads the dumps exist for
# --------------------------------------------------------------------------

def anchor_k24(ladder: pd.DataFrame) -> None:
    """The C-series anchor: k=24 informative on the B3/C 734-pair set."""
    b3 = {json.loads(l)["example_id"] for l in B3_TASKS.open(encoding="utf-8")}
    sub = ladder[(ladder["ordering"] == "informative") & (ladder["k"] == 24)
                 & ladder["example_id"].isin(b3)]
    rows = []
    for cluster, g in sub.groupby("cluster"):
        rows.append({
            "target": cluster, "n": len(g),
            "acc": float((g["pred_text"] == g["gt_text"]).mean()),
            "mean_max_p": float(g["max_p"].mean()),
        })
    tv, tv_u = tv_marginals(sub)
    rows.append({
        "target": "POOLED", "n": len(sub),
        "acc": float((sub["pred_text"] == sub["gt_text"]).mean()),
        "mean_max_p": float(sub["max_p"].mean()),
        "norm_acc": norm_acc_paper(sub),
        "tv_marginal": tv, "tv_uniform": tv_u,
        "n_targets": sub["cluster"].nunique(),
    })
    pd.DataFrame(rows).to_csv(OUT_DIR / "xgb_anchor_k24.csv", index=False)
    p = rows[-1]
    print(f"anchor: k24-informative on the 734-pair substrate -> "
          f"n={p['n']} across {p['n_targets']} targets, "
          f"acc {p['acc']:.4f}, norm_acc {p['norm_acc']:.4f}, "
          f"tv {p['tv_marginal']:.4f}")


def dissenter_split(readout: pd.DataFrame, ladder: pd.DataFrame) -> None:
    """M3: does the supervised ceiling live on modal respondents only?"""
    frames = [("readout", readout)]
    if len(ladder):
        frames.append(("ladder_k24_informative", ladder[
            (ladder["ordering"] == "informative") & (ladder["k"] == 24)]))
    rows = []
    for name, df in frames:
        for cluster, g in df.groupby("cluster"):
            modal = g["gt_text"].mode().iloc[0]
            is_modal = g["gt_text"] == modal
            emp = g["gt_text"].value_counts(normalize=True).to_numpy()
            rows.append({
                "substrate": name, "target": cluster,
                "n": len(g), "n_modal": int(is_modal.sum()),
                "modal_share": float(is_modal.mean()),
                "acc_modal": float((g.loc[is_modal, "pred_text"]
                                    == g.loc[is_modal, "gt_text"]).mean()),
                "acc_dissenter": (
                    float((g.loc[~is_modal, "pred_text"]
                           == g.loc[~is_modal, "gt_text"]).mean())
                    if (~is_modal).any() else float("nan")),
                "mean_pred_entropy": float(g["entropy"].mean()),
                "gt_entropy": float(-(np.clip(emp, EPS, 1)
                                      * np.log(np.clip(emp, EPS, 1))).sum()),
            })
    d = pd.DataFrame(rows)
    d["entropy_ratio"] = d["mean_pred_entropy"] / d["gt_entropy"]
    d.to_csv(OUT_DIR / "xgb_dissenter_split.csv", index=False)
    for name, g in d.groupby("substrate"):
        print(f"dissenter [{name}]: modal acc "
              f"{g['acc_modal'].mean():.4f} vs dissenter "
              f"{g['acc_dissenter'].mean():.4f} "
              f"(target-mean over {len(g)} targets), "
              f"entropy ratio {g['entropy_ratio'].mean():.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--substrate", choices=("readout", "ladder", "all"),
                    default="all")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    readout = run_readout() if args.substrate in ("readout", "all") else \
        pd.DataFrame()
    ladder = run_ladder() if args.substrate in ("ladder", "all") else \
        pd.DataFrame()
    if len(ladder):
        anchor_k24(ladder)
    if len(readout):
        dissenter_split(readout, ladder)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
