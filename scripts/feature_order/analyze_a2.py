"""A2 feature-order analysis, to the 13 Aug pre-registration (as
tightened by review the same day; locks in PAPER_STATE).

Written BEFORE the jobs land (the A3 pattern). The locked estimands:

- CO-PRIMARY, per arm, paired on base_id within one serving:
  (a) accuracy deltas for informative_last - informative_first and
      shuffled - informative_first (target-clustered bootstrap CI;
      order null within +-0.02, falsifier +-0.04), and
  (b) FLIP RATES for the same contrasts, read against the replicate
      ceiling (B3's lesson: a null delta can hide 17-28% flips).
- shuffled - informative_last is reported as a secondary contrast (it
  separates "position of the top features" from "any deviation from
  rank order") - no band was pre-registered for it.
- A null keeps the LADDER default (informative-first). No survey-order
  cell exists, so nothing here licenses "natural order" language.

    python scripts/feature_order/analyze_a2.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this
writes; a verify_a2_numbers.py pins them at landing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_sampling.analysis import (
    clustered_bootstrap_ci,
    normalized_accuracy,
    replicate_agreement,
)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT.parent / "analysis" / "feature_order"

ARMS = ("label_num", "echo_plain")
CELLS = ("informative_first", "informative_last", "shuffled")
CONTRASTS = (
    ("last-first", "informative_last", "informative_first"),
    ("shuf-first", "shuffled", "informative_first"),
    ("shuf-last", "shuffled", "informative_last"),
)


def _softmax(scores: dict) -> dict[str, float]:
    keys = list(scores)
    v = np.array([scores[k] for k in keys], dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return {}
    e = np.exp(v - v.max())
    p = e / e.sum()
    return dict(zip(keys, p.astype(float)))


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    inp = ROOT / "outputs" / "feature_order" / "inputs" / "a2_order_set.jsonl"
    res = (ROOT / "outputs" / "feature_order" / "results"
           / f"a2_order_results_{tag}.jsonl")

    meta: dict[str, dict] = {}
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    rows, raw = [], []
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        raw.append(r)
        m = meta[r["example_id"]]
        row = {
            "example_id": r["example_id"],
            "base_id": m["base_id"],
            "cell": m["arm_label"],
            "target": f"{m['survey']}|{m['target_code']}",
            "survey": m["survey"],
            "ground_truth": m["ground_truth"],
        }
        for arm in ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, miss, conf = None, True, np.nan
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
                probs = _softmax(scores)
                conf = max(probs.values()) if probs else np.nan
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_conf"] = conf
            row[f"{arm}_correct"] = (
                np.nan if pred is None
                else float(pred == m["ground_truth"]))
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
    """Distinct-M convention (as T0.1/XGB): distinct ground truths."""
    base = df[df["cell"] == "informative_first"]
    return {t: g["ground_truth"].nunique() for t, g in base.groupby("target")}


def norm_acc(sub: pd.DataFrame, arm: str, m_map: dict[str, int]) -> float:
    vals = []
    for t, g in sub.groupby("target"):
        m = m_map.get(t, 0)
        if m < 2:
            continue
        acc = g[f"{arm}_correct"].mean()
        if np.isfinite(acc):
            vals.append(normalized_accuracy(float(acc), m))
    return float(np.mean(vals)) if vals else float("nan")


def ece(sub: pd.DataFrame, arm: str, n_bins: int = 10) -> float:
    g = sub.dropna(subset=[f"{arm}_conf", f"{arm}_correct"])
    if g.empty:
        return float("nan")
    conf = g[f"{arm}_conf"].to_numpy()
    corr = g[f"{arm}_correct"].to_numpy()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(out)


def paired_contrast(df: pd.DataFrame, arm: str,
                    cell_a: str, cell_b: str) -> dict:
    """cell_a minus cell_b, paired on base_id; delta and flips co-primary."""
    a = df[df["cell"] == cell_a].set_index("base_id")
    b = df[df["cell"] == cell_b].set_index("base_id")
    shared = a.index.intersection(b.index)
    da = a.loc[shared, f"{arm}_correct"].astype(float)
    db = b.loc[shared, f"{arm}_correct"].astype(float)
    keep = da.notna() & db.notna()
    d = (da - db)[keep]
    clusters = a.loc[shared, "target"].to_numpy()[keep.to_numpy()]
    agree = (a.loc[shared, f"{arm}_pred"][keep]
             == b.loc[shared, f"{arm}_pred"][keep]).mean()
    lo, hi = clustered_bootstrap_ci(d.to_numpy(), clusters)
    per_cluster = pd.Series(d.to_numpy()).groupby(clusters).mean()
    return {
        "n_pairs": int(keep.sum()),
        "delta_acc": float(per_cluster.mean()),
        "ci_lo": lo, "ci_hi": hi,
        "flip_rate": float(1.0 - agree),
        "agree_rate": float(agree),
    }


def run(tag: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, raw = load(tag)
    m_map = m_by_target(df)
    print(f"{len(df)} scored rows ({df['base_id'].nunique()} base_ids), "
          f"{df['target'].nunique()} targets")

    levels, contrasts = [], []
    for arm in ARMS:
        for cell in CELLS:
            g = df[df["cell"] == cell]
            levels.append({
                "model": tag, "arm": arm, "cell": cell, "n": len(g),
                "miss_rate": float(g[f"{arm}_miss"].mean()),
                "acc_raw": float(g[f"{arm}_correct"].mean()),
                "norm_acc": norm_acc(g, arm, m_map),
                "mean_conf": float(g[f"{arm}_conf"].mean()),
                "ece": ece(g, arm),
            })
        for name, cell_a, cell_b in CONTRASTS:
            contrasts.append({
                "model": tag, "arm": arm, "contrast": name,
                **paired_contrast(df, arm, cell_a, cell_b),
            })
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })

    pd.DataFrame(levels).to_csv(OUTDIR / f"a2_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(
        OUTDIR / f"a2_contrasts_{tag}.csv", index=False)
    for stem in ("levels", "contrasts"):
        print(f"wrote {OUTDIR / f'a2_{stem}_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
