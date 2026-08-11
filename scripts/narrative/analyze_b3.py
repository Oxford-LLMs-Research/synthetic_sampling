"""B3 narrative-presentation analysis, to the 8 Aug pre-registration.

Joins the scoring set (arm_label: qa / narrative1 / narrative2 sharing
base_id) to the scored results and writes the tables the pre-registration
names: per-condition accuracy (T0.1 M convention), paired within-base_id
deltas against the qa baseline with target-clustered bootstrap CIs, the
narrative1-vs-narrative2 contrast as the wording-variance ceiling, flip
rates for the form-beyond-wording test, per-question TV, and the replicate
ceiling.

    python scripts/narrative/analyze_b3.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this writes;
`verify_b3_numbers.py` re-checks them by script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_sampling.analysis import (
    clustered_bootstrap_ci,
    normalized_accuracy,
    replicate_agreement,
)

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT.parent / "analysis" / "narrative"

ARMS = ("label_num", "echo_plain")
CONDITIONS = ("qa", "narrative1", "narrative2")


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    """Join the scoring set (conditions, options) to the scored file."""
    inp = ROOT / "outputs" / "narrative" / "inputs" / "narrative_label_set.jsonl"
    res = ROOT / "outputs" / "narrative" / "results" / f"narrative_label_results_{tag}.jsonl"

    meta: dict[str, dict] = {}
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = {
            "condition": r["arm_label"],
            "base_id": r["base_id"],
            "survey": r["survey"],
            "target_code": r["target_code"],
            "ground_truth": r["ground_truth"],
            "options": r["option_sets"]["original"],
        }

    rows, raw = [], []
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        raw.append(r)
        m = meta[r["example_id"]]
        row = {
            "example_id": r["example_id"],
            "base_id": m["base_id"],
            "condition": m["condition"],
            "target": f"{m['survey']}|{m['target_code']}",
            "survey": m["survey"],
            "ground_truth": m["ground_truth"],
            "n_options": len(m["options"]),
        }
        for arm in ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, miss = None, True
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_correct"] = float(pred == m["ground_truth"]) if pred is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
    """M = distinct answers GIVEN per target, M >= 2 (T0.1 convention)."""
    base = df[df["condition"] == "qa"]
    return {t: g["ground_truth"].nunique() for t, g in base.groupby("target")}


def norm_acc(sub: pd.DataFrame, arm: str, m_map: dict[str, int]) -> float:
    vals = []
    for t, g in sub.groupby("target"):
        m = m_map.get(t, 0)
        if m < 2:
            continue
        vals.append(normalized_accuracy(float(g[f"{arm}_correct"].mean()), m))
    return float(np.mean(vals)) if vals else float("nan")


def tv_by_target(sub: pd.DataFrame, arm: str) -> float:
    vals = []
    for _, g in sub.groupby("target"):
        preds = g[f"{arm}_pred"].dropna()
        if preds.empty:
            continue
        labels = set(preds) | set(g["ground_truth"])
        p = Counter(preds)
        q = Counter(g["ground_truth"])
        np_, nq_ = len(preds), len(g)
        vals.append(0.5 * sum(abs(p[k] / np_ - q[k] / nq_) for k in labels))
    return float(np.mean(vals)) if vals else float("nan")


def paired(df: pd.DataFrame, arm: str, cond: str, baseline: str) -> dict:
    """correct(cond) - correct(baseline) per base_id, target-clustered CI;
    agree_rate is the fraction of pairs predicting the same option."""
    a = df[df["condition"] == cond].set_index("base_id")
    b = df[df["condition"] == baseline].set_index("base_id")
    shared = a.index.intersection(b.index)
    d = (a.loc[shared, f"{arm}_correct"] - b.loc[shared, f"{arm}_correct"]).astype(float)
    clusters = a.loc[shared, "target"].to_numpy()
    agree = (a.loc[shared, f"{arm}_pred"] == b.loc[shared, f"{arm}_pred"]).mean()
    v = d.to_numpy()
    lo, hi = clustered_bootstrap_ci(v, clusters)
    per_cluster = pd.Series(v).groupby(clusters).mean()
    return {
        "n_pairs": int(len(shared)),
        "delta_acc": float(per_cluster.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "flip_rate": float(1.0 - agree),
        "agree_rate": float(agree),
    }


def run(tag: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, raw = load(tag)
    m_map = m_by_target(df)
    print(f"{len(df)} scored rows ({df['base_id'].nunique()} base_ids x "
          f"{df['condition'].nunique()} conditions + replicates), "
          f"{df['target'].nunique()} targets, "
          f"{sum(1 for m in m_map.values() if m < 2)} dropped for M<2")

    levels, contrasts = [], []
    for arm in ARMS:
        for cond in CONDITIONS:
            g = df[df["condition"] == cond]
            levels.append({
                "model": tag, "arm": arm, "condition": cond, "n": len(g),
                "miss_rate": float(g[f"{arm}_miss"].mean()),
                "acc_raw": float(g[f"{arm}_correct"].mean()),
                "norm_acc": norm_acc(g, arm, m_map),
                "tv": tv_by_target(g, arm),
            })
        # Endpoint 1: paired narrative-minus-qa, each narrative separately.
        for cond in ("narrative1", "narrative2"):
            contrasts.append({
                "model": tag, "arm": arm, "contrast": f"{cond}-qa",
                **paired(df, arm, cond, "qa"),
            })
        # Endpoint 2: narrative1 vs narrative2, the wording-variance ceiling.
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "narrative1-narrative2",
            **paired(df, arm, "narrative1", "narrative2"),
        })
        # Replicate ceiling for this serving (label read against the 64.4%
        # cross-serving floor, not 100%).
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })

    pd.DataFrame(levels).to_csv(OUTDIR / f"b3_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUTDIR / f"b3_contrasts_{tag}.csv", index=False)
    print(f"wrote {OUTDIR / f'b3_levels_{tag}.csv'}")
    print(f"wrote {OUTDIR / f'b3_contrasts_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
