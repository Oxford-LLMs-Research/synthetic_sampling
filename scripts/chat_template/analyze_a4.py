"""A4 chat-template analysis, to the 12 Aug pre-registration.

Joins B3's scoring set (arm_label: qa / narrative1 / narrative2 sharing
base_id) to the A4 results and writes: per-condition levels for BOTH label
readouts (raw label_num and chat_label_num through the model's own
template) with T0.1 norm_acc, TV, mean confidence and 10-bin ECE — the
pinned distributional outcomes; the presentation contrasts per readout
(pre-registered: narrative1-qa replicates within +-0.02 under the
template; a flip beyond +-0.04 confounds the raw-completion instrument);
the paired within-serving TEMPLATE contrast (chat minus raw on the same
instance, per condition — the template tax, descriptive); and the
replicate ceiling.

    python scripts/chat_template/analyze_a4.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this writes;
`verify_a4_numbers.py` re-checks them by script.
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
OUTDIR = ROOT.parent / "analysis" / "chat_template"

ARMS = ("label_num", "chat_label_num", "echo_plain")
LABEL_ARMS = ("label_num", "chat_label_num")
CONDITIONS = ("qa", "narrative1", "narrative2")


def _softmax_conf(scores: dict) -> float:
    v = np.array(list(scores.values()), dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return float("nan")
    e = np.exp(v - v.max())
    return float((e / e.sum()).max())


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    inp = ROOT / "outputs" / "narrative" / "inputs" / "narrative_label_set.jsonl"
    res = ROOT / "outputs" / "chat_template" / "results" / f"a4_label_results_{tag}.jsonl"

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
            pred, miss, conf = None, True, float("nan")
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
                if arm in LABEL_ARMS:
                    conf = _softmax_conf(scores)
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_correct"] = float(pred == m["ground_truth"]) if pred is not None else np.nan
            if arm in LABEL_ARMS:
                row[f"{arm}_conf"] = conf
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
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


def paired(df: pd.DataFrame, arm: str, cond: str, baseline: str) -> dict:
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


def template_contrast(df: pd.DataFrame, cond: str) -> dict:
    """chat_label_num minus label_num on the SAME instance (same serving)."""
    g = df[df["condition"] == cond]
    d = (g["chat_label_num_correct"] - g["label_num_correct"]).astype(float)
    keep = d.notna()
    v = d[keep].to_numpy()
    clusters = g.loc[keep, "target"].to_numpy()
    agree = (g.loc[keep, "chat_label_num_pred"]
             == g.loc[keep, "label_num_pred"]).mean()
    lo, hi = clustered_bootstrap_ci(v, clusters)
    per_cluster = pd.Series(v).groupby(clusters).mean()
    return {
        "n_pairs": int(keep.sum()),
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
    print(f"{len(df)} scored rows ({df['base_id'].nunique()} base_ids), "
          f"{df['target'].nunique()} targets")

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
                "mean_conf": (float(g[f"{arm}_conf"].mean())
                              if arm in LABEL_ARMS else float("nan")),
                "ece": ece(g, arm) if arm in LABEL_ARMS else float("nan"),
            })
        # Presentation contrasts per readout (pre-reg endpoint 1).
        for cond, base in (("narrative1", "qa"), ("narrative2", "qa"),
                           ("narrative1", "narrative2")):
            contrasts.append({
                "model": tag, "arm": arm, "contrast": f"{cond}-{base}",
                **paired(df, arm, cond, base),
            })
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })

    # The template tax, paired within serving (descriptive, no falsifier).
    for cond in CONDITIONS:
        contrasts.append({
            "model": tag, "arm": "chat-vs-raw",
            "contrast": f"template|{cond}",
            **template_contrast(df, cond),
        })

    pd.DataFrame(levels).to_csv(OUTDIR / f"a4_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUTDIR / f"a4_contrasts_{tag}.csv", index=False)
    for stem in ("levels", "contrasts"):
        print(f"wrote {OUTDIR / f'a4_{stem}_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
