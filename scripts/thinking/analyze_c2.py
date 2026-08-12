"""C2 thinking-toggle analysis, to the 12 Aug pre-registration.

Joins the assembled toggle set (arm_label: direct / thinking sharing
base_id) to the scored results and writes: per-condition levels for both
label readouts (raw label_num and chat_label_num, both scored with
thinking OFF) with T0.1 norm_acc, TV, mean confidence, 10-bin ECE; the
paired thinking-minus-direct contrast per readout (prediction within
+-0.02, falsifier +0.04); the replicate ceiling; and the elicitation
table from the stage-2 sidecar (parse outcomes, stated-answer accuracy,
stated-vs-label agreement, loop/empty/truncation census against C1's
pathology baseline).

    python scripts/thinking/analyze_c2.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this writes;
`verify_c2_numbers.py` re-checks them by script.
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
OUTDIR = ROOT.parent / "analysis" / "thinking"

ARMS = ("label_num", "chat_label_num")
CONDITIONS = ("direct", "thinking")
PARSE_OK = ("digit", "bare_digit", "option_text")


def _softmax_conf(scores: dict) -> float:
    v = np.array(list(scores.values()), dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return float("nan")
    e = np.exp(v - v.max())
    return float((e / e.sum()).max())


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    inp = ROOT / "outputs" / "thinking" / "inputs" / f"c2_label_set_{tag}.jsonl"
    res = ROOT / "outputs" / "thinking" / "results" / f"c2_label_results_{tag}.jsonl"

    meta: dict[str, dict] = {}
    for line in inp.open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = {
            "condition": r["arm_label"],
            "base_id": r["base_id"],
            "survey": r["survey"],
            "target_code": r["target_code"],
            "ground_truth": r["ground_truth"],
            "ground_truth_index": r["ground_truth_index"],
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
            "ground_truth_index": m["ground_truth_index"],
            "n_options": len(m["options"]),
        }
        for arm in ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, pred_idx, miss, conf = None, None, True, float("nan")
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
                pred_idx = cell.get("predicted_index")
                conf = _softmax_conf(scores)
            row[f"{arm}_pred"] = pred
            row[f"{arm}_pred_idx"] = pred_idx
            row[f"{arm}_miss"] = miss
            row[f"{arm}_correct"] = float(pred == m["ground_truth"]) if pred is not None else np.nan
            row[f"{arm}_conf"] = conf
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_by_target(df: pd.DataFrame) -> dict[str, int]:
    base = df[df["condition"] == "direct"]
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


def elicitation(tag: str, df: pd.DataFrame) -> list[dict]:
    """Sidecar census + stated answer vs the two injected label readouts."""
    sc = pd.read_csv(ROOT / "outputs" / "thinking" / "inputs"
                     / f"c2_label_set_{tag}_parse.csv")
    lab = df[df["condition"] == "thinking"].set_index("example_id")
    g = sc.copy()
    g["scored_id"] = g["example_id"] + "_ton"
    ok = g[g["parse"] != "generation_error"]
    parsed = ok[ok["parse"].isin(PARSE_OK)].dropna(subset=["stated_index"])
    j = parsed.join(lab, on="scored_id", rsuffix="_lab")
    stated = j["stated_index"].astype(float)
    think_words = pd.to_numeric(ok["think_words"], errors="coerce")
    loops = pd.to_numeric(ok["loop_markers"], errors="coerce")
    row = {
        "model": tag,
        "n_transcripts": int(len(g)),
        "n_generation_error": int((g["parse"] == "generation_error").sum()),
        "block_rate": float((ok["has_block"] == True).mean()),  # noqa: E712
        "closed_rate": float((ok["closed"] == True).mean()),  # noqa: E712
        "empty_rate": float(((ok["has_block"] == True)  # noqa: E712
                             & (think_words == 0)).mean()),
        "loop_rate": float((loops > 1).mean()),
        "parse_fail_rate": float(1.0 - len(parsed) / len(ok)),
        "median_think_words": float(think_words.median()),
        "stated_acc": float((stated == j["ground_truth_index"]).mean()),
        "stated_vs_label_agree": float(
            (stated == j["label_num_pred_idx"]).mean()),
        "stated_vs_chat_label_agree": float(
            (stated == j["chat_label_num_pred_idx"]).mean()),
        "n_parsed": int(len(j)),
    }
    return [row]


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
                "mean_conf": float(g[f"{arm}_conf"].mean()),
                "ece": ece(g, arm),
            })
        # Endpoint: thinking-minus-direct (within +-0.02, falsifier +0.04).
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "thinking-direct",
            **paired(df, arm, "thinking", "direct"),
        })
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })

    elic = elicitation(tag, df)

    pd.DataFrame(levels).to_csv(OUTDIR / f"c2_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUTDIR / f"c2_contrasts_{tag}.csv", index=False)
    pd.DataFrame(elic).to_csv(OUTDIR / f"c2_elicitation_{tag}.csv", index=False)
    for stem in ("levels", "contrasts", "elicitation"):
        print(f"wrote {OUTDIR / f'c2_{stem}_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
