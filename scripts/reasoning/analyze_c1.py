"""C1 reason-then-answer analysis, to the 8 Aug pre-registration.

Joins the assembled 2x2 scoring set (arm_label: qa / reasoned /
narrative_direct / narrative_reasoned sharing base_id) to the scored
results and writes the pre-registered tables: per-condition levels (T0.1 M
convention, plus label_num calibration: mean confidence and 10-bin ECE),
paired within-base_id contrasts with target-clustered bootstrap CIs
(reasoned-qa is endpoint 1: prediction within +-0.02, falsifier +0.04),
the 2x2 interaction (nr-nd)-(r-qa) over completed quads (non-additivity
signature beyond +-0.02), the replicate ceiling, and the elicitation table
(parse outcomes from the stage-2 sidecar — failures are DATA — stated-answer
accuracy, and stated-vs-label agreement).

    python scripts/reasoning/analyze_c1.py --tag qwen_qwen3-32b

Every number quoted anywhere else must come out of the CSVs this writes;
`verify_c1_numbers.py` re-checks them by script.
"""

from __future__ import annotations

import argparse
import json
import re
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
OUTDIR = ROOT.parent / "analysis" / "reasoning"

ARMS = ("label_num", "echo_plain")
CONDITIONS = ("qa", "reasoned", "narrative_direct", "narrative_reasoned")
# Sidecar cell -> the reasoned instance its transcript feeds.
PARSE_OK = ("digit", "option_text")

# Degenerate-loop census (12 Aug, post-hoc but descriptive): a transcript
# with >= 2 "Final answer" markers looped — restated its answer instead of
# stopping. The parse protocol reads the LAST marker so loops are invisible
# in the parse table; this table makes them countable, and the clean/looped
# contrast splits show whether endpoint 1 depends on them.
MARKER = re.compile(r"(?im)^[ \t]*final answer\s*[:\-]", re.MULTILINE)
DIGIT = re.compile(r"(?i)final answer\s*[:\-]?\s*(?:option\s*)?(\d+)")
MIN_SPLIT = 20  # emit clean/looped contrast rows only above this many pairs


def _softmax_conf(scores: dict) -> float:
    v = np.array(list(scores.values()), dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return float("nan")
    e = np.exp(v - v.max())
    return float((e / e.sum()).max())


def load(tag: str) -> tuple[pd.DataFrame, list[dict]]:
    """Join the scoring set (conditions, options) to the scored file."""
    inp = ROOT / "outputs" / "reasoning" / "inputs" / f"c1_label_set_{tag}.jsonl"
    res = ROOT / "outputs" / "reasoning" / "results" / f"c1_label_results_{tag}.jsonl"

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
                if arm == "label_num":
                    conf = _softmax_conf(scores)
            row[f"{arm}_pred"] = pred
            row[f"{arm}_pred_idx"] = pred_idx
            row[f"{arm}_miss"] = miss
            row[f"{arm}_correct"] = float(pred == m["ground_truth"]) if pred is not None else np.nan
            if arm == "label_num":
                row["label_conf"] = conf
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


def ece(sub: pd.DataFrame, n_bins: int = 10) -> float:
    """10-bin equal-width ECE of label_num confidence vs correctness."""
    g = sub.dropna(subset=["label_conf", "label_num_correct"])
    if g.empty:
        return float("nan")
    conf = g["label_conf"].to_numpy()
    corr = g["label_num_correct"].to_numpy()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(out)


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


def interaction(df: pd.DataFrame, arm: str) -> dict:
    """(narrative_reasoned - narrative_direct) - (reasoned - qa) per quad."""
    piv = df.pivot_table(index="base_id", columns="condition",
                         values=f"{arm}_correct", aggfunc="first")
    piv = piv.dropna(subset=list(CONDITIONS))
    tmap = df[df["condition"] == "qa"].set_index("base_id")["target"]
    v = ((piv["narrative_reasoned"] - piv["narrative_direct"])
         - (piv["reasoned"] - piv["qa"])).to_numpy(dtype=float)
    clusters = tmap.loc[piv.index].to_numpy()
    lo, hi = clustered_bootstrap_ci(v, clusters)
    per_cluster = pd.Series(v).groupby(clusters).mean()
    return {
        "n_pairs": int(len(piv)),
        "delta_acc": float(per_cluster.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "flip_rate": float("nan"),
        "agree_rate": float("nan"),
    }


def transcript_pathology(tag: str) -> tuple[list[dict], dict[str, set]]:
    """Per-cell loop census over the stage-1 transcripts.

    Returns the table rows and, per cell, the base_ids of looped (>= 2
    marker) transcripts for the robustness splits.
    """
    gen = ROOT / "outputs" / "reasoning" / "generated"
    rows, loops = [], {}
    for cell, fname in (("qa", f"reasoning_{tag}.jsonl"),
                        ("narrative", f"reasoning_narr1_{tag}.jsonl")):
        marks, words, length_fin, unstable = [], [], 0, 0
        looped_ids: set[str] = set()
        for line in (gen / fname).open(encoding="utf-8"):
            r = json.loads(line)
            if "error" in r:
                continue
            raw = r["reasoning_raw"]
            m = len(MARKER.findall(raw))
            marks.append(m)
            words.append(len(raw.split()))
            if r.get("finish_reason") == "length":
                length_fin += 1
            if m >= 2:
                base = re.sub(r"_narr1$", "", r["example_id"])
                looped_ids.add(base)
                if len(set(DIGIT.findall(raw))) > 1:
                    unstable += 1
        n = len(marks)
        ma = np.array(marks)
        rows.append({
            "model": tag, "cell": cell, "n": n,
            "length_rate": length_fin / n,
            "loop_rate": float((ma >= 2).mean()),
            "heavy_loop_rate": float((ma >= 5).mean()),
            "median_markers": float(np.median(ma)),
            "max_markers": int(ma.max()),
            "median_words": float(np.median(words)),
            "digit_instability_rate": (unstable / len(looped_ids)
                                       if looped_ids else float("nan")),
        })
        loops[cell] = looped_ids
    return rows, loops


def elicitation(tag: str, df: pd.DataFrame) -> list[dict]:
    """Parse outcomes (sidecar) + stated-answer readout vs the label readout.

    The sidecar's qa cell feeds `<eid>_reasoned`; its narrative cell (keyed
    `<eid>_narr1`) feeds `<eid>_narrreasoned`. An empty transcript is
    `no_marker` with zero reasoning words — Olmo's dominant failure mode.
    """
    sc = pd.read_csv(ROOT / "outputs" / "reasoning" / "inputs"
                     / f"c1_label_set_{tag}_parse.csv")
    lab = df[df["condition"].isin(("reasoned", "narrative_reasoned"))]
    lab = lab.set_index("example_id")
    out = []
    for cell, cond, suffix in (("qa", "reasoned", "_reasoned"),
                               ("narrative", "narrative_reasoned", "_narrreasoned")):
        g = sc[sc["cell"] == cell].copy()
        base = g["example_id"].str.replace(r"_narr1$", "", regex=True)
        g["scored_id"] = base + suffix
        counts = g["parse"].value_counts().to_dict()
        parsed = g[g["parse"].isin(PARSE_OK)].dropna(subset=["stated_index"])
        j = parsed.join(lab, on="scored_id", rsuffix="_lab")
        stated = j["stated_index"].astype(float)
        row = {
            "model": tag, "cell": cell, "condition": cond,
            "n_transcripts": int(len(g)),
            "parse_fail_rate": float(1.0 - len(parsed) / len(g)),
            "empty_rate": float(((g["parse"] == "no_marker")
                                 & (g["reasoning_words"].fillna(0) == 0)).mean()),
            "stated_acc": float((stated == j["ground_truth_index"]).mean()),
            "stated_vs_label_agree": float(
                (stated == j["label_num_pred_idx"]).mean()),
            "n_parsed": int(len(j)),
        }
        for k in ("digit", "option_text", "digit_out_of_range",
                  "unparseable_after_marker", "no_marker", "generation_error"):
            row[f"n_{k}"] = int(counts.get(k, 0))
        out.append(row)
    return out


def run(tag: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, raw = load(tag)
    m_map = m_by_target(df)
    print(f"{len(df)} scored rows ({df['base_id'].nunique()} base_ids), "
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
                "mean_conf": (float(g["label_conf"].mean())
                              if arm == "label_num" else float("nan")),
                "ece": ece(g) if arm == "label_num" else float("nan"),
            })
        # Endpoint 1: reasoned-minus-qa (prediction +-0.02, falsifier +0.04),
        # its narrative twin, and the within-serving B3 replication.
        for cond, base in (("reasoned", "qa"),
                           ("narrative_reasoned", "narrative_direct"),
                           ("narrative_direct", "qa")):
            contrasts.append({
                "model": tag, "arm": arm, "contrast": f"{cond}-{base}",
                **paired(df, arm, cond, base),
            })
        # Endpoint 2: the 2x2 interaction over completed quads.
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "interaction",
            **interaction(df, arm),
        })
        # Replicate ceiling for this serving (read against the 64.4%
        # cross-serving floor, not 100%).
        rate, n = replicate_agreement(raw, arm=arm)
        contrasts.append({
            "model": tag, "arm": arm, "contrast": "replicate",
            "n_pairs": n, "delta_acc": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate) else float("nan"),
            "agree_rate": rate,
        })

    # Loop-census robustness: the elicitation contrasts split by whether the
    # stage-1 transcript looped (label_num only; both halves need MIN_SPLIT
    # pairs, so Olmo — 1 looped transcript — gets no split rows).
    pathology, loops = transcript_pathology(tag)
    for cond, base, cell in (("reasoned", "qa", "qa"),
                             ("narrative_reasoned", "narrative_direct",
                              "narrative")):
        in_loop = df["base_id"].isin(loops[cell])
        for label, mask in (("looped", in_loop), ("clean", ~in_loop)):
            sub = df[mask]
            if min(len(sub[sub["condition"] == cond]),
                   len(sub[sub["condition"] == base])) < MIN_SPLIT:
                continue
            contrasts.append({
                "model": tag, "arm": "label_num",
                "contrast": f"{cond}-{base}|{label}",
                **paired(sub, "label_num", cond, base),
            })

    elic = elicitation(tag, df)

    pd.DataFrame(levels).to_csv(OUTDIR / f"c1_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUTDIR / f"c1_contrasts_{tag}.csv", index=False)
    pd.DataFrame(elic).to_csv(OUTDIR / f"c1_elicitation_{tag}.csv", index=False)
    pd.DataFrame(pathology).to_csv(OUTDIR / f"c1_transcripts_{tag}.csv",
                                   index=False)
    for stem in ("levels", "contrasts", "elicitation", "transcripts"):
        print(f"wrote {OUTDIR / f'c1_{stem}_{tag}.csv'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qwen_qwen3-32b")
    a = ap.parse_args(argv)
    run(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
