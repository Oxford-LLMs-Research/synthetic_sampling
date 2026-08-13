"""C3 sibling analysis, to the AMENDED 13 Aug protocol (see the C3
debrief in PAPER_STATE): reasoning as training under plausible
deployment, never a logprob read on an injected trace.

Cells:
  Instruct-2507  fresh direct scores (job 8560584): chat_label_num is
                 the sibling ceiling; label_num / echo_plain ride along.
  Thinking-2507  native chat generation, always-on thinking. ACCURACY is
                 the stated digit after </think> in draw 1 (seed 42).
                 DISTRIBUTION is the K=10 histogram of stated digits
                 (draws 1-10, seeds 42-51): what a synthetic survey
                 would actually sample. Majority vote is reported as a
                 separate estimand, never as "the" accuracy.

The sibling contrast is cross-checkpoint and cross-readout by design
(stated generation vs first-token chat logprobs). C2 measured the
stated-vs-chat-readout gap at -0.044 on fixed weights (0.531 vs 0.575,
Qwen3-32B); carry that as the known readout asymmetry when reading any
Thinking-minus-Instruct delta near that size.

Writes (WORK/analysis/thinking/):
  c3_levels_<tag>.csv       Instruct arms + Thinking stated K=1
  c3_contrasts_<tag>.csv    paired thinking-minus-instruct (chat + raw
                            readouts), replicate rows
  c3_census_<tag>.csv       per-draw elicitation census (block shape,
                            truncation, loops, parse)
  c3_aggregate_<tag>.csv    per-target TV: K=10 sampled shares vs human,
                            Instruct chat softmax shares vs human;
                            self-consistency; majority-vote row

    python scripts/thinking/analyze_c3.py
"""

from __future__ import annotations

import argparse
import importlib.util
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
from synthetic_sampling.scoring.thinking import parse_stated_chat, split_think

ROOT = Path(__file__).resolve().parents[2]
OUTER = ROOT.parent
GEN = ROOT / "outputs" / "thinking" / "generated"
RES = ROOT / "outputs" / "thinking" / "results"
LADDER = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"
B3_TASKS = ROOT / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
OUTDIR = OUTER / "analysis" / "thinking"

THINK_TAG = "qwen_qwen3-30b-a3b-thinking-2507"
INSTR_TAG = "qwen_qwen3-30b-a3b-instruct-2507"
K = 10
INSTR_ARMS = ("chat_label_num", "label_num", "echo_plain")
PARSE_OK = ("digit", "bare_digit", "option_text")

_spec = importlib.util.spec_from_file_location(
    "make_c2_set", Path(__file__).parent / "make_c2_set.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
LOOP_MARKER = _m.LOOP_MARKER


def _softmax_conf(scores: dict) -> tuple[float, dict]:
    keys = list(scores)
    v = np.array([scores[k] for k in keys], dtype=float)
    if not len(v) or not np.all(np.isfinite(v)):
        return float("nan"), {}
    e = np.exp(v - v.max())
    p = e / e.sum()
    return float(p.max()), dict(zip(keys, p.astype(float)))


def load_meta() -> dict[str, dict]:
    b3 = {json.loads(l)["example_id"] for l in B3_TASKS.open(encoding="utf-8")}
    meta = {}
    with LADDER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in b3:
                meta[r["example_id"]] = {
                    "target": f"{r['survey']}|{r['target_code']}",
                    "options": list(r["option_sets"]["original"]),
                    "gt_text": r["ground_truth"],
                    "gt_index": r["ground_truth_index"],
                }
    return meta


def load_draws(meta: dict) -> pd.DataFrame:
    rows = []
    for k in range(1, K + 1):
        suffix = "" if k == 1 else f"_k{k:02d}"
        path = GEN / f"thinking_{THINK_TAG}{suffix}.jsonl"
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            m = meta[r["example_id"]]
            raw = r.get("thinking_raw") or ""
            sp = split_think(raw)
            parsed = parse_stated_chat(sp["answer"], m["options"])
            think_words = len((sp["think"] or "").split())
            rows.append({
                "example_id": r["example_id"], "draw": k,
                "target": m["target"], "gt_index": m["gt_index"],
                "gt_text": m["gt_text"], "n_options": len(m["options"]),
                "error": "error" in r,
                "finish_reason": r.get("finish_reason"),
                "has_block": sp["has_block"], "closed": sp["closed"],
                "think_words": think_words,
                "loops": len(LOOP_MARKER.findall(sp["think"] or "")),
                "parse": parsed["parse"],
                "stated_index": parsed["stated_index"],
            })
    d = pd.DataFrame(rows)
    d["stated_text"] = [
        (meta[e]["options"][int(i)] if pd.notna(i) else None)
        for e, i in zip(d["example_id"], d["stated_index"])]
    d["correct"] = np.where(d["stated_index"].notna(),
                            (d["stated_index"] == d["gt_index"]).astype(float),
                            np.nan)
    return d


def load_instruct(meta: dict) -> tuple[pd.DataFrame, list[dict]]:
    rows, raw = [], []
    path = RES / f"c3_direct_results_{INSTR_TAG}.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        raw.append(r)
        base = r["example_id"][: -len("_direct")]
        m = meta[base]
        row = {"example_id": base, "target": m["target"],
               "gt_text": m["gt_text"], "n_options": len(m["options"])}
        for arm in INSTR_ARMS:
            cell = (r.get("results") or {}).get(f"original|{arm}")
            pred, miss, conf, probs = None, True, float("nan"), {}
            if cell and "error" not in cell:
                scores = cell.get("scores") or {}
                finite = [v for v in scores.values() if np.isfinite(v)]
                miss = len(finite) != len(scores) or not scores
                pred = cell.get("predicted")
                conf, probs = _softmax_conf(scores)
            row[f"{arm}_pred"] = pred
            row[f"{arm}_miss"] = miss
            row[f"{arm}_conf"] = conf
            row[f"{arm}_correct"] = (float(pred == m["gt_text"])
                                     if pred is not None else np.nan)
            row[f"{arm}_probs"] = probs
        rows.append(row)
    return pd.DataFrame(rows), raw


def m_map_of(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {t: g[col].nunique() for t, g in df.groupby("target")}


def norm_acc(df: pd.DataFrame, correct_col: str, m_map: dict) -> float:
    vals = []
    for t, g in df.groupby("target"):
        m = m_map.get(t, 0)
        acc = g[correct_col].mean()
        if m >= 2 and np.isfinite(acc):
            vals.append(normalized_accuracy(float(acc), m))
    return float(np.mean(vals)) if vals else float("nan")


def ece(df: pd.DataFrame, arm: str, n_bins: int = 10) -> float:
    g = df.dropna(subset=[f"{arm}_conf", f"{arm}_correct"])
    if g.empty:
        return float("nan")
    conf = g[f"{arm}_conf"].to_numpy()
    corr = g[f"{arm}_correct"].to_numpy()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean()
                     * abs(corr[bins == b].mean() - conf[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def tv_from_shares(pred_shares: dict, truths: pd.Series,
                   options: list[str]) -> float:
    emp = {o: float((truths == o).mean()) for o in options}
    keys = set(pred_shares) | set(emp)
    return 0.5 * sum(abs(pred_shares.get(k, 0.0) - emp.get(k, 0.0))
                     for k in keys)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    draws = load_draws(meta)
    instr, instr_raw = load_instruct(meta)
    print(f"{len(draws)} draw rows ({draws['example_id'].nunique()} pairs x "
          f"{draws['draw'].nunique()} draws), {len(instr)} instruct rows")

    gt_m = m_map_of(instr, "gt_text")

    # --- levels
    levels = []
    for arm in INSTR_ARMS:
        levels.append({
            "cell": f"instruct_{arm}", "n": len(instr),
            "miss_rate": float(instr[f"{arm}_miss"].mean()),
            "acc_raw": float(instr[f"{arm}_correct"].mean()),
            "norm_acc": norm_acc(instr, f"{arm}_correct", gt_m),
            "mean_conf": float(instr[f"{arm}_conf"].mean()),
            "ece": ece(instr, arm),
        })
    d1 = draws[draws["draw"] == 1].copy()
    levels.append({
        "cell": "thinking_stated_k1", "n": len(d1),
        "miss_rate": float(d1["stated_index"].isna().mean()),
        "acc_raw": float(d1["correct"].mean()),
        "norm_acc": norm_acc(d1, "correct", gt_m),
        "mean_conf": float("nan"), "ece": float("nan"),
    })
    # majority vote across K (separate estimand)
    mv = []
    for eid, g in draws.groupby("example_id"):
        stated = g["stated_text"].dropna()
        if stated.empty:
            continue
        top = stated.mode().iloc[0]
        mv.append({"example_id": eid, "target": meta[eid]["target"],
                   "correct": float(top == meta[eid]["gt_text"])})
    mvd = pd.DataFrame(mv)
    levels.append({
        "cell": f"thinking_majority_k{K}", "n": len(mvd),
        "miss_rate": float("nan"),
        "acc_raw": float(mvd["correct"].mean()),
        "norm_acc": norm_acc(mvd, "correct", gt_m),
        "mean_conf": float("nan"), "ece": float("nan"),
    })

    # --- paired contrasts: thinking stated K=1 vs instruct readouts
    contrasts = []
    d1i = d1.set_index("example_id")
    ii = instr.set_index("example_id")
    shared = d1i.index.intersection(ii.index)
    for arm in ("chat_label_num", "label_num"):
        a = d1i.loc[shared, "correct"].astype(float)
        b = ii.loc[shared, f"{arm}_correct"].astype(float)
        keep = a.notna() & b.notna()
        diff = (a - b)[keep].to_numpy()
        clusters = d1i.loc[shared, "target"].to_numpy()[keep.to_numpy()]
        agree = (d1i.loc[shared, "stated_text"][keep]
                 == ii.loc[shared, f"{arm}_pred"][keep]).mean()
        lo, hi = clustered_bootstrap_ci(diff, clusters)
        per_cluster = pd.Series(diff).groupby(clusters).mean()
        contrasts.append({
            "contrast": f"thinking_stated-instruct_{arm}",
            "n_pairs": int(keep.sum()),
            "delta_acc": float(per_cluster.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "flip_rate": float(1.0 - agree), "agree_rate": float(agree),
        })
    for arm in INSTR_ARMS:
        rate, n = replicate_agreement(instr_raw, arm=arm)
        contrasts.append({
            "contrast": f"instruct_{arm}_replicate", "n_pairs": n,
            "delta_acc": float("nan"), "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "flip_rate": float(1.0 - rate) if np.isfinite(rate)
            else float("nan"),
            "agree_rate": rate,
        })

    # --- per-draw elicitation census
    census = []
    for k, g in draws.groupby("draw"):
        parsed = g[g["parse"].isin(PARSE_OK)]
        census.append({
            "draw": k, "n": len(g),
            "n_error": int(g["error"].sum()),
            "block_rate": float((g["has_block"] == True).mean()),  # noqa: E712
            "closed_rate": float((g["closed"] == True).mean()),  # noqa: E712
            "truncated": int((g["finish_reason"] == "length").sum()),
            "loop_rate": float((g["loops"] > 1).mean()),
            "parse_fail_rate": float(1.0 - len(parsed) / len(g)),
            "median_think_words": float(g["think_words"].median()),
            "acc": float(g["correct"].mean()),
        })

    # --- aggregates: sampled shares vs human, instruct softmax vs human
    agg = []
    for target, g in instr.groupby("target"):
        eids = g["example_id"].tolist()
        options = meta[eids[0]]["options"]
        truths = g.set_index("example_id")["gt_text"]
        # thinking sampled shares: all draws, all pairs of this target
        gt_draws = draws[draws["target"] == target]
        stated = gt_draws["stated_text"].dropna()
        shares_t = {o: float((stated == o).mean()) for o in options}
        # instruct chat softmax mean distribution
        probs = [p for p in g["chat_label_num_probs"] if p]
        mean_p = {}
        if probs:
            for o in options:
                mean_p[o] = float(np.mean([p.get(o, 0.0) for p in probs]))
        # self-consistency: mean pairwise agreement of stated across draws
        cons = []
        for eid, ge in gt_draws.groupby("example_id"):
            st = ge["stated_text"].dropna().tolist()
            if len(st) >= 2:
                c = Counter(st)
                n = len(st)
                cons.append(sum(v * (v - 1) for v in c.values())
                            / (n * (n - 1)))
        agg.append({
            "target": target, "n_pairs": len(g),
            "tv_thinking_sampled": tv_from_shares(shares_t, truths, options),
            "tv_instruct_chat": (tv_from_shares(mean_p, truths, options)
                                 if mean_p else float("nan")),
            "self_consistency": float(np.mean(cons)) if cons else float("nan"),
        })

    tag = THINK_TAG
    pd.DataFrame(levels).to_csv(OUTDIR / f"c3_levels_{tag}.csv", index=False)
    pd.DataFrame(contrasts).to_csv(
        OUTDIR / f"c3_contrasts_{tag}.csv", index=False)
    pd.DataFrame(census).to_csv(OUTDIR / f"c3_census_{tag}.csv", index=False)
    pd.DataFrame(agg).to_csv(OUTDIR / f"c3_aggregate_{tag}.csv", index=False)

    for r in levels:
        print(f"{r['cell']:<28} n={r['n']:<4} acc={r['acc_raw']:.4f} "
              f"norm={r['norm_acc']:.4f}")
    for r in contrasts[:2]:
        print(f"{r['contrast']:<40} d={r['delta_acc']:+.4f} "
              f"CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"flip={r['flip_rate']:.3f}")
    cd = pd.DataFrame(census)
    print(f"census: blocks {cd['block_rate'].mean():.4f} closed "
          f"{cd['closed_rate'].mean():.4f} loops {cd['loop_rate'].mean():.4f} "
          f"parse_fail {cd['parse_fail_rate'].mean():.4f} "
          f"truncated {int(cd['truncated'].sum())}")
    ad = pd.DataFrame(agg)
    print(f"aggregate TV: thinking sampled {ad['tv_thinking_sampled'].mean():.4f} "
          f"vs instruct chat {ad['tv_instruct_chat'].mean():.4f}; "
          f"self-consistency {ad['self_consistency'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
