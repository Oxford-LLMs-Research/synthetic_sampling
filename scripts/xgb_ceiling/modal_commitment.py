"""Modal commitment: WHY the LLM under-performs XGB on modal respondents.

XGB-CEILING-FULL's dissenter split showed the supervised advantage lives
almost entirely on modal respondents (prompt-parity XGB 0.745 vs LLM
0.649; dissenters 0.393 vs 0.374). Since a base-rate predictor gets
modal respondents right for free, the deficit must come from the LLM
DEVIATING from the modal answer. This script diagnoses those deviations
on the identical 734 anchor instances, per XGB regime:

- deviation rate: how often each predictor's argmax is NOT the target's
  modal answer, split by respondent type (modal vs dissenter);
- deviation payoff: P(correct | deviated) — is straying from base rate
  ever earning its cost;
- near-miss anatomy (LLM only): when the LLM deviates, where the modal
  option sits in its softmax (rank, p_modal, margin to the chosen
  option) — distinguishes "modal narrowly out-ranked" (rank 2, thin
  margin: ranking almost right, distribution estimands survive) from
  "modal buried" (a real signal-conversion failure);
- overlap: do the LLM and XGB deviate on the SAME modal respondents.

Modal answer = mode of ground-truth text within target over the 734
anchor rows (ties broken by pandas mode order — first alphabetically),
matching matched_anchor_read.py's split. Headline splits are means of
per-target values (the pinned convention); pooled instance-level rates
are labeled "pooled".

Cross-serving note: supervised fits (no serving) vs one model serving —
no scores reused or compared across servings.

    python scripts/xgb_ceiling/modal_commitment.py

Writes WORK/analysis/xgb_ceiling/modal_commitment_{summary,by_target,
deviation_overlap}.csv; numbers pinned by verify_modal_commitment.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
AN = WORK / "analysis" / "xgb_ceiling"
B3_TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
C2_RESULTS = (REPO / "outputs" / "native_thinking" / "results"
              / "c2_label_results_qwen_qwen3-32b.jsonl")
LADDER_SET = WORK / "outputs_recovered" / "ladder_readout_set.jsonl"

REGIMES = ("grouped", "within", "within_prompt24")
LLM_TAG = "llm(c2_toff_label_num)"


def load_anchor_truth() -> dict[str, tuple[str, str]]:
    """example_id -> (ground_truth, cluster) for the 734 anchor rows.

    The anchor is B3's task list itself (all its ids are k=24 informative
    ladder rows) — NOT intersected with the small-cell xgb_ladder_dump,
    which drops the 74 rows of unusable cells (e.g. single-country QLEB7).
    """
    b3 = {json.loads(l)["example_id"] for l in B3_TASKS.open(encoding="utf-8")}
    gt = {}
    for line in LADDER_SET.open(encoding="utf-8"):
        r = json.loads(line)
        if r["example_id"] in b3:
            gt[r["example_id"]] = (r["ground_truth"],
                                   f"{r['survey']}|{r['target_code']}")
    if len(gt) != len(b3):
        raise SystemExit(f"FATAL truth for {len(gt)}/{len(b3)}")
    return gt


def load_llm(gt: dict) -> pd.DataFrame:
    rows = []
    for line in C2_RESULTS.open(encoding="utf-8"):
        r = json.loads(line)
        eid = r["example_id"]
        if not eid.endswith("_toff"):
            continue
        base = eid[: -len("_toff")]
        if base not in gt:
            continue
        cell = (r.get("results") or {}).get("original|label_num") or {}
        scores = cell.get("scores") or {}
        finite = {o: v for o, v in scores.items() if np.isfinite(v)}
        v = np.array(list(finite.values()), dtype=float)
        e = np.exp(v - v.max())
        probs = dict(zip(finite, e / e.sum()))
        truth, cluster = gt[base]
        rows.append({"example_id": base, "predictor": LLM_TAG,
                     "cluster": cluster, "gt_text": truth,
                     "pred_text": cell.get("predicted"), "probs": probs})
    if len(rows) != len(gt):
        raise SystemExit(f"FATAL llm matched {len(rows)}/{len(gt)}")
    return pd.DataFrame(rows)


def load_xgb(gt: dict) -> pd.DataFrame:
    rows = []
    for line in (AN / "xgb_full_anchor_dump.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["example_id"] not in gt:
            raise SystemExit(f"FATAL dump row not in anchor: {r['example_id']}")
        rows.append({"example_id": r["example_id"],
                     "predictor": f"xgb_{r['regime']}",
                     "cluster": r["cluster"], "gt_text": r["gt_text"],
                     "pred_text": r["pred_text"],
                     "probs": dict(zip(r["opts"], r["p"]))})
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame, modal: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["modal_text"] = df["cluster"].map(modal)
    df["is_modal_resp"] = df["gt_text"] == df["modal_text"]
    df["correct"] = (df["pred_text"] == df["gt_text"]).astype(float)
    df["deviates"] = df["pred_text"] != df["modal_text"]

    def prob_stats(row):
        p = row["probs"]
        pm = p.get(row["modal_text"], 0.0)
        pp = p.get(row["pred_text"], np.nan)
        ranked = sorted(p.values(), reverse=True)
        rank = 1 + sum(v > pm for v in p.values())
        return pd.Series({"p_modal": pm, "p_pred": pp,
                          "modal_rank": rank,
                          "margin": pp - pm if np.isfinite(pp) else np.nan})

    return pd.concat([df, df.apply(prob_stats, axis=1)], axis=1)


def per_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pred, cluster), g in df.groupby(["predictor", "cluster"]):
        m = g["is_modal_resp"]
        rows.append({
            "predictor": pred, "target": cluster, "n": len(g),
            "modal_share": float(m.mean()),
            "acc_modal": float(g.loc[m, "correct"].mean()),
            "acc_dissenter": (float(g.loc[~m, "correct"].mean())
                              if (~m).any() else np.nan),
            "dev_rate_modal": float(g.loc[m, "deviates"].mean()),
            "dev_rate_dissenter": (float(g.loc[~m, "deviates"].mean())
                                   if (~m).any() else np.nan),
            "p_modal_mean": float(g["p_modal"].mean()),
        })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred, g in df.groupby("predictor"):
        t = tgt[tgt["predictor"] == pred]
        dev = g[g["deviates"]]
        dev_m = g[g["deviates"] & g["is_modal_resp"]]
        rows.append({
            "predictor": pred, "n": len(g),
            "n_targets": g["cluster"].nunique(),
            # pinned-convention headline splits (per-target means)
            "acc_modal": float(t["acc_modal"].mean()),
            "acc_dissenter": float(t["acc_dissenter"].mean()),
            "dev_rate_modal": float(t["dev_rate_modal"].mean()),
            "dev_rate_dissenter": float(t["dev_rate_dissenter"].mean()),
            # pooled deviation anatomy
            "pooled_dev_rate": float(g["deviates"].mean()),
            "pooled_dev_payoff": (float(dev["correct"].mean())
                                  if len(dev) else np.nan),
            "pooled_dev_modal_resp_share": (
                float(dev["is_modal_resp"].mean()) if len(dev) else np.nan),
            "dev_modal_rank2_share": (
                float((dev_m["modal_rank"] == 2).mean())
                if len(dev_m) else np.nan),
            "dev_modal_rank_median": (
                float(dev_m["modal_rank"].median()) if len(dev_m) else np.nan),
            "dev_modal_p_modal_mean": (
                float(dev_m["p_modal"].mean()) if len(dev_m) else np.nan),
            "dev_modal_margin_mean": (
                float(dev_m["margin"].mean()) if len(dev_m) else np.nan),
            "mean_max_p": float(g["probs"].map(
                lambda p: max(p.values())).mean()),
            "mean_p_modal": float(g["p_modal"].mean()),
        })
    return pd.DataFrame(rows)


def overlap(llm: pd.DataFrame, xgb: pd.DataFrame) -> pd.DataFrame:
    """Deviation agreement on MODAL respondents, per regime."""
    rows = []
    lm = llm[llm["is_modal_resp"]].set_index("example_id")
    for regime in REGIMES:
        x = xgb[(xgb["predictor"] == f"xgb_{regime}")
                & xgb["is_modal_resp"]].set_index("example_id")
        shared = lm.index.intersection(x.index)
        a = lm.loc[shared, "deviates"]
        b = x.loc[shared, "deviates"]
        both = a & b
        rows.append({
            "regime": regime, "n_modal_shared": len(shared),
            "llm_dev_rate": float(a.mean()),
            "xgb_dev_rate": float(b.mean()),
            "both_dev_rate": float(both.mean()),
            "jaccard": (float(both.sum() / (a | b).sum())
                        if (a | b).any() else np.nan),
            "llm_acc_where_xgb_deviates": float(
                lm.loc[shared, "correct"][b].mean()) if b.any() else np.nan,
            "xgb_acc_where_llm_deviates": float(
                x.loc[shared, "correct"][a].mean()) if a.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    gt = load_anchor_truth()
    llm_raw = load_llm(gt)
    xgb_raw = load_xgb(gt)

    # Modal answer per target over the full 734 anchor rows (the
    # matched_anchor_read convention), applied to every predictor.
    modal = {c: g["gt_text"].mode().iloc[0]
             for c, g in llm_raw.groupby("cluster")}

    llm = annotate(llm_raw, modal)
    xgb = annotate(xgb_raw, modal)
    df = pd.concat([llm, xgb], ignore_index=True)

    tgt = per_target(df)
    summ = summary(df, tgt)
    ov = overlap(llm, xgb)

    keep = [c for c in tgt.columns]
    tgt[keep].to_csv(AN / "modal_commitment_by_target.csv", index=False)
    summ.to_csv(AN / "modal_commitment_summary.csv", index=False)
    ov.to_csv(AN / "modal_commitment_deviation_overlap.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(ov.to_string(index=False))
    for stem in ("summary", "by_target", "deviation_overlap"):
        print(f"wrote {AN / f'modal_commitment_{stem}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
