"""The matched ceiling read: model vs XGBoost on the IDENTICAL instances.

The k=24 anchor (xgb_anchor_k24.csv) is only readable next to a model
number computed on the same rows: C2's 0.3372 is on all 734 pairs, the
XGB anchor covers the 660 that sit in usable cells. This script restricts
Qwen3-32B's direct label_num readout (C2's toff cell, the substrate's
freshest direct scores) to exactly those 660 instances and applies the
same norm_acc convention (distinct-M per target, mean within target then
across). Writes xgb_anchor_matched_model.csv; pinned by
verify_xgb_ceiling_numbers.py.

Cross-serving note: this compares a supervised fit (no serving) to one
model serving — no scores are reused or compared across servings.

    python scripts/xgb_ceiling/matched_anchor_read.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
AN = WORK / "analysis" / "xgb_ceiling"
B3_TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
C2_RESULTS = (REPO / "outputs" / "thinking" / "results"
              / "c2_label_results_qwen_qwen3-32b.jsonl")
MODEL_TAG = "qwen3-32b_label_num_direct(c2_toff)"


def main() -> int:
    anchor_ids = set()
    for line in (AN / "xgb_ladder_dump.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["ordering"] == "informative" and r["k"] == 24:
            anchor_ids.add(r["example_id"])
    b3 = {json.loads(l)["example_id"]
          for l in B3_TASKS.open(encoding="utf-8")}
    anchor_ids &= b3

    gt = {}
    gt_all = {}
    with (WORK / "outputs_recovered" / "ladder_readout_set.jsonl").open(
            encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in b3:
                gt_all[r["example_id"]] = (r["ground_truth"],
                                           f"{r['survey']}|{r['target_code']}")
                if r["example_id"] in anchor_ids:
                    gt[r["example_id"]] = gt_all[r["example_id"]]

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
        truth, cluster = gt[base]
        rows.append({"cluster": cluster, "pred_text": cell.get("predicted"),
                     "gt_text": truth})
    df = pd.DataFrame(rows)
    if len(df) != len(anchor_ids):
        print(f"FATAL matched {len(df)} of {len(anchor_ids)} anchor rows")
        return 1

    M_by_t = df.groupby("cluster")["gt_text"].nunique()
    d = df.assign(M=df["cluster"].map(M_by_t))
    d = d[d["M"] >= 2]
    na = ((d["pred_text"] == d["gt_text"]).astype(float) - 1 / d["M"]) \
        / (1 - 1 / d["M"])
    norm = float(na.groupby(d["cluster"].to_numpy()).mean().mean())

    xgb = pd.read_csv(AN / "xgb_anchor_k24.csv")
    xgb_norm = float(xgb[xgb["target"] == "POOLED"].iloc[0]["norm_acc"])
    out = pd.DataFrame([{
        "model": MODEL_TAG, "n": len(df),
        "n_targets": df["cluster"].nunique(),
        "acc": float((df["pred_text"] == df["gt_text"]).mean()),
        "norm_acc": norm,
        "xgb_norm_acc": xgb_norm,
        "gap_model_minus_xgb": norm - xgb_norm,
    }])
    out.to_csv(AN / "xgb_anchor_matched_model.csv", index=False)
    r = out.iloc[0]
    print(f"matched anchor: n={r['n']} targets={r['n_targets']} "
          f"model norm {r['norm_acc']:.4f} vs xgb {r['xgb_norm_acc']:.4f} "
          f"(gap {r['gap_model_minus_xgb']:+.4f})")

    # The LLM side of the dissenter comparison, on ALL 734 anchor pairs
    # (same modal definition as xgb_full_dissenter_split): per-target
    # modal/dissenter accuracy + the all-734 norm_acc, so XGB-CEILING-FULL
    # is read against pinned LLM rows, not scrollback.
    rows_all = []
    for line in C2_RESULTS.open(encoding="utf-8"):
        r2 = json.loads(line)
        eid = r2["example_id"]
        if not eid.endswith("_toff"):
            continue
        base = eid[: -len("_toff")]
        if base not in gt_all:
            continue
        cell = (r2.get("results") or {}).get("original|label_num") or {}
        truth, cluster = gt_all[base]
        rows_all.append({"cluster": cluster,
                         "pred_text": cell.get("predicted"),
                         "gt_text": truth})
    da = pd.DataFrame(rows_all)
    split = []
    for cluster, g in da.groupby("cluster"):
        modal = g["gt_text"].mode().iloc[0]
        is_modal = g["gt_text"] == modal
        split.append({
            "target": cluster, "n": len(g),
            "modal_share": float(is_modal.mean()),
            "acc_modal": float((g.loc[is_modal, "pred_text"]
                                == g.loc[is_modal, "gt_text"]).mean()),
            "acc_dissenter": (
                float((g.loc[~is_modal, "pred_text"]
                       == g.loc[~is_modal, "gt_text"]).mean())
                if (~is_modal).any() else float("nan")),
        })
    M_by_t = da.groupby("cluster")["gt_text"].nunique()
    d2 = da.assign(M=da["cluster"].map(M_by_t))
    d2 = d2[d2["M"] >= 2]
    na2 = ((d2["pred_text"] == d2["gt_text"]).astype(float) - 1 / d2["M"]) \
        / (1 - 1 / d2["M"])
    norm734 = float(na2.groupby(d2["cluster"].to_numpy()).mean().mean())
    sd = pd.DataFrame(split)
    sd.to_csv(AN / "xgb_llm_dissenter_split.csv", index=False)
    print(f"llm all-734: norm {norm734:.4f}; dissenter split modal "
          f"{sd['acc_modal'].mean():.4f} vs dissenter "
          f"{sd['acc_dissenter'].mean():.4f} ({len(sd)} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
