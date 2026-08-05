"""Compare chat_number vs paper mean-token (echo) scoring on the same cells.

Joins two result jsonl files on (example_id, condition) and reports:
  - prediction agreement overall / by condition
  - accuracy by method x condition
  - whether year/placebo deltas agree in sign
  - flip agreement (does year change the pred under both methods?)

Usage:
    python compare_temporal_scoring_methods.py \\
        --echo ../outputs/colab_experiments/temporal_results_smoke.jsonl \\
        --chat ../outputs/colab_experiments/temporal_results_calib_chat.jsonl \\
        --instances ../outputs/colab_experiments/temporal_calibration_20.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_results(path: Path) -> dict[tuple, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["example_id"], r["condition"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--echo", required=True, help="paper perplexity / echo results")
    ap.add_argument("--chat", required=True, help="chat_number results")
    ap.add_argument("--instances", default=None,
                    help="optional instances jsonl for survey labels")
    args = ap.parse_args()

    echo = load_results(Path(args.echo))
    chat = load_results(Path(args.chat))
    survey = {}
    if args.instances:
        for line in open(args.instances, encoding="utf-8"):
            d = json.loads(line)
            survey[d["example_id"]] = d.get("survey")

    keys = sorted(set(echo) & set(chat))
    only_echo = len(set(echo) - set(chat))
    only_chat = len(set(chat) - set(echo))
    print(f"paired cells: {len(keys)}  (echo-only={only_echo}, chat-only={only_chat})")
    if not keys:
        raise SystemExit("no overlapping (example_id, condition) cells")

    rows = []
    for key in keys:
        e, c = echo[key], chat[key]
        eid, cond = key
        rows.append({
            "example_id": eid,
            "condition": cond,
            "survey": survey.get(eid),
            "gt": e.get("ground_truth") or e.get("answer"),
            "pred_echo": e["predicted"],
            "pred_chat": c["predicted"],
            "agree": e["predicted"] == c["predicted"],
            "correct_echo": bool(e["correct"]),
            "correct_chat": bool(c["correct"]),
        })
    df = pd.DataFrame(rows)

    print("\n=== Prediction agreement ===")
    print(f"  overall: {df['agree'].mean():.3f}  (n={len(df)})")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:20s}  {sub['agree'].mean():.3f}  n={len(sub)}")

    print("\n=== Accuracy by method x condition ===")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:20s}  echo={sub['correct_echo'].mean():.3f}  "
              f"chat={sub['correct_chat'].mean():.3f}")

    # Primary paired effect on instances with all 3 conditions under both methods
    primary = ["baseline", "with_year", "with_year_placebo"]
    by_eid = defaultdict(dict)
    for r in rows:
        by_eid[r["example_id"]][r["condition"]] = r
    complete = []
    for eid, m in by_eid.items():
        if all(c in m for c in primary):
            complete.append(eid)
    print(f"\n=== Effect agreement (instances with all primary conds: {len(complete)}) ===")
    if complete:
        def acc(method_correct_key, cond):
            return sum(by_eid[e][cond][method_correct_key] for e in complete) / len(complete)

        for label, key in (("echo", "correct_echo"), ("chat", "correct_chat")):
            b, y, p = acc(key, "baseline"), acc(key, "with_year"), acc(key, "with_year_placebo")
            print(f"  {label}: base={b:.3f} year={y:.3f} placebo={p:.3f}  "
                  f"d_year={y-b:+.3f} d_placebo={p-b:+.3f} contentful={y-p:+.3f}")

        # per-instance: does year flip pred? does placebo? agree across methods?
        flip_y_agree = flip_p_agree = 0
        pred_agree_base = 0
        for e in complete:
            be, ye, pe = by_eid[e]["baseline"], by_eid[e]["with_year"], by_eid[e]["with_year_placebo"]
            echo_fy = be["pred_echo"] != ye["pred_echo"]
            chat_fy = be["pred_chat"] != ye["pred_chat"]
            echo_fp = be["pred_echo"] != pe["pred_echo"]
            chat_fp = be["pred_chat"] != pe["pred_chat"]
            flip_y_agree += echo_fy == chat_fy
            flip_p_agree += echo_fp == chat_fp
            pred_agree_base += be["pred_echo"] == be["pred_chat"]
        n = len(complete)
        print(f"  baseline pred agreement:     {pred_agree_base/n:.3f}")
        print(f"  year-flip agreement:         {flip_y_agree/n:.3f}")
        print(f"  placebo-flip agreement:      {flip_p_agree/n:.3f}")

    if df["survey"].notna().any():
        print("\n=== Agreement by survey ===")
        for s, sub in df.groupby("survey"):
            print(f"  {s:18s}  {sub['agree'].mean():.3f}  n={len(sub)}")


if __name__ == "__main__":
    main()
