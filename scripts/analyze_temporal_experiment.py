"""Analyze temporal-context experiment results (paired within-instance).

Primary contrasts (all instances):
  with_year - baseline
  with_year_placebo - baseline
  with_year - with_year_placebo   # contentful year info

Secondary ablation (instances with with_date scored):
  with_date - baseline
  with_date - with_year

Also reports prediction flip rates vs baseline.

Usage:
    python analyze_temporal_experiment.py \\
        --instances ../outputs/colab_experiments/temporal_context_instances.jsonl \\
        --results temporal_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    instances = {d["example_id"]: d for d in load_jsonl(Path(args.instances))}
    results = load_jsonl(Path(args.results))
    if not results:
        raise SystemExit(f"no results in {args.results}")

    # (example_id, condition) -> row; last write wins if duplicates
    by_key: dict[tuple, dict] = {}
    for r in results:
        by_key[(r["example_id"], r["condition"])] = r

    # Pivot to wide per example_id
    conds_seen = sorted({c for _, c in by_key})
    rows = []
    for eid, inst in instances.items():
        rec = {
            "example_id": eid,
            "survey": inst["survey"],
            "survey_year": inst.get("survey_year"),
            "survey_year_placebo": inst.get("survey_year_placebo"),
            "has_interview_date": bool(inst.get("interview_date")),
            "date_precision": inst.get("date_precision"),
            "target_code": inst.get("target_code"),
        }
        complete_primary = True
        for cond in ("baseline", "with_year", "with_year_placebo"):
            r = by_key.get((eid, cond))
            if r is None:
                complete_primary = False
                rec[f"correct_{cond}"] = None
                rec[f"pred_{cond}"] = None
            else:
                rec[f"correct_{cond}"] = bool(r["correct"])
                rec[f"pred_{cond}"] = r["predicted"]
        r_date = by_key.get((eid, "with_date"))
        if r_date is not None:
            rec["correct_with_date"] = bool(r_date["correct"])
            rec["pred_with_date"] = r_date["predicted"]
        else:
            rec["correct_with_date"] = None
            rec["pred_with_date"] = None
        rec["complete_primary"] = complete_primary
        rows.append(rec)

    df = pd.DataFrame(rows)
    primary = df[df["complete_primary"]].copy()
    print(f"instances in file: {len(df)}")
    print(f"complete primary (baseline+year+placebo): {len(primary)}")
    print(f"conditions present in results: {conds_seen}")
    if primary.empty:
        raise SystemExit("no complete primary triples to analyze")

    def paired_ci(a: pd.Series, b: pd.Series, n_boot: int = 10000,
                  seed: int = 20260727) -> tuple[float, float]:
        """95% CI for mean(a) - mean(b), resampling instances (they are paired)."""
        import numpy as np
        d = (a.astype(float) - b.astype(float)).to_numpy()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].mean(axis=1)
        return tuple(np.percentile(boots, [2.5, 97.5]))

    def summarize(sub: pd.DataFrame, label: str) -> None:
        n = len(sub)
        acc = {
            "baseline": sub["correct_baseline"].mean(),
            "with_year": sub["correct_with_year"].mean(),
            "with_year_placebo": sub["correct_with_year_placebo"].mean(),
        }
        d_true = acc["with_year"] - acc["baseline"]
        d_placebo = acc["with_year_placebo"] - acc["baseline"]
        d_content = acc["with_year"] - acc["with_year_placebo"]
        lo_t, hi_t = paired_ci(sub["correct_with_year"], sub["correct_baseline"])
        lo_p, hi_p = paired_ci(sub["correct_with_year_placebo"], sub["correct_baseline"])
        lo_c, hi_c = paired_ci(sub["correct_with_year"], sub["correct_with_year_placebo"])
        flip_true = (sub["pred_with_year"] != sub["pred_baseline"]).mean()
        flip_placebo = (sub["pred_with_year_placebo"] != sub["pred_baseline"]).mean()
        print(f"\n=== {label} (n={n}) ===")
        print(f"  acc baseline          {acc['baseline']:.4f}")
        print(f"  acc with_year         {acc['with_year']:.4f}   "
              f"delta vs baseline {d_true:+.4f} [{lo_t:+.4f}, {hi_t:+.4f}]")
        print(f"  acc with_year_placebo {acc['with_year_placebo']:.4f}   "
              f"delta vs baseline {d_placebo:+.4f} [{lo_p:+.4f}, {hi_p:+.4f}]")
        print(f"  contentful (true-placebo) {d_content:+.4f} [{lo_c:+.4f}, {hi_c:+.4f}]")
        print(f"  flip rate true year   {flip_true:.4f}")
        print(f"  flip rate placebo     {flip_placebo:.4f}")

    summarize(primary, "OVERALL primary")
    for survey, sub in primary.groupby("survey"):
        summarize(sub, survey)

    # Fine-date ablation: need baseline + with_year + with_date
    fine = primary[primary["correct_with_date"].notna()].copy()
    if len(fine):
        print(f"\n=== FINE-DATE ablation (n={len(fine)}) ===")
        acc_b = fine["correct_baseline"].mean()
        acc_y = fine["correct_with_year"].mean()
        acc_d = fine["correct_with_date"].mean()
        print(f"  acc baseline   {acc_b:.4f}")
        print(f"  acc with_year  {acc_y:.4f}   delta vs baseline {acc_y - acc_b:+.4f}")
        print(f"  acc with_date  {acc_d:.4f}   delta vs baseline {acc_d - acc_b:+.4f}")
        print(f"  with_date - with_year {acc_d - acc_y:+.4f}")
        print(f"  flip date vs baseline "
              f"{(fine['pred_with_date'] != fine['pred_baseline']).mean():.4f}")
        print(f"  flip date vs year     "
              f"{(fine['pred_with_date'] != fine['pred_with_year']).mean():.4f}")
        print("\n  by survey (fine-date subset):")
        for survey, sub in fine.groupby("survey"):
            print(
                f"    {survey:18s} n={len(sub):4d}  "
                f"base={sub['correct_baseline'].mean():.3f}  "
                f"year={sub['correct_with_year'].mean():.3f}  "
                f"date={sub['correct_with_date'].mean():.3f}"
            )
    else:
        print("\n(no with_date results yet — skip fine-date ablation)")

    # Coverage of incomplete rows
    incomplete = df[~df["complete_primary"]]
    if len(incomplete):
        print(f"\nincomplete primary triples: {len(incomplete)} "
              f"(re-run runner to resume)")


if __name__ == "__main__":
    main()
