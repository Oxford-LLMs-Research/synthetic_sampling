"""Build analysis/xgboost_baseline/results_matched_to_llm.csv.

This file aligns the XGBoost per-question results with the LLM per-question
results so the two can be compared question by question
(analyze_llm_vs_xgb_per_question.py). It previously existed on disk with no
generating script, which meant the paper's per-question comparison could not be
regenerated.

Why a "matching" step existed at all: the LLM-side results once carried the
corrupted Arab Barometer and WVS target codes that repair_ids.py fixes (`Q725_5`
stored as `5`, `Q201B_13` as `13`), so 39 cells could not be joined on
target_code and were matched by question code instead. Both sides now carry
repaired codes, so a straight join on (survey, target_code, profile_type)
resolves everything that can be resolved, and the `via_qcodes` category no
longer arises. Four targets have no XGBoost result at all and are kept with a
null so the coverage stays visible rather than silently dropped.

Usage:
    python build_xgb_matched_to_llm.py [--compare]

`--compare` diffs the regenerated file against the one on disk instead of
overwriting it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
LLM = ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv"
XGB = ANALYSIS / "xgboost_baseline" / "results_merged.csv"
OUT = ANALYSIS / "xgboost_baseline" / "results_matched_to_llm.csv"

KEYS = ["survey", "target_code", "profile_type"]


def build() -> pd.DataFrame:
    llm = pd.read_csv(LLM)
    xgb = pd.read_csv(XGB)

    cells = llm.drop_duplicates(KEYS)[KEYS].copy()
    xgb_cells = xgb.drop_duplicates(KEYS)[KEYS + ["xgb_norm_acc"]]
    out = cells.merge(xgb_cells, on=KEYS, how="left")
    out = out.rename(columns={"xgb_norm_acc": "xgb_norm_acc_matched"})
    out["match_type"] = out["xgb_norm_acc_matched"].notna().map(
        {True: "direct", False: "no_xgb"})
    return out.sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="diff against the existing file instead of writing")
    args = ap.parse_args()

    fresh = build()
    n_direct = int((fresh["match_type"] == "direct").sum())
    print(f"{len(fresh)} LLM cells: {n_direct} matched to XGBoost, "
          f"{len(fresh) - n_direct} without an XGBoost result")
    missing = fresh[fresh["match_type"] == "no_xgb"]
    if len(missing):
        print("  no XGBoost result for:")
        for (s, t), g in missing.groupby(["survey", "target_code"]):
            print(f"    {s} {t} ({len(g)} profile levels)")

    if args.compare and OUT.exists():
        old = pd.read_csv(OUT)
        print(f"\nexisting file: {len(old)} rows, "
              f"types {old['match_type'].value_counts().to_dict()}")
        j = old.merge(fresh, on=KEYS, how="outer", suffixes=("_old", "_new"),
                      indicator=True)
        print(f"rows only in existing: {(j['_merge'] == 'left_only').sum()}")
        print(f"rows only in regenerated: {(j['_merge'] == 'right_only').sum()}")
        both = j[j["_merge"] == "both"]
        diff = both[(both["xgb_norm_acc_matched_old"].round(9)
                     != both["xgb_norm_acc_matched_new"].round(9))
                    & ~(both["xgb_norm_acc_matched_old"].isna()
                        & both["xgb_norm_acc_matched_new"].isna())]
        print(f"shared rows with a different value: {len(diff)}")
        if len(diff):
            print(diff[KEYS + ["xgb_norm_acc_matched_old",
                               "xgb_norm_acc_matched_new",
                               "match_type_old"]].head(12).to_string(index=False))
        return

    fresh.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
