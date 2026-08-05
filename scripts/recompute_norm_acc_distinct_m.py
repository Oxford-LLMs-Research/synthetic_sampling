"""Recompute normalized accuracy with M = distinct answer labels.

The harmonization verbalizes ESS 0-10 scales into seven anchors, so 26 of the
267 targets present option lists with more slots than labels: `stfdem` carries
thirteen entries ('Extremely dissatisfied' twice, 'Neither satisfied nor
dissatisfied' three times, ...) but only seven distinct strings. Scoring is over
option strings and duplicates receive identical scores, so the predictor faces a
seven-way choice and both its prediction and the ground truth are labels.
Dividing by thirteen therefore puts chance at 0.07 rather than 0 for those
questions and inflates their normalized accuracy.

Normalized accuracy is a post-hoc transform of raw accuracy, so nothing has to
be refit: every derived file stores the raw accuracy and the option count next
to it. This script rewrites `n_options` to the distinct-label count and
recomputes the normalized columns in place, keeping a `.pre_distinct_m` backup
of each file it touches.

The mixed-effects inputs are the exception: `n_options` is a regressor there and
the models must be refit afterwards (fit_mixed_effects_model.py, ~1.2 min).

Usage:
    python recompute_norm_acc_distinct_m.py [--check]     # --check = report only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
ANALYSIS = ROOT / "analysis"
JSONL_DIR = (ROOT / "synthetic_sampling" / "outputs"
             / "main_data_smaller_20_jan_26" / "main_data")

# file -> [(raw_accuracy_column, normalized_column), ...]
TARGETS = {
    ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv":
        [("raw_acc", "norm_acc")],
    ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv":
        [("raw_acc", "norm_acc")],
    ANALYSIS / "normalized_accuracy" / "majority_class_norm_acc.csv":
        [("majority_acc", "majority_norm_acc")],
    ANALYSIS / "xgboost_baseline" / "results_merged.csv":
        [("majority_acc", "majority_norm_acc"), ("xgb_acc", "xgb_norm_acc")],
}


def distinct_m() -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for path in sorted(JSONL_DIR.glob("*_instances.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                key = (d["survey"], d["target_code"])
                if key not in out:
                    out[key] = len(set(d["options"]))
    return out


def norm(acc: pd.Series, m: pd.Series) -> pd.Series:
    return (acc - 1.0 / m) / (1.0 - 1.0 / m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report the change without writing")
    args = ap.parse_args()

    m_map = distinct_m()
    print(f"distinct-label M for {len(m_map)} (survey, target) pairs")

    for path, cols in TARGETS.items():
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        df = pd.read_csv(path)
        key = list(zip(df["survey"], df["target_code"]))
        new_m = pd.Series([m_map.get(k) for k in key], index=df.index)
        unmapped = int(new_m.isna().sum())
        changed = int((new_m != df["n_options"]).sum())

        print(f"\n{path.name}: {len(df)} rows, {changed} with a changed M"
              f"{f', {unmapped} unmapped' if unmapped else ''}")
        for raw_col, norm_col in cols:
            old = df[norm_col].mean()
            fixed = norm(df[raw_col], new_m.fillna(df["n_options"]))
            print(f"   {norm_col:<20} {old:.4f} -> {fixed.mean():.4f} "
                  f"({fixed.mean() - old:+.4f})")
            if not args.check:
                df[norm_col] = fixed
        if not args.check:
            df["n_options"] = new_m.fillna(df["n_options"]).astype(int)
            shutil.copy2(path, path.with_suffix(path.suffix + ".pre_distinct_m"))
            df.to_csv(path, index=False)

    # results_matched_to_llm.csv stores only the normalized value and has no
    # generating script. Every row traces to results_merged: 714 by an exact
    # (survey, target, profile) key and the rest by a question-code match whose
    # rule is not recorded. Recover the latter by their stored value.
    matched = ANALYSIS / "xgboost_baseline" / "results_matched_to_llm.csv"
    merged = ANALYSIS / "xgboost_baseline" / "results_merged.csv"
    if matched.exists() and merged.exists() and not args.check:
        mt = pd.read_csv(matched)
        bak = merged.with_suffix(merged.suffix + ".pre_distinct_m")
        old_merged = pd.read_csv(bak) if bak.exists() else pd.read_csv(merged)
        new_merged = pd.read_csv(merged)
        keys = ["survey", "target_code", "profile_type"]
        lut = (old_merged[keys + ["xgb_norm_acc"]]
               .merge(new_merged[keys + ["xgb_norm_acc"]], on=keys,
                      suffixes=("_old", "_new")))
        direct = lut.set_index(keys)["xgb_norm_acc_new"]
        by_old = dict(zip(lut["xgb_norm_acc_old"].round(10),
                          lut["xgb_norm_acc_new"]))

        def fix(row):
            if pd.isna(row["xgb_norm_acc_matched"]):
                return row["xgb_norm_acc_matched"]
            k = (row["survey"], row["target_code"], row["profile_type"])
            if k in direct.index:
                return direct.loc[k]
            return by_old.get(round(row["xgb_norm_acc_matched"], 10),
                              row["xgb_norm_acc_matched"])

        before = mt["xgb_norm_acc_matched"].mean()
        mt["xgb_norm_acc_matched"] = mt.apply(fix, axis=1)
        shutil.copy2(matched, matched.with_suffix(matched.suffix + ".pre_distinct_m"))
        mt.to_csv(matched, index=False)
        print(f"\n{matched.name}: {before:.4f} -> "
              f"{mt['xgb_norm_acc_matched'].mean():.4f}")

    if args.check:
        print("\n--check: nothing written")
    else:
        print("\nRefit the mixed-effects models next; n_options is a regressor "
              "there and prepare_mixed_effects_data.py must be rerun first.")


if __name__ == "__main__":
    main()
