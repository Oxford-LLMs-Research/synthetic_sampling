"""
Merge partial XGBoost baseline results from separate survey runs.
Usage:
    python merge_xgb_results.py  # merges all results_*.csv in xgboost_baseline dir
"""
import os
from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "analysis/xgboost_baseline"

parts = [f for f in os.listdir(OUT_DIR) if f.startswith("results") and f.endswith(".csv")]
print(f"Found {len(parts)} result files: {parts}")

dfs = []
for fname in parts:
    df = pd.read_csv(OUT_DIR / fname)
    print(f"  {fname}: {len(df)} rows, surveys: {df['survey'].unique().tolist()}")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["survey", "target_code", "profile_type"])
print(f"\nMerged: {len(merged)} rows, surveys: {sorted(merged['survey'].unique().tolist())}")

merged.to_csv(OUT_DIR / "results_merged.csv", index=False)
print(f"Saved results_merged.csv")
print("\nMean norm_acc by profile_type:")
print(merged.groupby("profile_type")[["majority_acc", "xgb_acc", "majority_norm_acc", "xgb_norm_acc"]].mean().round(3))
