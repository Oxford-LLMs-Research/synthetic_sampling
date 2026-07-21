"""
Compute normalized accuracy for all LLM results.

Formula: norm_acc = (raw_acc - 1/M) / (1 - 1/M)
where M = number of answer options for that question.

Maps random chance to 0. Majority-class baseline and LLM results
become comparable across binary / 4-point / 10-point questions.

Sources:
  - analysis/<model>/results_data.csv  (predicted, correct, n_features, etc.)
  - synthetic_sampling/outputs/.../*.jsonl  (for n_options per question)

Outputs:
  analysis/normalized_accuracy/per_question_norm_acc.csv
  analysis/normalized_accuracy/aggregate_by_model.csv
  analysis/normalized_accuracy/majority_class_norm_acc.csv
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
JSONL_DIR = ROOT / "synthetic_sampling/outputs/main_data_smaller_20_jan_26/main_data"
ANALYSIS_DIR = ROOT / "analysis"
OUT_DIR = ROOT / "analysis/normalized_accuracy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_FILES = [
    "afrobarometer_instances.jsonl",
    "arabbarometer_instances.jsonl",
    "asianbarometer_instances.jsonl",
    "ess_wave_10_instances.jsonl",
    "ess_wave_11_instances.jsonl",
    "latinobarometer_instances.jsonl",
    "wvs_instances.jsonl",
]

MODEL_DIRS = [
    d for d in os.listdir(ANALYSIS_DIR)
    if (ANALYSIS_DIR / d / "results_data.csv").exists()
]


def normalized_accuracy(acc: float, n_options: int) -> float:
    if n_options <= 1:
        return np.nan
    chance = 1.0 / n_options
    return (acc - chance) / (1.0 - chance)


# ---------------------------------------------------------------------------
# Step 1: build (survey, target_code) -> n_options lookup from JSONL
# ---------------------------------------------------------------------------
print("Building n_options lookup from JSONL files...")
n_options_map = {}   # (survey, target_code) -> int

for fname in JSONL_FILES:
    path = JSONL_DIR / fname
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            inst = json.loads(line)
            key = (inst["survey"], inst["target_code"])
            if key not in n_options_map:
                n_options_map[key] = len(inst["options"])

print(f"  {len(n_options_map)} (survey, target) pairs with n_options")


# ---------------------------------------------------------------------------
# Step 2: compute per-question normalized accuracy for each model
# ---------------------------------------------------------------------------
print(f"\nProcessing {len(MODEL_DIRS)} model directories...")

per_question_rows = []
aggregate_rows = []

for model in sorted(MODEL_DIRS):
    csv_path = ANALYSIS_DIR / model / "results_data.csv"
    df = pd.read_csv(csv_path)

    # Add n_options
    df["n_options"] = df.apply(
        lambda r: n_options_map.get((r["survey"], r["target_code"]), np.nan), axis=1
    )
    missing_opts = df["n_options"].isna().sum()
    if missing_opts > 0:
        print(f"  [{model}] WARNING: {missing_opts} rows missing n_options")

    # Per-question accuracy (group by survey, target_code, profile_type)
    group_cols = ["survey", "target_code", "profile_type"]
    agg = df.groupby(group_cols).agg(
        n_respondents=("correct", "count"),
        raw_acc=("correct", "mean"),
        n_options=("n_options", "first"),
    ).reset_index()

    agg["norm_acc"] = agg.apply(
        lambda r: normalized_accuracy(r["raw_acc"], r["n_options"]), axis=1
    )
    agg["model"] = model
    per_question_rows.append(agg)

    # Aggregate per profile_type
    for pt, grp in agg.groupby("profile_type"):
        aggregate_rows.append({
            "model": model,
            "profile_type": pt,
            "mean_raw_acc": grp["raw_acc"].mean(),
            "mean_norm_acc": grp["norm_acc"].mean(),
            "n_questions": len(grp),
        })

    print(f"  [{model}] done, {len(agg)} (target, profile_type) cells")


per_question_df = pd.concat(per_question_rows, ignore_index=True)
aggregate_df = pd.DataFrame(aggregate_rows)

per_q_path = OUT_DIR / "per_question_norm_acc.csv"
agg_path = OUT_DIR / "aggregate_by_model.csv"
per_question_df.to_csv(per_q_path, index=False)
aggregate_df.to_csv(agg_path, index=False)
print(f"\nSaved per-question table: {len(per_question_df)} rows to {per_q_path}")
print(f"Saved aggregate table: {len(aggregate_df)} rows to {agg_path}")


# ---------------------------------------------------------------------------
# Step 3: majority-class normalized accuracy (per question, not per model)
# ---------------------------------------------------------------------------
print("\nComputing majority-class normalized accuracy...")

# Use one model's results_data as the reference for ground truth distributions
ref_model = sorted(MODEL_DIRS)[0]
ref_df = pd.read_csv(ANALYSIS_DIR / ref_model / "results_data.csv")
ref_df["n_options"] = ref_df.apply(
    lambda r: n_options_map.get((r["survey"], r["target_code"]), np.nan), axis=1
)

maj_rows = []
for (survey, target_code, profile_type), grp in ref_df.groupby(["survey", "target_code", "profile_type"]):
    modal_share = grp["ground_truth"].value_counts(normalize=True).iloc[0]
    n_opts = grp["n_options"].iloc[0]
    maj_rows.append({
        "survey": survey,
        "target_code": target_code,
        "profile_type": profile_type,
        "n_respondents": len(grp),
        "majority_acc": modal_share,
        "n_options": n_opts,
        "majority_norm_acc": normalized_accuracy(modal_share, n_opts),
    })

maj_df = pd.DataFrame(maj_rows)
maj_path = OUT_DIR / "majority_class_norm_acc.csv"
maj_df.to_csv(maj_path, index=False)
print(f"Saved majority-class table: {len(maj_df)} rows to {maj_path}")

# Quick summary
print("\nMean normalized accuracy by profile_type:")
summary = aggregate_df.groupby("profile_type")["mean_norm_acc"].mean()
print(summary.round(4))
print("\nMean majority-class normalized accuracy by profile_type:")
print(maj_df.groupby("profile_type")["majority_norm_acc"].mean().round(4))
