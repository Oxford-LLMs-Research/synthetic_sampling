"""
Within-country majority-class robustness check.

For each (model, survey, target_code, country) cell with >= MIN_N respondents,
compute whether the model's accuracy exceeds the within-country modal response
accuracy (i.e. always predicting the most common answer in that country).

This addresses reviewer iYUR's concern that cross-national majority-class
baselines conflate different political realities.

Outputs:
  analysis/within_country_majority/per_cell_results.csv   (one row per cell)
  analysis/within_country_majority/model_summary.csv      (one row per model)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
JSONL_DIR = ROOT / "synthetic_sampling/outputs/main_data_smaller_20_jan_26/main_data"
OUT_DIR = ROOT / "analysis/within_country_majority"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N = 20   # minimum respondents per (question x country) cell

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

# ---------------------------------------------------------------------------
# Build example_id -> country lookup from JSONL
# ---------------------------------------------------------------------------
print("Building example_id -> country lookup from JSONL files...")
id_to_country = {}
for fname in JSONL_FILES:
    path = JSONL_DIR / fname
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            inst = json.loads(line)
            id_to_country[inst["example_id"]] = inst["country"]
print(f"  {len(id_to_country):,} example_id -> country mappings loaded")

# ---------------------------------------------------------------------------
print(f"\nProcessing {len(MODEL_DIRS)} models...")

all_cell_rows = []
model_summary_rows = []

for model in sorted(MODEL_DIRS):
    csv_path = ANALYSIS_DIR / model / "results_data.csv"
    df = pd.read_csv(csv_path)

    # Join country via example_id -> country lookup
    df["country"] = df["example_id"].map(id_to_country)
    if df["country"].isna().sum() > 0:
        print(f"  [{model}] WARNING: {df['country'].isna().sum()} rows missing country mapping")

    # For each (target_code, country, profile_type) with enough respondents
    group_cols = ["survey", "target_code", "country", "profile_type"]
    cells = []

    for keys, grp in df.groupby(group_cols):
        if len(grp) < MIN_N:
            continue

        survey, target_code, country, profile_type = keys

        # Within-country majority-class accuracy = share of modal response
        modal_share = grp["ground_truth"].value_counts(normalize=True).iloc[0]

        # Model accuracy on this cell
        model_acc = grp["correct"].mean()

        cells.append({
            "model": model,
            "survey": survey,
            "target_code": target_code,
            "country": country,
            "profile_type": profile_type,
            "n_respondents": len(grp),
            "within_country_majority_acc": round(modal_share, 4),
            "model_acc": round(model_acc, 4),
            "beats_within_country_mode": int(model_acc > modal_share),
            "gap_vs_majority": round(model_acc - modal_share, 4),
        })

    all_cell_rows.extend(cells)
    cells_df = pd.DataFrame(cells)

    if cells_df.empty:
        print(f"  [{model}] no valid cells (all < {MIN_N} respondents)")
        continue

    # Model-level summary: % of cells where model beats within-country mode
    pct_beats = cells_df["beats_within_country_mode"].mean() * 100
    mean_gap = cells_df["gap_vs_majority"].mean()
    model_summary_rows.append({
        "model": model,
        "n_cells": len(cells_df),
        "pct_beats_within_country_mode": round(pct_beats, 2),
        "mean_gap_vs_majority": round(mean_gap, 4),
        "mean_model_acc": round(cells_df["model_acc"].mean(), 4),
        "mean_within_country_majority_acc": round(cells_df["within_country_majority_acc"].mean(), 4),
    })
    print(f"  [{model}] {len(cells_df)} cells, {pct_beats:.1f}% beat within-country mode")

# Save outputs
if all_cell_rows:
    cells_out = pd.DataFrame(all_cell_rows)
    per_cell_path = OUT_DIR / "per_cell_results.csv"
    cells_out.to_csv(per_cell_path, index=False)
    print(f"\nSaved per-cell results: {len(cells_out)} rows to {per_cell_path}")
else:
    print("\nWARNING: No cell results produced — results_data.csv may lack 'country' column")

if model_summary_rows:
    summary_df = pd.DataFrame(model_summary_rows)
    summary_path = OUT_DIR / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model summary: {len(summary_df)} rows to {summary_path}")
    print("\nModel summary:")
    print(summary_df.sort_values("mean_model_acc", ascending=False).to_string(index=False))
