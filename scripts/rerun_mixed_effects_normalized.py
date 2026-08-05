"""
Mixed effects model rerun with normalized accuracy (EMNLP revision).

Builds two versions:
  V1 (robustness): raw accuracy ~ n_options + modal_share + model + region +
                   topic_section + (1|survey)
  V2 (main paper): norm_acc ~ modal_share + model + region + topic_section +
                   (1|survey)

Where norm_acc = (raw_acc - 1/M) / (1 - 1/M), M = n_options.
In V2, n_options is baked into the outcome, so it drops as a control.
This makes topic/region coefficients interpretable without the n_options confound.

Data source: existing analysis/mixed_effects/mixed_effects_data.csv
Outputs:
  analysis/mixed_effects/mem_v1_raw_accuracy.txt
  analysis/mixed_effects/mem_v2_normalized_accuracy.txt
  analysis/mixed_effects/mem_variance_components.csv   (both versions)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
except ImportError:
    print("ERROR: statsmodels required. pip install statsmodels")
    sys.exit(1)

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MEM_DIR = ROOT / "synthetic_sampling/analysis/mixed_effects"
OUT_DIR = ROOT / "analysis/mixed_effects"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEM_DATA_PATH = MEM_DIR / "mixed_effects_data.csv"


def normalized_accuracy(raw_acc: float, n_options: float) -> float:
    if n_options <= 1 or np.isnan(n_options):
        return np.nan
    chance = 1.0 / n_options
    return (raw_acc - chance) / (1.0 - chance)


# ---------------------------------------------------------------------------
# Load existing mixed_effects_data.csv
# ---------------------------------------------------------------------------
print("Loading mixed_effects_data.csv ...")
df = pd.read_csv(MEM_DATA_PATH, encoding="latin-1", low_memory=False)
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

# Ensure n_options is numeric
df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce")
df["modal_share"] = pd.to_numeric(df["modal_share"], errors="coerce")
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")

# Drop rows with missing critical fields
n_before = len(df)
df = df.dropna(subset=["correct", "n_options", "modal_share", "model",
                        "region", "topic_section", "survey", "question"])
print(f"  {n_before - len(df):,} rows dropped (missing fields), {len(df):,} remain")

# ---------------------------------------------------------------------------
# Aggregate to per-(model, question, region, topic_section, survey) level
# This reduces 2.5M rows to a tractable ~100k aggregated cells and provides
# a continuous outcome (mean accuracy) rather than binary 0/1.
# ---------------------------------------------------------------------------
print("\nAggregating to per-cell level (model x question x region)...")
agg_cols = ["model", "question", "region", "topic_section", "survey",
            "target_code", "n_options", "modal_share"]

# modal_share and n_options are the same for all respondents in a cell
cell_df = df.groupby(agg_cols).agg(
    raw_acc=("correct", "mean"),
    n_respondents=("correct", "count"),
).reset_index()

cell_df["norm_acc"] = cell_df.apply(
    lambda r: normalized_accuracy(r["raw_acc"], r["n_options"]), axis=1
)

# Drop cells with NaN norm_acc
cell_df = cell_df.dropna(subset=["norm_acc"])
print(f"  Aggregated to {len(cell_df):,} cells")
print(f"  Models: {cell_df['model'].nunique()}, Questions: {cell_df['question'].nunique()}")
print(f"  Regions: {cell_df['region'].nunique()}, Surveys: {cell_df['survey'].nunique()}")

# Quick peek
print(f"\n  Mean raw_acc: {cell_df['raw_acc'].mean():.4f}")
print(f"  Mean norm_acc: {cell_df['norm_acc'].mean():.4f}")


def fit_and_summarize(data: pd.DataFrame, formula: str, groups: str,
                      label: str) -> dict:
    """Fit a linear mixed effects model and return summary stats."""
    print(f"\n{'='*60}")
    print(f"Fitting {label}")
    print(f"  Formula: {formula}")
    print(f"  Groups (random intercept): {groups}")
    print(f"  N cells: {len(data):,}")

    model = smf.mixedlm(formula, data=data, groups=data[groups])
    result = model.fit(reml=True, method="lbfgs")
    print(result.summary())

    # Variance components
    re_var = float(result.cov_re.iloc[0, 0])
    resid_var = float(result.scale)
    total_var = re_var + resid_var
    icc = re_var / total_var if total_var > 0 else np.nan

    summary_dict = {
        "label": label,
        "n_obs": len(data),
        "n_groups": data[groups].nunique(),
        "re_variance": round(re_var, 6),
        "residual_variance": round(resid_var, 6),
        "total_variance": round(total_var, 6),
        "icc": round(icc, 4),
        "aic": round(result.aic, 2),
        "bic": round(result.bic, 2),
        "converged": result.converged,
    }

    return summary_dict, result


# ---------------------------------------------------------------------------
# V1: Raw accuracy with n_options + modal_share controls
# ---------------------------------------------------------------------------
v1_formula = "raw_acc ~ n_options + modal_share + model + region + topic_section"
v1_summary, v1_result = fit_and_summarize(
    cell_df, v1_formula, groups="survey", label="V1 (raw accuracy)"
)

v1_txt = v1_result.summary().as_text()
v1_path = OUT_DIR / "mem_v1_raw_accuracy.txt"
with open(v1_path, "w", encoding="utf-8") as f:
    f.write(v1_txt)
print(f"\nSaved V1 summary to {v1_path}")

# ---------------------------------------------------------------------------
# V2: Normalized accuracy (n_options baked in)
# ---------------------------------------------------------------------------
v2_formula = "norm_acc ~ modal_share + model + region + topic_section"
v2_summary, v2_result = fit_and_summarize(
    cell_df, v2_formula, groups="survey", label="V2 (normalized accuracy)"
)

v2_txt = v2_result.summary().as_text()
v2_path = OUT_DIR / "mem_v2_normalized_accuracy.txt"
with open(v2_path, "w", encoding="utf-8") as f:
    f.write(v2_txt)
print(f"Saved V2 summary to {v2_path}")

# ---------------------------------------------------------------------------
# Variance components comparison table
# ---------------------------------------------------------------------------
var_df = pd.DataFrame([v1_summary, v2_summary])
var_path = OUT_DIR / "mem_variance_components.csv"
var_df.to_csv(var_path, index=False)
print(f"Saved variance components to {var_path}")

print("\nVariance components comparison:")
print(var_df[["label", "re_variance", "residual_variance", "icc", "converged"]].to_string(index=False))
