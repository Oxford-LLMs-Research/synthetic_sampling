"""Per-question comparison of LLM vs XGBoost normalized accuracy.

The paper's headline numbers (best LLM 16.1% vs XGBoost 34.5% normalized
accuracy) are aggregates. This script disaggregates the comparison to the
question level:

  1. For each (survey, target, profile_type) cell, compare each model's
     normalized accuracy with the matched XGBoost run on identical features.
  2. Report the share of cells where any/each LLM beats XGBoost, and the gap
     distribution.
  3. Bootstrap 95% CIs (resampling questions) for aggregate normalized
     accuracy of each model and XGBoost.

Inputs (relative to the analysis root):
  normalized_accuracy/per_question_norm_acc_fixed.csv
  xgboost_baseline/results_matched_to_llm.csv

Outputs to analysis/llm_vs_xgb/:
  per_cell_comparison.csv   one row per (survey, target, profile, model)
  summary_by_model.csv      win rates + mean gap per model
  bootstrap_cis.csv         95% CIs for aggregate norm acc per model + XGB
"""

import numpy as np
import pandas as pd
from pathlib import Path

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
OUT = ANALYSIS / "llm_vs_xgb"
OUT.mkdir(exist_ok=True)

N_BOOT = 10_000
SEED = 42


def main() -> None:
    llm = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
    xgb = pd.read_csv(ANALYSIS / "xgboost_baseline" / "results_matched_to_llm.csv")

    keys = ["survey", "target_code", "profile_type"]
    merged = llm.merge(xgb[keys + ["xgb_norm_acc_matched"]], on=keys, how="inner")
    merged["gap"] = merged["norm_acc"] - merged["xgb_norm_acc_matched"]
    merged["llm_wins"] = merged["gap"] > 0
    merged.to_csv(OUT / "per_cell_comparison.csv", index=False)

    summary = (
        merged.groupby("model")
        .agg(
            n_cells=("gap", "size"),
            mean_llm_norm_acc=("norm_acc", "mean"),
            mean_xgb_norm_acc=("xgb_norm_acc_matched", "mean"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            win_rate=("llm_wins", "mean"),
        )
        .sort_values("mean_llm_norm_acc", ascending=False)
    )
    summary.to_csv(OUT / "summary_by_model.csv")

    # Bootstrap over questions: resample (survey, target) units so that CIs
    # reflect question-level sampling variability, the level reviewers care
    # about, holding the respondent sample within each question fixed.
    rng = np.random.default_rng(SEED)
    rows = []
    q_units = merged[["survey", "target_code"]].drop_duplicates().reset_index(drop=True)

    wide = merged.pivot_table(
        index=["survey", "target_code"],
        columns="model",
        values="norm_acc",
        aggfunc="mean",
    )
    wide["xgboost"] = merged.groupby(["survey", "target_code"])["xgb_norm_acc_matched"].mean()
    wide = wide.dropna()
    mat = wide.to_numpy()
    n_q = mat.shape[0]
    idx = rng.integers(0, n_q, size=(N_BOOT, n_q))
    boot_means = mat[idx].mean(axis=1)  # (N_BOOT, n_models+1)

    for j, name in enumerate(wide.columns):
        lo, hi = np.percentile(boot_means[:, j], [2.5, 97.5])
        rows.append(
            {
                "model": name,
                "mean_norm_acc": mat[:, j].mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_questions": n_q,
                "n_boot": N_BOOT,
            }
        )
    pd.DataFrame(rows).sort_values("mean_norm_acc", ascending=False).to_csv(
        OUT / "bootstrap_cis.csv", index=False
    )

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3))
        print()
        print(pd.DataFrame(rows).sort_values("mean_norm_acc", ascending=False).round(3))


if __name__ == "__main__":
    main()
