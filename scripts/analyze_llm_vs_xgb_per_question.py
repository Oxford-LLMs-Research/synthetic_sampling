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

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
OUT = ANALYSIS / "llm_vs_xgb"
OUT.mkdir(exist_ok=True)
FIG_DIR = ANALYSIS / "figures" / "emnlp_revision"
AAAI_FIG_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")

N_BOOT = 10_000
SEED = 42
BEST_MODEL = "qwen3-32b"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def bootstrap_table(merged: pd.DataFrame, label: str) -> pd.DataFrame:
    """Question-level bootstrap CIs for each model and XGBoost."""
    rng = np.random.default_rng(SEED)
    wide = merged.pivot_table(index=["survey", "target_code"],
                              columns="model", values="norm_acc", aggfunc="mean")
    wide["xgboost"] = merged.groupby(["survey", "target_code"])["xgb_norm_acc_matched"].mean()
    wide = wide.dropna()
    mat = wide.to_numpy()
    idx = rng.integers(0, len(mat), size=(N_BOOT, len(mat)))
    boot = mat[idx].mean(axis=1)
    rows = []
    for j, name in enumerate(wide.columns):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        rows.append({"model": name, "mean_norm_acc": mat[:, j].mean(),
                     "ci_lo": lo, "ci_hi": hi,
                     "n_questions": len(mat), "subset": label})
    return pd.DataFrame(rows).sort_values("mean_norm_acc", ascending=False)


def plot_scatter(rich: pd.DataFrame) -> None:
    """Best LLM vs XGBoost per question at rich profiles."""
    g = rich[rich["model"] == BEST_MODEL]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    lims = (min(g["xgb_norm_acc_matched"].min(), g["norm_acc"].min()) - 0.04,
            max(g["xgb_norm_acc_matched"].max(), g["norm_acc"].max()) + 0.04)
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=0.7, alpha=0.8,
            zorder=1)
    ax.scatter(g["xgb_norm_acc_matched"], g["norm_acc"], s=13, alpha=0.55,
               color="#2E5090", edgecolor="none", zorder=3)
    win = (g["norm_acc"] > g["xgb_norm_acc_matched"]).mean()
    ax.text(0.03, 0.97, f"LLM above diagonal:\n{win:.0%} of questions",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlabel("XGBoost normalized accuracy (same features)", fontsize=9)
    ax.set_ylabel("Qwen 3 32B normalized accuracy", fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(linestyle="--", alpha=0.2, linewidth=0.4)
    plt.tight_layout(pad=0.5)
    base = FIG_DIR / "figure_llm_vs_xgb_scatter"
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")
    if AAAI_FIG_DIR.exists():
        shutil.copy2(base.with_suffix(".pdf"),
                     AAAI_FIG_DIR / "figure_llm_vs_xgb_scatter.pdf")


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
    ci_pooled = bootstrap_table(merged, "all_profiles")
    ci_pooled.to_csv(OUT / "bootstrap_cis.csv", index=False)

    rich = merged[merged["profile_type"] == "s6m4"]
    ci_rich = bootstrap_table(rich, "s6m4")
    ci_rich.to_csv(OUT / "bootstrap_cis_rich.csv", index=False)

    summary_rich = (
        rich.groupby("model")
        .agg(
            n_cells=("gap", "size"),
            mean_llm_norm_acc=("norm_acc", "mean"),
            mean_gap=("gap", "mean"),
            win_rate=("llm_wins", "mean"),
        )
        .sort_values("mean_llm_norm_acc", ascending=False)
    )
    summary_rich.to_csv(OUT / "summary_by_model_rich.csv")

    plot_scatter(rich)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("== all profiles ==")
        print(summary.round(3))
        print("\n== rich (s6m4) ==")
        print(summary_rich.round(3))
        print()
        print(ci_rich.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
