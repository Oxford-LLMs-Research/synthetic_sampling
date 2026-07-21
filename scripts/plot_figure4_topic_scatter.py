#!/usr/bin/env python3
"""
Figure 4: Topic scatter — normalized accuracy for all 267 individual questions.

Each dot = one target question.
x-axis: Mean normalized accuracy across all models, profile types, and respondents.
y-axis: Cross-region SD in normalized accuracy (how much predictability varies geographically).
Color:  7-section taxonomy (Political Attitudes, Institutional Trust, etc.)
Large markers: section-level means overlaid on top.

Source: analysis/normalized_accuracy/per_question_norm_acc.csv
        analysis/mixed_effects/mixed_effects_data.csv  (for region labels)
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
NORM_ACC_DIR = ROOT / "analysis/normalized_accuracy"
MEM_DATA     = ROOT / "synthetic_sampling/analysis/mixed_effects/mixed_effects_data.csv"
OUT_DIR      = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_DISPLAY = {
    "contemporary_issues":     "Contemporary Issues",
    "institutional_trust":     "Institutional Trust",
    "political_attitudes":     "Political Attitudes",
    "political_participation":  "Pol. Participation",
    "social_attitudes":        "Social Attitudes",
    "values_identity":         "Values & Identity",
    "wellbeing":               "Wellbeing",
}

SECTION_COLORS = {
    "contemporary_issues":     "#4C72B0",
    "institutional_trust":     "#DD8452",
    "political_attitudes":     "#55A868",
    "political_participation":  "#C44E52",
    "social_attitudes":        "#8172B2",
    "values_identity":         "#937860",
    "wellbeing":               "#DA8BC3",
}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_question_level_data():
    """
    Build per-question stats from two sources:
    1. per_question_norm_acc.csv for mean norm_acc (aggregated over models+profiles)
    2. mixed_effects_data.csv for region labels (to compute cross-region SD)
    """
    # ---- Source 1: per-question normalized accuracy ----
    per_q_path = NORM_ACC_DIR / "per_question_norm_acc.csv"
    if per_q_path.exists():
        per_q = pd.read_csv(per_q_path)
        print(f"  Loaded per_question_norm_acc.csv: {len(per_q):,} rows")
    else:
        raise FileNotFoundError(
            f"Not found: {per_q_path}\n"
            "Run compute_normalized_accuracy.py first."
        )

    # ---- Source 2: MEM data for region + section labels ----
    mem_cols = ["model", "correct", "region", "topic_section",
                "survey", "target_code", "n_options"]
    mem = pd.read_csv(MEM_DATA, encoding="latin-1", usecols=mem_cols)
    mem = mem[mem["region"] != "Unknown"].copy()
    mem["n_options"] = pd.to_numeric(mem["n_options"], errors="coerce")
    mem["correct"] = pd.to_numeric(mem["correct"], errors="coerce")
    mem = mem.dropna(subset=["correct", "n_options"])
    mem = mem[mem["n_options"] > 1]
    mem["norm_acc"] = (mem["correct"] - 1.0 / mem["n_options"]) / (1.0 - 1.0 / mem["n_options"])
    mem = mem.dropna(subset=["norm_acc"])
    print(f"  Loaded MEM data: {len(mem):,} rows")

    # Cross-region SD per question (average over models, then SD across regions)
    region_q = (mem.groupby(["survey", "target_code", "region"])["norm_acc"]
                   .mean()
                   .reset_index())
    cross_region_sd = (region_q.groupby(["survey", "target_code"])["norm_acc"]
                                .std()
                                .reset_index()
                                .rename(columns={"norm_acc": "cross_region_sd"}))

    # Section labels (take mode per question)
    section_labels = (mem.groupby(["survey", "target_code"])["topic_section"]
                         .first()
                         .reset_index())

    # ---- Merge: per_question_norm_acc mean over all models + profiles ----
    q_mean = (per_q.groupby(["survey", "target_code"])["norm_acc"]
                   .mean()
                   .reset_index()
                   .rename(columns={"norm_acc": "mean_norm_acc"}))

    result = (q_mean
              .merge(cross_region_sd, on=["survey", "target_code"], how="inner")
              .merge(section_labels,  on=["survey", "target_code"], how="inner"))

    result = result.dropna(subset=["mean_norm_acc", "cross_region_sd"])
    print(f"  Questions with both metrics: {len(result)}")
    return result


def make_figure(df):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    sections = [s for s in SECTION_COLORS if s in df["topic_section"].unique()]

    # ---- Individual question dots ----
    for section in sections:
        sub = df[df["topic_section"] == section]
        ax.scatter(
            sub["mean_norm_acc"], sub["cross_region_sd"],
            color=SECTION_COLORS[section],
            s=18, alpha=0.55, edgecolors="none", zorder=2,
            label=SECTION_DISPLAY.get(section, section),
        )

    # ---- Section-level mean markers (larger, bold edges) ----
    section_means = df.groupby("topic_section")[["mean_norm_acc", "cross_region_sd"]].mean()
    for section, row in section_means.iterrows():
        ax.scatter(
            row["mean_norm_acc"], row["cross_region_sd"],
            color=SECTION_COLORS.get(section, "#888888"),
            s=160, alpha=1.0, edgecolors="black", linewidths=1.2,
            marker="D", zorder=5,
        )
        # Annotate section means
        ax.annotate(
            SECTION_DISPLAY.get(section, section),
            xy=(row["mean_norm_acc"], row["cross_region_sd"]),
            xytext=(row["mean_norm_acc"] + 0.003, row["cross_region_sd"] + 0.002),
            fontsize=7.5, ha="left", va="bottom", color="#222222",
            zorder=6,
        )

    ax.set_xlabel("Mean Normalized Accuracy\n(averaged over all models, surveys, profile levels)", fontsize=10)
    ax.set_ylabel("Cross-Region SD in Normalized Accuracy\n(higher = more geographically variable)", fontsize=10)
    ax.grid(linestyle="--", alpha=0.20, linewidth=0.4)

    # ---- Legend: section colors (dots only) ----
    legend_handles = [
        mpatches.Patch(facecolor=SECTION_COLORS[s],
                       label=SECTION_DISPLAY.get(s, s), alpha=0.8)
        for s in sections
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper right",
              framealpha=0.85, edgecolor="#cccccc")

    # Note on n
    n_questions = len(df)
    ax.text(0.01, 0.01, f"n = {n_questions} target questions",
            transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")

    # Pad axes
    x_range = df["mean_norm_acc"].max() - df["mean_norm_acc"].min()
    y_range = df["cross_region_sd"].max() - df["cross_region_sd"].min()
    ax.set_xlim(df["mean_norm_acc"].min() - x_range * 0.05,
                df["mean_norm_acc"].max() + x_range * 0.28)  # extra room for labels
    ax.set_ylim(df["cross_region_sd"].min() - y_range * 0.08,
                df["cross_region_sd"].max() + y_range * 0.15)

    plt.tight_layout(pad=0.8)

    out_base = OUT_DIR / "figure4_topic_scatter"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "figure_topic_scatter.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


def main():
    print("Loading question-level data...")
    df = load_question_level_data()

    print("\nSection-level summary:")
    summ = df.groupby("topic_section")[["mean_norm_acc", "cross_region_sd"]].describe()
    print(summ.to_string())

    print("\nGenerating figure...")
    make_figure(df)
    print("Done.")


if __name__ == "__main__":
    main()
