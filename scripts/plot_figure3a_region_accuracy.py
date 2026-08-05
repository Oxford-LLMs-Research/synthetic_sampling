#!/usr/bin/env python3
"""
Figure 3(a): Region accuracy — normalized accuracy by world region.

Horizontal dot plot with 95% CIs, colored by continent.

Source: synthetic_sampling/analysis/mixed_effects/mixed_effects_data.csv
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import paperfig as pf

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MEM_DATA = (ROOT / "synthetic_sampling/analysis/mixed_effects/mixed_effects_data.csv")
OUT_DIR = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTINENT_MAP = {
    "Central Africa": "Africa",   "East Africa": "Africa",
    "North Africa": "Africa",     "Southern Africa": "Africa",
    "West Africa": "Africa",
    "Central America": "Americas", "North America": "Americas",
    "South America": "Americas",   "Caribbean": "Americas",
    "Central Asia": "Asia",        "East Asia": "Asia",
    "South Asia": "Asia",          "Southeast Asia": "Asia",
    "Middle East": "Asia",
    "Eastern Europe": "Europe",    "Northern Europe": "Europe",
    "Southern Europe": "Europe",   "Western Europe": "Europe",
    "Oceania": "Oceania",
}

CONTINENT_COLORS = {
    "Africa":   "#8B4513",
    "Americas": "#4A7C3F",
    "Asia":     "#6B4C93",
    "Europe":   "#2E5090",
    "Oceania":  "#e7298a",
}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "patch.linewidth": 0.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def normalized_accuracy(correct, n_options):
    chance = 1.0 / n_options
    return (correct - chance) / (1.0 - chance)


def main():
    pf.use_style()
    print("Loading MEM data...")
    df = pd.read_csv(MEM_DATA, encoding="latin-1",
                     usecols=["model", "correct", "region", "n_options"])

    # Drop Unknown and invalid rows
    df = df[df["region"] != "Unknown"].copy()
    df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct", "n_options"])
    df = df[df["n_options"] > 1]

    df["norm_acc"] = normalized_accuracy(df["correct"], df["n_options"])
    df = df.dropna(subset=["norm_acc"])
    print(f"  {len(df):,} rows after cleaning")

    # Aggregate by region: mean norm_acc and SE (binomial on norm_acc ~0)
    region_agg = (df.groupby("region")["norm_acc"]
                    .agg(["mean", "count", "std"])
                    .reset_index()
                    .rename(columns={"mean": "norm_acc_mean", "count": "n", "std": "norm_acc_std"}))
    region_agg["se"] = region_agg["norm_acc_std"] / np.sqrt(region_agg["n"])
    region_agg["ci95"] = 1.96 * region_agg["se"]
    region_agg["continent"] = region_agg["region"].map(CONTINENT_MAP).fillna("Other")
    region_agg = region_agg.sort_values("norm_acc_mean").reset_index(drop=True)

    # Per-model regional means (small dots behind the pooled mean)
    per_model = (df.groupby(["region", "model"])["norm_acc"]
                   .mean()
                   .reset_index())

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(pf.COL, 3.45), layout="constrained")
    y_pos = np.arange(len(region_agg))

    for i, row in region_agg.iterrows():
        color = CONTINENT_COLORS.get(row["continent"], "#666666")
        pm = per_model[per_model["region"] == row["region"]]["norm_acc"]
        # Range across models
        ax.hlines(i, pm.min(), pm.max(), color=color, alpha=0.3,
                  linewidth=1.0, zorder=2)
        # One small dot per model
        ax.scatter(pm, [i] * len(pm), color=color, s=6, alpha=0.6,
                   edgecolor="none", zorder=3)
        # 13-model pooled mean
        ax.scatter(row["norm_acc_mean"], i, color=color, s=26,
                   edgecolor="black", linewidth=0.4, zorder=4)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(region_agg["region"], fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Mean normalized accuracy", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    # Pad x-axis so the per-model range is visible
    xmax = per_model["norm_acc"].max()
    xmin = min(per_model["norm_acc"].min(), 0)
    pad = max(0.01, (xmax - xmin) * 0.05)
    ax.set_xlim(xmin - pad, xmax + pad)

    # Legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=k)
               for k, c in CONTINENT_COLORS.items()]
    from matplotlib.lines import Line2D
    patches += [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#666666",
               markeredgecolor="black", markeredgewidth=0.4, markersize=4.5,
               label="13-model mean"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#666666",
               markeredgecolor="none", markersize=2.5, alpha=0.7,
               label="One model"),
    ]
    # Below the axes: the two circle keys use the same marker as the data, and
    # inside the axes they sat at plausible accuracies on the African rows.
    fig.legend(handles=patches, loc="outside lower center", fontsize=9,
               frameon=False, handlelength=1.0, handletextpad=0.3,
               ncol=4, columnspacing=1.2)

    pf.save(fig, "figure_region_accuracy", pf.COL)

    # Print summary
    print("\nRegion summary (norm_acc):")
    for _, row in region_agg.iterrows():
        print(f"  {row['region']:<22} {row['norm_acc_mean']:+.4f} ± {row['ci95']:.4f}")


if __name__ == "__main__":
    main()
