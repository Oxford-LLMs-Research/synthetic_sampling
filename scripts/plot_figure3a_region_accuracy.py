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

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_pos = np.arange(len(region_agg))

    for i, row in region_agg.iterrows():
        color = CONTINENT_COLORS.get(row["continent"], "#666666")
        # Connecting line
        ax.hlines(i, 0, row["norm_acc_mean"], color="gray", alpha=0.15, linewidth=0.5)
        # Error bar
        ax.errorbar(row["norm_acc_mean"], i,
                    xerr=row["ci95"],
                    fmt="none", color="black",
                    capsize=2, capthick=0.5, elinewidth=0.5)
        # Colored dot
        ax.scatter(row["norm_acc_mean"], i, color=color, s=90,
                   edgecolor="black", linewidth=0.5, zorder=4)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(region_agg["region"], fontsize=8.5)
    ax.set_xlabel("Mean Normalized Accuracy (mean ± 95% CI)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)

    # Pad x-axis so CIs are visible
    xmax = (region_agg["norm_acc_mean"] + region_agg["ci95"]).max()
    xmin = (region_agg["norm_acc_mean"] - region_agg["ci95"]).min()
    pad = max(0.01, (xmax - xmin) * 0.05)
    ax.set_xlim(xmin - pad, xmax + pad)

    # Legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=k)
               for k, c in CONTINENT_COLORS.items()]
    legend = ax.legend(handles=patches, title="Continent", loc="lower right",
                       fontsize=8, title_fontsize=8.5, frameon=True, framealpha=0.9,
                       handlelength=1.0, handletextpad=0.4)
    legend.get_frame().set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

    out_base = OUT_DIR / "figure3a_region_accuracy"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "figure_region_accuracy.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")

    # Print summary
    print("\nRegion summary (norm_acc):")
    for _, row in region_agg.iterrows():
        print(f"  {row['region']:<22} {row['norm_acc_mean']:+.4f} ± {row['ci95']:.4f}")


if __name__ == "__main__":
    main()
