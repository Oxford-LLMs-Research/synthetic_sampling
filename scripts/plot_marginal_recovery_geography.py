#!/usr/bin/env python3
"""Geographic structure of aggregate marginal recovery (appendix figures).

Produces two figures from the marginal-recovery per-cell outputs
(analyze_marginal_recovery.py), rich profiles, 13-model mean:

  figure_marginal_recovery_region.pdf
      Dumbbell plot per world region: mean TV distance of model-predicted
      answer shares from the truth (filled dot) versus the pooled
      cross-country margin (open dot). The horizontal gap is the excess
      error attributable to the synthetic sample.

  figure_marginal_recovery_country.pdf
      Country-level scatter: x = TV(pooled margin, truth), i.e. how
      distinctive the country's opinions are; y = TV(model, truth).
      Points on the diagonal would mean the model is no better than the
      pooled margin; points below it beat the margin.

Also prints the region table and the correlation between region-level
excess TV and the conditional-stereotyping deltas (Appendix Table A5).

Style follows plot_figure3a_region_accuracy.py (serif 9pt, thin marks,
continent colors). Continents additionally get distinct marker shapes so
identity does not rest on color alone (Europe/Asia hues are close under CVD).
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
MR = ANALYSIS / "marginal_recovery"
OUT_DIR = ANALYSIS / "figures" / "emnlp_revision"
AAAI_FIG_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")
SCRIPTS = Path(__file__).resolve().parent

PROFILE = "s6m4"

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]

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
CONTINENT_MARKERS = {
    "Africa": "o", "Americas": "s", "Asia": "^", "Europe": "D", "Oceania": "P",
}

DISPLAY_NAMES = {
    "qwen3-32b": "Qwen 3 32B", "deepseek": "DeepSeek-V3",
    "olmo3_7b_dpo": "OLMo 3 7B inst.", "qwen3-4b": "Qwen 3 4B",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.", "gemma3-27b": "Gemma 3 27B",
    "gpt-oss": "GPT-OSS 120B", "olmo3_7b_base": "OLMo 3 7B base",
    "llama3.1_70b_instruct": "Llama 3.1 70B inst.",
    "olmo3_32b_base": "OLMo 3 32B base", "olmo3_32b_dpo": "OLMo 3 32B inst.",
    "llama3.1_70b_base": "Llama 3.1 70B base", "llama3.1_8b_base": "Llama 3.1 8B base",
}

# Conditional stereotyping deltas (explicit - implicit normalized accuracy),
# Appendix Table A5, 13-model average at medium richness.
STEREOTYPING_DELTA = {
    "Eastern Europe": -0.048, "Western Europe": -0.042, "Southern Europe": -0.029,
    "South Asia": -0.023, "Oceania": -0.023, "East Africa": -0.008,
    "West Africa": -0.002, "Southeast Asia": -0.001, "North Africa": -0.001,
    "Central Africa": 0.000, "Northern Europe": 0.002, "East Asia": 0.003,
    "Caribbean": 0.003, "North America": 0.003, "Middle East": 0.008,
    "Central America": 0.010, "South America": 0.014, "Southern Africa": 0.018,
    "Central Asia": 0.058,
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


def load_cells() -> pd.DataFrame:
    frames = []
    for m in MODELS:
        df = pd.read_csv(MR / f"per_cell_{m}.csv")
        df = df[df["profile_type"] == PROFILE]
        df["model"] = m
        frames.append(df)
    cells = pd.concat(frames, ignore_index=True)
    # 13-model mean per cell; tv_global is model-independent
    agg = (cells.groupby(["survey", "target_code", "country"])
           .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "first"),
                n=("n", "first"))
           .reset_index())
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))
    agg["region"] = agg["country"].map(region_map)
    agg = agg[agg["region"].notna() & (agg["region"] != "Unknown")]
    agg["continent"] = agg["region"].map(CONTINENT_MAP)
    return agg


def region_summary(cells: pd.DataFrame) -> pd.DataFrame:
    reg = (cells.groupby("region")
           .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                n_cells=("tv_model", "size"))
           .reset_index())
    reg["continent"] = reg["region"].map(CONTINENT_MAP)
    return reg.sort_values("tv_model").reset_index(drop=True)


def plot_region(cells: pd.DataFrame) -> pd.DataFrame:
    reg = region_summary(cells)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, row in reg.iterrows():
        color = CONTINENT_COLORS[row["continent"]]
        ax.hlines(i, row["tv_global"], row["tv_model"],
                  color=color, alpha=0.45, linewidth=1.2, zorder=2)
        ax.scatter(row["tv_global"], i, facecolor="white", edgecolor=color,
                   s=42, linewidth=1.0, zorder=3)
        ax.scatter(row["tv_model"], i, color=color, s=48,
                   edgecolor="black", linewidth=0.4, zorder=4)
    ax.set_yticks(np.arange(len(reg)))
    ax.set_yticklabels(reg["region"], fontsize=8)
    ax.set_xlabel("Mean TV distance from true answer shares", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)
    ax.set_xlim(0, None)

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#444444",
               markeredgecolor="black", markeredgewidth=0.4, markersize=6.5,
               label="Synthetic sample (13-model mean)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#444444", markeredgewidth=1.0, markersize=6,
               label="Pooled cross-country margin"),
    ]
    handles += [Line2D([], [], marker="s", linestyle="", markerfacecolor=c,
                       markeredgecolor="black", markeredgewidth=0.3,
                       markersize=5.5, label=k)
                for k, c in CONTINENT_COLORS.items()]
    leg = ax.legend(handles=handles, loc="upper center", fontsize=7,
                    frameon=False, handletextpad=0.4, labelspacing=0.35,
                    bbox_to_anchor=(0.5, -0.11), ncol=3, columnspacing=1.2)
    leg.get_frame().set_linewidth(0.4)

    plt.tight_layout(pad=0.5)
    base = OUT_DIR / "figure_marginal_recovery_region"
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")
    return reg


def plot_country(cells: pd.DataFrame) -> pd.DataFrame:
    cty = (cells.groupby("country")
           .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                n_cells=("tv_model", "size"), region=("region", "first"),
                continent=("continent", "first"))
           .reset_index())

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = max(cty["tv_model"].max(), cty["tv_global"].max()) * 1.06
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray",
            linewidth=0.7, alpha=0.7, zorder=1)
    ax.annotate("model = pooled margin", xy=(lim * 0.55, lim * 0.52),
                fontsize=7, color="gray", rotation=45,
                ha="center", va="top", rotation_mode="anchor")

    for cont, g in cty.groupby("continent"):
        ax.scatter(g["tv_global"], g["tv_model"],
                   color=CONTINENT_COLORS[cont],
                   marker=CONTINENT_MARKERS[cont],
                   s=26, alpha=0.85, edgecolor="black", linewidth=0.3,
                   label=cont, zorder=3)

    # Annotate extremes: most distinctive opinions, largest model error
    label_set = set(cty.nlargest(4, "tv_global")["country"]) \
        | set(cty.nlargest(3, "tv_model")["country"]) \
        | set(cty.nsmallest(2, "tv_model")["country"])
    for _, row in cty[cty["country"].isin(label_set)].iterrows():
        ax.annotate(row["country"], (row["tv_global"], row["tv_model"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=6.5)

    ax.set_xlabel("TV(pooled margin, truth): opinion distinctiveness", fontsize=9)
    ax.set_ylabel("TV(synthetic sample, truth), 13-model mean", fontsize=9)
    ax.grid(linestyle="--", alpha=0.2, linewidth=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    leg = ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
                    handletextpad=0.3, borderpad=0.5, labelspacing=0.35,
                    title="Continent", title_fontsize=7.5)
    leg.get_frame().set_linewidth(0.4)

    plt.tight_layout(pad=0.5)
    base = OUT_DIR / "figure_marginal_recovery_country"
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")
    return cty


def load_cells_permodel() -> pd.DataFrame:
    frames = []
    for m in MODELS:
        df = pd.read_csv(MR / f"per_cell_{m}.csv")
        df = df[df["profile_type"] == PROFILE].copy()
        df["model"] = m
        frames.append(df)
    cells = pd.concat(frames, ignore_index=True)
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))
    cells["region"] = cells["country"].map(region_map)
    cells = cells[cells["region"].notna() & (cells["region"] != "Unknown")]
    cells["continent"] = cells["region"].map(CONTINENT_MAP)
    return cells


def plot_region_combined(cells_pm: pd.DataFrame, order: list[str]) -> None:
    """One row per region: 13 small model dots, 13-model mean, baseline dot."""
    reg_pm = (cells_pm.groupby(["region", "model"])
              .agg(tv_model=("tv_model", "mean"))
              .reset_index())
    reg_base = (cells_pm.groupby("region")
                .agg(tv_global=("tv_global", "mean"))
                .reset_index())
    reg_pm = reg_pm.merge(reg_base, on="region")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, region in enumerate(order):
        g = reg_pm[reg_pm["region"] == region]
        color = CONTINENT_COLORS[CONTINENT_MAP[region]]
        lo, hi = g["tv_model"].min(), g["tv_model"].max()
        ax.hlines(i, lo, hi, color=color, alpha=0.35, linewidth=1.0, zorder=2)
        ax.scatter(g["tv_model"], [i] * len(g), color=color, s=13,
                   alpha=0.65, edgecolor="none", zorder=3)
        ax.scatter(g["tv_model"].mean(), i, color=color, s=58,
                   edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter(g["tv_global"].iloc[0], i, facecolor="white",
                   edgecolor=color, s=40, linewidth=1.0, zorder=4)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Mean TV distance from true answer shares", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)
    ax.set_xlim(0, None)

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#444444",
               markeredgecolor="black", markeredgewidth=0.5, markersize=7,
               label="13-model mean"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#444444",
               markeredgecolor="none", markersize=4,
               label="One model (13 dots per region)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#444444", markeredgewidth=1.0, markersize=6,
               label="Pooled cross-country margin"),
    ]
    ax.legend(handles=handles, loc="upper center", fontsize=7,
              frameon=False, handletextpad=0.4, labelspacing=0.35,
              bbox_to_anchor=(0.5, -0.11), ncol=3, columnspacing=1.0)

    plt.tight_layout(pad=0.5)
    base = OUT_DIR / "figure_marginal_recovery_region"
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf/.png (combined)")


def plot_country_grid(cells_pm: pd.DataFrame) -> None:
    """Small-multiples: country scatter per model, shared axes."""
    cty_pm = (cells_pm.groupby(["model", "country"])
              .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                   continent=("continent", "first"))
              .reset_index())
    model_order = (cty_pm.groupby("model")["tv_model"].mean()
                   .sort_values().index.tolist())
    lim = max(cty_pm["tv_model"].max(), cty_pm["tv_global"].max()) * 1.06

    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.4),
                             sharex=True, sharey=True)
    for ax in axes.flat:
        ax.set_axis_off()
    for k, model in enumerate(model_order):
        ax = axes.flat[k]
        ax.set_axis_on()
        g = cty_pm[cty_pm["model"] == model]
        ax.plot([0, lim], [0, lim], linestyle="--", color="gray",
                linewidth=0.5, alpha=0.7, zorder=1)
        for cont, gc in g.groupby("continent"):
            ax.scatter(gc["tv_global"], gc["tv_model"],
                       color=CONTINENT_COLORS[cont],
                       marker=CONTINENT_MARKERS[cont],
                       s=8, alpha=0.8, edgecolor="none", zorder=3)
        ax.set_title(DISPLAY_NAMES[model], fontsize=7.5, pad=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6, length=2)
        ax.grid(linestyle="--", alpha=0.15, linewidth=0.3)

    # Legend in the unused subplot slots
    ax_leg = axes.flat[13]
    handles = [Line2D([], [], marker=CONTINENT_MARKERS[c], linestyle="",
                      markerfacecolor=col, markeredgecolor="none", markersize=5,
                      label=c) for c, col in CONTINENT_COLORS.items()]
    handles.append(Line2D([], [], linestyle="--", color="gray", linewidth=0.8,
                          label="model = pooled margin"))
    ax_leg.legend(handles=handles, loc="center left", fontsize=7.5,
                  frameon=False, handletextpad=0.4, labelspacing=0.5)

    fig.supxlabel("TV(pooled margin, truth): opinion distinctiveness",
                  fontsize=9, y=0.04)
    fig.supylabel("TV(synthetic sample, truth)", fontsize=9, x=0.04)
    plt.tight_layout(pad=0.6, rect=(0.05, 0.05, 1, 1))
    base = OUT_DIR / "figure_marginal_recovery_country_permodel"
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


def main() -> None:
    cells = load_cells()
    reg = region_summary(cells)
    cty = plot_country(cells)

    cells_pm = load_cells_permodel()
    plot_region_combined(cells_pm, order=list(reg.sort_values("tv_model")["region"]))
    plot_country_grid(cells_pm)

    reg["excess"] = reg["tv_model"] - reg["tv_global"]
    reg["stereo_delta"] = reg["region"].map(STEREOTYPING_DELTA)
    reg.to_csv(MR / "region_summary.csv", index=False)
    cty.to_csv(MR / "country_summary.csv", index=False)

    print("\nRegion summary:")
    print(reg.sort_values("excess", ascending=False)
          .round(3).to_string(index=False))
    ok = reg["stereo_delta"].notna()
    r = np.corrcoef(reg.loc[ok, "excess"], reg.loc[ok, "stereo_delta"])[0, 1]
    print(f"\ncorr(region excess TV, stereotyping delta): r = {r:.3f} "
          f"(n = {ok.sum()} regions)")
    r2 = np.corrcoef(cty["tv_global"], cty["tv_model"])[0, 1]
    print(f"corr(country distinctiveness, model TV): r = {r2:.3f} "
          f"(n = {len(cty)} countries)")

    if AAAI_FIG_DIR.exists():
        for name in ("figure_marginal_recovery_region.pdf",
                     "figure_marginal_recovery_country.pdf",
                     "figure_marginal_recovery_country_permodel.pdf"):
            shutil.copy2(OUT_DIR / name, AAAI_FIG_DIR / name)
            print(f"Copied {name} to {AAAI_FIG_DIR}")


if __name__ == "__main__":
    main()
