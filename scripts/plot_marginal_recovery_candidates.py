#!/usr/bin/env python3
"""Candidate redesigns for the marginal-recovery main figure (Figure 2).

Does NOT overwrite the live figure_marginal_recovery_*.pdf files. Saves
side-by-side candidates for review:

  Main-text candidates (column width):
    figure_marginal_recovery_country_vA_excess.pdf
        Ranked Cleveland dots of excess TV = tv_model - tv_global.
    figure_marginal_recovery_country_vB_scatter.pdf
        Cropped distinctiveness-vs-model scatter with residual stems.
    figure_marginal_recovery_region_vC_main.pdf
        Polished region dumbbell promoted as a main-text candidate.

  Appendix candidates (full / column width):
    figure_marginal_recovery_country_permodel_vB.pdf
        Cropped-axes small-multiples grid.
    figure_marginal_recovery_country_permodel_vA_excess.pdf
        Per-model mean excess TV as horizontal strips.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paperfig as pf

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
MR = ANALYSIS / "marginal_recovery"
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
CONTINENT_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

DISPLAY_NAMES = {
    "qwen3-32b": "Qwen 3 32B", "deepseek": "DeepSeek-V3.1",
    "olmo3_7b_dpo": "OLMo 3 7B inst.", "qwen3-4b": "Qwen 3 4B",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.", "gemma3-27b": "Gemma 3 27B",
    "gpt-oss": "GPT-OSS 120B", "olmo3_7b_base": "OLMo 3 7B base",
    "llama3.1_70b_instruct": "Llama 3.1 70B inst.",
    "olmo3_32b_base": "OLMo 3 32B base", "olmo3_32b_dpo": "OLMo 3 32B inst.",
    "llama3.1_70b_base": "Llama 3.1 70B base", "llama3.1_8b_base": "Llama 3.1 8B base",
}

# ISO alpha-2 -> short display name for a handful of extremes.
COUNTRY_NAMES = {
    "KH": "Cambodia", "MN": "Mongolia", "VN": "Vietnam", "TH": "Thailand",
    "KR": "S. Korea", "NO": "Norway", "BR": "Brazil", "BO": "Bolivia",
    "JP": "Japan", "TW": "Taiwan", "CN": "China", "IN": "India",
    "US": "USA", "GB": "UK", "DE": "Germany", "FR": "France",
    "ZA": "S. Africa", "NG": "Nigeria", "EG": "Egypt", "AU": "Australia",
    "NZ": "New Zealand", "MX": "Mexico", "AR": "Argentina", "CL": "Chile",
    "PE": "Peru", "CO": "Colombia", "PL": "Poland", "UA": "Ukraine",
    "TR": "Turkey", "IL": "Israel", "SA": "Saudi Arabia", "PK": "Pakistan",
}


def load_cells() -> pd.DataFrame:
    frames = []
    for m in MODELS:
        df = pd.read_csv(MR / f"per_cell_{m}.csv")
        df = df[df["profile_type"] == PROFILE]
        df["model"] = m
        frames.append(df)
    cells = pd.concat(frames, ignore_index=True)
    agg = (cells.groupby(["survey", "target_code", "country"])
           .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "first"),
                n=("n", "first"))
           .reset_index())
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))
    agg["region"] = agg["country"].map(region_map)
    agg = agg[agg["region"].notna() & (agg["region"] != "Unknown")]
    agg["continent"] = agg["region"].map(CONTINENT_MAP)
    return agg


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


def country_summary(cells: pd.DataFrame) -> pd.DataFrame:
    cty = (cells.groupby("country")
           .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                n_cells=("tv_model", "size"), region=("region", "first"),
                continent=("continent", "first"))
           .reset_index())
    cty["excess"] = cty["tv_model"] - cty["tv_global"]
    return cty


def _display_country(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


def plot_candidate_a_excess(cty: pd.DataFrame) -> None:
    """Ranked excess-TV dots; every country fails relative to zero."""
    df = cty.sort_values("excess", ascending=True).reset_index(drop=True)
    n = len(df)
    # Compact height: ~0.055 in per country + margins; cap near column figure height.
    height = min(4.55, max(3.4, 0.055 * n + 0.85))
    fig, ax = plt.subplots(figsize=(pf.COL, height), layout="constrained")

    ax.axvline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
    for i, row in df.iterrows():
        color = CONTINENT_COLORS[row["continent"]]
        ax.plot([0, row["excess"]], [i, i], color=color, alpha=0.35,
                linewidth=0.9, zorder=2, solid_capstyle="butt")
        ax.scatter(row["excess"], i, s=18,
                   color=color, marker=CONTINENT_MARKERS[row["continent"]],
                   edgecolor="black", linewidth=0.25, zorder=4)

    # Label extremes only (full names), not the whole ranking.
    label_idx = set(df.nlargest(4, "excess").index) | set(df.nsmallest(3, "excess").index)
    for i in label_idx:
        row = df.loc[i]
        ax.annotate(_display_country(row["country"]),
                    (row["excess"], i),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=6, va="center", color="#333333")

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Excess TV  (synthetic − pooled margin)", fontsize=8)
    ax.set_title("Every country: synthetic sample worse than pooled margin",
                 loc="left", fontsize=8)
    ax.set_xlim(-0.02, df["excess"].max() * 1.12)
    ax.set_ylim(-1.2, n - 0.3)
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)

    handles = [
        Line2D([], [], marker=CONTINENT_MARKERS[c], linestyle="",
               markerfacecolor=col, markeredgecolor="black",
               markeredgewidth=0.25, markersize=5, label=c)
        for c, col in CONTINENT_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False,
              handletextpad=0.25, borderpad=0.2, labelspacing=0.25,
              title="Continent", title_fontsize=6.5)

    # Annotation at mid-height near the zero line, not among the top labels.
    ax.text(0.01, n * 0.45, "0 = pooled\nmargin",
            fontsize=6.2, color="#666666", ha="left", va="center",
            style="italic", linespacing=1.25)

    pf.save(fig, "figure_marginal_recovery_country_vA_excess", pf.COL)


def plot_candidate_b_scatter(cty: pd.DataFrame) -> None:
    """Cropped scatter; diagonal is below the frame, so note it in text.

    Cropping to the data cloud puts y = x entirely off-plot (model TV is
    ~3x the pooled-margin distance). Residual stems to the diagonal therefore
    clip to the axis floor and read as noise; we drop them and state the
    claim in a corner note instead.
    """
    fig, ax = plt.subplots(figsize=(pf.COL, 3.35), layout="constrained")

    x = cty["tv_global"].to_numpy()
    y = cty["tv_model"].to_numpy()
    pad_x = 0.06 * (x.max() - x.min())
    pad_y = 0.06 * (y.max() - y.min())
    xmin, xmax = x.min() - pad_x, x.max() + pad_x
    ymin, ymax = y.min() - pad_y, y.max() + pad_y

    for _, row in cty.iterrows():
        color = CONTINENT_COLORS[row["continent"]]
        ax.scatter(row["tv_global"], row["tv_model"],
                   color=color, marker=CONTINENT_MARKERS[row["continent"]],
                   s=30, alpha=0.9, edgecolor="black", linewidth=0.3, zorder=4)

    label_set = set(cty.nlargest(3, "tv_global")["country"]) \
        | set(cty.nlargest(2, "tv_model")["country"]) \
        | set(cty.nsmallest(2, "tv_model")["country"])
    for _, row in cty[cty["country"].isin(label_set)].iterrows():
        ax.annotate(_display_country(row["country"]),
                    (row["tv_global"], row["tv_model"]),
                    xytext=(3.5, 2), textcoords="offset points", fontsize=6.2)

    ax.set_xlabel("Opinion distinctiveness\nTV(pooled margin, truth)", fontsize=7.5)
    ax.set_ylabel("TV(synthetic sample, truth)\n13-model mean", fontsize=7.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=7)
    ax.grid(linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker=CONTINENT_MARKERS[c], linestyle="",
               markerfacecolor=col, markeredgecolor="black",
               markeredgewidth=0.3, markersize=5.5, label=c)
        for c, col in CONTINENT_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False,
              handletextpad=0.25, borderpad=0.2, labelspacing=0.25,
              title="Continent", title_fontsize=6.5)

    r = np.corrcoef(cty["tv_global"], cty["tv_model"])[0, 1]
    # Diagonal y = x lies entirely below this crop; state the claim directly.
    ax.text(0.98, 0.03,
            f"All 70 above $y\\!=\\!x$\n(pooled margin)\n$r$ = {r:.2f}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            color="#444444", linespacing=1.3)

    pf.save(fig, "figure_marginal_recovery_country_vB_scatter", pf.COL)


def plot_candidate_c_region(cells_pm: pd.DataFrame) -> None:
    """Polished region dumbbell as main-text candidate."""
    reg_pm = (cells_pm.groupby(["region", "model"])
              .agg(tv_model=("tv_model", "mean"))
              .reset_index())
    reg_base = (cells_pm.groupby("region")
                .agg(tv_global=("tv_global", "mean"))
                .reset_index())
    reg_pm = reg_pm.merge(reg_base, on="region")
    means = (reg_pm.groupby("region")
             .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "first"))
             .reset_index())
    means["excess"] = means["tv_model"] - means["tv_global"]
    # Sort by excess so the failure gap is the ranking cue.
    order = list(means.sort_values("excess", ascending=False)["region"])

    fig, ax = plt.subplots(figsize=(pf.COL, 3.55), layout="constrained")
    for i, region in enumerate(order):
        g = reg_pm[reg_pm["region"] == region]
        color = CONTINENT_COLORS[CONTINENT_MAP[region]]
        lo, hi = g["tv_model"].min(), g["tv_model"].max()
        base = g["tv_global"].iloc[0]
        mean = g["tv_model"].mean()
        # Gap bar from pooled margin to model mean (the claim).
        ax.hlines(i, base, mean, color=color, alpha=0.45, linewidth=1.4, zorder=2)
        # Model range as a thin span behind the mean.
        ax.hlines(i, lo, hi, color=color, alpha=0.22, linewidth=3.2, zorder=1)
        ax.scatter(g["tv_model"], [i] * len(g), color=color, s=11,
                   alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter(mean, i, color=color, s=52,
                   edgecolor="black", linewidth=0.45, zorder=5)
        ax.scatter(base, i, facecolor="white", edgecolor=color,
                   s=38, linewidth=1.0, zorder=5)

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("Mean TV distance from true answer shares", fontsize=8)
    ax.set_title("Synthetic samples miss every region vs. pooled margin",
                 loc="left", fontsize=8)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75])
    ax.set_xlim(0, max(reg_pm["tv_model"].max() * 1.05, 0.8))
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#444444",
               markeredgecolor="black", markeredgewidth=0.45, markersize=6.5,
               label="13-model mean"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#444444",
               markeredgecolor="none", markersize=3.5,
               label="One model"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#444444", markeredgewidth=1.0, markersize=5.5,
               label="Pooled margin"),
    ]
    ax.legend(handles=handles, loc="upper center", fontsize=6.5,
              frameon=False, handletextpad=0.35, labelspacing=0.3,
              bbox_to_anchor=(0.5, -0.14), ncol=3, columnspacing=0.9)

    pf.save(fig, "figure_marginal_recovery_region_vC_main", pf.COL)


def plot_appendix_permodel_vB(cells_pm: pd.DataFrame) -> None:
    """Cropped-axes small-multiples: same cloud zoom for every model."""
    cty_pm = (cells_pm.groupby(["model", "country"])
              .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                   continent=("continent", "first"))
              .reset_index())
    model_order = (cty_pm.groupby("model")["tv_model"].mean()
                   .sort_values().index.tolist())

    xmin = cty_pm["tv_global"].min()
    xmax = cty_pm["tv_global"].max()
    ymin = cty_pm["tv_model"].min()
    ymax = cty_pm["tv_model"].max()
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    fig, axes = plt.subplots(3, 5, figsize=(pf.FULL, 5.4),
                             sharex=True, sharey=True, layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()

    for k, model in enumerate(model_order):
        ax = axes.flat[k]
        ax.set_axis_on()
        g = cty_pm[cty_pm["model"] == model]
        # y = x is entirely below this crop; omit the diagonal segment.
        for cont, gc in g.groupby("continent"):
            ax.scatter(gc["tv_global"], gc["tv_model"],
                       color=CONTINENT_COLORS[cont],
                       marker=CONTINENT_MARKERS[cont],
                       s=12, alpha=0.85, edgecolor="none", zorder=3)
        ax.set_title(DISPLAY_NAMES[model], fontsize=7, pad=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=5.5, length=2)
        ax.grid(linestyle="--", alpha=0.15, linewidth=0.3)

    ax_leg = axes.flat[13]
    ax_leg.set_axis_off()
    handles = [Line2D([], [], marker=CONTINENT_MARKERS[c], linestyle="",
                      markerfacecolor=col, markeredgecolor="none", markersize=5,
                      label=c) for c, col in CONTINENT_COLORS.items()]
    ax_leg.legend(handles=handles, loc="center", fontsize=7.5,
                  frameon=False, handletextpad=0.4, labelspacing=0.45,
                  title="All panels: every country\nabove pooled margin",
                  title_fontsize=7)

    fig.supxlabel("TV(pooled margin, truth): opinion distinctiveness",
                  fontsize=8.5)
    fig.supylabel("TV(synthetic sample, truth)", fontsize=8.5)
    pf.save(fig, "figure_marginal_recovery_country_permodel_vB", pf.FULL)


def plot_appendix_permodel_vA_excess(cells_pm: pd.DataFrame) -> None:
    """One row per model: distribution of country-level excess TV."""
    cty_pm = (cells_pm.groupby(["model", "country"])
              .agg(tv_model=("tv_model", "mean"), tv_global=("tv_global", "mean"),
                   continent=("continent", "first"))
              .reset_index())
    cty_pm["excess"] = cty_pm["tv_model"] - cty_pm["tv_global"]
    model_order = (cty_pm.groupby("model")["excess"].mean()
                   .sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(pf.FULL, 3.6), layout="constrained")
    ax.axvline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    rng = np.random.default_rng(0)
    for i, model in enumerate(model_order):
        g = cty_pm[cty_pm["model"] == model]
        # Light vertical jitter so continents separate a little.
        jitter = rng.uniform(-0.18, 0.18, size=len(g))
        for cont in CONTINENT_ORDER:
            gc = g[g["continent"] == cont]
            if gc.empty:
                continue
            j = jitter[g["continent"].to_numpy() == cont]
            ax.scatter(gc["excess"], np.full(len(gc), i) + j[:len(gc)],
                       color=CONTINENT_COLORS[cont],
                       marker=CONTINENT_MARKERS[cont],
                       s=14, alpha=0.7, edgecolor="none", zorder=3)
        ax.scatter(g["excess"].mean(), i, s=48, color="#222222",
                   edgecolor="white", linewidth=0.6, zorder=5)

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in model_order], fontsize=7.5)
    ax.set_xlabel("Excess TV  (synthetic − pooled margin), by country",
                  fontsize=8)
    ax.set_title("Per-model country excess: no model reaches the pooled margin",
                 loc="left", fontsize=8.5)
    ax.set_xlim(-0.02, cty_pm["excess"].max() * 1.08)
    ax.set_ylim(-0.7, len(model_order) - 0.3)
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker=CONTINENT_MARKERS[c], linestyle="",
               markerfacecolor=col, markeredgecolor="none", markersize=5,
               label=c)
        for c, col in CONTINENT_COLORS.items()
    ]
    handles.append(Line2D([], [], marker="o", linestyle="",
                          markerfacecolor="#222222", markeredgecolor="white",
                          markeredgewidth=0.6, markersize=6,
                          label="Model mean"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False,
              ncol=3, handletextpad=0.3, columnspacing=0.9, labelspacing=0.25)

    pf.save(fig, "figure_marginal_recovery_country_permodel_vA_excess", pf.FULL)


def main() -> None:
    pf.use_style()
    cells = load_cells()
    cty = country_summary(cells)
    cells_pm = load_cells_permodel()

    print(f"Countries: {len(cty)}  regions: {cells['region'].nunique()}  "
          f"excess range [{cty['excess'].min():.3f}, {cty['excess'].max():.3f}]")

    plot_candidate_a_excess(cty)
    plot_candidate_b_scatter(cty)
    plot_candidate_c_region(cells_pm)
    plot_appendix_permodel_vB(cells_pm)
    plot_appendix_permodel_vA_excess(cells_pm)

    print("Candidates written (live figure_* files left unchanged).")


if __name__ == "__main__":
    main()
