#!/usr/bin/env python3
"""
Effect of naming the respondent's country, by region: observation vs experiment.

Bars are the observational contrast, the change in normalized accuracy between
profiles that happen to carry a country item and profiles that do not
(explicit − implicit). Small dots overlay the per-model values (13 per region);
dots sharing the sign of the regional mean are filled and the rest hollow.

Diamonds and whiskers are the controlled experiment, which appends a country to
a fixed profile and contrasts the true country against one from another region.
Plotting both is the point: the observational contrast is between different
profiles, and the experiment is too imprecise per region to adjudicate it, so
the figure has to show the disagreement rather than assert either reading.

Computed from the mixed-effects instance data (rich profiles):
  analysis/mixed_effects/mixed_effects_data_country_in_profile.csv   (explicit)
  analysis/mixed_effects/mixed_effects_data_no_country_in_profile.csv (implicit)
  analysis/country_injection/by_region.csv                           (experiment)
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paperfig as pf

REPO = Path(__file__).resolve().parents[1]
MEM_DIR = REPO / "analysis" / "mixed_effects"
OUT_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis\figures\emnlp_revision")
AAAI_FIG_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")
EXPERIMENT = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis\country_injection\by_region.csv")


# Blue/orange rather than green/red: the pair survives the common forms of
# colour-vision deficiency, and green/red also read as a verdict on the region.
COLOR_HELPS = "#2E5090"   # country label helps
COLOR_HURTS = "#D9822B"   # country label hurts
COLOR_ZERO  = "#BBBBBB"   # inside the no-effect band
DOT_COLOR   = "#333333"
NULL_BAND   = 0.01        # |delta| below this is not a detectable effect
EXP_C       = "#111111"   # controlled experiment, overlaid on the bars


def load_norm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1",
                     usecols=["model", "correct", "region", "n_options"])
    df = df[df["region"] != "Unknown"].copy()
    df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct", "n_options"])
    df = df[df["n_options"] > 1]
    df["norm_acc"] = (df["correct"] - 1 / df["n_options"]) / (1 - 1 / df["n_options"])
    return df


def main():
    pf.use_style()
    explicit = load_norm(MEM_DIR / "mixed_effects_data_country_in_profile.csv")
    implicit = load_norm(MEM_DIR / "mixed_effects_data_no_country_in_profile.csv")

    # Per-model deltas per region
    ex_pm = explicit.groupby(["region", "model"])["norm_acc"].mean()
    im_pm = implicit.groupby(["region", "model"])["norm_acc"].mean()
    delta_pm = (ex_pm - im_pm).reset_index().rename(columns={"norm_acc": "delta"})

    # 13-model mean delta per region (pooled means, matching Table A5)
    delta_mean = (explicit.groupby("region")["norm_acc"].mean()
                  - implicit.groupby("region")["norm_acc"].mean()).sort_values()

    regions = list(delta_mean.index)
    n = len(regions)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(pf.COL, 3.30), layout="constrained")

    # Most regions sit inside a band where the effect is not distinguishable
    # from nothing; shading it stops the eye from ranking that middle.
    ax.axvspan(-NULL_BAND, NULL_BAND, color="#F0F0F0", zorder=0)

    # Experimental effect: the same quantity measured by appending the country
    # to a fixed profile and contrasting it against a country from another
    # region. Overlaying it is the point of the figure, because the two do not
    # agree: plotting only the observational bars would assert a regional
    # pattern that the controlled test does not reproduce.
    exp = pd.read_csv(EXPERIMENT).set_index("region")

    rng = np.random.default_rng(0)
    for i, region in enumerate(regions):
        delta = delta_mean[region]
        if abs(delta) < NULL_BAND:
            color = COLOR_ZERO
        elif delta > 0:
            color = COLOR_HELPS
        else:
            color = COLOR_HURTS
        ax.barh(y[i], delta, height=0.62, color=color, alpha=0.85, zorder=3)

        if region in exp.index:
            row = exp.loc[region]
            ax.plot([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                    color=EXP_C, linewidth=0.9, alpha=0.75, zorder=5,
                    solid_capstyle="butt")
            ax.scatter(row["experimental"], y[i], marker="D", s=13,
                       facecolor="white", edgecolor=EXP_C, linewidth=1.0,
                       zorder=6)

        pm = delta_pm[delta_pm["region"] == region]["delta"].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(pm))
        agree = np.sign(pm) == np.sign(delta)
        ax.scatter(pm[agree], y[i] + jitter[agree], color=DOT_COLOR, s=5,
                   alpha=0.55, edgecolor="none", zorder=4)
        ax.scatter(pm[~agree], y[i] + jitter[~agree], facecolor="none", s=5,
                   alpha=0.55, edgecolor=DOT_COLOR, linewidth=0.4, zorder=4)

    ax.axvline(0, color="black", linewidth=0.8, zorder=5)

    # The observational extremes are what the controlled test fails to
    # reproduce, so they are labelled with the disagreement rather than with
    # the observational value on its own.
    for region in (regions[0], regions[-1]):
        obs = delta_mean[region]
        i = regions.index(region)
        if region not in exp.index:
            continue
        label = f"obs {obs:+.3f}\nexp {exp.loc[region, 'experimental']:+.3f}"
        pm = delta_pm[delta_pm["region"] == region]["delta"]
        if obs < 0:
            ax.text(min(obs, pm.min()) - 0.005, y[i], label, ha="right",
                    va="center", fontsize=9, color=EXP_C, linespacing=1.25)
        else:
            ax.text(max(obs, pm.max()) + 0.005, y[i], label, ha="left",
                    va="center", fontsize=9, color=EXP_C, linespacing=1.25)

    ax.set_yticks(y)
    ax.set_yticklabels(regions, fontsize=9)
    ax.set_xlabel("Change in normalized accuracy\n"
                  "(explicit country − implicit)", fontsize=9)
    lo = min(delta_pm["delta"].min(), delta_mean.min())
    hi = max(delta_pm["delta"].max(), delta_mean.max())
    pad = 0.05 * (hi - lo)
    # Room on both ends for the obs/exp annotations, which are drawn outward
    # from the extreme bars and were clipped at column width. The left margin
    # used to also hold the legend; that has moved below the axes.
    ax.set_xlim(lo - pad - 0.048, hi + pad + 0.050)
    ax.set_ylim(-1.4, n - 0.4)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    # The key goes below the axes. Inside them its diamond sat at a region's
    # own row, reading as that region's estimate rather than as a key.
    leg_handles = [
        Line2D([], [], marker="D", linestyle="-", markerfacecolor="white",
               markeredgecolor=EXP_C, markeredgewidth=1.0, markersize=4,
               color=EXP_C, linewidth=0.9, label="experiment, 95% CI"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=DOT_COLOR,
               markeredgecolor="none", markersize=3, alpha=0.55,
               label="one dot per model"),
    ]
    fig.legend(handles=leg_handles, fontsize=9, loc="outside lower center",
               frameon=False, handlelength=1.4, borderpad=0.1,
               handletextpad=0.4, ncol=2, columnspacing=1.4)

    pf.save(fig, "figure_country_label_by_region", pf.COL)

    # Cross-model agreement stats for the caption / prose
    sign_agree = (delta_pm.assign(mean_sign=delta_pm["region"].map(np.sign(delta_mean)))
                  .assign(agree=lambda d: np.sign(d["delta"]) == d["mean_sign"])
                  .groupby("region")["agree"].mean())
    print("\nShare of models agreeing with mean sign:")
    print(sign_agree.round(2).to_string())


if __name__ == "__main__":
    main()
