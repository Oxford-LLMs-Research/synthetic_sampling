#!/usr/bin/env python3
"""
Figure 3b: Conditional stereotyping effect — mean-delta bars + per-model dots.

Shows the change in normalized accuracy when the respondent's country is
made explicit versus kept implicit in the profile (explicit − implicit).
Positive delta (green bar) = country label helps; negative (red) = hurts.
Small dots overlay the per-model deltas (13 per region) so cross-model
agreement is visible directly.

Computed from the mixed-effects instance data (rich profiles):
  analysis/mixed_effects/mixed_effects_data_country_in_profile.csv   (explicit)
  analysis/mixed_effects/mixed_effects_data_no_country_in_profile.csv (implicit)
The 13-model mean deltas reproduce Appendix Table A5 exactly.
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
MEM_DIR = REPO / "analysis" / "mixed_effects"
OUT_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis\figures\emnlp_revision")
AAAI_FIG_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")

plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

COLOR_HELPS = "#4CAF50"   # green
COLOR_HURTS = "#E57373"   # soft red
COLOR_ZERO  = "#BBBBBB"   # near-zero
DOT_COLOR   = "#333333"


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

    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    for i, region in enumerate(regions):
        delta = delta_mean[region]
        if abs(delta) < 0.001:
            color = COLOR_ZERO
        elif delta > 0:
            color = COLOR_HELPS
        else:
            color = COLOR_HURTS
        ax.barh(y[i], delta, height=0.65, color=color, alpha=0.88, zorder=3)
        pm = delta_pm[delta_pm["region"] == region]["delta"]
        ax.scatter(pm, np.full(len(pm), y[i]), color=DOT_COLOR, s=9,
                   alpha=0.55, edgecolor="none", zorder=4)

    ax.axvline(0, color="black", linewidth=0.8, zorder=5)

    # Annotate extreme mean deltas
    for region in (regions[0], regions[-1]):
        delta = delta_mean[region]
        i = regions.index(region)
        if delta < 0:
            ax.text(min(delta, delta_pm[delta_pm["region"] == region]["delta"].min())
                    - 0.004, y[i], f"{delta:+.3f}", ha="right", va="center",
                    fontsize=7.5, color="#B71C1C", fontweight="bold")
        else:
            ax.text(max(delta, delta_pm[delta_pm["region"] == region]["delta"].max())
                    + 0.004, y[i], f"{delta:+.3f}", ha="left", va="center",
                    fontsize=7.5, color="#2E7D32", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(regions, fontsize=8.5)
    ax.set_xlabel("Change in normalized accuracy\n(explicit country − implicit; positive = explicit helps)",
                  fontsize=8.5)
    lo = min(delta_pm["delta"].min(), delta_mean.min())
    hi = max(delta_pm["delta"].max(), delta_mean.max())
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad - 0.006, hi + pad + 0.006)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)

    leg_handles = [
        mpatches.Patch(facecolor=COLOR_HELPS, alpha=0.88, label="Explicit helps (13-model mean)"),
        mpatches.Patch(facecolor=COLOR_HURTS, alpha=0.88, label="Explicit hurts (13-model mean)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=DOT_COLOR,
               markeredgecolor="none", markersize=3.5, alpha=0.7,
               label="One model (13 dots per region)"),
    ]
    ax.legend(handles=leg_handles, fontsize=7.5, loc="lower right",
              frameon=True, framealpha=0.9, handlelength=1.0, borderpad=0.5)

    plt.tight_layout(pad=0.6)

    out_base = OUT_DIR / "figure3b_conditional_stereotyping"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")

    if AAAI_FIG_DIR.exists():
        dest = AAAI_FIG_DIR / "conditional_stereotyping_dumbbell.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")

    # Cross-model agreement stats for the caption / prose
    sign_agree = (delta_pm.assign(mean_sign=delta_pm["region"].map(np.sign(delta_mean)))
                  .assign(agree=lambda d: np.sign(d["delta"]) == d["mean_sign"])
                  .groupby("region")["agree"].mean())
    print("\nShare of models agreeing with mean sign:")
    print(sign_agree.round(2).to_string())


if __name__ == "__main__":
    main()
