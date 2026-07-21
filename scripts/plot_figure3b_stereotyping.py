#!/usr/bin/env python3
"""
Figure 3b: Conditional stereotyping effect — delta bar chart.

Shows the change in normalized accuracy when the respondent's country is
made explicit versus kept implicit in the profile (explicit − implicit).
Positive delta (green) = country label helps; negative delta (red) = hurts.

Values are from Appendix Table A5 (tab:country_conditioning), averaged
across all 13 models at medium profile richness (12 features).
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/figures/emnlp_revision"
LATEX_FIG_DIR = (ROOT / "paper/emnlp/Association_for_Computational_Linguistics__ACL__conference"
                 "/latex/figures")

# From Appendix Table A5: region → (implicit acc, explicit acc, delta)
REGION_DATA = [
    # (region, implicit, explicit, delta)  sorted by delta ascending (most negative first)
    ("Eastern Europe",   0.150,  0.101, -0.048),
    ("Western Europe",   0.139,  0.097, -0.042),
    ("Southern Europe",  0.167,  0.138, -0.029),
    ("South Asia",       0.010, -0.013, -0.023),
    ("Oceania",          0.036,  0.013, -0.023),
    ("East Africa",      0.018,  0.010, -0.008),
    ("West Africa",      0.014,  0.012, -0.002),
    ("Southeast Asia",   0.082,  0.081, -0.001),
    ("North Africa",     0.010,  0.009, -0.001),
    ("Central Africa",   0.011,  0.011,  0.000),
    ("Northern Europe",  0.147,  0.149,  0.002),
    ("East Asia",        0.079,  0.082,  0.003),
    ("Caribbean",        0.083,  0.086,  0.003),
    ("North America",    0.075,  0.079,  0.003),
    ("Middle East",      0.032,  0.040,  0.008),
    ("Central America",  0.127,  0.137,  0.010),
    ("South America",    0.128,  0.141,  0.014),
    ("Southern Africa",  0.005,  0.023,  0.018),
    ("Central Asia",     0.038,  0.096,  0.058),
]

# Sort bottom-to-top: most negative at bottom, most positive at top
REGION_DATA = sorted(REGION_DATA, key=lambda x: x[3])

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


def main():
    regions = [r[0] for r in REGION_DATA]
    deltas  = [r[3] for r in REGION_DATA]
    n = len(regions)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    for i, (region, delta) in enumerate(zip(regions, deltas)):
        if abs(delta) < 0.001:
            color = COLOR_ZERO
        elif delta > 0:
            color = COLOR_HELPS
        else:
            color = COLOR_HURTS
        ax.barh(y[i], delta, height=0.65, color=color, alpha=0.88, zorder=3)

    # Reference line at 0
    ax.axvline(0, color="black", linewidth=0.8, zorder=4)

    # Annotate extremes
    bottom_i = deltas.index(min(deltas))
    top_i    = deltas.index(max(deltas))
    ax.text(deltas[bottom_i] - 0.002, y[bottom_i],
            f"{deltas[bottom_i]:+.3f}", ha="right", va="center", fontsize=7.5,
            color="#B71C1C", fontweight="bold")
    ax.text(deltas[top_i] + 0.002, y[top_i],
            f"{deltas[top_i]:+.3f}", ha="left", va="center", fontsize=7.5,
            color="#2E7D32", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(regions, fontsize=8.5)
    ax.set_xlabel("Change in normalized accuracy\n(explicit country − implicit; positive = explicit helps)",
                  fontsize=8.5)
    ax.set_xlim(-0.07, 0.08)
    ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)

    # Compact legend
    leg_handles = [
        mpatches.Patch(facecolor=COLOR_HELPS, alpha=0.88, label="Explicit helps"),
        mpatches.Patch(facecolor=COLOR_HURTS, alpha=0.88, label="Explicit hurts"),
        mpatches.Patch(facecolor=COLOR_ZERO,  alpha=0.88, label="Neutral (|Δ| < 0.001)"),
    ]
    ax.legend(handles=leg_handles, fontsize=7.5, loc="lower right",
              frameon=True, framealpha=0.9, handlelength=1.0, borderpad=0.5)

    plt.tight_layout(pad=0.6)

    out_base = OUT_DIR / "figure3b_conditional_stereotyping"
    plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")

    if LATEX_FIG_DIR.exists():
        dest = LATEX_FIG_DIR / "conditional_stereotyping_dumbbell.pdf"
        shutil.copy2(out_base.with_suffix(".pdf"), dest)
        print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
