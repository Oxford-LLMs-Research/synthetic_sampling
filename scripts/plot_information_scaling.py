#!/usr/bin/env python3
"""Main-paper Figure 1: accuracy against the amount of information in the profile.

Carries three claims in one panel that previously took a full-width per-model
figure plus a subsection with no figure at all:

  - no model approaches either no-model baseline (the two reference lines),
  - the fitted baseline does not clear the majority class either,
  - and information helps without being close to enough.

That third claim needs care, and the first version of this figure got it wrong.
Every model does improve from 6 to 24 features, and the gap to XGBoost does
narrow, from 0.221 to 0.184. Drawn as bare rising lines the panel therefore
reads as "more information is better", which is true and is not the point. So
the gap is drawn: an arrow at each richness level, labelled with the factor by
which XGBoost still leads. It goes 3.1x, 2.6x, 2.2x, and the honest reading is
that quadrupling the profile closes about a sixth of it.

Model identity is carried the way the rest of the paper carries it: family by
color, instruct versus base by line style, solid with a filled marker against
dashed with an open one. An earlier draft dropped both to a single color for
want of anywhere to put a key. That was the wrong trade -- the base models are
the two lines that sit at chance, and a reader who cannot tell which they are
cannot see that the floor of this panel is a pretraining artifact rather than
a scale effect.

The key sits below the axes. AAAI requires every label in an illustration to
be at least nine point, and at that size an eight-entry key no longer fits any
interior band without covering the majority-class line.

Source: analysis/normalized_accuracy/per_question_norm_acc.csv
        analysis/normalized_accuracy/majority_class_norm_acc.csv
        analysis/xgboost_baseline/results_merged.csv
Output: analysis/figures/emnlp_revision/figure_information_scaling.pdf
"""
from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import paperfig as pf

A = pf.ROOT / "analysis"
ORDER = ["s3m2", "s4m3", "s6m4"]
LABELS = ["6", "12", "24"]


def load():
    acc = (pd.read_csv(A / "normalized_accuracy" / "per_question_norm_acc.csv")
           .groupby(["model", "profile_type"])["norm_acc"].mean().unstack())
    acc = acc[ORDER]
    acc = acc[acc.index.isin(pf.MODEL_META)]

    maj = (pd.read_csv(A / "normalized_accuracy" / "majority_class_norm_acc.csv")
           .groupby("profile_type")["majority_norm_acc"].mean().reindex(ORDER))
    xgb = (pd.read_csv(A / "xgboost_baseline" / "results_merged.csv")
           .groupby("profile_type")["xgb_norm_acc"].mean().reindex(ORDER))
    return acc, maj, xgb


def main() -> None:
    pf.use_style()
    acc, maj, xgb = load()
    x = range(len(ORDER))

    fig, ax = plt.subplots(figsize=(pf.COL, 2.80), layout="constrained")

    # The two no-model baselines, drawn as the ceilings they are. Majority class
    # is flat by construction: it never sees a profile.
    ax.plot(x, maj.values, color=pf.BASELINE_C, linestyle="--", linewidth=1.3,
            zorder=4)
    ax.plot(x, xgb.values, color=pf.XGB_C, linestyle="-.", linewidth=1.3,
            zorder=4)
    # Between the first two gap arrows, which stand at x = 0 and x = 1. Flush
    # left they sat on top of the first arrow.
    ax.text(0.30, maj.iloc[0] + 0.007, f"Majority {maj.iloc[0]:.2f}",
            fontsize=9, color=pf.BASELINE_C, ha="left", va="bottom")
    ax.text(0.30, xgb.iloc[0] - 0.017, f"XGBoost {xgb.iloc[-1]:.2f}",
            fontsize=9, color=pf.XGB_C, ha="left", va="top")

    for name, row in acc.iterrows():
        _, family, instruct = pf.MODEL_META[name]
        c = pf.FAMILY_COLORS[family]
        ax.plot(x, row.values, color=c, linewidth=0.9, alpha=0.85,
                linestyle="-" if instruct else (0, (3, 1.4)),
                marker="o", markersize=2.6,
                markerfacecolor=c if instruct else "white",
                markeredgecolor=c, markeredgewidth=0.8, zorder=3)

    # The gap is the message, so it is measured at every level rather than left
    # to the reader to subtract. Without these the rising lines read as
    # "information helps", which is true and is not what the panel is for.
    for i, p in enumerate(ORDER):
        best = acc[p].max()
        ax.annotate("", xy=(i, best + 0.006), xytext=(i, xgb.iloc[i] - 0.006),
                    arrowprops=dict(arrowstyle="<->", color="#8A8A8A", lw=0.7,
                                    shrinkA=0, shrinkB=0))
        ax.text(i + 0.05, (best + xgb.iloc[i]) / 2,
                f"{xgb.iloc[i] / best:.1f}$\\times$", fontsize=9,
                color="#5A5A5A", ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Features in the profile")
    ax.set_ylabel("Normalized accuracy  (0 = chance)")
    ax.set_xlim(-0.14, 2.30)
    ax.set_ylim(-0.03, 0.41)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.grid(axis="y", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#999999", linewidth=0.7, zorder=1)

    # Six families in three columns, then the style contrast in a fourth, so
    # the caption does not have to carry it. Column-major fill puts the two
    # style entries together in the last column.
    handles = [mlines.Line2D([], [], color=c, linewidth=1.2, label=f)
               for f, c in pf.FAMILY_COLORS.items()]
    handles += [
        mlines.Line2D([], [], color="#666666", linewidth=1.2, marker="o",
                      markersize=2.6, label="instruct"),
        mlines.Line2D([], [], color="#666666", linewidth=1.2,
                      linestyle=(0, (3, 1.4)), marker="o", markersize=2.6,
                      markerfacecolor="white", markeredgecolor="#666666",
                      markeredgewidth=0.8, label="base"),
    ]
    opts = dict(pf.LEGEND_BOX)
    opts.update(ncol=4, handlelength=1.4, columnspacing=1.0, borderpad=0.35,
                labelspacing=0.3)
    leg = fig.legend(handles=handles, loc="outside lower center", **opts)
    leg.get_frame().set_linewidth(0.5)

    pf.save(fig, "figure_information_scaling", pf.COL)
    for p, l in zip(ORDER, LABELS):
        print(f"  {l:>2} feat: best {acc[p].max():.3f}  mean {acc[p].mean():.3f}"
              f"  xgb {xgb[p]:.3f}  gap {xgb[p] - acc[p].max():.3f}"
              f"  ratio {xgb[p] / acc[p].max():.1f}x")
    print(f"  XGBoost at 6 features ({xgb.iloc[0]:.3f}) beats every model at 24 "
          f"({acc[ORDER[-1]].max():.3f}): {xgb.iloc[0] > acc[ORDER[-1]].max()}")


if __name__ == "__main__":
    main()
