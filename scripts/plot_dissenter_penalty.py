"""Figure: the dissenter penalty, the mode-following signature.

Redraw notes (v2). The first version had three clarity problems, all fixed
here:

  1. Family color encoded the dot, so majority-versus-dissenter (the only
     contrast the figure exists to show) was carried by fill alone, at 5.6pt.
     Color now encodes the contrast itself: blue = holds the country's majority
     view, red = dissents, identically in both panels.
  2. The two panels used different accuracy ranges, so the reader could not
     carry a value across. They now share one range.
  3. The result worth remembering, that accuracy on dissenters goes *below*
     random chance once local consensus is strong, was left for the reader to
     infer from an unlabeled axis. The sub-chance region is now shaded and the
     two endpoints are annotated.

Panel (a) One row per model. Filled dot = normalized accuracy on respondents
whose answer matches their question x country modal answer; open dot =
dissenters. Sorted by majority-group accuracy, so the widening of the gap down
the ranking is visible: the better a model is at this task, the larger its
penalty.

Panel (b) The mechanism. As the modal answer's share of a cell rises, accuracy
on majority-view respondents climbs steeply while accuracy on dissenters falls
through zero. A predictor that tracked individuals would improve on both; one
that tracks the mode improves on the majority by abandoning everyone else.

Source: analysis/equity_audit/dissenter_penalty_by_{model,share}.csv
Output: analysis/figures/emnlp_revision/figure_dissenter_penalty.pdf, copied
into the AAAI paper tree. Drawn at exactly \\textwidth (no rescaling).
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import paperfig as pf

EQ = pf.ROOT / "analysis" / "equity_audit"

DISPLAY = {k: v[0] for k, v in pf.MODEL_META.items()}

# The stratifier, spelled out. "Share" alone left the reader asking share of
# whom: it is the human survey respondents who answered that question, pooled
# over every country where it was fielded, which is also the population whose
# modal answer defines the split. Percentages move onto the ticks so the axis
# label can spend its width naming the group instead of the unit.
BINS = ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"]
BIN_TICKS = ["<40%", "40–60%", "60–80%", ">80%"]
SHARE_LABEL = ("Survey respondents giving the\n"
               "question's most common answer")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(EQ / "dissenter_penalty_by_model.csv")
    m["label"] = m["model"].map(DISPLAY)
    m = m.sort_values("norm_modal").reset_index(drop=True)

    s = pd.read_csv(EQ / "dissenter_penalty_by_share.csv",
                    header=[0, 1], index_col=0)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return m, s


def consensus_only(s: pd.DataFrame) -> None:
    """Main-paper figure: the mechanism panel on its own, at column width.

    The per-model dumbbell that used to sit beside it is one scalar per model,
    which is Table 1's shape, so it moved there. The two-panel version below is
    still built for the appendix, where the per-model detail is discussed.
    """
    order = BINS
    s = s.reindex(order)
    x = list(range(len(order)))
    lo, hi = -0.24, 0.38

    fig, ax = plt.subplots(figsize=(pf.COL, 2.35), layout="constrained")
    ax.axhspan(lo, 0, color="#F6F0F0", zorder=0)
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    ax.plot(x, s["norm_True"], marker="o", markersize=4.2, color=pf.MODAL_C,
            linewidth=1.6, zorder=3)
    ax.plot(x, s["norm_False"], marker="o", markersize=4.2, color=pf.DISS_C,
            linewidth=1.6, markerfacecolor="white", markeredgewidth=1.2,
            zorder=3)

    for col, colour, name, va in (
            ("norm_True", pf.MODAL_C, "Gives the\ncommon answer", "bottom"),
            ("norm_False", pf.DISS_C, "Does not", "top")):
        ax.annotate(f"{s[col].iloc[-1]:+.2f}", xy=(3, s[col].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=9,
                    color=colour, ha="left", va="center")
        # Lifted clear of its own curve: at nine point the two-line majority
        # label sat on the blue line's first segment.
        ax.text(0.06, s[col].iloc[0] + (0.055 if va == "bottom" else -0.025),
                name, fontsize=9, color=colour, ha="left", va=va,
                linespacing=1.15)

    ax.text(0.06, lo + 0.02, "worse than guessing", fontsize=9,
            color="#8B3A3A", ha="left", va="bottom", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_TICKS)
    ax.set_xlabel(SHARE_LABEL, fontsize=9)
    ax.set_ylabel("Normalized accuracy")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.12, 3.55)
    ax.grid(axis="y", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    pf.save(fig, "figure_consensus_penalty", pf.COL)


def main() -> None:
    pf.use_style()
    m, s = load()
    consensus_only(s.copy())

    # Panel (b) needs headroom below zero for the sub-chance dissenter curve;
    # panel (a) has only one model below zero, so giving it the same floor
    # would spend a third of its width on blank shading.
    lo, hi = -0.23, 0.37
    # Panel (a) needs its own ceiling: it was reusing panel (b)'s y-limit, which
    # left a quarter of its width empty beyond the highest model.
    lo_a = -0.045
    hi_a = max(m["norm_modal"].max(), m["norm_dissenter"].max()) + 0.03

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(pf.FULL, 2.70),
        gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08},
        layout="constrained")

    # ---- panel (a): per-model dumbbell ---------------------------------
    n = len(m)
    ax1.axvspan(lo_a, 0, color="#F6F0F0", zorder=0)
    ax1.axvline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    for i, r in m.iterrows():
        ax1.plot([r["norm_dissenter"], r["norm_modal"]], [i, i],
                 color="#B8BCC4", linewidth=1.6, zorder=2,
                 solid_capstyle="round")
        ax1.scatter(r["norm_dissenter"], i, s=20, facecolor="white",
                    edgecolor=pf.DISS_C, linewidth=1.2, zorder=4)
        ax1.scatter(r["norm_modal"], i, s=23, color=pf.MODAL_C,
                    edgecolor="none", zorder=4)

    ax1.set_yticks(range(n))
    ax1.set_yticklabels(m["label"], fontsize=9)
    # The gloss on zero lives on Figure 1, where the reader first meets the
    # metric; here the shaded band and its label already mark it.
    ax1.set_xlabel("Normalized accuracy")
    # Say what is measured. The split is the cell's own majority; whether the
    # majority the models follow is country-specific is a separate question,
    # and the answer is no, so neither "their country" nor "the common answer"
    # would describe this panel accurately.
    ax1.set_title("(a) Every model is better on people who give the "
                  "common answer", loc="left", fontsize=9)
    ax1.set_xlim(lo_a, hi_a)
    ax1.set_ylim(-0.85, n - 0.25)
    ax1.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax1.set_axisbelow(True)

    # ---- panel (b): mechanism ------------------------------------------
    order = BINS
    s = s.reindex(order)
    x = list(range(len(order)))

    ax2.axhspan(lo, 0, color="#F6F0F0", zorder=0)
    ax2.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    ax2.plot(x, s["norm_True"], marker="o", markersize=4.5, color=pf.MODAL_C,
             linewidth=1.6, zorder=3)
    ax2.plot(x, s["norm_False"], marker="o", markersize=4.5, color=pf.DISS_C,
             linewidth=1.6, markerfacecolor="white", markeredgewidth=1.2,
             zorder=3)

    # Both end labels sit to the right of their final point, on the same
    # baseline as it. Placed above or below they either crossed the curve or
    # ran into the frame.
    for col, colour in (("norm_True", pf.MODAL_C), ("norm_False", pf.DISS_C)):
        ax2.annotate(f"{s[col].iloc[-1]:+.2f}",
                     xy=(3, s[col].iloc[-1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=9,
                     color=colour, ha="left", va="center")
    # Callout sits in the empty lower-left of the shaded band, clear of the curve.
    ax2.text(0.35, lo + 0.025, "worse than guessing", fontsize=9,
             color="#8B3A3A", ha="left", va="bottom", style="italic")

    ax2.set_xticks(x)
    ax2.set_xticklabels(BIN_TICKS)
    ax2.set_xlabel(SHARE_LABEL)
    ax2.set_ylabel("Normalized accuracy")
    ax2.set_title("(b) Consensus helps one group, costs the other",
                  loc="left", fontsize=9)
    ax2.set_ylim(lo, hi)
    ax2.set_xlim(-0.2, 3.62)
    ax2.grid(axis="y", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax2.set_axisbelow(True)

    # Group shares are read off the data so the legend cannot drift from the
    # split the analysis actually used.
    modal_pct = 100 * s["n_True"].sum() / (s["n_True"].sum() + s["n_False"].sum())

    # One legend for both panels: the two colors mean the same thing in each.
    # Upper-left of panel (b) stays empty of both curves through the first two
    # bins; the box keeps the two dots from reading as points on them.
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=pf.MODAL_C, markeredgecolor="none",
                      label=f"Gives the common answer ({modal_pct:.1f}%)"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor="white", markeredgecolor=pf.DISS_C,
                      markeredgewidth=1.2, label=f"Does not ({100 - modal_pct:.1f}%)"),
    ]
    # 6.6pt, not the default 7: at 7 the box's lower right corner met the blue
    # curve on its rise into the last bin.
    pf.boxed_legend(ax2, handles, loc="upper left", fontsize=9,
                    bbox_to_anchor=(0.0, 1.0))

    pf.save(fig, "figure_dissenter_penalty", pf.FULL)

    print("\npanel (b) values:")
    print(s[["norm_True", "norm_False", "n_True", "n_False"]].round(3).to_string())


if __name__ == "__main__":
    main()
