#!/usr/bin/env python3
"""Figure 1 v5 - full-width, shared model axis.

Replaces the v4 scatter of normalized accuracy against variance ratio. The
scatter required a reader to decode 13 overlapping labels to find which model
was which, and it made the paper's headline comparison (every model sits far
below both no-model baselines) into two reference lines that were easy to miss.

The redesign puts the model names once, on a shared vertical axis, and gives
each claim its own panel:

  (a) normalized accuracy, with the majority-class and XGBoost baselines drawn
      as the ceilings the models fail to reach. The shaded span between the
      best model and XGBoost is the gap the paper is about.
  (b) variance ratio on the same row order, so flattening can be read per model
      without re-locating it in a cloud of points.

Drawn at exactly \\textwidth so \\includegraphics applies no rescaling.
"""
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import paperfig as pf

ANALYSIS = pf.ROOT / "analysis"
NORM_ACC = ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv"
MAJORITY = ANALYSIS / "normalized_accuracy" / "majority_class_norm_acc.csv"
XGB = ANALYSIS / "xgboost_baseline" / "results_merged.csv"
# Entropy ratios come from the soft/hard analysis rather than the older
# vr_jsd_by_model.csv cache, so this panel, the main text and the appendix table
# all report one computation. The two agree to within 0.028 with a Spearman
# correlation of 0.99, but agreeing exactly is worth more than that.
ENTROPY = ANALYSIS / "soft_hard" / "soft_hard_by_model.csv"

PT = "s6m4"

# soft_hard keys models by display name; everything else keys by directory name.
_DISPLAY_TO_KEY = {v[0]: k for k, v in pf.MODEL_META.items()}


def load():
    acc = (pd.read_csv(NORM_ACC).query("profile_type == @PT")
           .groupby("model")["norm_acc"].mean().rename("norm_acc"))
    maj = pd.read_csv(MAJORITY).query("profile_type == @PT")["majority_norm_acc"].mean()
    xgb = pd.read_csv(XGB).query("profile_type == @PT")["xgb_norm_acc"].mean()
    er = pd.read_csv(ENTROPY)
    er["key"] = er["model"].map(_DISPLAY_TO_KEY)
    missing = er[er["key"].isna()]["model"].tolist()
    if missing:
        raise KeyError(f"unmapped model names in {ENTROPY.name}: {missing}")
    vr = er.set_index("key")["er_hard"]

    df = pd.DataFrame({"norm_acc": acc, "vr": vr})
    df = df[df.index.isin(pf.MODEL_META)].copy()
    df["display"] = df.index.map(lambda m: pf.MODEL_META[m][0])
    df["family"] = df.index.map(lambda m: pf.MODEL_META[m][1])
    df["is_instruct"] = df.index.map(lambda m: pf.MODEL_META[m][2])
    df["color"] = df["family"].map(pf.FAMILY_COLORS)
    df["face"] = df.apply(
        lambda r: r["color"] if r["is_instruct"] else "white", axis=1)
    return df.sort_values("norm_acc").reset_index(drop=True), maj, xgb


def main():
    pf.use_style()
    df, maj, xgb = load()
    n = len(df)
    y = range(n)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(pf.FULL, 2.28), sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.06},
        layout="constrained")

    # ---- panel (a): normalized accuracy --------------------------------
    best = df["norm_acc"].max()
    ax1.axvspan(best, xgb, color="#F2F4F7", zorder=0)

    for i, r in df.iterrows():
        ax1.plot([0, r["norm_acc"]], [i, i], color=r["color"],
                 alpha=0.30, linewidth=1.1, zorder=2, solid_capstyle="butt")
        ax1.scatter(r["norm_acc"], i, s=22, facecolor=r["face"],
                    edgecolor=r["color"], linewidth=1.2, zorder=4)

    ax1.axvline(0, color="#999999", linewidth=0.7, zorder=1)
    ax1.axvline(xgb, color=pf.XGB_C, linestyle="-.", linewidth=1.2, zorder=3)
    ax1.axvline(maj, color=pf.BASELINE_C, linestyle="--", linewidth=1.2, zorder=3)

    # Both baseline labels run leftward from their own line, stacked. Putting
    # the majority label to the right of its line pushed it past the data range
    # and into the gap between the panels, which squeezed panel (b).
    ax1.text(maj - 0.006, n - 0.30, f"Majority {maj:.2f}", fontsize=6.5,
             color=pf.BASELINE_C, ha="right", va="center")
    ax1.text(xgb - 0.006, n - 1.35, f"XGBoost {xgb:.2f}", fontsize=6.5,
             color=pf.XGB_C, ha="right", va="center")
    ax1.annotate("", xy=(best, -0.85), xytext=(xgb, -0.85),
                 arrowprops=dict(arrowstyle="<->", color="#7A7A7A", lw=0.8))
    ax1.text((best + xgb) / 2, -0.85, r"2$\times$ gap", fontsize=6.5,
             color="#5A5A5A", ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    ax1.set_yticks(list(y))
    ax1.set_yticklabels(df["display"], fontsize=6.8)
    ax1.set_xlabel("Normalized accuracy  (0 = random chance)")
    ax1.set_title("(a) No model approaches a no-model baseline", loc="left",
                  fontsize=8.5)
    # The ceiling is the majority line itself: nothing is drawn beyond it now
    # that both baseline labels sit inside.
    ax1.set_xlim(-0.028, maj + 0.010)
    ax1.set_ylim(-1.6, n + 0.15)
    ax1.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax1.set_axisbelow(True)
    # Light row guides so panel (b) can be tracked without repeating names.
    for i in y:
        ax1.axhline(i, color="#E6E6E6", linewidth=0.4, zorder=0)

    # ---- panel (b): variance ratio -------------------------------------
    # No shading here: every model falls below VR = 1, so a band over the
    # whole data range would mark nothing. The deficit is drawn per model as
    # the segment running back to the VR = 1 reference.
    for i in y:
        ax2.axhline(i, color="#E6E6E6", linewidth=0.4, zorder=0)
    for i, r in df.iterrows():
        ax2.plot([1.0, r["vr"]], [i, i], color=r["color"], alpha=0.30,
                 linewidth=1.1, zorder=2, solid_capstyle="butt")
        ax2.scatter(r["vr"], i, s=22, facecolor=r["face"],
                    edgecolor=r["color"], linewidth=1.2, zorder=4)

    ax2.axvline(1.0, color=pf.BASELINE_C, linestyle=":", linewidth=1.2, zorder=3)
    # Left of its line, like the two in panel (a); centered it overran the
    # right edge of the figure.
    ax2.text(0.994, n - 0.15, "human diversity", fontsize=6.5,
             color=pf.BASELINE_C, ha="right", va="bottom")
    # The "shortfall in diversity" arrow that used to sit here said what the
    # axis label already says, and it crowded the instruct/base key.

    ax2.set_xlabel("Entropy ratio  (< 1 = less diverse than people)")
    ax2.set_title("(b) All flatten opinion diversity", loc="left", fontsize=8.5)
    ax2.set_xlim(0.40, 1.06)
    ax2.set_ylim(-1.6, n + 0.15)
    ax2.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)

    # No key. Every open marker belongs to a row whose own label ends in
    # "base", so an Instruct/Base legend restates the y axis, and the only
    # interior space it could occupy is the shaded gap band in panel (a),
    # which a white box would punch a hole in.

    pf.save(fig, "figure1_v5", pf.FULL)
    print(f"  models {n}  best {df['norm_acc'].max():.3f}  "
          f"xgb {xgb:.3f}  maj {maj:.3f}  vr max {df['vr'].max():.3f}")


if __name__ == "__main__":
    main()
