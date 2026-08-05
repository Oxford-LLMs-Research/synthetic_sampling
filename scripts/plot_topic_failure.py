#!/usr/bin/env python3
"""Per-topic accuracy: the models against XGBoost on the same questions.

One row per fine-grained topic, sorted by model accuracy. The bar between the
two markers is the shortfall. XGBoost stays well above chance on every topic,
including the six where the models fall below it, so those topics are hard for
the models rather than hard in themselves.

A dumbbell rather than a scatter: every topic can be named, the shortfall is a
length rather than a distance from a diagonal, and which topics cross zero is
readable without a legend.

Input: analysis/topic_difficulty/by_topic.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paperfig as pf

TD = pf.ROOT / "analysis" / "topic_difficulty"
AAAI_FIG_DIR = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")

# The per-model dots carry family color (Figure 1's palette), so the two
# summary markers have to sit outside that palette to stay readable as
# summaries. Black against blue does that; the earlier red mean was a near
# match for GPT-OSS.
LLM_C = "#1A1A1A"
XGB_C = "#2E5090"
BAR_C = "#BFC5CC"

RENAME = {"Media Information": "Media & information",
          "International Relations": "International relations",
          "Climate Environment": "Climate & environment",
          "Security Safety": "Security & safety",
          "Institutional Confidence": "Institutional confidence",
          "Corruption Perceptions": "Corruption perceptions",
          "Government Performance": "Government performance"}


def main() -> None:
    pf.use_style()
    t = pd.read_csv(TD / "by_topic.csv").sort_values("llm").reset_index(drop=True)
    t["name"] = (t["tag"].str.replace("_", " ").str.capitalize()
                 .replace({k.capitalize(): v for k, v in RENAME.items()}))
    y = range(len(t))

    # Rotated to full width. Thirty-one topic labels at the nine-point floor
    # AAAI requires need ~4.7in of rows read vertically, which made a
    # column-width version 6.4in tall and unplaceable: a float that deep leaves
    # its column nearly empty. Read horizontally the same labels sit under the
    # axis at 45 degrees, the accuracy axis needs only its own range rather than
    # one slot per topic, and the float lands at the top of a page.
    fig, ax = plt.subplots(figsize=(pf.FULL, 3.05), layout="constrained")

    # Low enough for the per-model points: two of them reach -0.31, and an
    # axis that clipped them would understate the spread it is there to show.
    y_lo = -0.36
    ax.axhspan(y_lo, 0, color="#F6F0F0", zorder=0)
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    ax.vlines(y, t["llm"], t["xgb"], color=BAR_C, linewidth=1.7, zorder=2)

    # One dot per model behind the mean: the mean alone cannot show whether a
    # topic is below chance for all thirteen or only for some. Colored by
    # family on Figure 1's palette, which turns the cloud from a spread into a
    # ranking the reader can follow across topics -- Qwen and DeepSeek lead
    # almost everywhere, the Llama base models trail almost everywhere.
    pm = pd.read_csv(TD / "by_topic_model.csv")
    row_of = {tag: i for i, tag in enumerate(t["tag"])}
    rng = np.random.default_rng(0)
    pm = pm[pm["tag"].isin(row_of)].copy()
    pm["x"] = [row_of[tg] for tg in pm["tag"]] + rng.uniform(-0.22, 0.22, len(pm))
    pm["family"] = pm["model"].map(lambda m: pf.MODEL_META[m][1])
    pm["instruct"] = pm["model"].map(lambda m: pf.MODEL_META[m][2])
    # Lightly muted: the per-model cloud is context, the two summary markers
    # are the comparison. Full saturation let thirteen colors compete with
    # them; too much muting lost the family ranking the cloud exists to show.
    # The size gap to the summary markers now carries most of the hierarchy.
    for (fam, inst), g in pm.groupby(["family", "instruct"]):
        c = pf.FAMILY_COLORS[fam]
        ax.scatter(g["x"], g["llm"], s=4.3, alpha=0.68, linewidth=0.4,
                   facecolor=c if inst else "white", edgecolor=c, zorder=2.5)

    ax.scatter(y, t["llm"], s=21, color=LLM_C, edgecolor="white",
               linewidth=0.55, zorder=3)
    ax.scatter(y, t["xgb"], s=21, color=XGB_C, edgecolor="white",
               linewidth=0.55, zorder=3)

    ax.set_xticks(list(y))
    ax.set_xticklabels(t["name"], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.set_ylabel("Normalized accuracy")
    ax.set_xlim(-0.8, len(t) - 0.2)
    ax.set_ylim(y_lo, 1.06)
    ax.grid(axis="y", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    # In the shaded band, at the end where no topic dips into it.
    ax.text(len(t) - 0.6, y_lo + 0.02, "worse than random guessing", fontsize=9,
            color="#8B3A3A", style="italic", ha="right", va="bottom")

    # One key, not two. Rotating freed the band above the highest XGBoost
    # value (0.73), which at full width holds all ten entries in one row.
    # Column-major fill keeps the pairs grouped: the two summary markers, then
    # the instruct/base contrast in Figure 1's wording, then the six families.
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=4.4,
               markerfacecolor=LLM_C, markeredgecolor="white",
               markeredgewidth=0.5, label="13-model mean"),
        Line2D([], [], marker="o", linestyle="", markersize=2.3,
               markerfacecolor="#8C8C8C", markeredgecolor="#8C8C8C",
               markeredgewidth=0.4, alpha=0.68, label="instruct"),
        Line2D([], [], marker="o", linestyle="", markersize=4.4,
               markerfacecolor=XGB_C, markeredgecolor="white",
               markeredgewidth=0.5, label="XGBoost"),
        Line2D([], [], marker="o", linestyle="", markersize=2.3,
               markerfacecolor="white", markeredgecolor="#8C8C8C",
               markeredgewidth=0.4, label="base"),
    ]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=2.6,
                       markerfacecolor=c, markeredgecolor=c,
                       markeredgewidth=0.4, alpha=0.68, label=f)
                for f, c in pf.FAMILY_COLORS.items()]
    opts = dict(pf.LEGEND_BOX)
    opts.update(ncol=5, borderpad=0.4, labelspacing=0.32,
                columnspacing=1.2, handletextpad=0.4)
    leg = ax.legend(handles=handles, loc="upper center", **opts)
    leg.get_frame().set_linewidth(0.5)

    pf.save(fig, "figure_topic_failure", pf.FULL)

    AAAI_FIG_DIR.mkdir(parents=True, exist_ok=True)
    src = pf.OUT_DIR / "figure_topic_failure.pdf"
    (AAAI_FIG_DIR / src.name).write_bytes(src.read_bytes())
    print(f"  copied to {AAAI_FIG_DIR}")
    print(f"  topics below chance for the models: {int((t['llm'] < 0).sum())}")
    print(f"  XGBoost range: {t['xgb'].min():.3f} to {t['xgb'].max():.3f}")
    print(f"  XGBoost above the models on all {len(t)} topics: "
          f"{bool((t['xgb'] > t['llm']).all())}")


if __name__ == "__main__":
    main()
