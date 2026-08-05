#!/usr/bin/env python3
"""Appendix figure: marginal recovery over a wider pool of countries.

The main-paper comparison keeps only country-question cells with at least 30
respondents, which is what limits it to 70 of the 130 countries. That floor
exists because a whole answer distribution is noisier to estimate than a single
modal answer, but it also selects for large, well-surveyed countries, so the
obvious question is whether the result is an artifact of that selection.

This redraws the same comparison at a floor of 15, which admits 111 countries.
Both distributions shift right, as they must when smaller cells make the human
shares themselves noisier, and the gap between them is unchanged.

Source: analysis/marginal_recovery_wide/per_cell_*.csv
        (analyze_marginal_recovery_wide.py, which leaves the main outputs alone)
Output: analysis/figures/emnlp_revision/figure_marginal_recovery_wide.pdf
"""
from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paperfig as pf

WIDE = pf.ROOT / "analysis" / "marginal_recovery_wide"
MAIN = pf.ROOT / "analysis" / "marginal_recovery"
PROFILE = "s6m4"
MARGIN_C = "#3C4650"
KEYS = ["survey", "target_code", "country"]


def load(d):
    frames = []
    for f in sorted(glob.glob(str(d / "per_cell_*.csv"))):
        x = pd.read_csv(f)
        frames.append(x[x["profile_type"] == PROFILE])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    pf.use_style()
    wide, main = load(WIDE), load(MAIN)

    margin = wide.drop_duplicates(KEYS)["tv_global"]
    synth = wide["tv_model"]

    fig, ax = plt.subplots(figsize=(pf.COL, 2.35), layout="constrained")
    bins = np.arange(0, 1.02, 0.02)
    for v, colour in ((margin, MARGIN_C), (synth, pf.DISS_C)):
        ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)),
                color=colour, alpha=0.75, zorder=3)

    for v, colour, name, at in ((margin, MARGIN_C, "Pooled margin", 0.17),
                                (synth, pf.DISS_C, "Synthetic sample", 0.66)):
        med = v.median()
        ax.axvline(med, color=colour, linestyle=":", linewidth=0.9, zorder=4)
        ax.text(at, 16.0, f"{name}\nmedian {med:.2f}", fontsize=9,
                color=colour, ha="center", va="top", linespacing=1.2, zorder=5)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 16.4)
    ax.set_xlabel("Distance from the true answer shares (TV)\n"
                  "0 = exactly right, 1 = no answer in common")
    ax.set_ylabel("% of country–\nquestion cells")
    ax.grid(axis="x", linestyle="--", alpha=pf.GRID_A, linewidth=0.4)
    ax.set_axisbelow(True)

    pf.save(fig, "figure_marginal_recovery_wide", pf.COL)

    for tag, d in (("min 30", main), ("min 15", d_ := wide)):
        g = d.drop_duplicates(KEYS)
        print(f"  {tag}: {d.groupby(KEYS).ngroups:5d} cells, "
              f"{d['country'].nunique():3d} countries, "
              f"margin {g['tv_global'].median():.3f}, "
              f"synth {d['tv_model'].median():.3f}, "
              f"models win {100 * d['model_beats_global'].mean():.1f}%")


if __name__ == "__main__":
    main()
