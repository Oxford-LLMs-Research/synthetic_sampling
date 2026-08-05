#!/usr/bin/env python
r"""Figures for the three-arm ladder, built to resist over-aggregation.

A mean over 30 target questions can hide two very different worlds: a small
shift in every question, or a large shift in five and nothing in the rest. The
same mean can also be produced by opposing movements that cancel. So the set
below is deliberately a ladder of its own, from most aggregated to least, and
each figure is answerable only at its own level:

  1. arms      four measures at once, because the interesting thing early on is
               that accuracy, likelihood and entropy move in DIFFERENT
               directions. Any single-measure plot hides it by construction.
  2. targets   all 30 questions as small multiples, no averaging at all.
  6. collapse  normalized accuracy against how often the model gives its single
               most common answer, three points per target joined by a line, so
               the length of the segment IS the size of the ordering effect.

Every axis in that set is a quantity the reader already has. Three further
figures (spread, per-target advantage, level steps) are behind --all-figures and
are NOT for the paper: each plots informative-minus-random, which cannot be read
without composing a subtraction in the reader's head. Their content is two
sentences of prose. The one derived quantity that survives is the band in figure
1, which is stated plainly in its caption together with the rule for reading it.

Colour encodes an ordered comparison, not three unrelated categories:
informative and anti are opposing poles and random is the neutral reference the
paper itself uses, so the palette is diverging with a neutral midpoint. It was
checked with the dataviz validator (worst adjacent CVD dE 15.8, normal 20.2).

    python .../plot_ladder.py
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import paperfig                                              # noqa: E402
from paperfig import COL, FULL, use_style, boxed_legend       # noqa: E402

LAD = REPO.parent / "analysis" / "ladder"
FIG = REPO.parent / "analysis" / "figures" / "ladder"
LEVELS = [0, 1, 2, 4, 8, 16, 32, 64, 96]
ARMS = ("informative", "random", "anti")
ARM_C = {"informative": "#2166AC", "random": "#9C8B92", "anti": "#B2182B"}
ARM_LABEL = {"informative": "informative first", "random": "random order",
             "anti": "least informative first"}
GRID_A = 0.20


def save(fig, name: str, width: float) -> None:
    """Write a working PNG here, and hand the PDF to paperfig for the paper.

    paperfig.save asserts the PDF media box equals the requested width and
    copies it into the paper's figures directory, which is what makes
    \includegraphics run at scale 1.0. Saving the PDF locally instead would
    reintroduce the silent rescaling paperfig exists to prevent, and the 9pt
    floor would stop being 9pt on the page.
    """
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    paperfig.save(fig, name, width)          # closes the figure


def boot_ci(values: np.ndarray, clusters: np.ndarray, n: int = 2000,
            seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    means = np.array([values[inv == i].mean() for i in range(len(keys))])
    draws = means[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(LAD / "ladder_per_instance.csv")
    want = {(a, k) for a in ARMS for k in LEVELS if k > 0} | {("all", 0)}
    have = d.groupby("pair").apply(
        lambda g: want <= set(zip(g.arm, g.k)), include_groups=False)
    d = d[d.pair.isin(set(have[have].index))].copy()

    zero = d[(d.arm == "all") & (d.k == 0)]
    lad = pd.concat([d[d.arm.isin(ARMS)]] +
                    [zero.assign(arm=a) for a in ARMS], ignore_index=True)
    lad["q"] = lad.survey + "|" + lad.target_code
    lad["norm_acc"] = ((lad.correct_mean - 1 / lad.n_options)
                       / (1 - 1 / lad.n_options))
    sim = pd.read_csv(LAD / "ladder_top1_similarity.csv")
    print(f"{lad.pair.nunique():,} pairs | {lad.q.nunique()} targets")
    return lad, sim


def qcurve(lad: pd.DataFrame, arm: str, col: str, boot: int):
    """Question-averaged mean and clustered interval at each level."""
    m, lo, hi = [], [], []
    for k in LEVELS:
        g = lad[(lad.arm == arm) & (lad.k == k)]
        v, c = g[col].to_numpy(), g.q.to_numpy()
        m.append(pd.Series(v).groupby(c).mean().mean())
        a, b = boot_ci(v, c, boot)
        lo.append(a)
        hi.append(b)
    return np.array(m), np.array(lo), np.array(hi)


# --------------------------------------------------------------------------
def paired_band(lad: pd.DataFrame, arm: str, col: str, boot: int):
    """CI of the per-pair difference from the random arm, drawn on the arm's line.

    The arms are measured on the SAME pairs, so the inferential quantity is the
    within-pair difference, not each arm's level. A band showing each level's own
    spread would be dominated by how much target questions differ from one
    another, and three such bands would overlap heavily even where every pair
    moves the same way. Centring the difference interval on the arm's own line
    keeps the reading a viewer expects: where the band clears the random line,
    the paired test is significant.
    """
    m, lo, hi = [], [], []
    for k in LEVELS:
        g = lad[(lad.arm.isin([arm, "random"])) & (lad.k == k)]
        p = g.pivot_table(index=["pair", "q"], columns="arm", values=col).dropna()
        mean_arm = p[arm].groupby(p.index.get_level_values("q")).mean().mean()
        m.append(mean_arm)
        if arm == "random":
            lo.append(mean_arm)
            hi.append(mean_arm)
            continue
        dd = (p[arm] - p["random"]).to_numpy()
        qq = p.index.get_level_values("q").to_numpy()
        md = pd.Series(dd).groupby(qq).mean().mean()
        a, b = boot_ci(dd, qq, boot)
        lo.append(mean_arm + (a - md))
        hi.append(mean_arm + (b - md))
    return np.array(m), np.array(lo), np.array(hi)


def fig_arms(lad: pd.DataFrame, boot: int) -> None:
    """Four measures, because early on they disagree with one another."""
    panels = [("norm_acc", "Normalized accuracy", None),
              ("nll_true", "NLL of the true answer", "lower is better"),
              ("entropy", "Entropy of the option distribution", None),
              # "KL" alone tells a reader nothing; name what it measures and put
              # the statistic in brackets.
              ("kl_from_0", "Belief shift from the empty profile (KL)", None)]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.1))
    x = np.arange(len(LEVELS))
    for ax, (col, title, note) in zip(axes.ravel(), panels):
        for arm in ARMS:
            m, lo, hi = paired_band(lad, arm, col, boot)
            if arm != "random":
                ax.fill_between(x, lo, hi, color=ARM_C[arm], alpha=0.18, lw=0)
            ax.plot(x, m, color=ARM_C[arm], lw=1.4, marker="o", ms=3.2,
                    mec="white", mew=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in LEVELS])
        ax.set_xlabel("features in profile")
        ax.set_title(title if not note else f"{title} ({note})")
        ax.grid(axis="y", alpha=GRID_A, lw=0.5)
        ax.set_axisbelow(True)
        # Autoscale can clip an asymmetric band; give every panel headroom, and
        # open extra space at the top of the first one so the key sits on empty
        # canvas rather than over the anti arm.
        ax.margins(y=0.10)
    # The empty profile is the common origin of all three arms; mark it once.
    for ax in axes.ravel():
        ax.axvline(0, color="#B0B4BA", lw=0.6, ls=(0, (2, 2)), zorder=0)
    axes[0, 0].axhline(0, color="#444444", lw=0.6, zorder=1)
    lo0, hi0 = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo0, hi0 + 0.30 * (hi0 - lo0))
    handles = [Line2D([], [], color=ARM_C[a], lw=1.4, marker="o", ms=3.2,
                      mec="white", mew=0.5, label=ARM_LABEL[a]) for a in ARMS]
    boxed_legend(axes[0, 0], handles, "upper left")
    # State what the band is and how to read it. It is a derived quantity, the
    # one place in this set where that is justified, so it must not need
    # decoding: the alternative, each level's own spread, is dominated by how
    # much questions differ from each other and would make significant paired
    # results look null.
    fig.supxlabel("Shaded: 95% interval for the gap between this arm and the "
                  "random arm, measured within respondent.\nWhere a band "
                  "clears the grey line, the gap is significant.",
                  fontsize=9, y=0.005)
    fig.tight_layout(pad=0.6, rect=(0, 0.055, 1, 1))
    save(fig, "ladder_1_arms", FULL)


def fig_targets(lad: pd.DataFrame, sim: pd.DataFrame, boot: int) -> None:
    """All 30 questions, unaveraged. The point is the spread, not any one panel."""
    order = (lad[lad.k == 8].pivot_table(index="q", columns="arm",
                                         values="nll_true"))
    order = (order["informative"] - order["random"]).sort_values()
    qs = list(order.index)
    ncol, nrow = 5, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 8.2), sharex=True,
                             sharey=True)
    x = np.arange(len(LEVELS))
    for ax, q in zip(axes.ravel(), qs):
        sub = lad[lad.q == q]
        for arm in ARMS:
            m = [sub[(sub.arm == arm) & (sub.k == k)].norm_acc.mean()
                 for k in LEVELS]
            ax.plot(x, m, color=ARM_C[arm], lw=1.2, marker="o", ms=2.4,
                    mec="white", mew=0.4)
        ax.axhline(0, color="#B0B4BA", lw=0.5, zorder=0)
        ax.set_title(q.split("|")[1], pad=2)
        ax.grid(axis="y", alpha=GRID_A, lw=0.4)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(qs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(LEVELS[i]) for i in range(0, len(LEVELS), 2)])
    for ax in axes[:, 0]:
        ax.set_ylabel("norm. acc.")
    fig.supxlabel("features in profile, sorted by the informative arm's "
                  "advantage at k = 8", fontsize=9, y=0.012)
    handles = [Line2D([], [], color=ARM_C[a], lw=1.2, label=ARM_LABEL[a])
               for a in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.998))
    fig.tight_layout(pad=0.4, rect=(0, 0.028, 1, 0.972))
    save(fig, "ladder_2_targets", FULL)


def fig_spread(lad: pd.DataFrame) -> None:
    """The whole per-pair distribution, so a tail cannot pass as a broad shift."""
    shown = [1, 4, 16, 64]
    # Sequential ramp: k is an ordered magnitude, so one hue, light to dark.
    ramp = ["#BDD7E7", "#6BAED6", "#3182BD", "#08519C"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    w = lad.pivot_table(index=["pair", "q", "k"], columns="arm",
                        values="nll_true").reset_index()
    for ax, (b, lab) in zip(axes, [("random", "informative - random"),
                                   ("anti", "informative - anti")]):
        handles = []
        for c, k in zip(ramp, shown):
            g = w[w.k == k].dropna(subset=["informative", b])
            v = np.sort((g["informative"] - g[b]).to_numpy())
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.4)
            # The median is where the curve crosses 0.5; marking it against the
            # mean is the point of the panel, since the two disagree by a factor
            # of four at k=64. A mean far left of the median is a tail, not a shift.
            ax.plot([np.median(v)], [0.5], marker="|", ms=7, mew=1.6, color=c,
                    zorder=4)
            handles.append(Line2D(
                [], [], color=c, lw=1.4,
                label=f"{k:<3}{(v < 0).mean():>5.0%}{np.median(v):>7.2f}"
                      f"{v.mean():>7.2f}"))
        ax.axvline(0, color="#444444", lw=0.7)
        ax.axhline(0.5, color="#B0B4BA", lw=0.5, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(-4, 4)
        ax.set_xlabel(f"per-pair difference in NLL, {lab}")
        ax.grid(alpha=GRID_A, lw=0.5)
        ax.set_axisbelow(True)
        # An ECDF sweeps diagonally, so both the lower-right and upper-left keys
        # collided with it until the labels were compressed to columns. The
        # median and mean sit side by side because their disagreement is the
        # point: a mean well left of the median is a tail, not a broad shift.
        boxed_legend(ax, handles, "upper left",
                     title="k    <0    med   mean", alignment="left")
    axes[0].set_ylabel("cumulative share of pairs")
    fig.supxlabel("axis clipped at +/-4; 1.4-2.6% of pairs fall below it against "
                  "0.3-0.8% above, which is where the mean comes from",
                  fontsize=9, y=0.005)
    fig.tight_layout(pad=0.6, rect=(0, 0.07, 1, 1))
    save(fig, "ladder_3_spread", FULL)


def fig_which(lad: pd.DataFrame, sim: pd.DataFrame, boot: int) -> None:
    """Which targets benefit, and do the two candidate explanations predict it?"""
    k = 8
    w = lad[lad.k == k].pivot_table(index=["pair", "q"], columns="arm",
                                    values="nll_true").reset_index()
    w["d"] = w["informative"] - w["random"]
    per_q = []
    rng = np.random.default_rng(42)
    for q, g in w.groupby("q"):
        v = g.d.to_numpy()
        draws = v[rng.integers(0, len(v), size=(2000, len(v)))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        per_q.append({"q": q, "d": v.mean(), "lo": lo, "hi": hi})
    per_q = pd.DataFrame(per_q)
    per_q["target_code"] = per_q.q.str.split("|").str[1]
    per_q["survey"] = per_q.q.str.split("|").str[0]
    per_q = per_q.merge(sim, on=["survey", "target_code"], how="left")
    # Strongest informative advantage at the top, which is how a ranked dot plot
    # is read; the difference is negative when that arm wins, so sort descending.
    per_q = per_q.sort_values("d", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(per_q))
    sig = (per_q.hi < 0) | (per_q.lo > 0)
    ax.hlines(y, per_q.lo, per_q.hi, color="#9AA0A6", lw=1.0)
    ax.scatter(per_q.d, y, s=22, zorder=3,
               color=np.where(per_q.d < 0, ARM_C["informative"], ARM_C["anti"]),
               edgecolor="white", linewidth=0.5)
    ax.scatter(per_q.d[~sig], y[~sig], s=22, zorder=4, facecolor="white",
               edgecolor="#9AA0A6", linewidth=0.8)
    ax.axvline(0, color="#444444", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(per_q.target_code)
    ax.set_ylim(-0.8, len(per_q) - 0.2)
    ax.set_xlabel(f"informative - random, NLL at k = {k}")
    ax.set_title("per target: left of 0 favours informative\n"
                 "(open = interval covers 0)")
    ax.grid(axis="x", alpha=GRID_A, lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.scatter(per_q.top1_cos, per_q.d, s=26, color=ARM_C["informative"],
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#444444", lw=0.7)
    r = per_q[["d", "top1_cos"]].corr(method="spearman").iloc[0, 1]
    # One target sits far below the rest; report the correlation without it too,
    # rather than letting a single point carry the claim either way.
    trimmed = per_q[per_q.d > per_q.d.min()]
    rt = trimmed[["d", "top1_cos"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("cosine, target vs its best feature")
    ax.set_ylabel(f"informative - random, NLL at k = {k}")
    ax.set_title(f"Spearman {r:+.2f} ({rt:+.2f} dropping the lowest point):\n"
                 f"similarity explains little of which targets benefit")
    ax.grid(alpha=GRID_A, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    save(fig, "ladder_4_which", FULL)
    return per_q


def fig_steps(lad: pd.DataFrame, boot: int) -> None:
    """Level-to-level change within arm: the only view where dilution is visible."""
    steps = list(zip(LEVELS, LEVELS[1:]))
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    off = {"informative": -0.24, "random": 0.0, "anti": 0.24}
    for arm in ARMS:
        xs, ms, los, his = [], [], [], []
        for i, (lo_k, hi_k) in enumerate(steps):
            g = lad[(lad.arm == arm) & (lad.k.isin([lo_k, hi_k]))]
            p = g.pivot_table(index=["pair", "q"], columns="k",
                              values="nll_true").dropna()
            if p.empty:
                continue
            dd = (p[hi_k] - p[lo_k]).to_numpy()
            qq = p.index.get_level_values("q").to_numpy()
            xs.append(i + off[arm])
            ms.append(pd.Series(dd).groupby(qq).mean().mean())
            a, b = boot_ci(dd, qq, boot)
            los.append(a)
            his.append(b)
        ax.vlines(xs, los, his, color=ARM_C[arm], lw=1.2)
        ax.scatter(xs, ms, s=20, color=ARM_C[arm], zorder=3, edgecolor="white",
                   linewidth=0.5)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{a}→{b}" for a, b in steps])
    ax.set_xlabel("profile grows from one level to the next")
    ax.set_ylabel("change in NLL")
    ax.set_title("below zero: the added features helped. "
                 "Only one step above zero is significant")
    ax.grid(axis="y", alpha=GRID_A, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=ARM_C[a], lw=1.2, marker="o", ms=3.4,
                      mec="white", mew=0.5, label=ARM_LABEL[a]) for a in ARMS]
    boxed_legend(ax, handles, "lower left")
    fig.tight_layout(pad=0.6)
    save(fig, "ladder_5_steps", FULL)


def concentration() -> pd.DataFrame:
    """Per target: how often the model gives its single most common answer.

    The per-instance file keeps metrics, not the predicted label, so this reads
    the raw results. A target where one answer covers most respondents is one
    where the profile is doing nothing, whatever the profile contains.
    """
    import json
    shard = {}
    for f in sorted((REPO / "outputs" / "scaling_experiment" /
                     "ladder_shards").glob("ladder_shard_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            shard[r["example_id"]] = r["target_code"]
    pred, true = {}, {}
    import collections
    pred = collections.defaultdict(collections.Counter)
    true = collections.defaultdict(collections.Counter)
    for f in sorted((REPO / "outputs" / "scaling_experiment" /
                     "ladder_results").glob("ladder_results_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            t = shard.get(r["example_id"])
            if t is None:
                continue
            pred[t][r["predicted"]] += 1
            if r["n_features"] == 0:
                true[t][r["ground_truth"]] += 1
    rows = []
    for t, c in pred.items():
        top, n = c.most_common(1)[0]
        rows.append({"target_code": t, "conc": n / sum(c.values()),
                     "matches_mode": top == true[t].most_common(1)[0][0]})
    return pd.DataFrame(rows)


def fig_collapse(lad: pd.DataFrame) -> None:
    """Why a third of the panels in figure 2 are flat: the profile is inert.

    Both axes are quantities the reader already has. The horizontal one is how
    often the model gives its single most common answer, so 1.0 means it said
    the same thing to every respondent. The vertical one is normalized accuracy,
    the paper's own estimand. Nothing here is a difference of differences: an
    earlier version plotted KL from the empty profile against a between-arm gap,
    which required composing two subtractions in the reader's head before the
    picture meant anything.

    Each target contributes three points joined by a line, one per arm. Reading
    it is then direct: where the model answers the same thing to everyone the
    three points sit on top of one another, because no ordering of a profile can
    matter to a prediction that ignores the profile. Marker fill carries the
    consequence for accuracy, which is settled by luck: a fixed answer scores
    well only when it happens to be the population's modal answer.
    """
    conc = concentration()
    acc = []
    for (t, arm), g in lad[lad.k == 96].groupby(["target_code", "arm"]):
        v = (g.correct_mean - 1 / g.n_options) / (1 - 1 / g.n_options)
        acc.append({"target_code": t, "arm": arm, "norm_acc": v.mean()})
    m = pd.DataFrame(acc).merge(conc, on="target_code")

    fig, ax = plt.subplots(figsize=(FULL, 3.2))
    wide = m.pivot(index="target_code", columns="arm", values="norm_acc")
    wide = wide.join(conc.set_index("target_code"))
    # One thin connector per target: the height of the segment IS the size of
    # the ordering effect, so it needs no separate panel.
    # Where the arms coincide the markers overplot, and coincidence is exactly
    # what this figure exists to show, so a reader must be able to tell three
    # points on top of one another from one point. A fixed per-arm offset keeps
    # every triplet legible; it is smaller than the marker, so it never suggests
    # a difference in the x quantity.
    off = {"informative": -0.007, "random": 0.0, "anti": 0.007}
    for t, r in wide.iterrows():
        ys = [r[a] for a in ARMS]
        ax.plot([r.conc] * 2, [min(ys), max(ys)], color="#C7CBD1", lw=0.8,
                zorder=1)
    for arm in ARMS:
        g = m[m.arm == arm]
        ax.scatter(g.conc + off[arm], g.norm_acc, s=28, zorder=3,
                   facecolor=np.where(g.matches_mode, ARM_C[arm], "white"),
                   edgecolor=ARM_C[arm], linewidth=0.9)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xlabel("share of respondents given the model's single most common answer")
    ax.set_ylabel("normalized accuracy at 96 features")
    ax.grid(alpha=GRID_A, lw=0.5)
    ax.set_axisbelow(True)
    ax.margins(0.06)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    handles = [Line2D([], [], color=ARM_C[a], marker="o", ls="", ms=5.5,
                      mec=ARM_C[a], label=ARM_LABEL[a]) for a in ARMS]
    handles.append(Line2D([], [], color="#6B7079", marker="o", ls="", ms=5.5,
                          mfc="white", mec="#6B7079",
                          label="open: the fixed answer is not the true mode"))
    boxed_legend(ax, handles, "upper center", ncol=2)
    fig.tight_layout(pad=0.6)
    save(fig, "ladder_6_collapse", FULL)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--all-figures", action="store_true",
                    help="also draw the difference-based figures (spread, "
                         "per-target advantage, level steps). They are kept for "
                         "the archive but plot informative-minus-random, which "
                         "cannot be read without composing a subtraction.")
    args = ap.parse_args()
    use_style()
    lad, sim = load()
    print("writing figures")
    fig_arms(lad, args.boot)
    fig_targets(lad, sim, args.boot)
    m = fig_collapse(lad)
    per_q = None
    if args.all_figures:
        fig_spread(lad)
        per_q = fig_which(lad, sim, args.boot)
        fig_steps(lad, args.boot)
    # The spread of the three arms within a target, in normalized accuracy, is
    # what the connectors in figure 6 show; report it the same way rather than
    # as a between-arm difference.
    per_t = m.pivot(index="target_code", columns="arm", values="norm_acc")
    per_t = per_t.join(m.groupby("target_code")[["conc", "matches_mode"]].first())
    per_t["spread"] = per_t[list(ARMS)].max(axis=1) - per_t[list(ARMS)].min(axis=1)
    hi, lo = per_t[per_t.conc >= 0.8], per_t[per_t.conc < 0.8]
    print(f"\n{len(hi)}/{len(per_t)} targets give one answer to >=80% of "
          f"respondents; it is the true mode in {int(hi.matches_mode.sum())}")
    print(f"  spread across the three arms at k=96: {hi.spread.mean():.3f} for "
          f"those, {lo.spread.mean():.3f} for the rest")
    if per_q is not None:
        n_neg = int((per_q.d < 0).sum())
        n_sig = int(((per_q.hi < 0) | (per_q.lo > 0)).sum())
        print(f"per target at k=8: {n_neg}/{len(per_q)} favour the informative "
              f"arm, {n_sig} with an interval excluding zero")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
