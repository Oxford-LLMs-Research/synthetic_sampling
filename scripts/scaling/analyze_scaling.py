#!/usr/bin/env python
"""Profile-scaling experiment: does the k^0.28 curve hold at 48 and 96 features?

Reads Qwen 3 32B results for the two new levels alongside the existing 6/12/24
results and answers three questions:

  1. Does a single power law fit all five points, or does the curve bend?
  2. Where do the observed values fall against the predictions registered
     before the run (0.187 at k=48, 0.227 at k=96)?
  3. Do the entropy ratio and the dissenter penalty move with k? The paper's
     mechanism claim predicts they do not: more information should make the
     model somewhat more accurate without making it stop following the mode.

Everything is computed on a FIXED respondent x target set, the pairs present at
every level including k=96. Otherwise the curve would confound profile size
with respondent composition, since the respondents who can supply 96 features
are exactly those who answered more of the questionnaire.

Nothing the main analysis reads is modified; results go to
analysis/scaling_experiment/.

Usage:
    python .../analyze_scaling.py --results <dir with qwen scaling results>
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ANALYSIS = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
OUT = ANALYSIS / "scaling_experiment"
MODEL = "qwen3-32b"

# k -> profile_type tag. The first three come from the main experiment.
LEVELS = {6: "s3m2", 12: "s4m3", 24: "s6m4", 48: "s6m4x48", 96: "s6m4x96"}

# Registered before the run, from acc = 0.0639 * k^0.278 fitted on 6/12/24.
PREREGISTERED = {48: 0.187, 96: 0.227}
FLAT_NULL = 0.156


def norm_acc(correct: pd.Series, n_options: pd.Series) -> pd.Series:
    chance = 1.0 / n_options
    return (correct - chance) / (1.0 - chance)


def mechanism(df: pd.DataFrame, levels: dict[int, str]) -> pd.DataFrame:
    """Does more information stop the model following the mode?

    Three measures per level, each matching the method already used in the
    paper so the numbers are comparable to the ones it reports:

      entropy ratio     H(predicted labels) / H(recorded answers), per question
                        then averaged. Below 1 means the spread of views is
                        flattened. Hard labels only: results_data.csv carries
                        no option_logprobs, so the soft version of
                        analyze_soft_hard_heterogeneity.py cannot be computed
                        for the 6/12/24 levels.
      mode rate         share of predictions equal to the question's modal
                        recorded answer, pooled over countries. This is the
                        mechanism claim stated directly.
      penalty           normalized accuracy on respondents who gave the modal
                        answer minus those who did not (the question split of
                        analyze_dissenter_penalty.py).

    The paper's mechanism claim predicts accuracy rises with k while these
    three barely move.
    """
    from scipy.stats import entropy

    # Modal recorded answer per question. Answers do not depend on the level,
    # so this is computed once and applied to all of them.
    one = df[df["profile_type"] == LEVELS[24]]
    modes = (one.groupby(["survey", "target_code"])["ground_truth"]
             .agg(lambda s: s.mode().iloc[0]).rename("modal_answer").reset_index())
    df = df.merge(modes, on=["survey", "target_code"], how="left")
    df["is_modal"] = df["ground_truth"] == df["modal_answer"]
    df["pred_modal"] = df["predicted"] == df["modal_answer"]

    rows = []
    for k, tag in levels.items():
        g = df[df["profile_type"] == tag]

        ers = []
        for _, cell in g.groupby(["survey", "target_code"]):
            e = cell["ground_truth"].value_counts(normalize=True).to_numpy()
            h = cell["predicted"].value_counts(normalize=True).to_numpy()
            h_emp = entropy(e[e > 0])
            if h_emp > 0:
                ers.append(entropy(h[h > 0]) / h_emp)

        pen = (g.loc[g["is_modal"], "norm_acc"].mean()
               - g.loc[~g["is_modal"], "norm_acc"].mean())
        rows.append({"k": k, "entropy_ratio": float(np.mean(ers)),
                     "mode_rate": float(g["pred_modal"].mean()),
                     "penalty_norm": float(pen)})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def load_main() -> pd.DataFrame:
    """Per-instance results for the three existing levels."""
    from repair_ids import repair
    df = pd.read_csv(
        ANALYSIS / MODEL / "results_data.csv",
        usecols=["example_id", "survey", "respondent_id", "target_code",
                 "profile_type", "ground_truth", "predicted", "correct"],
        dtype=str, low_memory=False)
    df = repair(df, verbose=False)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(float)
    return df[df["profile_type"].isin(["s3m2", "s4m3", "s6m4"])]


def load_scaling(results_dir: Path) -> pd.DataFrame:
    """Per-instance results for k = 48 and 96, from the cluster run."""
    frames = []
    for p in sorted(results_dir.glob("*scaling*results*.csv")):
        frames.append(pd.read_csv(p, dtype=str, low_memory=False))
    if not frames:
        raise SystemExit(f"no scaling results found under {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true").astype(float)
    return df


def n_options_map() -> pd.DataFrame:
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc_fixed.csv")
    return q.groupby(["survey", "target_code"])["n_options"].first().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()

    df = pd.concat([load_main(), load_scaling(args.results)], ignore_index=True)
    df = df.merge(n_options_map(), on=["survey", "target_code"], how="left")
    df = df[df["n_options"] > 1]
    df["norm_acc"] = norm_acc(df["correct"], df["n_options"])

    # Fixed pair set: present at every level, so the curve is within-respondent.
    df["pair"] = df["survey"] + "|" + df["respondent_id"] + "|" + df["target_code"]
    per_level = {k: set(df[df["profile_type"] == tag]["pair"])
                 for k, tag in LEVELS.items()}
    print("pairs per level:", {k: len(v) for k, v in per_level.items()})

    # A level with no data is a shard that has not finished, not a level with
    # nothing in it. Intersecting over it would empty the pair set silently, so
    # it is dropped and named instead.
    levels = {k: tag for k, tag in LEVELS.items() if per_level[k]}
    missing = [k for k in LEVELS if not per_level[k]]
    if missing:
        print(f"NOT YET SCORED, excluded from the curve: k={missing}")
    common = set.intersection(*(per_level[k] for k in levels))
    print(f"common to the {len(levels)} available levels: {len(common):,}")
    if not common:
        raise SystemExit("no pairs common to every available level")
    df = df[df["pair"].isin(common)]

    # Per question then over questions, matching the main analysis.
    rows = []
    for k, tag in levels.items():
        g = df[df["profile_type"] == tag]
        per_q = g.groupby(["survey", "target_code"])["norm_acc"].mean()
        rows.append({"k": k, "norm_acc": per_q.mean(), "n_questions": len(per_q),
                     "n_instances": len(g)})
    curve = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Power law on the three original points, then on all five.
    def fit(sub):
        s, b = np.polyfit(np.log(sub["k"]), np.log(sub["norm_acc"]), 1)
        r2 = np.corrcoef(np.log(sub["k"]), np.log(sub["norm_acc"]))[0, 1] ** 2
        return s, np.exp(b), r2

    s3, a3, r3 = fit(curve[curve["k"] <= 24])
    s5, a5, r5 = fit(curve)
    curve["predicted_from_6_12_24"] = a3 * curve["k"] ** s3

    print()
    print(curve.round(4).to_string(index=False))
    print()
    print(f"fit on 6/12/24 : acc = {a3:.4f} * k^{s3:.3f}  (R2 {r3:.4f})")
    print(f"fit on all five: acc = {a5:.4f} * k^{s5:.3f}  (R2 {r5:.4f})")
    print(f"exponent change: {s5 - s3:+.3f}  "
          f"({'flattening' if s5 < s3 else 'holding or steepening'})")

    print()
    print("against predictions registered before the run:")
    for k, pred in PREREGISTERED.items():
        if k not in levels:
            print(f"  k={k:<3} not yet scored")
            continue
        obs = float(curve.loc[curve["k"] == k, "norm_acc"].iloc[0])
        print(f"  k={k:<3} observed {obs:.3f}   registered {pred:.3f}   "
              f"flat null {FLAT_NULL:.3f}   "
              f"shortfall vs registered {obs - pred:+.3f}")

    # Question-clustered bootstrap on the two new levels.
    print()
    print("95% CIs (question-clustered bootstrap, 2,000 resamples):")
    rng = np.random.default_rng(42)
    for k, tag in levels.items():
        g = df[df["profile_type"] == tag]
        per_q = g.groupby(["survey", "target_code"])["norm_acc"].mean().values
        boot = [rng.choice(per_q, len(per_q), replace=True).mean()
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  k={k:<3} {per_q.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

    # Does the extra information change the behaviour, or only the score?
    mech = mechanism(df, levels)
    print()
    print("mechanism: accuracy may rise, but does mode-following fall?")
    print(mech.round(4).to_string(index=False))
    base = mech[mech["k"] == 24].iloc[0]
    top = mech[mech["k"] == max(levels)].iloc[0]
    print(f"  24 -> {max(levels)}:  entropy ratio {top['entropy_ratio'] - base['entropy_ratio']:+.3f}"
          f"   mode rate {top['mode_rate'] - base['mode_rate']:+.3f}"
          f"   penalty {top['penalty_norm'] - base['penalty_norm']:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT / "scaling_curve.csv", index=False)
    mech.to_csv(OUT / "scaling_mechanism.csv", index=False)
    print(f"\nwrote {OUT / 'scaling_curve.csv'}")
    print(f"wrote {OUT / 'scaling_mechanism.csv'}")


if __name__ == "__main__":
    main()
