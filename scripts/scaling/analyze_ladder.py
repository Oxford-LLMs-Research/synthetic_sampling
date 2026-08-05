#!/usr/bin/env python
r"""How far does a profile move the model, and does the ORDER of arrival matter?

The scaling experiment measured four profile sizes on two serving stacks with a
generator defect on three surveys. This measures nine sizes from zero upward on
one stack, with every level a verified prefix of the next, so the trajectory can
be read directly. It also splits each trajectory three ways.

The three arms give the same respondent-target pair the same features in three
orders: the oracle's most predictive first, a seeded permutation, and the same
ranking read from the bottom. Because the feature SET is held fixed and only the
order varies, a difference between arms at a given k is attributable to which
features arrived, not how many. That is the one comparison the main experiment
cannot make, since it draws features at random by design.

Five questions, in order of how much they matter:

  1. Does the order of arrival matter? If the model is extracting the signal a
     feature carries, the informative arm should reach a given belief sooner
     than the random arm, and the anti arm later. If the three arms are
     indistinguishable, the profile is acting as a prompt-length or topical
     cue rather than as evidence about this person. Arms are compared as
     per-pair differences at the same k, which removes pair difficulty
     entirely, and the bootstrap clusters on the target question, since the
     paper's own error analysis found questions, not respondents, set the width.
  2. Does the profile move the model at all? k=0 is the same prompt with an
     empty profile slot, so KL(p_k || p_0) is the distance the profile content
     travels, and the share of instances whose argmax ever changes is the
     bluntest version of the same thing.
  3. Is the movement smooth, or does it arrive in steps? The scaling run showed
     an unexplained jump between 24 and 48 features. Levels 16, 32 and 64
     bracket that, on one stack.
  4. Where does it saturate? The all-features level is shared by the three arms
     by construction, so it is both the endpoint and a measurement-noise floor.
  5. Is the answer sensitive to how options are scored? option_n_tokens makes
     the unnormalised sum recoverable, and k=0 supplies a null for a contrastive
     score, so four rules with known and opposing biases can be compared:

       mean          the paper's rule, mean token logprob. Favours long options.
       sum           mean x tokens. Favours short options.
       contrastive   mean minus its own k=0 value, which cancels the option's
                     inherent surface likelihood, the thing driving the length
                     preference. Residual bias runs the other way and is smaller.
       contrastive-sum  the same on the sum scale.

     Agreement across all four is a robustness statement that does not require
     any of them to be correct in the abstract.

    python .../analyze_ladder.py
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
RESULTS = SCALE / "ladder_results"
SHARDS = SCALE / "ladder_shards"
OUT = REPO.parent / "analysis" / "ladder"
LEVELS = [0, 1, 2, 4, 8, 16, 32, 64, 96]
ARMS = ("informative", "random", "anti")
RULES = ("mean", "sum", "contrastive", "contrastive_sum")


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def boot_ci(values: np.ndarray, clusters: np.ndarray, n: int = 2000,
            seed: int = 42) -> tuple[float, float]:
    """Question-clustered bootstrap: resample target questions, not observations.

    Each draw averages within a question before averaging across questions, so
    the interval is built around the same estimator the tables report and a
    question with many respondents cannot dominate it.
    """
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    by = [values[inv == i] for i in range(len(keys))]
    means = np.array([b.mean() for b in by])
    draws = means[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def qmean(g: pd.DataFrame, col: str) -> float:
    """Average within target question, then across, matching the bootstrap."""
    return g.groupby(["survey", "target_code"])[col].mean().mean()


def load() -> pd.DataFrame:
    """One row per (pair, arm, level), with all four scoring rules resolved."""
    # Join keys come from the shard files, not from parsing example_id: target
    # codes contain underscores and that is exactly the corruption repair_ids.py
    # exists to undo elsewhere in this project.
    meta = {}
    for p in sorted(SHARDS.glob("ladder_shard_*.jsonl")):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                meta[d["example_id"]] = (d["survey"], d["target_code"], d["id"])
    if not meta:
        raise SystemExit(f"no shard inputs under {SHARDS}")
    print(f"{len(meta):,} prompts in the shard files")

    q = pd.read_csv(REPO.parent / "analysis" / "normalized_accuracy" /
                    "per_question_norm_acc.csv")
    nopt = q.groupby(["survey", "target_code"])["n_options"].first().to_dict()

    rows, dropped = [], collections.Counter()
    for p in sorted(RESULTS.glob("ladder_results_*.jsonl")):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                m = meta.get(r["example_id"])
                if m is None:
                    dropped["no shard record"] += 1
                    continue
                survey, target, rid = m
                lp = r.get("option_logprobs") or {}
                nt = r.get("option_n_tokens") or {}
                opts = [o for o in lp if np.isfinite(lp[o])]
                if len(opts) < 2:
                    dropped["fewer than two scored options"] += 1
                    continue
                mean = np.array([lp[o] for o in opts], float)
                toks = np.array([nt.get(o, 1) for o in opts], float)
                # profile_type is ladder_<arm>_k<level>; the shared k=0 and the
                # all-features endpoint both carry the arm name "all".
                arm = r["profile_type"].split("_")[1]
                rows.append({
                    "pair": f"{survey}|{rid}|{target}", "arm": arm,
                    "survey": survey, "target_code": target,
                    "k": int(r["n_features"]),
                    "opts": tuple(opts), "mean": mean, "sum": mean * toks,
                    "gt": r.get("ground_truth"),
                    "n_options": nopt.get((survey, target), len(set(opts))),
                })
    for k, v in dropped.items():
        print(f"  dropped, {k}: {v:,}")
    df = pd.DataFrame(rows)
    print(f"{len(df):,} scored | {df['pair'].nunique():,} pairs | "
          f"{df.groupby(['survey','target_code']).ngroups} targets | "
          f"arms {dict(df.arm.value_counts())}")
    return df


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    """Score every record against its own pair's k=0 null."""
    by = {(r["pair"], r["arm"], r["k"]): r for _, r in df.iterrows()}
    base = {p: by.get((p, "all", 0)) for p in df["pair"].unique()}

    recs, dropped = [], collections.Counter()
    for (pair, arm, k), r in by.items():
        b = base.get(pair)
        if b is None:
            dropped["no k=0 null"] += 1
            continue
        if b["opts"] != r["opts"]:
            dropped["option set differs from k=0"] += 1
            continue
        p_mean, p0 = softmax(r["mean"]), softmax(b["mean"])
        j = {o: i for i, o in enumerate(r["opts"])}.get(r["gt"])
        scores = {
            "mean": r["mean"],
            "sum": r["sum"],
            "contrastive": r["mean"] - b["mean"],
            "contrastive_sum": r["sum"] - b["sum"],
        }
        rec = {"pair": pair, "arm": arm, "survey": r["survey"],
               "target_code": r["target_code"], "k": k,
               "n_options": r["n_options"],
               "kl_from_0": float((p_mean * np.log(p_mean / p0)).sum()),
               "entropy": float(-(p_mean * np.log(p_mean)).sum()),
               "nll_true": float(-np.log(p_mean[j])) if j is not None else np.nan,
               "moved": r["opts"][int(np.argmax(r["mean"]))]
                        != b["opts"][int(np.argmax(b["mean"]))]}
        for name, s in scores.items():
            # At k=0 the contrastive scores are a level minus itself, so every
            # option is exactly zero and argmax just returns the first one.
            # That is tie-breaking, not a measurement, so it is not reported.
            if k == 0 and name.startswith("contrastive"):
                rec[f"correct_{name}"] = np.nan
                continue
            rec[f"correct_{name}"] = float(r["opts"][int(np.argmax(s))] == r["gt"])
            if j is not None:
                rec[f"nll_{name}"] = float(-np.log(softmax(s)[j]))
        recs.append(rec)
    for k, v in dropped.items():
        print(f"  dropped, {k}: {v:,}")
    return pd.DataFrame(recs)


def fixed_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Pairs observed at every arm and every level, so all curves compare like with like.

    Levels are skipped when a respondent has fewer usable features than the
    level asks for, so without this the k=96 row would be measured on the
    feature-rich respondents alone and the curve would confound richness with
    level.
    """
    want = {(a, k) for a in ARMS for k in LEVELS if k > 0} | {("all", 0)}
    have = d.groupby("pair").apply(
        lambda g: want <= set(zip(g.arm, g.k)), include_groups=False)
    keep = set(have[have].index)
    print(f"\nfixed pair set: {len(keep):,} of {d.pair.nunique():,} pairs "
          f"reach every level in every arm")
    return d[d.pair.isin(keep)].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    d = resolve(load())
    fx = fixed_pairs(d)

    # k=0 is a single shared measurement. Give each arm its own copy so every
    # trajectory starts from the same origin without scoring it three times.
    zero = fx[(fx.arm == "all") & (fx.k == 0)]
    lad = pd.concat([fx[fx.arm.isin(ARMS)]] +
                    [zero.assign(arm=a) for a in ARMS], ignore_index=True)

    def norm(col: str, g: pd.DataFrame) -> float:
        """Normalized accuracy, (acc - 1/M) / (1 - 1/M), question-averaged."""
        v = (g[col] - 1 / g["n_options"]) / (1 - 1 / g["n_options"])
        return qmean(g.assign(_v=v), "_v")

    print("\n=== 1. does the order of arrival matter? ===")
    print("NLL the model assigns the respondent's true answer; lower is better")
    print(f"{'k':>4}" + "".join(f"{a:>14}" for a in ARMS))
    curve = []
    for k in LEVELS:
        cells = []
        for a in ARMS:
            g = lad[(lad.arm == a) & (lad.k == k)]
            m = qmean(g, "nll_true")
            cells.append(f"{m:>14.4f}")
            curve.append({"arm": a, "k": k, "nll_true": m,
                          "norm_acc": norm("correct_mean", g),
                          "entropy": qmean(g, "entropy"),
                          "kl_from_0": qmean(g, "kl_from_0"),
                          "moved": qmean(g, "moved"),
                          "n_pairs": g.pair.nunique()})
        print(f"{k:>4}" + "".join(cells))

    print("\nsame, as normalized accuracy")
    print(f"{'k':>4}" + "".join(f"{a:>14}" for a in ARMS))
    for k in LEVELS:
        print(f"{k:>4}" + "".join(
            f"{norm('correct_mean', lad[(lad.arm == a) & (lad.k == k)]):>14.4f}"
            for a in ARMS))

    print("\npaired contrasts on NLL, same pair and level, question-clustered CI")
    print("negative favours the first arm: it assigns the true answer more mass")
    wide = lad.pivot_table(index=["pair", "survey", "target_code", "k"],
                           columns="arm", values="nll_true").reset_index()
    wide["q"] = wide.survey + "|" + wide.target_code
    contrasts = []
    for a, b in (("informative", "random"), ("anti", "random"),
                 ("informative", "anti")):
        print(f"\n  {a} - {b}")
        for k in LEVELS:
            if k == 0:
                continue
            g = wide[wide.k == k].dropna(subset=[a, b])
            if g.empty:
                continue
            dd = (g[a] - g[b]).to_numpy()
            m = pd.Series(dd).groupby(g.q.to_numpy()).mean().mean()
            lo, hi = boot_ci(dd, g.q.to_numpy(), args.boot)
            star = "*" if (lo > 0) or (hi < 0) else " "
            print(f"    k={k:>3}  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}] {star}")
            contrasts.append({"contrast": f"{a}-{b}", "k": k, "delta": m,
                              "lo": lo, "hi": hi, "n_pairs": len(g)})

    print("\n=== 2-4. the trajectory (random arm, the paper's own design) ===")
    print(f"{'k':>4}{'norm acc':>10}{'NLL true':>10}{'entropy':>9}"
          f"{'KL from 0':>11}{'argmax moved':>14}")
    tab = {}
    for k in LEVELS:
        g = lad[(lad.arm == "random") & (lad.k == k)]
        if g.empty:
            continue
        tab[k] = norm("correct_mean", g)
        print(f"{k:>4}{tab[k]:>10.4f}{qmean(g, 'nll_true'):>10.4f}"
              f"{qmean(g, 'entropy'):>9.4f}{qmean(g, 'kl_from_0'):>11.4f}"
              f"{qmean(g, 'moved'):>13.1%}")

    ks = [k for k in LEVELS if k in tab and k > 0]
    print("\n  change per doubling, from k=1:")
    for lo_k, hi_k in zip(ks, ks[1:]):
        e = np.log(max(tab[hi_k], 1e-9) / max(tab[lo_k], 1e-9)) / np.log(hi_k / lo_k)
        print(f"    {lo_k:>3} -> {hi_k:<3} norm acc {tab[hi_k]-tab[lo_k]:+.4f}   "
              f"local exponent {e:6.3f}")

    endpoint = fx[(fx.arm == "all") & (fx.k > 0)]
    if not endpoint.empty:
        print(f"\n  all-features endpoint, shared by the three arms: "
              f"norm acc {norm('correct_mean', endpoint):.4f}, "
              f"NLL {qmean(endpoint, 'nll_true'):.4f}, "
              f"median {int(endpoint.k.median())} features")

    print("\n=== 5. does the scoring rule change the answer? ===")
    print("normalized accuracy, random arm")
    print(f"{'k':>4}" + "".join(f"{r:>16}" for r in RULES))
    for k in LEVELS:
        g = lad[(lad.arm == "random") & (lad.k == k)]
        if g.empty:
            continue
        print(f"{k:>4}" + "".join(f"{norm('correct_' + r, g):>16.4f}" for r in RULES))

    print("\ninformative - random on NLL, under each rule")
    print(f"{'k':>4}" + "".join(f"{r:>16}" for r in RULES))
    robust = []
    for k in LEVELS:
        if k == 0:
            continue
        cells = []
        for rule in RULES:
            w = lad.pivot_table(index=["pair", "survey", "target_code", "k"],
                                columns="arm", values=f"nll_{rule}").reset_index()
            g = w[(w.k == k)].dropna(subset=["informative", "random"])
            if g.empty:
                cells.append(f"{'-':>16}")
                continue
            qq = (g.survey + "|" + g.target_code).to_numpy()
            dd = (g["informative"] - g["random"]).to_numpy()
            m = pd.Series(dd).groupby(qq).mean().mean()
            lo, hi = boot_ci(dd, qq, args.boot)
            cells.append(f"{m:+.4f}{'*' if (lo > 0 or hi < 0) else ' '}".rjust(16))
            robust.append({"k": k, "rule": rule, "delta": m, "lo": lo, "hi": hi})
        print(f"{k:>4}" + "".join(cells))

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "ladder_per_instance.csv", index=False)
    pd.DataFrame(curve).to_csv(OUT / "ladder_curve.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "ladder_contrasts.csv", index=False)
    pd.DataFrame(robust).to_csv(OUT / "ladder_robustness.csv", index=False)
    print(f"\nwrote ladder_per_instance.csv ({len(d):,} rows), ladder_curve.csv, "
          f"ladder_contrasts.csv and ladder_robustness.csv to {OUT}")


if __name__ == "__main__":
    main()
