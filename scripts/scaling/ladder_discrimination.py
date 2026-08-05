#!/usr/bin/env python
r"""Does the model rank RESPONDENTS correctly even when it ranks OPTIONS wrongly?

Argmax accuracy asks which option wins. That question is contaminated by how
likely each option's wording is in the abstract: "I voted in the election" can
lose to "I did not vote" for every respondent simply because the phrase is less
probable in this context, regardless of who is being described. A constant
offset like that destroys accuracy while leaving person-level information
intact.

Discrimination asks a different question that the offset cannot touch. Fix one
option. Across respondents, is its score higher for the people who actually gave
it? That is an AUC, and it is invariant to adding any constant to that option's
score. So:

    accuracy low, AUC ~ 0.5   the model has no information about these people
    accuracy low, AUC > 0.5   it has information; the wording offset hides it

Scores are centred within respondent before comparison, which removes each
prompt's overall likelihood level and leaves only the relative standing of the
options.

The null is a permutation, not the k=0 level. The prompt is built from the
profile and the target question alone, so with an empty profile every respondent
for a target receives a byte-identical prompt and the AUC is undefined. It came
out far from 0.5 anyway, because identical prompts scored on different GPUs
differ by up to 0.22 nats (median 0.05, against a typical 3.5 nat gap between
options), so a k=0 "AUC" only measures which shard a respondent landed in.
Shuffling the answers across respondents keeps the scores and destroys the
pairing, which is the null this question actually needs.

    python .../ladder_discrimination.py
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
OUT = REPO.parent / "analysis" / "ladder"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; invariant to any constant added to `scores`."""
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so identical scores cannot manufacture separation.
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def collect(level: str) -> dict:
    """Records at one level, keyed by target then pair."""
    shard = {}
    for f in sorted((SCALE / "ladder_shards").glob("ladder_shard_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            shard[r["example_id"]] = (r["target_code"], r["survey"],
                                      f"{r['survey']}|{r['id']}|{r['target_code']}")
    out = collections.defaultdict(list)
    for f in sorted((SCALE / "ladder_results").glob("ladder_results_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            m = shard.get(r["example_id"])
            if m is None or r["profile_type"] != level:
                continue
            out[m[0]].append(r)
    return out


def _weighted_auc(cent: dict, truth: np.ndarray, opts: list) -> float:
    aucs, wts = [], []
    for o in opts:
        y = (truth == o).astype(int)
        a = auc(cent[o], y)
        if not np.isnan(a):
            aucs.append(a)
            wts.append(y.sum())
    if not aucs:
        return float("nan")
    return float(np.average(aucs, weights=wts))


def discrimination(recs: list, n_perm: int = 500,
                   seed: int = 42) -> tuple[float, float, float, int]:
    """Weighted AUC over one target's options, with a permutation null.

    Returns (auc, null 2.5th, null 97.5th, n). Permuting the answers across
    respondents holds the scores fixed and destroys only the pairing, so the
    band is what this estimator returns when the model knows nothing.
    """
    opts = sorted(recs[0]["option_logprobs"])
    keep = [r for r in recs if set(r["option_logprobs"]) == set(opts)]
    if not keep:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array([[r["option_logprobs"][o] for o in opts] for r in keep])
    cent_m = arr - arr.mean(axis=1, keepdims=True)
    cent = {o: cent_m[:, i] for i, o in enumerate(opts)}
    truth = np.array([r["ground_truth"] for r in keep])

    obs = _weighted_auc(cent, truth, opts)
    rng = np.random.default_rng(seed)
    null = np.array([_weighted_auc(cent, rng.permutation(truth), opts)
                     for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    lo, hi = np.percentile(null, [2.5, 97.5]) if len(null) else (np.nan, np.nan)
    return obs, float(lo), float(hi), len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="ladder_random_k096")
    args = ap.parse_args()

    full = collect(args.level)
    print(f"level {args.level}: {len(full)} targets\n")

    rows = []
    for t, recs in full.items():
        a, lo, hi, n = discrimination(recs)
        acc = np.mean([r["correct"] for r in recs])
        M = len(recs[0]["option_logprobs"])
        rows.append({"target_code": t, "n": n, "M": M, "acc": acc,
                     "norm_acc": (acc - 1 / M) / (1 - 1 / M),
                     "auc": a, "null_lo": lo, "null_hi": hi,
                     "above_null": a > hi})
    d = pd.DataFrame(rows).sort_values("norm_acc")

    print(f"{'target':<12}{'M':>3}{'norm acc':>10}{'AUC':>8}{'null 95%':>16}")
    for _, r in d.iterrows():
        flag = ""
        if r.norm_acc < 0.05 and r.above_null:
            flag = "  <- at or below chance on accuracy, yet discriminates"
        elif r.norm_acc > 0.3 and not r.above_null:
            flag = "  <- scores well without discriminating"
        print(f"{r.target_code:<12}{r.M:>3}{r.norm_acc:>10.3f}{r.auc:>8.3f}"
              f"   [{r.null_lo:.3f},{r.null_hi:.3f}]{flag}")

    print(f"\nmean AUC {d.auc.mean():.3f}; "
          f"{int(d.above_null.sum())}/{len(d)} targets above their permutation null")
    below = d[d.norm_acc < 0.05]
    print(f"targets at or below chance on accuracy: {len(below)}, "
          f"mean AUC {below.auc.mean():.3f}, "
          f"{int(below.above_null.sum())} above null")
    print("\nthe two failures are different: accuracy without discrimination is "
          "guessing the\nmode; discrimination without accuracy is a wording "
          "offset hiding real signal.")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "ladder_discrimination.csv", index=False)
    print(f"\nwrote {OUT / 'ladder_discrimination.csv'}")


if __name__ == "__main__":
    main()
