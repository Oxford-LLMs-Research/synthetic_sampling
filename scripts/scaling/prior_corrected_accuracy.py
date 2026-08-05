#!/usr/bin/env python
r"""How much of the paper's gap is information, and how much is calibration?

The paper compares an LLM's argmax against a gradient-boosted classifier and
against predicting each question's modal answer. That comparison is not
symmetric in one respect. XGBoost is FITTED on respondents from the same
population, so it knows each answer's base rate for free. The language model
knows no such thing: its implicit prior over the options is how likely each
option's WORDING is, which has no reason to match how often people choose it.

So the measured gap mixes two things:

    information   does the model know anything about this person?
    calibration   does it know how often people give each answer?

This script gives the model the second for free and re-measures. Within each
question, respondents are split into folds; on the training fold the empirical
log frequency of each answer is computed, and a single scalar temperature is
fitted; on the held-out fold the prediction is

    argmax_o  [ alpha * centred_llm_score(o) + log p_train(o) ]

Centring removes each respondent's overall prompt likelihood, so what alpha
scales is only the model's relative preference among options. Three quantities
are then comparable on identical held-out respondents:

    raw          the paper's metric, argmax of the raw scores
    prior only   alpha = 0, which is exactly the majority baseline
    corrected    the fitted alpha

If corrected beats prior-only, the model carries information beyond base rates
and the paper's headline understates it. If it does not, the paper's conclusion
stands as measured and the wording offset is not hiding anything usable.

    python .../prior_corrected_accuracy.py --model qwen3-32b --level s6m4
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"
ANALYSIS = REPO.parent / "analysis"
OUT = ANALYSIS / "ladder"
ALPHAS = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 24)])


def target_matcher(survey: str):
    """Resolve target_code without splitting example_id on underscores."""
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv")
    codes = sorted(q[q.survey == survey].target_code.unique(), key=len, reverse=True)

    def match(eid: str, level: str):
        if not eid.endswith("_" + level):
            return None
        body = eid[:-(len(level) + 1)]
        for c in codes:
            if body.endswith("_" + c):
                return c
        return None
    return match


def evaluate(arr: np.ndarray, y: np.ndarray, opts: list, folds: int,
             seed: int = 42) -> dict:
    """Cross-fitted accuracy under the three rules, on identical held-out rows."""
    n = len(y)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % folds
    cent = arr - arr.mean(axis=1, keepdims=True)
    idx = {o: i for i, o in enumerate(opts)}
    yi = np.array([idx[v] for v in y])

    raw_hit, prior_hit, corr_hit, alphas = [], [], [], []
    for f in range(folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # Base rates from the training respondents only.
        cnt = np.bincount(yi[tr], minlength=len(opts)).astype(float)
        logp = np.log((cnt + 0.5) / (cnt.sum() + 0.5 * len(opts)))
        # One temperature, chosen on the training fold by log-likelihood.
        best, best_ll = 0.0, -np.inf
        for a in ALPHAS:
            s = a * cent[tr] + logp
            s = s - s.max(axis=1, keepdims=True)
            ll = (s[np.arange(tr.sum()), yi[tr]]
                  - np.log(np.exp(s).sum(axis=1))).mean()
            if ll > best_ll:
                best_ll, best = ll, a
        alphas.append(best)
        raw_hit.append((arr[te].argmax(axis=1) == yi[te]).mean())
        prior_hit.append((np.full(te.sum(), logp.argmax()) == yi[te]).mean())
        corr_hit.append(((best * cent[te] + logp).argmax(axis=1) == yi[te]).mean())
    if not raw_hit:
        return {}
    M = len(set(y))
    norm = lambda a: (a - 1 / M) / (1 - 1 / M)
    return {"M": M, "n": n, "alpha": float(np.mean(alphas)),
            "raw": norm(float(np.mean(raw_hit))),
            "prior_only": norm(float(np.mean(prior_hit))),
            "corrected": norm(float(np.mean(corr_hit)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--level", default="s6m4")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    mdir = RESULTS / args.model
    rows = []
    for f in sorted(mdir.glob("*.jsonl")):
        # The file prefix is the checkpoint name, which is NOT always the
        # directory name: llama3.1-8b-base/ holds llama3.1-8b_survey_*.jsonl,
        # gemma-3-27b-instruct/ holds gemma-3-27b-it_*. Stripping the directory
        # name left survey unparsable and silently produced zero questions.
        m = re.search(r"_survey_(.+)_results$", f.stem)
        if m is None:
            print(f"  skipping unparsable filename {f.name}")
            continue
        survey = m.group(1)
        match = target_matcher(survey)
        byq = collections.defaultdict(list)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            t = match(r["example_id"], args.level)
            if t is not None:
                byq[t].append(r)
        for t, recs in byq.items():
            opts = sorted(recs[0]["option_logprobs"])
            keep = [r for r in recs if set(r["option_logprobs"]) == set(opts)
                    and r["ground_truth"] in opts]
            if len(keep) < 40 or len({r["ground_truth"] for r in keep}) < 2:
                continue
            arr = np.array([[r["option_logprobs"][o] for o in opts] for r in keep])
            y = np.array([r["ground_truth"] for r in keep])
            out = evaluate(arr, y, opts, args.folds)
            if out:
                rows.append({"survey": survey, "target_code": t, **out})
        print(f"  {survey}: {len(byq)} questions", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n=== {args.model}, level {args.level}, {len(d)} questions, "
          f"{d.n.sum():,} respondents ===")
    print("normalized accuracy, question-averaged, cross-fitted on held-out "
          "respondents\n")
    print(f"  raw argmax (the paper's metric)        {d.raw.mean():>8.4f}")
    print(f"  base rates only (majority baseline)    {d.prior_only.mean():>8.4f}")
    print(f"  base rates + model, fitted temperature {d.corrected.mean():>8.4f}")
    gain = d.corrected - d.prior_only
    # Question-clustered interval on the gain over base rates alone.
    rng = np.random.default_rng(42)
    g = gain.to_numpy()
    draws = g[rng.integers(0, len(g), size=(4000, len(g)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\n  gain of the model over base rates alone {gain.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"  questions where the model adds anything: "
          f"{(gain > 0).sum()}/{len(d)}")
    print(f"  mean fitted temperature                 {d.alpha.mean():.3f} "
          f"(0 would mean the model is ignored)")
    print(f"  questions where the fit sets alpha = 0  {(d.alpha == 0).sum()}/{len(d)}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"prior_corrected_{args.model}_{args.level}.csv"
    d.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
