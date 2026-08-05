#!/usr/bin/env python
r"""Does the paper's headline survive an offset-free metric?

The paper scores an option by the mean log-probability of its own tokens and
takes the argmax. That ranking is contaminated by how likely each option's
WORDING is, independent of the respondent: a phrase can lose for everyone
because it is unusual English in this context, not because the model believes it
is false of these people. A constant per-option offset like that can destroy
argmax accuracy while leaving person-level information intact.

Discrimination removes it by construction. Centre each respondent's option
scores, fix one option, and ask across respondents whether it scores higher for
the people who actually chose it. Adding any constant to that option cannot
change the answer, so this measures only what the model knows about individuals.

    accuracy low, AUC ~ 0.5   the model has no information about these people
    accuracy low, AUC > 0.5   it has information the argmax throws away

The null is a permutation of answers across respondents within a question, which
holds the scores fixed and destroys only the pairing.

    python .../main_grid_discrimination.py --model qwen3-32b --level s6m4
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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def weighted_auc(cent: np.ndarray, truth: np.ndarray, opts: list) -> float:
    aucs, wts = [], []
    for i, o in enumerate(opts):
        y = (truth == o).astype(int)
        a = auc(cent[:, i], y)
        if not np.isnan(a):
            aucs.append(a)
            wts.append(y.sum())
    if not aucs:
        return float("nan")
    return float(np.average(aucs, weights=wts))


def target_matcher(survey: str) -> callable:
    """Resolve target_code from example_id without splitting on underscores.

    23 Arab Barometer target codes contain an underscore, which is exactly the
    corruption repair_ids.py exists to undo. Matching against the known code
    list, longest first, cannot make that mistake.
    """
    q = pd.read_csv(ANALYSIS / "normalized_accuracy" / "per_question_norm_acc.csv")
    codes = sorted(q[q.survey == survey].target_code.unique(), key=len, reverse=True)

    def match(example_id: str, level: str) -> str | None:
        body = example_id[:-(len(level) + 1)] if example_id.endswith("_" + level) else None
        if body is None:
            return None
        for c in codes:
            if body.endswith("_" + c):
                return c
        return None
    return match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--level", default="s6m4")
    ap.add_argument("--perm", type=int, default=300)
    args = ap.parse_args()

    mdir = RESULTS / args.model
    if not mdir.exists():
        raise SystemExit(f"no results at {mdir}")

    rows = []
    for f in sorted(mdir.glob("*.jsonl")):
        # The file prefix is the checkpoint name, not always the directory
        # name (llama3.1-8b-base/ holds llama3.1-8b_survey_*.jsonl).
        m = re.search(r"_survey_(.+)_results$", f.stem)
        if m is None:
            continue
        survey = m.group(1)
        match = target_matcher(survey)
        byq: dict[str, list] = collections.defaultdict(list)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            t = match(r["example_id"], args.level)
            if t is None:
                continue
            byq[t].append(r)
        rng = np.random.default_rng(42)
        for t, recs in byq.items():
            opts = sorted(recs[0]["option_logprobs"])
            keep = [r for r in recs if set(r["option_logprobs"]) == set(opts)]
            if len(keep) < 20:
                continue
            arr = np.array([[r["option_logprobs"][o] for o in opts] for r in keep])
            cent = arr - arr.mean(axis=1, keepdims=True)
            truth = np.array([r["ground_truth"] for r in keep])
            # M is the number of DISTINCT answer labels given, as in the paper.
            # A question where every respondent gave the same answer has M=1:
            # normalized accuracy is undefined and AUC has no negative class.
            M = len(set(truth))
            if M < 2:
                continue
            obs = weighted_auc(cent, truth, opts)
            null = np.array([weighted_auc(cent, rng.permutation(truth), opts)
                             for _ in range(args.perm)])
            null = null[~np.isnan(null)]
            acc = float(np.mean([r["correct"] for r in keep]))
            rows.append({
                "survey": survey, "target_code": t, "n": len(keep), "M": M,
                "acc": acc, "norm_acc": (acc - 1 / M) / (1 - 1 / M),
                "auc": obs,
                "null_hi": float(np.percentile(null, 97.5)) if len(null) else np.nan,
                "mode_acc": float(pd.Series(truth).value_counts(normalize=True).max()),
            })
        print(f"  {survey}: {len(byq)} questions", flush=True)

    d = pd.DataFrame(rows)
    d["above_null"] = d.auc > d.null_hi
    d["norm_mode"] = (d.mode_acc - 1 / d.M) / (1 - 1 / d.M)

    print(f"\n=== {args.model}, level {args.level}, "
          f"{len(d)} questions, {d.n.sum():,} instances ===")
    print(f"normalized accuracy (question-averaged) : {d.norm_acc.mean():.4f}")
    print(f"majority baseline, same questions       : {d.norm_mode.mean():.4f}")
    print(f"mean discrimination AUC                 : {d.auc.mean():.4f}")
    print(f"questions discriminating above null     : "
          f"{int(d.above_null.sum())}/{len(d)}  ({d.above_null.mean():.0%})")
    below = d[d.norm_acc <= 0]
    print(f"questions at or below chance on accuracy: {len(below)}, "
          f"of which {int(below.above_null.sum())} still discriminate")
    print(f"correlation, norm acc vs AUC (Spearman) : "
          f"{d[['norm_acc','auc']].corr(method='spearman').iloc[0,1]:+.3f}")

    print("\nworst 10 questions by accuracy, with their AUC:")
    print(f"{'survey':<16}{'target':<12}{'norm acc':>10}{'AUC':>8}{'null':>8}")
    for _, r in d.nsmallest(10, "norm_acc").iterrows():
        print(f"{r.survey:<16}{r.target_code:<12}{r.norm_acc:>10.3f}"
              f"{r.auc:>8.3f}{r.null_hi:>8.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"main_grid_discrimination_{args.model}_{args.level}.csv"
    d.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
