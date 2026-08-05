#!/usr/bin/env python
r"""Test 2 (options hidden vs shown), per question rather than pooled.

The supplement concludes that showing the answer options "changes WHICH option
the model selects far more than it changes HOW OFTEN the selection is right",
on the grounds that mean accuracy moves by at most 1.2 points between
conditions. That mean is taken over questions. A mean can be flat while
individual questions move a great deal in opposite directions, and the readout
experiment found exactly that on one question outside this sample: Afrobarometer
Q13 goes from 0.10 accuracy with options hidden to 0.63 with the same scoring
rule and the options shown.

So this recomputes the same comparison without pooling. The hidden condition is
not in the stored options-context results, which hold only the shown ones, so it
comes from the main run joined on base_id plus profile_type.

    python .../test2_per_question.py
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
TESTS = REPO / "outputs"
PERP = REPO.parent / "perplexity_test" / "results" / "results"
BASE = REPO.parent / "perplexity_test" / "base_run" / "base_run"
RESULTS = REPO.parent / "results"
OUT = REPO.parent / "analysis" / "readout"

MODELS = ["llama3.1-70b-instruct", "llama3.1-8b-instruct",
          "olmo3-32b-dpo", "olmo3-7b-dpo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=8,
                    help="questions with fewer instances than this are reported "
                         "but excluded from the movement summary")
    args = ap.parse_args()

    tests = {}
    for line in open(TESTS / "options_context_test.jsonl", encoding="utf-8"):
        r = json.loads(line)
        tests[r["example_id"]] = r
    print(f"{len(tests):,} options-context instances")

    rows = []
    for model in MODELS:
        # The stored options-context results carry only the shown conditions.
        # Hidden lives in perplexity_test/base_run, which scored the validation
        # samples themselves; the main run covers only the 40 questions the
        # validation pool shares with the 267 targets, which is why joining to
        # it recovered 108 of 2,035 instances and 14 questions.
        hidden = {}
        base = BASE / f"{model}_survey_results_base.jsonl"
        if base.exists():
            for line in open(base, encoding="utf-8"):
                r = json.loads(line)
                hidden[r["example_id"]] = r
        for f in sorted((RESULTS / model).glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                hidden.setdefault(r["example_id"], r)
        p = PERP / f"{model}_options_context_results.jsonl"
        if not p.exists():
            print(f"  {model}: no options-context results")
            continue
        miss = 0
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            t = tests.get(r["example_id"])
            if t is None:
                continue
            key = f"{t['base_id']}_{t['profile_type']}"
            h = hidden.get(key)
            shown = (r.get("results") or {}).get("shown_natural")
            if h is None or shown is None or shown.get("predicted") is None:
                miss += 1
                continue
            gt = t["ground_truth"]
            rows.append({
                "model": model, "survey": t["survey"],
                "target_code": t["target_code"],
                "q": f"{t['survey']}|{t['target_code']}",
                "n_options": t.get("n_options") or len(t.get("options") or []),
                "hidden": float(h["predicted"] == gt),
                "shown": float(shown["predicted"] == gt),
                "same": float(h["predicted"] == shown["predicted"]),
            })
        print(f"  {model}: joined {sum(1 for x in rows if x['model'] == model):,}"
              f", unmatched {miss}")

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("nothing joined")

    print(f"\npooled, as the supplement reports it")
    print(f"  hidden {d.hidden.mean():.4f}   shown {d.shown.mean():.4f}   "
          f"difference {d.shown.mean() - d.hidden.mean():+.4f}")
    print(f"  predictions unchanged on {d.same.mean():.1%} of instances")

    # Per question, averaged over the four models so one model cannot drive it.
    # Group on the question alone. An earlier version grouped on (q, n_options)
    # and called the result "questions", which split the 25 questions whose
    # option set varies by respondent (occupation, language at home) into one
    # group per option count and inflated 282 questions into 329. Both columns
    # here are raw accuracy, so nothing needs n_options held constant.
    per = (d.groupby("q")[["hidden", "shown", "same"]].mean()
           .join(d.groupby("q").size().rename("n"))
           .join(d.groupby("q").n_options.median().rename("n_options"))
           .reset_index())
    per["delta"] = per.shown - per.hidden
    big = per[per.n >= args.min_n]
    print(f"\nper question ({len(per)} questions, {len(big)} with n >= {args.min_n})")
    print(f"  mean of the per-question differences {big.delta.mean():+.4f}")
    print(f"  MEAN ABSOLUTE difference             {big.delta.abs().mean():.4f}")
    print(f"  questions improving by >0.10         {(big.delta > 0.10).sum()}")
    print(f"  questions worsening by >0.10         {(big.delta < -0.10).sum()}")
    print(f"  questions moving by >0.20 either way {(big.delta.abs() > 0.20).sum()}")
    print(f"  share of questions moving <0.05      {(big.delta.abs() < 0.05).mean():.0%}")

    print(f"\nlargest gains from showing the options")
    print(f"{'question':<28}{'n':>5}{'hidden':>9}{'shown':>8}{'delta':>9}")
    for _, r in big.nlargest(8, "delta").iterrows():
        print(f"{r.q:<28}{int(r.n):>5}{r.hidden:>9.3f}{r.shown:>8.3f}{r.delta:>+9.3f}")
    print(f"\nlargest losses")
    for _, r in big.nsmallest(8, "delta").iterrows():
        print(f"{r.q:<28}{int(r.n):>5}{r.hidden:>9.3f}{r.shown:>8.3f}{r.delta:>+9.3f}")

    # The question the pooled mean answers is whether gains and losses cancel.
    pos = big[big.delta > 0].delta.sum()
    neg = big[big.delta < 0].delta.sum()
    print(f"\ncancellation: gains total {pos:+.2f}, losses total {neg:+.2f}, "
          f"net {pos + neg:+.2f}")
    print("if the totals are large and the net is small, the pooled mean is "
          "hiding movement\nrather than showing stability")

    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "test2_per_question.csv", index=False)
    print(f"\nwrote {OUT / 'test2_per_question.csv'}")


if __name__ == "__main__":
    main()
