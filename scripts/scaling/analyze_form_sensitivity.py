#!/usr/bin/env python
r"""Is the paper's scoring rule measuring meaning, or wording?

Three manipulations were prepared and scored earlier and never analysed. Each
holds the respondent, the profile and the question fixed and changes only how
the answer options are presented, so any change in the prediction is a property
of the measurement rather than of the person.

  surface form   the options are paraphrased, preserving meaning
                 ("A great deal" -> "A lot"). A model reading MEANING answers
                 the same; a model reading WORDING does not. This is the direct
                 test of whether echo scoring measures what it is meant to.

  options shown  the paper's prompt never shows the option set; it asks an open
                 question and scores candidate continuations. Here the options
                 are listed. This separates "sees the choices" from "how the
                 answer is read", which any echo-versus-labels comparison
                 otherwise confounds.

  option order   natural versus reversed listing, once the options are shown.
                 Echo scoring is order-invariant when the options are hidden,
                 which is the paper's stated reason for choosing it, but that
                 guarantee lapses as soon as they appear in the prompt.

Agreement is computed on the option INDEX, so a paraphrased option counts as the
same answer as its original. Accuracy is normalized as in the paper, and both
are averaged within question before across, matching the paper's estimator.

    python .../analyze_form_sensitivity.py
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
OUT = REPO.parent / "analysis" / "ladder"
MODELS = ["llama3.1-70b-instruct", "llama3.1-8b-instruct",
          "olmo3-32b-dpo", "olmo3-7b-dpo"]


def load_tests(name: str) -> dict:
    out = {}
    with open(TESTS / f"{name}_test.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["example_id"]] = r
    return out


def qmean(rows: list[dict], col: str) -> float:
    d = pd.DataFrame(rows)
    if d.empty or col not in d:
        return float("nan")
    return d.groupby(["survey", "target_code"])[col].mean().mean()


def boot(rows: list[dict], col: str, n: int = 2000, seed: int = 42) -> tuple:
    d = pd.DataFrame(rows)
    if d.empty:
        return (np.nan, np.nan)
    per_q = d.groupby(["survey", "target_code"])[col].mean().to_numpy()
    if len(per_q) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = per_q[rng.integers(0, len(per_q), size=(n, len(per_q)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    argparse.ArgumentParser().parse_args()
    sf_tests = load_tests("surface_form")
    oc_tests = load_tests("options_context")

    summary = []
    for model in MODELS:
        # ---- surface form: paraphrase the options, keep the meaning ----------
        rows = []
        p = PERP / f"{model}_surface_form_improved_results.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                t = sf_tests.get(r["example_id"])
                if t is None:
                    continue
                res = r["results"]
                if not {"original", "synonym"} <= set(res):
                    continue
                a, b = res["original"], res["synonym"]
                # The synonym set is index-aligned with the original, so a
                # meaning-preserving model must return the same INDEX.
                if a.get("predicted_index") is None or b.get("predicted_index") is None:
                    continue
                gt, M = t["ground_truth_index"], len(t["option_sets"]["original"])
                rows.append({
                    "survey": t["survey"], "target_code": t["target_code"],
                    "agree": float(a["predicted_index"] == b["predicted_index"]),
                    "acc_orig": float(a["predicted_index"] == gt),
                    "acc_syn": float(b["predicted_index"] == gt),
                    "M": M})
        if rows:
            M = np.mean([r["M"] for r in rows])
            norm = lambda v: (v - 1 / M) / (1 - 1 / M)
            lo, hi = boot(rows, "agree")
            summary.append({
                "model": model, "test": "paraphrase options", "n": len(rows),
                "agreement": qmean(rows, "agree"), "lo": lo, "hi": hi,
                "acc_a": norm(qmean(rows, "acc_orig")),
                "acc_b": norm(qmean(rows, "acc_syn"))})

        # ---- options shown, natural versus reversed order --------------------
        rows = []
        p = PERP / f"{model}_options_context_results.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                t = oc_tests.get(r["example_id"])
                if t is None:
                    continue
                res = r["results"]
                if not {"shown_natural", "shown_reversed"} <= set(res):
                    continue
                a, b = res["shown_natural"], res["shown_reversed"]
                if a.get("predicted") is None or b.get("predicted") is None:
                    continue
                gt = t["ground_truth"]
                rows.append({
                    "survey": t["survey"], "target_code": t["target_code"],
                    "agree": float(a["predicted"] == b["predicted"]),
                    "acc_a": float(a["predicted"] == gt),
                    "acc_b": float(b["predicted"] == gt),
                    "M": t["n_options"]})
        if rows:
            M = np.mean([r["M"] for r in rows])
            norm = lambda v: (v - 1 / M) / (1 - 1 / M)
            lo, hi = boot(rows, "agree")
            summary.append({
                "model": model, "test": "reverse option order", "n": len(rows),
                "agreement": qmean(rows, "agree"), "lo": lo, "hi": hi,
                "acc_a": norm(qmean(rows, "acc_a")),
                "acc_b": norm(qmean(rows, "acc_b"))})

    d = pd.DataFrame(summary)
    print("Same respondent, same question, same meaning. Only the presentation "
          "changes.\n")
    print(f"{'model':<24}{'manipulation':<22}{'n':>6}{'same answer':>14}"
          f"{'95% CI':>18}{'acc before':>12}{'acc after':>11}")
    for _, r in d.iterrows():
        print(f"{r.model:<24}{r.test:<22}{r.n:>6}{r.agreement:>13.1%}"
              f"   [{r.lo:.3f},{r.hi:.3f}]{r.acc_a:>12.3f}{r.acc_b:>11.3f}")

    for test in d.test.unique():
        s = d[d.test == test]
        print(f"\n  {test}: mean agreement {s.agreement.mean():.1%}, "
              f"mean accuracy {s.acc_a.mean():.3f} -> {s.acc_b.mean():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "form_sensitivity.csv", index=False)
    print(f"\nwrote {OUT / 'form_sensitivity.csv'}")


if __name__ == "__main__":
    main()
