#!/usr/bin/env python
r"""Score the blind round-trip validation of the paraphrase sets.

A different model than the generator receives, per set, the original options
and the paraphrases, each independently shuffled, with no alignment
information, and must return a bijective mapping. A set passes when the
mapping is a bijection AND matches the true alignment for every option. A set
that fails goes back for regeneration; a set failing twice is excluded at the
question level (a broken option breaks the whole set's alignment).

Inputs (outputs/scaling_experiment/paraphrase/):
  roundtrip_input.json    what the validator saw
  roundtrip_key.json      the true permutations (never shown to the validator)
  roundtrip_output.json   the validator's mappings:
                          [{"set_id": ..., "mapping": [j_for_a0, j_for_a1, ...]}]
                          mapping[i] = position in list_b of list_a[i]'s partner

Output: roundtrip_results.csv (per set: pass, n_correct, n_options)

    python score_roundtrip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PARA = REPO / "outputs" / "scaling_experiment" / "paraphrase"


def main() -> None:
    key = json.load(open(PARA / "roundtrip_key.json", encoding="utf-8"))
    outs = json.load(open(PARA / "roundtrip_output.json", encoding="utf-8"))

    rows = []
    for r in outs:
        sid = r["set_id"]
        k = key.get(sid)
        if k is None:
            rows.append({"set_id": sid, "pass": False, "reason": "unknown set",
                         "n_correct": 0, "n_options": 0})
            continue
        pa, pb = k["perm_a"], k["perm_b"]
        mapping = r["mapping"]
        n = len(pa)
        bij = (len(mapping) == n and sorted(mapping) == list(range(n)))
        # list_a position i holds original index pa[i]; its true partner sits
        # at the list_b position j with pb[j] == pa[i].
        truth = {i: pb.index(pa[i]) for i in range(n)}
        correct = sum(1 for i in range(min(len(mapping), n))
                      if mapping[i] == truth[i])
        rows.append({"set_id": sid, "pass": bool(bij and correct == n),
                     "reason": "" if bij else "not a bijection",
                     "n_correct": correct, "n_options": n})

    d = pd.DataFrame(rows).sort_values("set_id")
    d.to_csv(PARA / "roundtrip_results.csv", index=False)
    n_pass = int(d["pass"].sum())
    print(f"{n_pass}/{len(d)} sets pass the blind round-trip")
    fails = d[~d["pass"]]
    if len(fails):
        print("\nfailed sets (regenerate, then re-validate; exclude at the "
              "question level on a second failure):")
        for _, r in fails.iterrows():
            print(f"  {r.set_id}: {r.n_correct}/{r.n_options} correct "
                  f"{('(' + r.reason + ')') if r.reason else ''}")
        sys.exit(1)
    print("all sets validated")


if __name__ == "__main__":
    main()
