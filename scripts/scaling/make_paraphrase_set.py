#!/usr/bin/env python
r"""Merge the validated paraphrase sets into the readout instances.

Each instance gains option_sets["paraphrase"], aligned index-for-index with
its own original option list, so the scorer's per-set predicted_index is
directly comparable across the two sets. Instances whose (question, option
list) belongs to an excluded set are dropped and counted: set041's question
(source typo duplicate) and set001's variant (residual missingness code 94.0)
as of 5 Aug; the authoritative list is paraphrase_exclusions.json.

The scorer silently drops any instance whose option sets differ in length
(score_formats.py line ~349). That must never happen here, so this script
asserts equal lengths for every instance it writes and exits non-zero if the
written count plus the excluded count does not reconcile to the input count.

Output: outputs/scaling_experiment/paraphrase_set.jsonl

    python make_paraphrase_set.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
PARA = SCALE / "paraphrase"


def main() -> None:
    inp = json.load(open(PARA / "paraphrase_input.json", encoding="utf-8"))
    raw = {r["set_id"]: r for r in
           json.load(open(PARA / "paraphrase_raw.json", encoding="utf-8"))}
    excl = json.load(open(PARA / "paraphrase_exclusions.json", encoding="utf-8"))
    excluded_ids = {e["set_id"] for e in excl}
    # question-level exclusions remove every variant of that question
    q_excluded = {e["question"] for e in excl if e.get("level") != "variant"}

    by_key = {(s["question"], tuple(s["options"])): s["set_id"] for s in inp}

    n_in = n_out = 0
    dropped = collections.Counter()
    out_path = SCALE / "paraphrase_set.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for line in open(SCALE / "readout_set.jsonl", encoding="utf-8"):
            n_in += 1
            r = json.loads(line)
            opts = r["option_sets"]["original"]
            sid = by_key.get((r["target_question"], tuple(opts)))
            if sid is None:
                dropped["no set_id (should be impossible)"] += 1
                continue
            if sid in excluded_ids or r["target_question"] in q_excluded:
                dropped[f"excluded {sid}"] += 1
                continue
            para = raw[sid]["paraphrases"]
            assert len(para) == len(opts), f"{sid}: length mismatch"
            r["option_sets"]["paraphrase"] = list(para)
            r["paraphrase_set_id"] = sid
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"{n_in} instances read, {n_out} written to {out_path.name}")
    for k, v in dropped.most_common():
        print(f"  dropped {v}: {k}")
    if n_out + sum(dropped.values()) != n_in:
        sys.exit("reconciliation failed")
    if dropped.get("no set_id (should be impossible)"):
        sys.exit("instances failed the set join; investigate before scoring")
    print("reconciled; every written instance has equal-length option sets")


if __name__ == "__main__":
    main()
