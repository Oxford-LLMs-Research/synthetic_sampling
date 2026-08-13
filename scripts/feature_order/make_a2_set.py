"""A2: assemble the feature-order scoring set (RUN_CATALOGUE A2).

The question: does WHERE a feature sits in the profile change whether the
model uses it? Untested under a clean readout (the January order test was
echo-based). The ladder's `questions` dict is ordered by informative rank,
most informative first (prefix property verified 13 Aug: k001 is a prefix of
k008 is a prefix of k024 on all 1,489 complete pairs), so the ladder default
IS the informative-first cell.

Substrate: the same seeded sample build_narrative_tasks.py drew for B3
(SEED 42, 30 pairs per target, k=24 informative rung, duplicate-option
hygiene targets excluded): 734 pairs, 25 targets — identical to B3's
sample. B3's 11 narrative-validation exclusions are irrelevant here (no
prose in the loop), so all 734 pairs are used, not just the 723 survivors.

Cells, three instances per pair sharing base_id, scored in ONE serving:

  <eid>_ordfirst   informative_first  questions as the ladder orders them
  <eid>_ordlast    informative_last   the exact reverse
  <eid>_ordshuf    shuffled           a per-pair seeded permutation,
                                      re-drawn if it collides with either

Feature order is carried by dict insertion order end to end: the ladder set
preserves it, json round-trips it, and prompts.render_profile iterates it.

Usage:
  python scripts/feature_order/make_a2_set.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
DEFAULT_IN = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"
OUT_DIR = REPO / "outputs" / "feature_order" / "inputs"

SEED = 42          # build_narrative_tasks.py's seed: reproduces B3's sample
PAIRS_PER_TARGET = 30


def sample_pairs(ladder_set: Path, pairs_per_target: int) -> list[dict]:
    """The exact sampling walk of build_narrative_tasks.py (B3 stage 1)."""
    by_target: dict[tuple, list[dict]] = defaultdict(list)
    with open(ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["example_id"].endswith("_ladder_informative_k024"):
                continue
            opts = r["option_sets"]["original"]
            if len(set(opts)) != len(opts):
                continue  # duplicate-option hygiene targets
            by_target[(r["survey"], r["target_code"])].append(r)

    rng = random.Random(SEED)
    picked: list[dict] = []
    for key in sorted(by_target):
        rows = sorted(by_target[key], key=lambda r: r["example_id"])
        take = rng.sample(rows, min(pairs_per_target, len(rows)))
        picked.extend(sorted(take, key=lambda r: r["example_id"]))
    return picked


def order_cells(questions: dict, eid: str) -> dict[str, dict]:
    """The three orderings of one pair's feature dict."""
    items = list(questions.items())
    if len(items) < 3:
        # With <3 features no permutation differs from both first and
        # last; a k=24 substrate can never hit this, so it is malformed.
        raise ValueError(f"{eid}: {len(items)} features, need >= 3")
    first = dict(items)
    last = dict(reversed(items))
    rng = random.Random(f"{SEED}:{eid}")
    shuf = list(items)
    while True:
        rng.shuffle(shuf)
        if shuf != items and shuf != list(reversed(items)):
            break
    return {"ordfirst": first, "ordlast": last, "ordshuf": dict(shuf)}


ARM_LABELS = {"ordfirst": "informative_first", "ordlast": "informative_last",
              "ordshuf": "shuffled"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladder-set", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "a2_order_set.jsonl")
    ap.add_argument("--pairs-per-target", type=int, default=PAIRS_PER_TARGET)
    args = ap.parse_args(argv)

    picked = sample_pairs(args.ladder_set, args.pairs_per_target)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    targets = set()
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for src in picked:
            eid = src["example_id"]
            targets.add((src["survey"], src["target_code"]))
            common = {
                "base_id": eid,
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "option_sets": src["option_sets"],
                "ground_truth": src["ground_truth"],
                "ground_truth_index": src["ground_truth_index"],
            }
            # The three cells stay adjacent so LIMIT (canary) keeps whole
            # triples and the awk-every-50th smoke spans all cells.
            for cell, qs in order_cells(src["questions"], eid).items():
                fh.write(json.dumps({
                    "example_id": f"{eid}_{cell}",
                    "arm_label": ARM_LABELS[cell],
                    "questions": qs, **common}, ensure_ascii=False) + "\n")
                n += 1

    manifest = {
        "seed": SEED,
        "substrate": "ladder_readout k024 informative, B3 sampling walk",
        "pairs_per_target": args.pairs_per_target,
        "n_pairs": len(picked), "n_targets": len(targets),
        "n_instances": n, "cells": ARM_LABELS,
    }
    (args.out.parent / "a2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(picked)} pairs x 3 cells = {n} instances, "
          f"{len(targets)} targets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
