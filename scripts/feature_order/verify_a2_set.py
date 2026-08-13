"""Pin the A2 feature-order set before it is staged (mirror of B3's
verify_narrative_set.py). Exit 0 or the set does not ship.

Checks:
  1. 734 pairs x 3 cells = 2,202 rows, 25 targets, cells adjacent per pair.
  2. Per pair: the three cells carry the same feature SET with the same
     answers; ordfirst matches the ladder source order exactly; ordlast is
     its exact reverse; ordshuf differs from both.
  3. Common fields (options, truth, target question, ...) identical across
     a pair's three rows and equal to the ladder source row.
  4. No duplicate-option target present; every row's ground_truth resolves
     to a real option index.
  5. build_prompt renders the intended order: for a sample of pairs, the
     first feature of each cell's dict is the first "Q:" line in the
     rendered label_num prompt, and orders differ across cells.

Usage:
  python scripts/feature_order/verify_a2_set.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from synthetic_sampling.scoring.prompts import build_prompt  # noqa: E402

LADDER = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"
SET = REPO / "outputs" / "feature_order" / "inputs" / "a2_order_set.jsonl"
# B3's task list: A2's substrate claim is IDENTITY with B3's sample, not
# just matching counts — a different seed would still hit 734/25.
B3_TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"

N_PAIRS = 734
N_TARGETS = 25
CELLS = ("ordfirst", "ordlast", "ordshuf")


def fail(msg: str, bad: list[str]) -> None:
    bad.append(msg)


def main() -> int:
    bad: list[str] = []

    rows = [json.loads(l) for l in SET.open(encoding="utf-8")]
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        eid, cell = r["example_id"].rsplit("_", 1)
        by_pair[eid][cell] = r

    if len(rows) != N_PAIRS * 3:
        fail(f"{len(rows)} rows, want {N_PAIRS * 3}", bad)
    if len(by_pair) != N_PAIRS:
        fail(f"{len(by_pair)} pairs, want {N_PAIRS}", bad)
    targets = {(r["survey"], r["target_code"]) for r in rows}
    if len(targets) != N_TARGETS:
        fail(f"{len(targets)} targets, want {N_TARGETS}", bad)

    b3_ids = {json.loads(l)["example_id"]
              for l in B3_TASKS.open(encoding="utf-8")}
    if set(by_pair) != b3_ids:
        only_a2 = len(set(by_pair) - b3_ids)
        only_b3 = len(b3_ids - set(by_pair))
        fail(f"base_ids are not B3's sample: {only_a2} only in A2, "
             f"{only_b3} only in B3", bad)

    # Adjacency: each consecutive block of three rows is one pair.
    for i in range(0, len(rows), 3):
        block = rows[i:i + 3]
        if len({r["base_id"] for r in block}) != 1:
            fail(f"rows {i}..{i + 2} not one pair", bad)
            break

    source: dict[str, dict] = {}
    with open(LADDER, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in by_pair:
                source[r["example_id"]] = r

    common_keys = ("survey", "target_code", "id", "country",
                   "target_question", "option_sets", "ground_truth",
                   "ground_truth_index")
    for eid, cells in by_pair.items():
        if set(cells) != set(CELLS):
            fail(f"{eid}: cells {sorted(cells)}", bad)
            continue
        src = source.get(eid)
        if src is None:
            fail(f"{eid}: not in ladder source", bad)
            continue
        src_items = list(src["questions"].items())
        first = list(cells["ordfirst"]["questions"].items())
        last = list(cells["ordlast"]["questions"].items())
        shuf = list(cells["ordshuf"]["questions"].items())
        if first != src_items:
            fail(f"{eid}: ordfirst is not the source order", bad)
        if last != list(reversed(src_items)):
            fail(f"{eid}: ordlast is not the exact reverse", bad)
        if shuf == first or shuf == last:
            fail(f"{eid}: ordshuf collides with first/last", bad)
        if sorted(shuf) != sorted(src_items):
            fail(f"{eid}: ordshuf changes content, not just order", bad)
        for cell, r in cells.items():
            for k in common_keys:
                if r.get(k) != src.get(k):
                    fail(f"{eid}/{cell}: field {k} differs from source", bad)
        opts = src["option_sets"]["original"]
        if len(set(opts)) != len(opts):
            fail(f"{eid}: duplicate-option target leaked in", bad)
        if opts[src["ground_truth_index"]] != src["ground_truth"]:
            fail(f"{eid}: ground_truth_index does not resolve", bad)

    # Prompt rendering: order actually reaches the prompt.
    for eid in sorted(by_pair)[::97]:
        rendered = {}
        for cell, r in by_pair[eid].items():
            opts = r["option_sets"]["original"]
            p = build_prompt(r, opts, "label_num")
            qs = list(r["questions"])
            pos = [p.find(f"Q: {q}\n") for q in qs]
            if any(x < 0 for x in pos) or pos != sorted(pos):
                fail(f"{eid}/{cell}: prompt order != dict order", bad)
            rendered[cell] = p
        if len(set(rendered.values())) != 3:
            fail(f"{eid}: cells render identical prompts", bad)

    for m in bad[:20]:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures; {len(rows)} rows, {len(by_pair)} pairs, "
          f"{len(targets)} targets")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
