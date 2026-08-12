"""C3: build the Instruct-sibling direct scoring set.

Same 734-pair qa substrate as C2/C3 Thinking, no reasoning field — the
Instruct-2507 ceiling against which the Thinking checkpoint is read.
Fresh scores only (reuse rule: instances yes, scores never across servings).

Usage:
  python scripts/thinking/make_c3_direct_set.py \\
      --out outputs/thinking/inputs/c3_direct_set_qwen_qwen3-30b-a3b-instruct-2507.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--ladder-set", type=Path, default=LADDER_SET)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap pairs (canary). Deterministic: sorted eids.")
    args = ap.parse_args(argv)

    eids = set()
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            eids.add(json.loads(line)["example_id"])
    rows = []
    with open(args.ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in eids:
                rows.append(r)
    rows.sort(key=lambda r: r["example_id"])
    if args.limit is not None:
        rows = rows[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for src in rows:
            fh.write(json.dumps({
                "example_id": f"{src['example_id']}_direct",
                "arm_label": "direct",
                "base_id": src["example_id"],
                "survey": src["survey"], "target_code": src["target_code"],
                "id": src.get("id"), "country": src.get("country"),
                "target_question": src["target_question"],
                "option_sets": src["option_sets"],
                "ground_truth": src["ground_truth"],
                "ground_truth_index": src["ground_truth_index"],
                "questions": src["questions"],
            }, ensure_ascii=False) + "\n")
    print(f"{len(rows)} direct instances -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
