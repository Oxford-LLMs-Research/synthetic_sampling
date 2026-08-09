"""B3: split a regeneration list into SIBLING-FREE agent workloads.

Tranche-2 finding (8 Aug): when both drafts of a pair land in the same
regeneration agent, they converge — same context, same attention list, same
corrective instruction — and the pair then fails the near-copy gate (6 of 8
tranche-2 near-copies arose exactly this way, two at Jaccard 0.92-0.93).
The two drafts of a pair are supposed to be INDEPENDENT; that is what makes
narrative1-vs-narrative2 agreement a wording-variance ceiling. So the same
rule the validation splitter uses applies here: a pair's drafts never share
an agent.

Reads the scorer's narrative_regen_list.jsonl plus the tranche task file,
emits regen_part_NN.jsonl carrying {task_id, questions, attention}.

Usage:
  python scripts/narrative/make_regen_parts.py \
      --tranche outputs/narrative/generated/tranche3 --size 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tranche", type=Path, required=True,
                    help="tranche directory holding narrative_regen_list.jsonl")
    ap.add_argument("--tasks", type=Path, default=None,
                    help="default: <tranche>/<name>_tasks.jsonl")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--prefix", default="regen_part")
    args = ap.parse_args(argv)

    tasks_path = args.tasks or (args.tranche / f"{args.tranche.name}_tasks.jsonl")
    tasks = {}
    with open(tasks_path, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = t

    items = []
    with open(args.tranche / "narrative_regen_list.jsonl", encoding="utf-8") as fh:
        for line in fh:
            f = json.loads(line)
            t = tasks[f["task_id"]]
            qs = list(t["questions"].items())
            items.append({
                "task_id": f["task_id"], "questions": t["questions"],
                "attention": [{"question": qs[i - 1][0], "answer": qs[i - 1][1]}
                              for i in f["wrong_questions"]],
            })

    parts: list[list[dict]] = []
    for it in items:
        base = it["task_id"].rsplit("#", 1)[0]
        for p in parts:
            if len(p) < args.size and all(
                    x["task_id"].rsplit("#", 1)[0] != base for x in p):
                p.append(it)
                break
        else:
            parts.append([it])

    for i, p in enumerate(parts):
        bases = [x["task_id"].rsplit("#", 1)[0] for x in p]
        assert len(bases) == len(set(bases)), f"sibling clash in part {i}"
        out = args.tranche / f"{args.prefix}_{i:02d}.jsonl"
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            for x in p:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {out.name}: {len(p)}")
    print(f"{len(items)} drafts -> {len(parts)} sibling-free parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
