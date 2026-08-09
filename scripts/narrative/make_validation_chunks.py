"""B3 full run: split blind validation tasks into shuffled validator chunks.

Wave-1 amendment (recorded in PAPER_STATE 8 Aug): the two drafts of a pair
must never share a validator context — wave 1 had pair-adjacent chunks and
one validator's off-by-one cascade cost 8 spurious misses. This splitter
shuffles tasks deterministically (seed 42) and then places each task into
the next chunk that does not already hold its sibling (same example_id
prefix of the task_id, i.e. everything before '#').

Usage:
  python scripts/narrative/make_validation_chunks.py \
      --tasks outputs/narrative/generated/tranche1/narrative_validation_tasks.jsonl \
      --out-dir outputs/narrative/generated/tranche1/vchunks
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 42
CHUNK = 5


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path, required=True,
                    help="narrative_validation_tasks.jsonl from check_narratives")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK)
    args = ap.parse_args(argv)

    with open(args.tasks, encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh]
    random.Random(SEED).shuffle(tasks)

    chunks: list[list[dict]] = []
    for t in tasks:
        base = t["task_id"].rsplit("#", 1)[0]
        for c in chunks:
            if len(c) < args.chunk_size and all(
                    x["task_id"].rsplit("#", 1)[0] != base for x in c):
                c.append(t)
                break
        else:
            chunks.append([t])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks):
        with open(args.out_dir / f"vchunk_{i:03d}.jsonl", "w",
                  encoding="utf-8", newline="\n") as fh:
            for t in c:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Invariant: no chunk holds both drafts of a pair.
    for i, c in enumerate(chunks):
        bases = [t["task_id"].rsplit("#", 1)[0] for t in c]
        assert len(bases) == len(set(bases)), f"sibling clash in chunk {i}"
    print(f"{len(tasks)} tasks -> {len(chunks)} chunks of <= "
          f"{args.chunk_size} in {args.out_dir} (sibling-free, seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
