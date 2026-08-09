"""B3 full run: deterministic tranche layout for the remaining pairs.

Wave 1 (50 pairs) is complete; its Sonnet drafts are experiment drafts. The
remaining pairs are shuffled once (seed 42), split into three tranches, and
each tranche's tasks are laid out as generation batches of 10 with the two
drafts of a pair NEVER in the same batch (draft-1 batches first, then
draft-2), so no generator context ever sees a sibling. Batch files carry the
same row shape as wave 1 ({task_id, questions}); the frozen generation
prompt is untouched.

Usage:
  python scripts/narrative/make_tranches.py            # writes all 3 layouts
  python scripts/narrative/make_tranches.py --tranche 1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "outputs" / "narrative" / "inputs"
GEN_DIR = REPO / "outputs" / "narrative" / "generated"
SEED = 42
N_TRANCHES = 3
BATCH = 10


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tranche", type=int, default=None,
                    help="1-based tranche to write (default: all)")
    args = ap.parse_args(argv)

    tasks_by_pair: dict[str, dict[int, dict]] = {}
    with open(IN_DIR / "narrative_tasks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks_by_pair.setdefault(t["example_id"], {})[t["draft"]] = t

    wave1 = json.loads((GEN_DIR / "wave1" / "wave1_manifest.json")
                       .read_text(encoding="utf-8"))
    done = set(wave1["pairs"])
    remaining = sorted(p for p in tasks_by_pair if p not in done)
    random.Random(SEED).shuffle(remaining)
    print(f"{len(tasks_by_pair)} pairs total, {len(done)} in wave 1, "
          f"{len(remaining)} remaining")

    per = -(-len(remaining) // N_TRANCHES)
    for tn in range(1, N_TRANCHES + 1):
        if args.tranche and tn != args.tranche:
            continue
        pairs = remaining[(tn - 1) * per:tn * per]
        out_dir = GEN_DIR / f"tranche{tn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        batches: dict[str, list[str]] = {}
        bi = 0
        for draft in (1, 2):
            tasks = [tasks_by_pair[p][draft] for p in pairs]
            for i in range(0, len(tasks), BATCH):
                chunk = tasks[i:i + BATCH]
                name = f"gen_batch_{bi:02d}"
                with open(out_dir / f"{name}.jsonl", "w",
                          encoding="utf-8", newline="\n") as fh:
                    for t in chunk:
                        fh.write(json.dumps(
                            {"task_id": t["task_id"],
                             "questions": t["questions"]},
                            ensure_ascii=False) + "\n")
                batches[name] = [t["task_id"] for t in chunk]
                bi += 1
        manifest = {"tranche": tn, "seed": SEED, "pairs": pairs,
                    "batches": batches}
        (out_dir / f"tranche{tn}_manifest.json").write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
        print(f"tranche {tn}: {len(pairs)} pairs, {bi} batches -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
