"""C2 instrument gates, run between assembly and scoring.

Aborts the job on INSTRUMENT failure only; the scientific quantities the
pre-registration names (loop rate, empty rate, parse rate) are printed and
recorded but never gate — a failed prediction is a finding, not a bug.

Fatal (instrument):
- generation errors on >10% of pairs (the serving is sick)
- think block present on <50% of transcripts (the toggle did not engage,
  or the template pre-opens the block and the splitter needs adapting —
  either way scoring would measure the wrong thing; read the trace file)

Usage: python scripts/native_thinking/check_c2_gates.py <c2_label_set.jsonl>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

MAX_ERROR_RATE = 0.10
MIN_BLOCK_RATE = 0.50


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: check_c2_gates.py <c2_label_set.jsonl>", file=sys.stderr)
        return 2
    sidecar = Path(args[0]).with_name(Path(args[0]).stem + "_parse.csv")
    rows = list(csv.DictReader(open(sidecar, encoding="utf-8")))
    n = len(rows)
    if not n:
        print("FATAL empty sidecar")
        return 1
    errs = [r for r in rows if r["parse"] == "generation_error"]
    ok = [r for r in rows if r["parse"] != "generation_error"]
    blocks = [r for r in ok if r["has_block"] == "True"]
    closed = [r for r in blocks if r["closed"] == "True"]
    empty = [r for r in ok if r["has_block"] == "True"
             and r["think_words"] == "0"]
    looped = [r for r in ok if r["loop_markers"] not in ("", "0", "1")]
    parsed = [r for r in ok if r["parse"] in
              ("digit", "bare_digit", "option_text")]

    def rate(part, whole):
        return len(part) / len(whole) if whole else float("nan")

    print(f"C2 GATES over {n} transcripts:")
    print(f"  generation_error : {len(errs)}/{n} ({rate(errs, rows):.1%})")
    print(f"  think block      : {len(blocks)}/{len(ok)} ({rate(blocks, ok):.1%})")
    print(f"  block closed     : {len(closed)}/{len(blocks)} ({rate(closed, blocks):.1%})")
    print(f"  empty think      : {len(empty)}/{len(ok)} ({rate(empty, ok):.1%})  [pre-reg: <5% with loops]")
    print(f"  looped (>1 restatement in think): {len(looped)}/{len(ok)} ({rate(looped, ok):.1%})  [pre-reg: <5% with empties]")
    print(f"  stated parse ok  : {len(parsed)}/{len(ok)} ({rate(parsed, ok):.1%})")

    fatal = []
    if rate(errs, rows) > MAX_ERROR_RATE:
        fatal.append("generation error rate above 10%")
    if rate(blocks, ok) < MIN_BLOCK_RATE:
        fatal.append("think block present on <50% of transcripts — toggle "
                     "not engaging or template pre-opens the block; read "
                     "the *_trace.json before rerunning")
    for f in fatal:
        print(f"FATAL {f}")
    print("VERDICT", "FAIL" if fatal else "PASS")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
