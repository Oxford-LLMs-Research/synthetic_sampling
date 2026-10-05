"""Split a Phase 2 instance file by option count.

The label_num readout matches the first generated token, and on most of the
roster (Qwen, Gemma, Nemotron) option numbers from 10 up are two tokens
(scripts/phase2/check_label_tokens.py). Instances with 10 or more options
are therefore held back from the grid until the readout handles them
(decided 5 Oct 2026). Lines are copied byte for byte, in input order.

    python scripts/phase2/split_long_lists.py outputs/phase2/inputs/instances_r0-50.jsonl

writes instances_r0-50_short.jsonl (under 10 options, the grid input) and
instances_r0-50_long.jsonl (10 or more) next to the input.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

LONG_FROM = 10


def main() -> None:
    src = Path(sys.argv[1])
    outs = {k: src.with_name(f"{src.stem}_{k}.jsonl") for k in ("short", "long")}
    n = collections.Counter()
    targets = collections.Counter()
    sha = {k: hashlib.sha256() for k in outs}
    with open(src, "rb") as fh, \
            open(outs["short"], "wb") as short, open(outs["long"], "wb") as long_:
        for line in fh:
            r = json.loads(line)
            k = "long" if len(r["options"]) >= LONG_FROM else "short"
            (long_ if k == "long" else short).write(line)
            sha[k].update(line)
            n[k] += 1
            if k == "long":
                targets[(r["survey"], r["target_code"], len(r["options"]))] += 1
    for k, p in outs.items():
        print(f"{k:5s} {n[k]:7d} instances  sha256 {sha[k].hexdigest()}  {p}")
    for (survey, code, m), c in sorted(targets.items()):
        print(f"  held back: {survey} {code} ({m} options) {c}")


if __name__ == "__main__":
    main()
