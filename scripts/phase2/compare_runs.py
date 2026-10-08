"""Agreement between two scoring runs of the same instances.

    python scripts/phase2/compare_runs.py RUN_A RUN_B

RUN_A and RUN_B are score files or folders of `*_shard*of*.jsonl` files.
Rows are matched on `example_id`. Reports, for the `label_num` arm on the
`original` option set, how many shared instances got the same predicted
option, and the largest and mean absolute difference in option scores.

Used to check that a serving setting (attention backend, all-reduce
implementation, vLLM version) leaves predictions unchanged before it is
applied to part of the grid. Standard library only, so it runs on a login
node.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

ARM = "original|label_num"


def load(path: str) -> dict:
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*_shard*of*.jsonl"))))
    if not files:
        sys.exit(f"no score files under {path}")
    out = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rec = r["results"].get(ARM)
                if rec is not None and "error" not in rec:
                    out[r["example_id"]] = rec
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("no shared example_id")
    same = 0
    diffs = []
    for k in shared:
        same += a[k]["predicted_index"] == b[k]["predicted_index"]
        sa, sb = a[k]["scores"], b[k]["scores"]
        for opt in set(sa) & set(sb):
            if math.isfinite(sa[opt]) and math.isfinite(sb[opt]):
                diffs.append(abs(sa[opt] - sb[opt]))
    print(f"rows A {len(a)}  rows B {len(b)}  shared {len(shared)}")
    print(f"same prediction {same} of {len(shared)} "
          f"({100.0 * same / len(shared):.1f}%)")
    if diffs:
        print(f"option score abs diff: mean {sum(diffs) / len(diffs):.4f}  "
              f"max {max(diffs):.4f}  (n {len(diffs)})")


if __name__ == "__main__":
    main()
