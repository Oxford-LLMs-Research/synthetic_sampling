#!/usr/bin/env python
"""One-file coverage verdict for a scoring run, appended to its STATUS file.

An arm that silently returned nothing for most instances is the failure mode
that survives a 48-row smoke, so coverage per arm is checked over the full
output. Usage: grid_coverage_report.py <results.jsonl> <input.jsonl>
"""
from __future__ import annotations

import collections
import json
import math
import sys


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    want = sum(1 for _ in open(sys.argv[2], encoding="utf-8"))
    print(f"RESULTS {len(rows)}/{want} instances written")
    ok: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    nonfinite: collections.Counter = collections.Counter()
    for r in rows:
        for key, d in r["results"].items():
            arm = key.split("|", 1)[1]
            tot[arm] += 1
            if "error" in d:
                continue
            if d.get("scores"):
                ok[arm] += 1
                nonfinite[arm] += sum(1 for v in d["scores"].values()
                                      if not math.isfinite(v))
    for arm in sorted(tot):
        print(f"ARM {arm:<20} usable {ok[arm]:>6}/{tot[arm]:<6} "
              f"({ok[arm] / tot[arm]:.0%})  non-finite scores {nonfinite[arm]}")
    low = [a for a in tot if ok[a] / tot[a] < 0.9]
    print("VERDICT " + ("INCOMPLETE, low coverage in " + ", ".join(low) if low
                        else "all arms above 90% coverage"))


if __name__ == "__main__":
    main()
