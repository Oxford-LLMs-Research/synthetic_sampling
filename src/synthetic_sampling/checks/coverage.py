"""Coverage report for a scoring run."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union


def coverage_report(
    results_path: Union[str, Path],
    input_path: Union[str, Path],
    min_frac: float = 0.9,
) -> Tuple[str, bool]:
    """Return (text report, ok). ok is False if any arm is below min_frac."""
    results_path, input_path = Path(results_path), Path(input_path)
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    want = sum(1 for _ in open(input_path, encoding="utf-8"))
    lines = [f"RESULTS {len(rows)}/{want} instances written"]
    ok_c: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    nonfinite: collections.Counter = collections.Counter()
    for r in rows:
        for key, d in r.get("results", {}).items():
            arm = key.split("|", 1)[1] if "|" in key else key
            tot[arm] += 1
            if "error" in d:
                continue
            if d.get("scores"):
                ok_c[arm] += 1
                nonfinite[arm] += sum(
                    1 for v in d["scores"].values() if not math.isfinite(v))
    for arm in sorted(tot):
        lines.append(
            f"ARM {arm:<20} usable {ok_c[arm]:>6}/{tot[arm]:<6} "
            f"({ok_c[arm] / tot[arm]:.0%})  non-finite scores {nonfinite[arm]}")
    low = [a for a in tot if ok_c[a] / tot[a] < min_frac]
    ok = not low
    lines.append(
        "VERDICT " + (
            "INCOMPLETE, low coverage in " + ", ".join(low) if low
            else f"all arms above {min_frac:.0%} coverage"))
    return "\n".join(lines), ok


def main(argv: List[str] | None = None) -> int:
    import sys
    args = list(argv or sys.argv[1:])
    if len(args) != 2:
        print("Usage: coverage <results.jsonl> <input.jsonl>", file=sys.stderr)
        return 2
    text, ok = coverage_report(args[0], args[1])
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
