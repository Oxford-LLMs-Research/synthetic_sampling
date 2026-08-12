"""Smoke gate for a scoring run (fatal vs advisory)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Union

CRITICAL_ARM = "label_num"
MAX_LABEL_MISS = float(os.environ.get("READOUT_MAX_LABEL_MISS", "0.10"))
MIN_ROWS = 40


def check_smoke(rows: List[dict]) -> Tuple[List[str], List[str]]:
    """Return (fatal, notes). Fatal means the run cannot produce its result."""
    fatal: List[str] = []
    notes: List[str] = []

    if len(rows) < MIN_ROWS:
        fatal.append(f"expected one row per question, got {len(rows)}")
        return fatal, notes

    errs = sorted({
        k for r in rows for k, v in r.get("results", {}).items() if "error" in v
    })
    if errs:
        fatal.append(f"arms returned errors: {errs[:6]}")

    # Every label-readout arm must clear the miss threshold INDEPENDENTLY
    # (label_num, chat_label_num, ...). Pooling them would let a healthy raw
    # arm mask a dead chat arm — the A4 failure mode.
    arms_present = sorted({
        k.split("|", 1)[1] if "|" in k else k
        for r in rows for k in r.get("results", {})
    })
    label_arms = [a for a in arms_present if a.endswith(CRITICAL_ARM)]
    if not label_arms:
        notes.append(
            f"{CRITICAL_ARM} not among this run's arms; label check skipped")
    for arm in label_arms:
        vals = [
            v for r in rows for k, d in r["results"].items()
            if (k.split("|", 1)[1] if "|" in k else k) == arm
            for v in d.get("scores", {}).values()
        ]
        miss = sum(1 for v in vals if not math.isfinite(v))
        rate = miss / len(vals) if vals else 1.0
        notes.append(
            f"{arm}: {miss}/{len(vals)} options got no label "
            f"logprob ({rate:.1%})")
        if not vals or rate > MAX_LABEL_MISS:
            fatal.append(
                f"{arm}: {miss}/{len(vals)} options ({rate:.0%}) "
                "got no label logprob; the tokeniser emits labels the "
                "matcher misses")

    for arm in ("echo_qonly", "echo_ctxfree"):
        vals = [
            v for r in rows for k, d in r.get("results", {}).items()
            if k.endswith(arm) for v in d.get("scores", {}).values()
        ]
        if not vals:
            continue
        miss = sum(1 for v in vals if not math.isfinite(v))
        if miss:
            notes.append(f"{arm}: {miss}/{len(vals)} non-finite scores")

    return fatal, notes


def check_smoke_file(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return check_smoke(rows)


def main(argv: List[str] | None = None) -> int:
    import sys
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: smoke <results.jsonl>", file=sys.stderr)
        return 2
    fatal, notes = check_smoke_file(args[0])
    for n in notes:
        print(f"NOTE  {n}")
    for f in fatal:
        print(f"FATAL {f}")
    if fatal:
        print("VERDICT FAIL")
        return 1
    print("VERDICT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
