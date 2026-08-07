"""CLI: run analysis core / checks over scoring JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_sampling.analysis import summarize_controls
from synthetic_sampling.checks import check_smoke_file, coverage_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze / check scoring results")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="Smoke gate on results JSONL")
    p_smoke.add_argument("results", type=Path)

    p_cov = sub.add_parser("coverage", help="Coverage report")
    p_cov.add_argument("results", type=Path)
    p_cov.add_argument("input", type=Path)

    p_rep = sub.add_parser("replicate", help="Replicate agreement summary")
    p_rep.add_argument("results", type=Path)
    p_rep.add_argument("--arm", default="label_num")

    args = ap.parse_args(argv)

    if args.cmd == "smoke":
        fatal, notes = check_smoke_file(args.results)
        for n in notes:
            print(f"NOTE  {n}")
        for f in fatal:
            print(f"FATAL {f}")
        print("VERDICT " + ("FAIL" if fatal else "PASS"))
        return 1 if fatal else 0

    if args.cmd == "coverage":
        text, ok = coverage_report(args.results, args.input)
        print(text)
        return 0 if ok else 1

    if args.cmd == "replicate":
        rows = [json.loads(l) for l in open(args.results, encoding="utf-8")]
        print(json.dumps(summarize_controls(rows, arm=args.arm), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
