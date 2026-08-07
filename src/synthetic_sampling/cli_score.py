"""CLI: score instances with label_num (default) / echo / PMI arms."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_sampling.scoring import DEFAULT_ARMS, parse_arms, run_scoring


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score survey instances via /completions")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help=f"Comma-separated arms (default: {','.join(DEFAULT_ARMS)})",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replicate-frac", type=float, default=0.1,
                    help="Fraction also scored as original_replicate")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    args = ap.parse_args(argv)

    n = run_scoring(
        input_path=args.input,
        out_path=args.out,
        base_url=args.base_url,
        model=args.model,
        arms=parse_arms(args.arms),
        workers=args.workers,
        limit=args.limit,
        replicate_frac=args.replicate_frac,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
