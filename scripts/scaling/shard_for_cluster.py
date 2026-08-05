#!/usr/bin/env python
"""Split the scaling instances into N shards for data-parallel scoring.

Balances on estimated prefill cost, not line count. A k=96 instance carries
roughly twice the prompt of a k=48 one, so round-robin by line would leave the
shards up to 2x apart in wall time; longest-processing-time-first assignment
gets them within a fraction of a percent.

Instances are independent, so any partition is valid: shards can be scored on
separate GPUs and the results concatenated.

Usage:
    python .../shard_for_cluster.py --shards 4
    python .../shard_for_cluster.py --shards 8 --out /data/.../shards
"""
from __future__ import annotations

import argparse
import heapq
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = REPO / "outputs" / "scaling_experiment"

# Options are scored per instance; if the scorer shares the prompt KV cache
# across them the cost is ~prompt + n_options * option_len, which this
# approximates well enough for load balancing.
CHARS_PER_TOKEN = 4.0


def cost(rec: dict) -> float:
    prompt = sum(len(k) + len(str(v)) for k, v in rec["questions"].items())
    prompt += len(rec.get("target_question") or "")
    opts = sum(len(str(o)) for o in (rec.get("options") or []))
    return (prompt + opts) / CHARS_PER_TOKEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", type=Path, default=SRC / "shards")
    args = ap.parse_args()

    files = sorted(SRC.glob("*_scaling_instances.jsonl"))
    if not files:
        raise SystemExit(f"no instance files under {SRC}")

    records = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                rec = json.loads(line)
                records.append((cost(rec), line))
    print(f"{len(records)} instances from {len(files)} files")

    # LPT: heaviest first into the currently lightest shard.
    records.sort(key=lambda r: -r[0])
    heap = [(0.0, i) for i in range(args.shards)]
    heapq.heapify(heap)
    buckets: list[list[str]] = [[] for _ in range(args.shards)]
    totals = [0.0] * args.shards
    for c, line in records:
        load, i = heapq.heappop(heap)
        buckets[i].append(line)
        totals[i] = load + c
        heapq.heappush(heap, (load + c, i))

    args.out.mkdir(parents=True, exist_ok=True)
    print()
    print("%-8s %10s %14s" % ("shard", "instances", "est. tokens"))
    for i, (b, t) in enumerate(zip(buckets, totals)):
        p = args.out / f"scaling_shard_{i:02d}.jsonl"
        p.write_text("".join(b), encoding="utf-8")
        print("%-8d %10d %14s" % (i, len(b), f"{t:,.0f}"))
    spread = (max(totals) - min(totals)) / max(totals) * 100
    print(f"\ntotal est. prefill tokens {sum(totals):,.0f}"
          f"   shard imbalance {spread:.2f}%")
    print(f"wrote {args.shards} shards to {args.out}")


if __name__ == "__main__":
    main()
