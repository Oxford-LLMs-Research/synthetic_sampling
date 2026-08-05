#!/usr/bin/env python
r"""Build a k=24 control set so the scaling curve's seam can be interpreted.

The curve's five points do not come from one place. k = 6, 12 and 24 are the
main experiment's results; k = 48 and 96 were scored on the cluster with vLLM.
The local exponent per doubling is 0.233, 0.339, 0.545, 0.316 -- flat either
side of 24->48 and anomalous exactly at it, which is exactly where the serving
stack changes. So "the curve steepens above 24" and "the cluster backend scores
higher than whatever served the main run" predict the same five numbers, and
the experiment as run cannot tell them apart.

This builds the missing control: the main run's OWN k=24 instances, re-scored on
the cluster. Same respondents, same targets, same 24 features, same prompt, same
scorer. The only thing that differs is the backend, so any gap is the backend.

The profiles are taken from the main instance files rather than regenerated, so
they are byte-identical to what the published results were computed on -- no
reliance on build_core() reproducing the expansion path.

Instances are restricted to the fixed pair set the scaling curve uses, and
stratified by question so all 266 are covered.

    python .../make_control_shards.py --n 16000 --shards 4
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
SCALED = REPO / "outputs" / "scaling_experiment" / "results"
OUT = REPO / "outputs" / "scaling_experiment" / "control_shards"
CORE_TAG = "s6m4"


def pair_set() -> set[tuple[str, str, str]]:
    """(survey, respondent_id, target_code) present at k=96, the binding level."""
    import pandas as pd
    df = pd.read_csv(SCALED / "scaling_results_all.csv",
                     usecols=["survey", "respondent_id", "target_code", "n_features"],
                     dtype=str, low_memory=False)
    df = df[df["n_features"] == "96"]
    return set(map(tuple, df[["survey", "respondent_id", "target_code"]].to_numpy()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep = pair_set()
    print(f"{len(keep):,} pairs in the k=96 fixed set")

    by_q: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    seen = 0
    for f in sorted(MAIN.glob("*_instances.jsonl")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                if d["profile_type"] != CORE_TAG:
                    continue
                key = (d["survey"], d["id"], d["target_code"])
                if key not in keep:
                    continue
                seen += 1
                d["n_features"] = len(d.get("questions") or {})
                by_q[(d["survey"], d["target_code"])].append(d)
    print(f"{seen:,} core instances matched, across {len(by_q)} questions")
    if not by_q:
        raise SystemExit("no instances matched; check MAIN and the pair set")

    # Even per-question quota, so every question is represented and the paired
    # comparison is not dominated by the questions with the most respondents.
    cap = max(1, args.n // len(by_q))
    rng = random.Random(args.seed)
    picked: list[dict] = []
    for q in sorted(by_q):
        rows = sorted(by_q[q], key=lambda d: d["example_id"])
        picked.extend(rows if len(rows) <= cap else rng.sample(rows, cap))
    rng.shuffle(picked)
    n_feat = collections.Counter(d["n_features"] for d in picked)
    print(f"cap {cap}/question -> {len(picked):,} instances")
    print("  realised profile size: "
          + ", ".join(f"{k} feat {v:,}" for k, v in sorted(n_feat.items()))[:200])

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("control_shard_*.jsonl"):
        old.unlink()
    counts = [0] * args.shards
    handles = [open(OUT / f"control_shard_{i:02d}.jsonl", "w", encoding="utf-8")
               for i in range(args.shards)]
    try:
        for i, d in enumerate(picked):
            s = i % args.shards
            handles[s].write(json.dumps(d, ensure_ascii=False) + "\n")
            counts[s] += 1
    finally:
        for h in handles:
            h.close()

    for i, c in enumerate(counts):
        p = OUT / f"control_shard_{i:02d}.jsonl"
        print(f"  {p.name}  {c:,} instances  {p.stat().st_size / 1e6:.1f} MB")
    print(f"\nwrote {sum(counts):,} instances to {OUT}")


if __name__ == "__main__":
    main()
