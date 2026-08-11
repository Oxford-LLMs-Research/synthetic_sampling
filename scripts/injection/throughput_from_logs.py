"""Dense-vs-MoE serving throughput, parsed from the job logs.

Closes the A6 speed benchmark: the catalogue left the clean 4-arm dense ratio
to "the A1 logs when they complete". Reads the `done: N instances in T min`
line of each main run (the second such line; the first is the smoke gate) and
writes one row per job.

The comparison is only meaningful because the substrates match on what drives
cost -- ~5.0 scored cells per instance, ~4.8-5.1 options, ~2,620-2,652 chars of
profile plus question -- and because all three jobs landed on htc-g058. Both
facts are checked here, not assumed: the per-instance load is recomputed from
the result files, and the node comes from the recorded sacct output.

    python scripts/injection/throughput_from_logs.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT.parent / "outputs_recovered" / "a1_logs"
MOE_LOG = ROOT.parent / "outputs_recovered" / "readout_moe" / "a6-readout-moe-8498233.out"
OUTDIR = ROOT.parent / "analysis" / "injection"

DONE = re.compile(r"done: (\d+) instances in ([\d.]+) min")

# sacct -j 8498233,8498370,8498891 --format=JobID,NodeList, run 9 Aug 2026.
NODES = {"8498233": "htc-g058", "8498370": "htc-g058", "8498891": "htc-g058"}

JOBS = [
    ("dense_32b", "Qwen/Qwen3-32B", "country_injection", "8498891",
     LOGS / "a1-country-qwen3-32b-8498891.out",
     ROOT / "outputs/country_injection/results/country_injection_label_results_qwen_qwen3-32b.jsonl"),
    ("dense_32b", "Qwen/Qwen3-32B", "temporal_context", "8498370",
     LOGS / "a1-temporal-qwen3-32b-8498370.out",
     ROOT / "outputs/temporal_context/results/temporal_context_label_results_qwen_qwen3-32b.jsonl"),
    ("moe_30b_a3b", "Qwen/Qwen3-30B-A3B-Instruct-2507", "readout_moe", "8498233",
     MOE_LOG,
     ROOT / "outputs/readout_moe/results/readout_results_qwen_qwen3-30b-a3b-instruct-2507.jsonl"),
]


def main_run(log: Path) -> tuple[int, float]:
    """(instances, minutes) of the main run -- the LAST `done:` line."""
    hits = DONE.findall(log.read_text(encoding="utf-8"))
    if not hits:
        raise SystemExit(f"no `done:` line in {log}")
    n, mins = hits[-1]
    return int(n), float(mins)


def per_instance_load(results: Path) -> tuple[float, float]:
    """(mean scored cells per instance, mean options) -- the cost drivers."""
    cells, opts = [], []
    for line in results.open(encoding="utf-8"):
        res = (json.loads(line).get("results") or {})
        cells.append(len(res))
        first = next(iter(res.values()))
        opts.append(len(first.get("scores") or {}))
    return st.mean(cells), st.mean(opts)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, model, experiment, job, log, results in JOBS:
        n, mins = main_run(log)
        cells, opts = per_instance_load(results)
        rows.append({
            "kind": kind, "model": model, "experiment": experiment, "job": job,
            "node": NODES[job], "instances": n, "minutes": mins,
            "inst_per_s": n / (mins * 60.0),
            "cells_per_instance": cells, "mean_options": opts,
        })

    df = pd.DataFrame(rows)
    moe = df[df.kind == "moe_30b_a3b"]["inst_per_s"].iloc[0]
    df["speedup_vs_this_row"] = moe / df["inst_per_s"]

    nodes = set(df["node"])
    load = df["cells_per_instance"]
    print(f"nodes: {nodes} ({'same node' if len(nodes) == 1 else 'MIXED -- ratio not clean'})")
    print(f"cells/instance spread: {load.min():.3f}-{load.max():.3f} "
          f"({'comparable' if load.max() - load.min() < 0.1 else 'NOT comparable'})")
    print(df.round(4).to_string(index=False))

    out = OUTDIR / "a1_throughput.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
