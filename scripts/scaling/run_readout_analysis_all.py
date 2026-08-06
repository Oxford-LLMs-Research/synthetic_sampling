#!/usr/bin/env python
"""Run the full readout analysis suite over every results file present.

One command instead of five per model, so the grid's outputs cannot end up
half-analysed. Steps per file, each tolerated individually (a failed step is
reported, not fatal, because a grid file for a base model may legitimately
lack an arm a step needs):

    analyze_readout.py          summary, contrasts, replicate (per tag)
    readout_claims_check.py     entropy ratio, dissenter penalty (per tag)
    consensus_analysis.py       cross-elicitation agreement (per tag)
    analyze_pmi.py              PMI decomposition, for files carrying the
                                neutral arms (readout_pmi_* and grid files)
    readout_aggregate_checks.py marginal recovery / defaults / anchoring
                                (runs once; loops over files itself)

    python .../run_readout_analysis_all.py [--perm 300]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
RESULTS_DIR = REPO / "outputs" / "scaling_experiment" / "readout_results"


def run(step: str, *args: str) -> bool:
    cmd = [sys.executable, str(SCRIPTS / step), *args]
    print(f"\n{'#' * 74}\n# {step} {' '.join(args)}\n{'#' * 74}", flush=True)
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=300,
                    help="permutations for analyze_readout's AUC null")
    args = ap.parse_args()

    full = (sorted(RESULTS_DIR.glob("readout_results_*.jsonl"))
            + sorted(RESULTS_DIR.glob("readout_grid_*.jsonl")))
    pmi = (sorted(RESULTS_DIR.glob("readout_pmi_*.jsonl"))
           + sorted(RESULTS_DIR.glob("readout_grid_*.jsonl")))
    if not full and not pmi:
        sys.exit(f"no results files under {RESULTS_DIR}")

    outcomes: list[tuple[str, str, bool]] = []
    for f in full:
        for step, extra in (("analyze_readout.py", ["--perm", str(args.perm)]),
                            ("readout_claims_check.py", []),
                            ("consensus_analysis.py", [])):
            ok = run(step, "--results", str(f), *extra)
            outcomes.append((f.name, step, ok))
    for f in pmi:
        ok = run("analyze_pmi.py", "--results", str(f))
        outcomes.append((f.name, "analyze_pmi.py", ok))
    ok = run("readout_aggregate_checks.py")
    outcomes.append(("(all files)", "readout_aggregate_checks.py", ok))

    print(f"\n{'=' * 74}\nsuite summary\n{'=' * 74}")
    for fname, step, ok in outcomes:
        print(f"  {'OK  ' if ok else 'FAIL'} {step:<30} {fname}")
    n_fail = sum(1 for _, _, ok in outcomes if not ok)
    print(f"\n{len(outcomes) - n_fail}/{len(outcomes)} steps succeeded")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
