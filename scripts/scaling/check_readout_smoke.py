#!/usr/bin/env python
r"""Is this readout run worth spending the GPU hours on?

Run against the smoke output on the cluster, before the full 4,800-instance run
starts. Exits non-zero only when continuing would waste the whole allocation.

The point of this file existing rather than living in a heredoc inside the
sbatch is that the check is itself code, and code that can abort a job is code
that can be wrong. Twice now the cluster has been the place we found that out:

  the guard was zero-tolerance on missing label logprobs, so Olmo 3.1 32B
  Instruct DPO was killed over 4 misses in 458 (0.9%) after a clean startup and
  a clean smoke. It cost a 40-minute model load and a queue slot.

So the rule here is: FATAL means "the run cannot produce the result it exists to
produce". Anything narrower is printed and the run continues, because a warning
costs nothing and a false abort costs a cycle. Verify against a stored run
before shipping a change:

    python check_readout_smoke.py --self-test <any readout_results_*.jsonl>

which asserts a known-good file passes, and that a file with the label arm
blanked fails. Both take a second and neither needs a GPU.

    python check_readout_smoke.py <smoke.jsonl>
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# The arm the experiment rests on. Echo scoring is what the paper already did;
# the label readout is the thing being tested, so if it is dead the run is
# pointless. Everything else can be missing and the run is still worth having.
CRITICAL_ARM = "label_num"
MAX_LABEL_MISS = 0.10
MIN_ROWS = 40


def check(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (fatal, notes). Fatal aborts the job; notes are recorded only."""
    fatal: list[str] = []
    notes: list[str] = []

    # The smoke input is every 100th line of a file grouped 100-per-question, so
    # a short file means it did not span questions and the format was validated
    # against one question only. That is how a bad prompt format reached a full
    # run once already.
    if len(rows) < MIN_ROWS:
        fatal.append(f"expected one row per question, got {len(rows)}")
        return fatal, notes

    errs = sorted({k for r in rows for k, v in r["results"].items() if "error" in v})
    if errs:
        fatal.append(f"arms returned errors: {errs[:6]}")

    vals = [v for r in rows for k, d in r["results"].items()
            if k.endswith(CRITICAL_ARM) for v in d.get("scores", {}).values()]
    miss = sum(1 for v in vals if not math.isfinite(v))
    rate = miss / len(vals) if vals else 1.0
    notes.append(f"{CRITICAL_ARM}: {miss}/{len(vals)} options got no label "
                 f"logprob ({rate:.1%})")
    # Total failure means the tokeniser puts the label somewhere the matcher
    # never looks; on Qwen 3 32B that was 72 of 72 before the trailing-space
    # fix. A handful of misses is a different thing: those instances drop out of
    # the analysis and cost nothing.
    if not vals or rate > MAX_LABEL_MISS:
        fatal.append(f"{CRITICAL_ARM}: {miss}/{len(vals)} options ({rate:.0%}) "
                     "got no label logprob; the tokeniser emits labels the "
                     "matcher misses")

    # Everything below is advisory. Each affects one auxiliary arm or one
    # secondary quantity, never the four scoring arms the analysis needs.
    gs = [d for r in rows for k, d in r["results"].items()
          if k.endswith("generate_sampled")]
    if not gs or not all(d.get("n_draws") == 8 for d in gs):
        notes.append("WARN n>1 sampling not honoured; generate_sampled will be "
                     "single-draw, the other arms are unaffected")
    if not any(k.startswith("original_replicate") for r in rows for k in r["results"]):
        notes.append("WARN replicate condition absent; the measurement's own "
                     "noise floor will not be estimable from this run")

    g = [d for r in rows for k, d in r["results"].items() if k.endswith("|generate")]
    matched = sum(1 for d in g if d.get("predicted") is not None)
    drawn = sum(d.get("n_draws", 0) for d in gs)
    srate = sum(d.get("n_matched", 0) for d in gs) / drawn if drawn else 0.0
    notes.append(f"greedy matched {matched}/{len(g)}, sampled match rate {srate:.0%}")
    if g and matched < 0.5 * len(g):
        # Some instruct models never emit an option verbatim. Olmo 3.1 32B
        # matched 0 of 96 where Qwen 3 32B matched 56. That is a fact about the
        # model worth recording, not a reason to discard five working arms.
        notes.append("WARN generate arms unreliable, inspect 'raw'; the four "
                     "scoring arms are unaffected")
    return fatal, notes


def self_test(path: Path) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:200]
    if len(rows) < MIN_ROWS:
        print(f"self-test needs at least {MIN_ROWS} rows, {path} has {len(rows)}")
        return 1
    ok = True

    fatal, notes = check(rows)
    print(f"known-good file: {'PASS' if not fatal else 'FAIL ' + str(fatal)}")
    ok &= not fatal

    blanked = copy.deepcopy(rows)
    for r in blanked:
        for k, d in r["results"].items():
            if k.endswith(CRITICAL_ARM):
                d["scores"] = {o: float("nan") for o in d.get("scores", {})}
    fatal, _ = check(blanked)
    print(f"label arm blanked: {'PASS (caught)' if fatal else 'FAIL (missed)'}")
    ok &= bool(fatal)

    truncated = rows[:MIN_ROWS - 1]
    fatal, _ = check(truncated)
    print(f"short smoke:       {'PASS (caught)' if fatal else 'FAIL (missed)'}")
    ok &= bool(fatal)

    # One miss must NOT abort. This is the regression that killed the Olmo run.
    one_miss = copy.deepcopy(rows)
    for r in one_miss[:1]:
        for k, d in r["results"].items():
            if k.endswith(CRITICAL_ARM) and d.get("scores"):
                first = next(iter(d["scores"]))
                d["scores"][first] = float("nan")
                break
    fatal, _ = check(one_miss)
    print(f"single miss:       {'PASS (tolerated)' if not fatal else 'FAIL ' + str(fatal)}")
    ok &= not fatal

    print("\nself-test", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--self-test", action="store_true",
                    help="validate this checker against a stored run, no GPU needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args.results))

    rows = [json.loads(l) for l in open(args.results, encoding="utf-8")]
    fatal, notes = check(rows)
    for n in notes:
        print(f"SMOKE {n}")
    if fatal:
        print("SMOKE FATAL " + " | ".join(fatal))
        raise SystemExit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
