#!/bin/bash
# Submit the B3 validated-narrative presentation battery (RUN_CATALOGUE B3,
# substrate completed 8 Aug): one job per model over the single assembled set
# holding all three arms, so qa / narrative1 / narrative2 are scored in ONE
# serving. That is the whole point — the reuse rule forbids comparing scores
# across servings, and both contrasts here are within-pair:
#
#   narrative1 vs narrative2   the wording-variance ceiling
#   narrative  vs qa           the format effect, read against that ceiling
#
# Never split the arms across jobs or shards-by-arm; the runner's shards are
# by example_id and keep a base_id's three rows in the same serving only if
# SHARD_COUNT is unset. Run from the repo root on ARC after `pip install -e .`
# and after outputs/narrative/inputs/narrative_label_set.jsonl is staged.
#
#   ./scripts/cluster/submit_b3.sh
#   B3_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" ./scripts/cluster/submit_b3.sh
#
# B3_MODELS overrides the roster (space-separated HF ids). The MoE is in the
# default roster because A6 certified it on 8 Aug (fidelity PASS, calibration
# temperature-fixable) and it costs ~10 min of serving here, so a third model
# is close to free. It earns its slot on design grounds too: the roster then
# spans 4B-active MoE / 32B dense x two model families, and B3 is powered for
# a NULL — "presentation does not move the ceiling" is far harder to dismiss
# on three architectures than on two. Its absolute accuracy sits below the
# dense 32Bs (norm 0.235), so read it WITHIN model, never as a level shift
# against them; every B3 contrast is within-pair within-serving anyway.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${B3_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO Qwen/Qwen3-30B-A3B-Instruct-2507}"
INPUT="$ROOT/outputs/narrative/inputs/narrative_label_set.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "missing $INPUT (build it with scripts/narrative/make_narrative_set.py)"
  exit 1
fi

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  OUT="$ROOT/outputs/narrative/results/narrative_label_results_${TAG}.jsonl"
  MODEL="$MODEL" INPUT="$INPUT" OUT="$OUT" REPLICATE_FRAC=0.25 \
    sbatch --export=ALL --job-name="b3-${TAG##*_}" \
    --time=08:00:00 "$ROOT/scripts/cluster/run_score.sbatch"
done
