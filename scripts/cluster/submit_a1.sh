#!/bin/bash
# Submit the A1 injection retests (RUN_CATALOGUE A1, pre-registered 8 Aug):
# country and temporal instance files x the A1 roster, one serving per job,
# default arms, 25 percent replicate. Run from the repo root on ARC after
# `pip install -e .` and after the input files are staged.
#
#   ./scripts/cluster/submit_a1.sh
#   A1_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" ./scripts/cluster/submit_a1.sh
#
# A1_MODELS overrides the roster (space-separated HF ids). The MoE candidate
# joins only after its readout battery certifies label_num (see A6).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${A1_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO}"

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  for EXP in country_injection temporal_context; do
    INPUT="$ROOT/outputs/$EXP/inputs/${EXP}_label_set.jsonl"
    OUT="$ROOT/outputs/$EXP/results/${EXP}_label_results_${TAG}.jsonl"
    MODEL="$MODEL" INPUT="$INPUT" OUT="$OUT" REPLICATE_FRAC=0.25 \
      sbatch --export=ALL --job-name="a1-${EXP%%_*}-${TAG##*_}" \
      --time=06:00:00 "$ROOT/scripts/cluster/run_score.sbatch"
  done
done
