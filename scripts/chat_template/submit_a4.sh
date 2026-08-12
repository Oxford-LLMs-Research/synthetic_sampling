#!/bin/bash
# Submit the A4 chat-template validation (RUN_CATALOGUE A4 expanded,
# pre-registered 12 Aug). One job per model scores the B3 narrative set
# under BOTH prompt regimes in one serving: the raw-completion arms
# (label_num + echo controls) and chat_label_num through the model's own
# chat template, so template-vs-raw is a paired within-serving contrast.
#
# Run from the repo root on ARC after `pip install -e .`, with
#   outputs/narrative/inputs/narrative_label_set.jsonl   staged.
#
#   ./scripts/chat_template/submit_a4.sh
#   A4_MODELS="Qwen/Qwen3-32B" ./scripts/chat_template/submit_a4.sh
#
# Qwen3-32B's template defaults to thinking ON; A4 is the non-reasoning
# arm, so it gets enable_thinking=false. The other rosters' templates have
# no toggle and get no kwargs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${A4_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO Qwen/Qwen3-30B-A3B-Instruct-2507}"

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  KWARGS=""
  if [ "$MODEL" = "Qwen/Qwen3-32B" ]; then
    KWARGS='{"enable_thinking": false}'
  fi
  MODEL="$MODEL" TAG="$TAG" REPLICATE_FRAC=0.25 CHAT_KWARGS="$KWARGS" \
    sbatch --export=ALL --job-name="a4-${TAG##*_}" \
    "$ROOT/scripts/chat_template/run_a4.sbatch"
done
