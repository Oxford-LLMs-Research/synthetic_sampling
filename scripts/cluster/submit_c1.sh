#!/bin/bash
# Submit the C1 reason-then-answer battery (RUN_CATALOGUE C1, pre-registered
# 8 Aug with same-day amendments; the presentation x elicitation 2x2 added
# before any run). One job per model runs generation AND scoring in a single
# serving, so all four cells (qa / reasoned / narrative_direct /
# narrative_reasoned) are paired within-serving by construction. The
# narrative cells inherit B3's substrate and its 11 exclusions; a pair whose
# narrative was excluded simply has no 2x2 cells.
#
# Run from the repo root on ARC after `pip install -e .`, with
#   outputs/narrative/inputs/narrative_label_set.jsonl   (B3 scoring set)
#   $LADDER_SET (default ../outputs_recovered/ladder_readout_set.jsonl)
# staged.
#
#   ./scripts/cluster/submit_c1.sh
#   C1_MODELS="Qwen/Qwen3-32B" ./scripts/cluster/submit_c1.sh
#
# Roster mirrors B3: the two dense 32Bs plus the certified MoE — the 2x2 is
# read within model, and three architectures make the (predicted) null hard
# to dismiss. Stage 1 is sampled generation (the transcripts are
# non-regenerable data; pull them with the results).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${C1_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO Qwen/Qwen3-30B-A3B-Instruct-2507}"

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  MODEL="$MODEL" TAG="$TAG" REPLICATE_FRAC=0.25 \
    sbatch --export=ALL --job-name="c1-${TAG##*_}" \
    --time=08:00:00 "$ROOT/scripts/cluster/run_c1.sbatch"
done
