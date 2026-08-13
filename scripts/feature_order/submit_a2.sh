#!/bin/bash
# Submit A2, the feature-order battery (RUN_CATALOGUE A2, pre-registered
# 13 Aug). One job per model over the single assembled set holding all
# three order cells (informative_first / informative_last / shuffled), so
# every contrast is within-pair inside ONE serving. Never split cells
# across jobs; SHARD_COUNT stays unset.
#
# Arms are label_num + echo_plain only: the PMI premises are
# profile-independent, hence identical across order cells, and A2 makes
# no cross-experiment PMI claim.
#
# Run from the repo root on ARC after `pip install -e .`, with
#   outputs/feature_order/inputs/a2_order_set.jsonl   staged
# (build locally with scripts/feature_order/make_a2_set.py and verify with
# verify_a2_set.py before staging).
#
#   ./scripts/feature_order/submit_a2.sh
#   A2_MODELS="Qwen/Qwen3-32B" ./scripts/feature_order/submit_a2.sh
#   A2_LIMIT=30 ./scripts/feature_order/submit_a2.sh     # canary
#
# A2_LIMIT caps instances and quarantines outputs under a _canary tag so a
# later full run cannot resume from canary rows. Use a multiple of 3: the
# set keeps a pair's three cells adjacent, so the cap keeps whole triples.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${A2_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO Qwen/Qwen3-30B-A3B-Instruct-2507}"
INPUT="$ROOT/outputs/feature_order/inputs/a2_order_set.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "missing $INPUT (build with scripts/feature_order/make_a2_set.py)"
  exit 1
fi

SUFFIX=""
if [ -n "${A2_LIMIT:-}" ]; then
  SUFFIX="_canary"
  export LIMIT="$A2_LIMIT"
  echo "CANARY MODE: LIMIT=$A2_LIMIT, outputs tagged ${SUFFIX}"
fi

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  OUT="$ROOT/outputs/feature_order/results/a2_order_results_${TAG}${SUFFIX}.jsonl"
  MODEL="$MODEL" INPUT="$INPUT" OUT="$OUT" \
    ARMS="label_num,echo_plain" REPLICATE_FRAC=0.25 \
    sbatch --export=ALL --job-name="a2-${TAG##*_}" \
    --time=08:00:00 "$ROOT/scripts/cluster/run_score.sbatch"
done
