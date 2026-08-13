#!/bin/bash
# Submit A3, the default-option (DK) battery (RUN_CATALOGUE A3, amended and
# pre-registered 13 Aug: TWO cells, dk_present / dk_absent — the latin
# square cancels absolute option position, so the catalogue's forced-last
# cell would measure nothing it is meant to). One job per model over the
# single assembled set holding both cells, so the contrast is within-pair
# inside ONE serving. Never split cells across jobs; SHARD_COUNT stays
# unset.
#
# Arms are label_num + echo_plain only (PMI premises make no A3 claim).
# NOTE the two cells have different option COUNTS by construction (DK
# removed), so per-instance request cost differs across cells; that is the
# treatment, not a bug.
#
# Run from the repo root on ARC after `pip install -e .`, with
#   outputs/default_options/inputs/a3_dk_set.jsonl   staged
# (build locally with scripts/default_options/make_a3_set.py and verify
# with verify_a3_set.py before staging).
#
#   ./scripts/default_options/submit_a3.sh
#   A3_MODELS="Qwen/Qwen3-32B" ./scripts/default_options/submit_a3.sh
#   A3_LIMIT=20 ./scripts/default_options/submit_a3.sh   # canary
#
# A3_LIMIT caps instances and quarantines outputs under a _canary tag so a
# later full run cannot resume from canary rows. Use an EVEN number: the
# set keeps a pair's two cells adjacent, so the cap keeps whole pairs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

MODELS="${A3_MODELS:-Qwen/Qwen3-32B allenai/Olmo-3.1-32B-Instruct-DPO Qwen/Qwen3-30B-A3B-Instruct-2507}"
INPUT="$ROOT/outputs/default_options/inputs/a3_dk_set.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "missing $INPUT (build with scripts/default_options/make_a3_set.py)"
  exit 1
fi

SUFFIX=""
if [ -n "${A3_LIMIT:-}" ]; then
  SUFFIX="_canary"
  export LIMIT="$A3_LIMIT"
  echo "CANARY MODE: LIMIT=$A3_LIMIT, outputs tagged ${SUFFIX}"
fi

for MODEL in $MODELS; do
  TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
  OUT="$ROOT/outputs/default_options/results/a3_dk_results_${TAG}${SUFFIX}.jsonl"
  MODEL="$MODEL" INPUT="$INPUT" OUT="$OUT" \
    ARMS="label_num,echo_plain" REPLICATE_FRAC=0.25 \
    sbatch --export=ALL --job-name="a3-${TAG##*_}" \
    --time=08:00:00 "$ROOT/scripts/cluster/run_score.sbatch"
done
