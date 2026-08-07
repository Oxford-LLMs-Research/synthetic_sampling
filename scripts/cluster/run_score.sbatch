#!/bin/bash
#SBATCH --job-name=ss-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Minimal-core scoring job. Keep this file LF-only (never Path.write_text on Windows).
# Required env: MODEL, INPUT, OUT, optional BASE_URL / ARMS / REPLICATE_FRAC.

set -euo pipefail

MODEL="${MODEL:?set MODEL}"
INPUT="${INPUT:?set INPUT}"
OUT="${OUT:?set OUT}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
ARMS="${ARMS:-label_num,echo_plain,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.1}"
PORT="${PORT:-8000}"

mkdir -p "$(dirname "$OUT")" logs

echo "Starting vLLM for $MODEL on port $PORT"
vllm serve "$MODEL" --port "$PORT" --dtype auto &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Wait for /v1/models
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 5
done

ss-score \
  --input "$INPUT" \
  --out "$OUT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --arms "$ARMS" \
  --replicate-frac "$REPLICATE_FRAC" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${SHARD_INDEX:+--shard-index "$SHARD_INDEX"} \
  ${SHARD_COUNT:+--shard-count "$SHARD_COUNT"}

ss-analyze coverage "$OUT" "$INPUT" || true
