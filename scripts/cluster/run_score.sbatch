#!/bin/bash
#SBATCH --job-name=ss-score
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Minimal-core scoring job, ARC HTC pattern (docs/CLUSTER_HANDOFF.md in the
# paper workspace): wake env, serve vLLM offline, gate on a design-spanning
# smoke, then run. Resume-safe: the runner skips example_ids already in OUT.
# Keep this file LF-only (never Path.write_text on Windows).
# Required env: MODEL, INPUT, OUT. Optional: ARMS, REPLICATE_FRAC, DTYPE,
# MAX_MODEL_LEN, LIMIT, SHARD_INDEX, SHARD_COUNT, PORT, VLLM_READY_TIMEOUT_SEC.

set -euo pipefail

MODEL="${MODEL:?set MODEL}"
INPUT="${INPUT:?set INPUT}"
OUT="${OUT:?set OUT}"
ARMS="${ARMS:-label_num,echo_plain,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# Re-wake the ARC environment (guarded, so the script still runs elsewhere).
if command -v module >/dev/null 2>&1; then
  module load Python/3.11.5-GCCcore-13.2.0
fi
if [ -n "${DATA:-}" ] && [ -f "$DATA/vllm-env/bin/activate" ]; then
  source "$DATA/vllm-env/bin/activate"
  export HF_HOME="$DATA/hf_cache"
fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$(dirname "$OUT")" logs

echo "Starting vLLM for $MODEL on port $PORT (dtype $DTYPE)"
vllm serve "$MODEL" --port "$PORT" --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
  > "$(dirname "$OUT")/vllm_serve_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Wait until ready, or until vLLM dies. Slurm --time is the real backstop.
# Optional VLLM_READY_TIMEOUT_SEC caps the wait (unset = no curl timeout).
elapsed=0
while ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM died during startup, see its log"; exit 1
  fi
  if [ -n "${VLLM_READY_TIMEOUT_SEC:-}" ] && [ "$elapsed" -ge "$VLLM_READY_TIMEOUT_SEC" ]; then
    echo "vLLM never came up within ${VLLM_READY_TIMEOUT_SEC}s"; exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

# Smoke gate spanning the design, not just its head.
SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
awk 'NR % 50 == 1' "$INPUT" > "$SMOKE_IN"
ss-score --input "$SMOKE_IN" --out "$SMOKE_OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --arms "$ARMS" \
  --replicate-frac 0 --workers 16
ss-analyze smoke "$SMOKE_OUT" || { echo "SMOKE FAILED, aborting"; exit 1; }

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
echo "done"
