#!/bin/bash
#SBATCH --job-name=ss-c3-inst
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C3 Instruct sibling (RUN_CATALOGUE C3): fresh direct scores on
# Qwen3-30B-A3B-Instruct-2507 over the same 734-pair substrate. Do NOT
# reuse A4/A6 numbers (reuse rule). Keep LF-only.
# Optional: MODEL, TAG, C3_LIMIT (>=40 for canary smoke), ARMS, REPLICATE_FRAC,
# DTYPE, MAX_MODEL_LEN, PORT, VLLM_READY_TIMEOUT_SEC.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TAG="${TAG:-$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')}"
C3_LIMIT="${C3_LIMIT:-}"
if [ -n "$C3_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
ARMS="${ARMS:-label_num,chat_label_num}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.25}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
LADDER_SET="${LADDER_SET:-$REPO/../outputs_recovered/ladder_readout_set.jsonl}"
TASKS="$REPO/outputs/narrative/inputs/narrative_tasks.jsonl"
INPUT="$REPO/outputs/thinking/inputs/c3_direct_set_${TAG}.jsonl"
OUT="$REPO/outputs/thinking/results/c3_direct_results_${TAG}.jsonl"

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

for f in "$LADDER_SET" "$TASKS"; do
  if [ ! -f "$f" ]; then
    echo "FATAL missing input $f"; exit 1
  fi
done
mkdir -p "$(dirname "$INPUT")" "$(dirname "$OUT")" logs

echo "RUNSTAMP repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP arms=$ARMS replicate=$REPLICATE_FRAC input=$INPUT out=$OUT"
echo "RUNSTAMP c3 instruct: direct cells, fresh scores"

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "FATAL port $PORT on $(hostname) is already serving; refusing to attach"
  exit 1
fi

echo "Starting vLLM for $MODEL on port $PORT (dtype $DTYPE)"
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
  > "$(dirname "$OUT")/vllm_serve_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

elapsed=0
while ! curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -qF "$MODEL"; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM died during startup, see its log"; exit 1
  fi
  if [ -n "${VLLM_READY_TIMEOUT_SEC:-}" ] && [ "$elapsed" -ge "$VLLM_READY_TIMEOUT_SEC" ]; then
    echo "vLLM never came up within ${VLLM_READY_TIMEOUT_SEC}s"; exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

python "$REPO/scripts/thinking/make_c3_direct_set.py" \
  --tasks "$TASKS" --ladder-set "$LADDER_SET" \
  --out "$INPUT" \
  ${C3_LIMIT:+--limit "$C3_LIMIT"}

SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
if [ -n "$C3_LIMIT" ]; then
  cp "$INPUT" "$SMOKE_IN"
else
  awk 'NR % 30 == 1' "$INPUT" > "$SMOKE_IN"
fi
# Instruct-2507 is non-thinking; no chat_template_kwargs.
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
  --workers 32

ss-analyze coverage "$OUT" "$INPUT" || true
echo "done"
