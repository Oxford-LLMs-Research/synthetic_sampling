#!/bin/bash
#SBATCH --job-name=ss-c3-k10
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C3 Thinking K=10 native generations (calibration / aggregate shares).
# Draw 1 is the existing seed-42 file (do NOT overwrite it). This job
# loads vLLM once and writes draws 2-10 to separate jsonl files
# (seed = 41+k, so k=2 -> 43, k=10 -> 51). Generate only: no assemble,
# no hybrid label_num score. Resume is per-file by example_id.
# Optional: MODEL, TAG, C3_LIMIT, DTYPE, MAX_MODEL_LEN, PORT,
# VLLM_READY_TIMEOUT_SEC. Keep LF-only.
#
#   sbatch --export=ALL --job-name=c3-think-k10 scripts/native_thinking/run_c3_thinking_k10.sbatch

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
TAG="${TAG:-$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')}"
C3_LIMIT="${C3_LIMIT:-}"
if [ -n "$C3_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -z "${LADDER_SET:-}" ]; then
  if [ -n "${DATA:-}" ] && [ -f "$DATA/data/ladder_readout_set.jsonl" ]; then
    LADDER_SET="$DATA/data/ladder_readout_set.jsonl"
  else
    LADDER_SET="$REPO/../outputs_recovered/ladder_readout_set.jsonl"
  fi
fi
TASKS="$REPO/outputs/narrative/inputs/narrative_tasks.jsonl"
GENDIR="$REPO/outputs/native_thinking/generated"
K1="$GENDIR/thinking_${TAG}.jsonl"

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
mkdir -p "$GENDIR" logs

echo "RUNSTAMP repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP c3 thinking K=10 generate-only: draw1=$K1 (seed 42, not rewritten)"
echo "RUNSTAMP draws 2-10 -> ${GENDIR}/thinking_${TAG}_kNN.jsonl seeds 43-51 max_tokens=8192"

if [ -z "$C3_LIMIT" ]; then
  if [ ! -f "$K1" ]; then
    echo "FATAL draw-1 file missing: $K1"; exit 1
  fi
  n1=$(wc -l < "$K1")
  echo "RUNSTAMP draw1_lines=$n1"
fi

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "FATAL port $PORT on $(hostname) is already serving; refusing to attach"
  exit 1
fi

echo "Starting vLLM for $MODEL on port $PORT (dtype $DTYPE)"
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
  > "$GENDIR/vllm_serve_${SLURM_JOB_ID:-local}.log" 2>&1 &
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

# k=2..10, seed=41+k (43..51). Per-file resume: a rerun skips finished draws.
for k in $(seq 2 10); do
  seed=$((41 + k))
  kk=$(printf '%02d' "$k")
  out="$GENDIR/thinking_${TAG}_k${kk}.jsonl"
  echo "RUNSTAMP draw=$k seed=$seed out=$out"
  python "$REPO/scripts/native_thinking/generate_thinking.py" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --ladder-set "$LADDER_SET" --tasks "$TASKS" \
    --out "$out" --seed "$seed" --trace 0 \
    --chat-template-kwargs "" --max-tokens 8192 \
    ${C3_LIMIT:+--limit "$C3_LIMIT"}
  echo "draw $k done lines=$(wc -l < "$out")"
done

echo "done"
