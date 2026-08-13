#!/bin/bash
#SBATCH --job-name=ss-c2
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C2 native thinking toggle (RUN_CATALOGUE C2, pre-registered 12 Aug): one
# serving generates the thinking transcripts (chat template, toggle ON),
# assembles the paired toggle set, and scores BOTH cells with thinking OFF
# (locked protocol: the reasoning is injected text, never a live-thinking
# shallow read). Same weights; generation uses a think-then-digit instruction
# (not the scoring "No reasoning" readout lock). Keep this file LF-only.
# Required env: none (Qwen3-32B is the toggle-bearing model). Optional:
# MODEL, TAG, ARMS, REPLICATE_FRAC, DTYPE, MAX_MODEL_LEN, PORT,
# VLLM_READY_TIMEOUT_SEC, C2_LIMIT.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
TAG="${TAG:-$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')}"
# Canary mode: C2_LIMIT=N caps generation and scoring and suffixes every
# output path so canary artifacts (a different serving) can never be
# resumed into the real run. Use C2_LIMIT >= 20 so 2 cells clear the
# 40-row smoke minimum.
C2_LIMIT="${C2_LIMIT:-}"
if [ -n "$C2_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
ARMS="${ARMS:-label_num,chat_label_num}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.25}"
CHAT_KWARGS='{"enable_thinking": false}'
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# Ladder-set resolution: explicit env wins; else the ARC staging location
# ($DATA/data/), else the local-workstation layout. The unset-var default
# has killed two submissions at the fail-fast check; never rely on it on ARC.
if [ -z "${LADDER_SET:-}" ]; then
  if [ -n "${DATA:-}" ] && [ -f "$DATA/data/ladder_readout_set.jsonl" ]; then
    LADDER_SET="$DATA/data/ladder_readout_set.jsonl"
  else
    LADDER_SET="$REPO/../outputs_recovered/ladder_readout_set.jsonl"
  fi
fi
TASKS="$REPO/outputs/narrative/inputs/narrative_tasks.jsonl"
GEN="$REPO/outputs/native_thinking/generated/thinking_${TAG}.jsonl"
INPUT="$REPO/outputs/native_thinking/inputs/c2_label_set_${TAG}.jsonl"
OUT="$REPO/outputs/native_thinking/results/c2_label_results_${TAG}.jsonl"

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
mkdir -p "$(dirname "$GEN")" "$(dirname "$INPUT")" "$(dirname "$OUT")" logs

echo "RUNSTAMP repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP arms=$ARMS replicate=$REPLICATE_FRAC input=$INPUT out=$OUT"
echo "RUNSTAMP c2 stage1: chat thinking ON, sampled t=0.6 top_p=0.95 top_k=20 seed=42 max_tokens=4096"
echo "RUNSTAMP c2 scoring: thinking OFF (chat_kwargs=$CHAT_KWARGS)"

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

# Stage 1: thinking transcripts (toggle ON), trace the first 5 verbatim.
python "$REPO/scripts/native_thinking/generate_thinking.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --ladder-set "$LADDER_SET" --tasks "$TASKS" \
  --out "$GEN" --trace 5 \
  ${C2_LIMIT:+--limit "$C2_LIMIT"}

# Stage 2: assemble the paired toggle set (sidecar is data).
python "$REPO/scripts/native_thinking/make_c2_set.py" \
  --transcripts "$GEN" --tasks "$TASKS" --ladder-set "$LADDER_SET" \
  --out "$INPUT"

# Canary gates beyond the standard smoke: think-block presence, closure,
# non-empty think content, stated-answer parse rate.
python "$REPO/scripts/native_thinking/check_c2_gates.py" "$INPUT" \
  || { echo "C2 GATES FAILED, aborting"; exit 1; }

# Smoke gate spanning the design, then the full scoring run (thinking OFF).
SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
if [ -n "$C2_LIMIT" ]; then
  cp "$INPUT" "$SMOKE_IN"
else
  awk 'NR % 30 == 1' "$INPUT" > "$SMOKE_IN"
fi
ss-score --input "$SMOKE_IN" --out "$SMOKE_OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --arms "$ARMS" \
  --replicate-frac 0 --workers 16 \
  --chat-template-kwargs "$CHAT_KWARGS"
ss-analyze smoke "$SMOKE_OUT" || { echo "SMOKE FAILED, aborting"; exit 1; }

ss-score \
  --input "$INPUT" \
  --out "$OUT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --arms "$ARMS" \
  --replicate-frac "$REPLICATE_FRAC" \
  --workers 32 \
  --chat-template-kwargs "$CHAT_KWARGS"

ss-analyze coverage "$OUT" "$INPUT" || true
echo "done"
