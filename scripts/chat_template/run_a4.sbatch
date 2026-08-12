#!/bin/bash
#SBATCH --job-name=ss-a4
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# A4 chat-template validation (RUN_CATALOGUE A4 expanded, pre-registered
# 12 Aug): the raw-completion arms and the chat_label_num arm score the SAME
# instances in ONE serving, so template-vs-raw is a paired within-serving
# contrast, and the presentation contrasts (qa vs narrative) replicate — or
# fail to — under the model's own tuned format. Keep this file LF-only
# (never Path.write_text on Windows). Required env: MODEL, TAG. Optional:
# ARMS, REPLICATE_FRAC, DTYPE, MAX_MODEL_LEN, PORT, VLLM_READY_TIMEOUT_SEC,
# CHAT_KWARGS (JSON forwarded to the chat template, e.g.
# '{"enable_thinking": false}' on Qwen3-32B whose template defaults to
# thinking ON — A4 is the non-reasoning arm).

set -euo pipefail

MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
# Canary mode: A4_LIMIT=N caps the scoring at N instances and suffixes the
# output paths so canary scores (a different serving) can never be resumed
# into the real run.
A4_LIMIT="${A4_LIMIT:-}"
if [ -n "$A4_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
ARMS="${ARMS:-label_num,chat_label_num,echo_plain,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.25}"
CHAT_KWARGS="${CHAT_KWARGS:-}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT="$REPO/outputs/narrative/inputs/narrative_label_set.jsonl"
OUT="$REPO/outputs/chat_template/results/a4_label_results_${TAG}.jsonl"

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

if [ ! -f "$INPUT" ]; then
  echo "FATAL missing input $INPUT"; exit 1
fi
mkdir -p "$(dirname "$OUT")" logs

echo "RUNSTAMP repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP arms=$ARMS replicate=$REPLICATE_FRAC input=$INPUT out=$OUT"
echo "RUNSTAMP a4 chat_kwargs=${CHAT_KWARGS:-none}"

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

# Smoke gate spanning the design, then the full scoring run. In canary mode
# the capped set is tiny, so the smoke covers all of it (the gate needs
# >= 40 rows; use A4_LIMIT >= 40).
SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
if [ -n "$A4_LIMIT" ]; then
  head -n "$A4_LIMIT" "$INPUT" > "$SMOKE_IN"
else
  awk 'NR % 50 == 1' "$INPUT" > "$SMOKE_IN"
fi
ss-score --input "$SMOKE_IN" --out "$SMOKE_OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --arms "$ARMS" \
  --replicate-frac 0 --workers 16 \
  ${CHAT_KWARGS:+--chat-template-kwargs "$CHAT_KWARGS"}
ss-analyze smoke "$SMOKE_OUT" || { echo "SMOKE FAILED, aborting"; exit 1; }

ss-score \
  --input "$INPUT" \
  --out "$OUT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --arms "$ARMS" \
  --replicate-frac "$REPLICATE_FRAC" \
  --workers 32 \
  ${CHAT_KWARGS:+--chat-template-kwargs "$CHAT_KWARGS"} \
  ${A4_LIMIT:+--limit "$A4_LIMIT"}

ss-analyze coverage "$OUT" "$INPUT" || true
echo "done"
