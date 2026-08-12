#!/bin/bash
#SBATCH --job-name=ss-c3-think
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C3 Thinking checkpoint (RUN_CATALOGUE C3): generate native think transcripts
# on Qwen3-30B-A3B-Thinking-2507, assemble thinking-only instances (injected
# think content, no stated answer), gates, then score with label_num via
# /completions. Chat shallow readout is NOT used here: Thinking-2507 cannot
# disable thinking, so chat_label_num would re-open a think block. The
# Instruct sibling is a separate job (run_c3_instruct.sbatch). Keep LF-only.
# Optional: MODEL, TAG, C3_LIMIT (>=40 for canary smoke), ARMS, REPLICATE_FRAC,
# DTYPE, MAX_MODEL_LEN, PORT, VLLM_READY_TIMEOUT_SEC.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
TAG="${TAG:-$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '_')}"
C3_LIMIT="${C3_LIMIT:-}"
if [ -n "$C3_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
# Completions readouts only — see header. echo_plain rides as the
# paper-method / serving control, as on every other experiment.
ARMS="${ARMS:-label_num,echo_plain}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.25}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# Ladder-set resolution: explicit env wins; else the ARC staging location
# ($DATA/data/), else the local-workstation layout.
if [ -z "${LADDER_SET:-}" ]; then
  if [ -n "${DATA:-}" ] && [ -f "$DATA/data/ladder_readout_set.jsonl" ]; then
    LADDER_SET="$DATA/data/ladder_readout_set.jsonl"
  else
    LADDER_SET="$REPO/../outputs_recovered/ladder_readout_set.jsonl"
  fi
fi
TASKS="$REPO/outputs/narrative/inputs/narrative_tasks.jsonl"
GEN="$REPO/outputs/thinking/generated/thinking_${TAG}.jsonl"
INPUT="$REPO/outputs/thinking/inputs/c3_thinking_set_${TAG}.jsonl"
OUT="$REPO/outputs/thinking/results/c3_thinking_results_${TAG}.jsonl"

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
echo "RUNSTAMP c3 thinking: generate (no chat_template_kwargs; always-on, max_tokens=8192) then score via /completions"

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

# Empty --chat-template-kwargs: Thinking-2507 is always-on; the kwarg is a
# no-op. max_tokens 8192: this checkpoint thinks LONG, and a generation cut
# inside its (pre-opened) think block is the shape the assembler has to
# rescue via finish_reason — keep it rare, not common.
python "$REPO/scripts/thinking/generate_thinking.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --ladder-set "$LADDER_SET" --tasks "$TASKS" \
  --out "$GEN" --trace 5 --chat-template-kwargs "" \
  --max-tokens 8192 \
  ${C3_LIMIT:+--limit "$C3_LIMIT"}

python "$REPO/scripts/thinking/make_c3_set.py" \
  --transcripts "$GEN" --tasks "$TASKS" --ladder-set "$LADDER_SET" \
  --out "$INPUT"

python "$REPO/scripts/thinking/check_c2_gates.py" "$INPUT" \
  || { echo "C3 GATES FAILED, aborting"; exit 1; }

SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
if [ -n "$C3_LIMIT" ]; then
  cp "$INPUT" "$SMOKE_IN"
else
  awk 'NR % 30 == 1' "$INPUT" > "$SMOKE_IN"
fi
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
