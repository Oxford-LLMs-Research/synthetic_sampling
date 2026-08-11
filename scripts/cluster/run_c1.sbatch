#!/bin/bash
#SBATCH --job-name=ss-c1
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C1 reason-then-answer job (RUN_CATALOGUE C1, pre-registered 8 Aug): one
# serving per model runs BOTH stages and the scoring, so every cell of the
# presentation x elicitation 2x2 (qa / reasoned / narrative_direct /
# narrative_reasoned) is paired within one serving. Never split stages or
# arms across jobs. Keep this file LF-only (never Path.write_text on
# Windows). Required env: MODEL, TAG. Optional: ARMS, REPLICATE_FRAC, DTYPE,
# MAX_MODEL_LEN, PORT, VLLM_READY_TIMEOUT_SEC, LADDER_SET.

set -euo pipefail

MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
# Canary mode: C1_LIMIT=N caps both generation stages and the scoring at N
# instances, and suffixes every output path so canary transcripts (a
# different serving) can never be resumed into the real run.
C1_LIMIT="${C1_LIMIT:-}"
if [ -n "$C1_LIMIT" ]; then
  TAG="${TAG}_canary"
fi
ARMS="${ARMS:-label_num,echo_plain,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.25}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
LADDER_SET="${LADDER_SET:-$REPO/../outputs_recovered/ladder_readout_set.jsonl}"
NARR_SET="$REPO/outputs/narrative/inputs/narrative_label_set.jsonl"
GEN_DIR="$REPO/outputs/reasoning/generated"
QA_TRANS="$GEN_DIR/reasoning_${TAG}.jsonl"
NARR_TRANS="$GEN_DIR/reasoning_narr1_${TAG}.jsonl"
INPUT="$REPO/outputs/reasoning/inputs/c1_label_set_${TAG}.jsonl"
OUT="$REPO/outputs/reasoning/results/c1_label_results_${TAG}.jsonl"

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

for f in "$LADDER_SET" "$NARR_SET"; do
  if [ ! -f "$f" ]; then
    echo "FATAL missing input $f"; exit 1
  fi
done
mkdir -p "$GEN_DIR" "$(dirname "$INPUT")" "$(dirname "$OUT")" logs

echo "RUNSTAMP repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP arms=$ARMS replicate=$REPLICATE_FRAC input=$INPUT out=$OUT"
echo "RUNSTAMP c1 stage1: sampled t=0.6 top_p=0.95 top_k=20 seed=42 max_tokens=2048"

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

# Stage 1a: reasoning transcripts over the qa substrate (734 pairs).
python "$REPO/scripts/reasoning/generate_reasoning.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --ladder-set "$LADDER_SET" --out "$QA_TRANS" \
  ${C1_LIMIT:+--limit "$C1_LIMIT"}

# Stage 1b: reasoning transcripts over the validated narrative1 arm (723).
python "$REPO/scripts/reasoning/generate_reasoning.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --input "$NARR_SET" --input-arm-label narrative1 \
  --out "$NARR_TRANS" \
  ${C1_LIMIT:+--limit "$C1_LIMIT"}

# Stage 2: assemble the 2x2 scoring set (parse sidecar is data).
python "$REPO/scripts/reasoning/make_reasoned_set.py" \
  --transcripts "$QA_TRANS" \
  --narrative-transcripts "$NARR_TRANS" --narrative-set "$NARR_SET" \
  --ladder-set "$LADDER_SET" --out "$INPUT"

# Smoke gate spanning the design, then the full scoring run. In canary mode
# the assembled set is tiny, so the smoke covers all of it (the gate needs
# >= 40 rows; use C1_LIMIT >= 10 so 4 arms x limit clears that).
SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
if [ -n "$C1_LIMIT" ]; then
  cp "$INPUT" "$SMOKE_IN"
else
  awk 'NR % 50 == 1' "$INPUT" > "$SMOKE_IN"
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
  --workers 32 \
  ${C1_LIMIT:+--limit "$C1_LIMIT"}

ss-analyze coverage "$OUT" "$INPUT" || true
echo "done"
