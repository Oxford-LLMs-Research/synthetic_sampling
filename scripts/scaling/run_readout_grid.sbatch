#!/bin/bash
#SBATCH -J readout-grid
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/grid_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/grid_%j.err

# The readout experiment extended to the model grid: is "the elicitation is a
# factor of the measurement" a property of three models or of the class?
#
# Arms per model (override via env ARMS=...):
#   echo_plain         the paper's exact condition, fresh same-serving control
#   echo_listed        options listed, same scoring rule
#   label_num          numbered readout, cyclic Latin square
#   label_num_natural  numbered, natural order
#   echo_qonly         PMI neutral premise: question, empty profile
#   echo_ctxfree       PMI neutral premise: bare "Answer:"
# plus a 25% replicate. Generate arms are dropped (57.7% self-agreement).
#
# Usage:  sbatch [--gres=gpu:h100:N --time=HH:MM:SS] run_readout_grid.sbatch <HF model id> [TP]
#   TP defaults to 1 and must match the gpu count in --gres.
# Model identities and substitutions: docs/MODEL_CHECKPOINTS.md in the paper
# workspace. Base models: export READOUT_MAX_LABEL_MISS=1.0 to demote the
# label-miss guard to advisory; a base model that cannot emit a bare digit is a
# FINDING to record, not a reason to lose the echo arms.
#
# Output family is readout_grid_* (NEW name): score_formats.py resumes by
# example_id, so pointing --out at an existing readout_results_* file would
# mark everything done and score nothing.

set -euo pipefail

DATA=/data/polf-sula/nuff1496

MODEL=${1:?usage: sbatch run_readout_grid.sbatch model [tensor-parallel]}
TP=${2:-1}
TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
PORT=$((8400 + SLURM_JOB_ID % 200))
LOG=$DATA/vllm_grid_${SLURM_JOB_ID}.log

IN=$DATA/data/readout_set.jsonl
OUT=$DATA/data/readout_grid_${TAG}.jsonl
ARMS=${ARMS:-echo_plain,echo_listed,label_num,label_num_natural,echo_qonly,echo_ctxfree}

mkdir -p "$DATA/logs" "$DATA/data"

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/vllm-env/bin/activate"

export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export NEBIUS_API_KEY=dummy

echo "=== $(date) node=$(hostname) job=$SLURM_JOB_ID model=$MODEL tp=$TP arms=$ARMS ==="
nvidia-smi -L
test -s "$IN" || { echo "FATAL: input missing or empty: $IN"; exit 1; }
wc -l "$IN"

# Same serving parameters as every other run. One exception is forced by the
# checkpoint itself: gpt-oss ships MXFP4 weights and refuses --dtype bfloat16,
# so that flag is omitted there and vLLM keeps the native dtype. Recorded in
# docs/MODEL_CHECKPOINTS.md.
DTYPE_ARGS="--dtype bfloat16"
case "$MODEL" in
  *gpt-oss*) DTYPE_ARGS="" ;;
esac

HF_HUB_OFFLINE=1 vllm serve "$MODEL" \
  $DTYPE_ARGS --max-model-len 16384 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.92 --port "$PORT" \
  > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM pid=$VLLM_PID log=$LOG"

cleanup() {
  echo "=== shutting down vLLM pid=$VLLM_PID ==="
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for vLLM on :$PORT ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM ready after ~$((i * 20))s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died during startup; last 80 log lines:"; tail -n 80 "$LOG" || true; exit 1
  fi
  sleep 20
done

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "vLLM did not become ready in time; last 80 log lines:"
  tail -n 80 "$LOG" || true; exit 1
fi

STATUS=$DATA/data/readout_grid_STATUS_${TAG}.txt
: > "$STATUS"
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }
note "job $SLURM_JOB_ID on $(hostname), model $MODEL, tp $TP, arms $ARMS"

echo "=== smoke (one respondent from each question) ==="
SMOKE=$DATA/data/grid_smoke_${TAG}.jsonl
MINI=$DATA/data/grid_smoke_in_${TAG}.jsonl
rm -f "$SMOKE"
awk 'NR % 100 == 1' "$IN" > "$MINI"
wc -l "$MINI"
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$MINI" --out "$SMOKE" --arms "$ARMS" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --workers 16 --replicate-frac 1.0
if ! python "$DATA/scripts/scaling/check_readout_smoke.py" "$SMOKE" >> "$STATUS" 2>&1
then
  note "FAILED at smoke test, see $STATUS and $LOG"
  exit 1
fi
note "smoke passed"

echo "=== full run ==="
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$IN" --out "$OUT" --arms "$ARMS" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --workers 32 --replicate-frac 0.25

python "$DATA/scripts/scaling/grid_coverage_report.py" "$OUT" "$IN" >> "$STATUS" 2>&1

note "run finished, $(wc -l < "$OUT") instances"
echo "=== $(date) done ==="
cat "$STATUS"
