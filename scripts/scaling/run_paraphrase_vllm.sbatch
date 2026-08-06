#!/bin/bash
#SBATCH -J paraphrase-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/paraphrase_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/paraphrase_%j.err

# Experiment A, finally: is the wording instability in the SCORING RULE or in
# the MODEL? Validated meaning-preserving paraphrases of the answer options
# (blind round-trip 55/56; two source-hygiene exclusions; provenance in
# outputs/scaling_experiment/paraphrase/) scored against the originals under
# BOTH readouts, in one serving:
#
#   echo_plain   the paper's rule. Historical result: paraphrase flips ~63%
#                of predictions (measured with an invalid manipulation; this
#                run replaces that number with a valid one)
#   label_num    the numbered readout. Falsifiable prediction: if the fluency
#                story is right, paraphrase sensitivity collapses here toward
#                the replicate ceiling
#
# ~100k requests: 4,699 instances x 2 option sets x 2 arms + 25% replicate.
# Fresh OUT name family (paraphrase_results_*): resume is by example_id.

set -euo pipefail

DATA=/data/polf-sula/nuff1496

MODEL=${1:-Qwen/Qwen3-32B}
TP=${2:-1}
TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
PORT=$((8400 + SLURM_JOB_ID % 200))
LOG=$DATA/vllm_paraphrase_${SLURM_JOB_ID}.log

IN=$DATA/data/paraphrase_set.jsonl
OUT=$DATA/data/paraphrase_results_${TAG}.jsonl
ARMS=echo_plain,label_num

mkdir -p "$DATA/logs" "$DATA/data"

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/vllm-env/bin/activate"

export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export NEBIUS_API_KEY=dummy

echo "=== $(date) node=$(hostname) job=$SLURM_JOB_ID model=$MODEL arms=$ARMS ==="
nvidia-smi -L
test -s "$IN" || { echo "FATAL: input missing or empty: $IN"; exit 1; }
wc -l "$IN"

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

STATUS=$DATA/data/paraphrase_STATUS_${TAG}.txt
: > "$STATUS"
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }
note "job $SLURM_JOB_ID on $(hostname), model $MODEL, arms $ARMS"

# paraphrase_set.jsonl preserves the readout set's 100-per-question grouping
# (minus the two exclusions), so every 100th line still spans questions.
echo "=== smoke (one respondent from ~each question) ==="
SMOKE=$DATA/data/paraphrase_smoke_${TAG}.jsonl
MINI=$DATA/data/paraphrase_smoke_in_${TAG}.jsonl
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
