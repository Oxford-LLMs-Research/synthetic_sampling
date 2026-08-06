#!/bin/bash
#SBATCH -J ladder-readout
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/ladder_readout_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/ladder_readout_%j.err

# Does the model's ceiling rise under INFORMATIVE features once the readout is
# fixed? The paraphrase and PMI results scoped the central negative claim to
# randomly drawn profiles; this run crosses the ladder with the elicitation.
#
# Input: ladder_readout_set.jsonl, 1,504 respondent-target pairs, the ladder's
# own verified feature sequences, rungs k=0/1/8/24/48/96 (24 = s6m4, the
# paper's operating point) + full endpoint, informative and random orderings.
# Arms: echo_plain (paper's rule, fresh control), echo_listed, label_num,
# echo_ctxfree (cached; gives pmi_free per rung; k=0 doubles as the qonly
# premise). ~390k requests + 25% replicate; ~5 h scoring after load.
#
# Fresh OUT name family; score_formats.py resumes by example_id.

set -euo pipefail

DATA=/data/polf-sula/nuff1496

MODEL=${1:-Qwen/Qwen3-32B}
TP=${2:-1}
TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
PORT=$((8400 + SLURM_JOB_ID % 200))
LOG=$DATA/vllm_ladder_readout_${SLURM_JOB_ID}.log

IN=$DATA/data/ladder_readout_set.jsonl
OUT=$DATA/data/ladder_readout_results_${TAG}.jsonl
ARMS=echo_plain,echo_listed,label_num,echo_ctxfree

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

STATUS=$DATA/data/ladder_readout_STATUS_${TAG}.txt
: > "$STATUS"
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }
note "job $SLURM_JOB_ID on $(hostname), model $MODEL, arms $ARMS"

# The input is grouped ~12 lines per pair, sorted by pair; every 300th line
# spans pairs, targets, rungs and orderings. ~60 smoke rows.
echo "=== smoke ==="
SMOKE=$DATA/data/ladder_readout_smoke_${TAG}.jsonl
MINI=$DATA/data/ladder_readout_smoke_in_${TAG}.jsonl
rm -f "$SMOKE"
awk 'NR % 300 == 1' "$IN" > "$MINI"
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
