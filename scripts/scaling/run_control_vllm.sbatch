#!/bin/bash
#SBATCH -J scaling-control
#SBATCH -p short
#SBATCH --array=0-3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/control_%A_%a.out
#SBATCH -e /data/polf-sula/nuff1496/logs/control_%A_%a.err

# Backend control for the profile-scaling experiment.
#
# The scaling curve's k = 6, 12 and 24 points come from the main experiment and
# k = 48, 96 from this cluster. The local exponent per doubling is 0.233, 0.339,
# 0.545, 0.316: flat on either side of 24->48 and anomalous exactly at it, which
# is where the serving stack changes. So the observed steepening is confounded
# with the backend.
#
# This re-scores the main run's OWN k=24 instances here. Same respondents,
# targets, features, prompt and scorer; only the backend differs. Comparing the
# result to the main run's k=24 number separates the two explanations.
#
# 15,217 instances over four shards, roughly a third the per-shard work of the
# scaling run at half the prompt length, so ~1h of scoring plus startup. The
# 3h walltime is deliberate slack; resume works if it is hit anyway.
#
# Identical to run_scaling_vllm.sbatch in every serving parameter. Do not tune
# anything here: matching the scaling run's engine settings is the whole point.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
SHARD=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
PORT=$((8100 + SLURM_ARRAY_TASK_ID))
LOG=$DATA/vllm_control_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

IN=$DATA/data/control_shard_${SHARD}.jsonl
OUT=$DATA/data/control_results_${SHARD}.jsonl

mkdir -p "$DATA/logs" "$DATA/data"

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/vllm-env/bin/activate"

export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export NEBIUS_API_KEY=dummy

echo "=== $(date) node=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID ==="
nvidia-smi -L
echo "shard=$SHARD port=$PORT"
echo "in=$IN"
test -s "$IN" || { echo "FATAL: shard file missing or empty: $IN"; exit 1; }
wc -l "$IN"

HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3-32B \
  --dtype bfloat16 --max-model-len 16384 \
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

grep -iE "prefix.cach" "$LOG" | head -5 || true

echo "=== smoke (20 instances) ==="
SMOKE=$DATA/data/control_smoke_${SHARD}.jsonl
rm -f "$SMOKE"
python "$DATA/scripts/scaling/score_scaling.py" \
  --input "$IN" --out "$SMOKE" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --workers 32 --limit 20
python - "$SMOKE" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
assert len(rows) == 20, f"expected 20 scored, got {len(rows)}"
assert all(r["option_logprobs"] for r in rows), "empty option_logprobs"
assert all(r["predicted"] in r["options"] for r in rows), "prediction not among options"
assert all(-30 < min(r["option_logprobs"].values()) < 0 for r in rows), \
    "logprobs out of plausible range for a MEAN token logprob"
print("smoke OK:", len(rows), "instances,",
      sum(r["correct"] for r in rows), "correct")
PY

echo "=== full shard $SHARD (resumes from $OUT) ==="
python "$DATA/scripts/scaling/score_scaling.py" \
  --input "$IN" --out "$OUT" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --workers 32

echo "=== done $(date) ==="
wc -l "$OUT"
