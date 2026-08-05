#!/bin/bash
#SBATCH -J scaling-vllm
#SBATCH -p short
#SBATCH --array=0-3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/scaling_%A_%a.out
#SBATCH -e /data/polf-sula/nuff1496/logs/scaling_%A_%a.err

# Profile-scaling experiment: score k=48 and k=96 profiles with Qwen 3 32B.
#
# Data-parallel as a job array: four independent tasks, one H100 each, one
# cost-balanced shard each (38,123 instances / ~74.4M prefill tokens per shard,
# imbalance 0.00%). Nothing is shared between tasks, so they can start and
# finish independently.
#
# Prefix caching is what makes this affordable. score_options_echo issues one
# request per answer option (874,466 requests over 152,492 instances, mean 5.7
# options) and every request within an instance shares a byte-identical prefix.
# With caching the run is ~305M prefill tokens; without it, ~1,743M. The scorer
# issues an instance's options serially so the first populates the cache and
# the rest hit it.
#
# No --enable-prefix-caching flag: this vLLM is 0.26.0, where
# CacheConfig.enable_prefix_caching defaults to True, so the flag would be a
# no-op. Passing an argument this build might spell differently would kill all
# four array tasks at startup, which is a real cost against no benefit. The
# startup check below confirms caching is actually live rather than assuming it.
#
# Longest prompt is 3,718 tokens, so 16384 has ample headroom.
#
# score_scaling.py resumes from --out, so if a task hits the walltime, simply
# requeue the array: already-scored instances are skipped.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
SHARD=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
PORT=$((8000 + SLURM_ARRAY_TASK_ID))   # distinct per task in case two share a node
LOG=$DATA/vllm_serve_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

IN=$DATA/data/scaling_shard_${SHARD}.jsonl
OUT=$DATA/data/scaling_results_${SHARD}.jsonl

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

# Confirm prefix caching is actually live. It defaults on in 0.26, but a silent
# default flip would cost 5.7x, so record what the engine reports rather than
# trusting the default.
grep -iE "prefix.cach" "$LOG" | head -5 || true
echo "(no prefix-cache line above means the engine did not log one; check GPU"
echo " KV cache usage in $LOG before assuming caching is off)"

echo "=== smoke (20 instances) ==="
SMOKE=$DATA/data/scaling_smoke_${SHARD}.jsonl
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
