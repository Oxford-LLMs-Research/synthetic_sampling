#!/bin/bash
#SBATCH -J ladder-vllm
#SBATCH -p short
#SBATCH --array=0-3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# The ladder's 96-feature levels prefill far more than the profile-scaling run's
# did, and the echo path cannot reuse a prefix cache, so throughput fell to
# 0.9 inst/s and two of four shards were cut off at six hours. Ask for the
# partition maximum; a resume run should override this downwards on the command
# line, since backfill schedules short requests sooner.
#SBATCH --time=12:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/ladder_%A_%a.out
#SBATCH -e /data/polf-sula/nuff1496/logs/ladder_%A_%a.err

# Per-feature ladder, three arms: how the model's belief over the answer options
# moves as a profile grows, and whether it moves sooner when useful information
# arrives sooner.
#
# For each respondent-target pair the same question is scored at 0, 1, 2, 4, 8,
# 16, 32, 64, 96 features and once more with every feature the respondent
# answered. The feature SET is identical across arms; only the order differs:
#
#   informative  the oracle's most predictive first (per target x country cell)
#   random       a seeded permutation, the paper's own design
#   anti         the same ranking read from the bottom
#
# k=0 and the all-features level are shared across arms and emitted once, so a
# pair costs 26 prompts rather than 30. The all-features level is where the three
# arms coincide by construction, which supplies the analysis's own noise floor.
#
# Respondents come only from the oracle's reserved 20% split, so the informative
# arm is never scored on the people whose data chose its ranking.
#
# Profiles are built by the project's own RespondentProfileGenerator with the
# same MISSING_VALUE_PATTERNS the main dataset used, so the option sets match
# the paper's exactly rather than raw metadata's.
#
# Same serving parameters as every other run here. Do not tune them: two
# servings of this model agree on aggregate accuracy but differ by ~0.9 nats in
# mean option logprob and pick the same option only 64% of the time, so a ladder
# split across configurations would be uninterpretable.
#
# score_scaling.py resumes from --out, so requeue the array if a task hits the
# walltime; already-scored instances are skipped.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
SHARD=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
PORT=$((8200 + SLURM_ARRAY_TASK_ID))
LOG=$DATA/vllm_ladder_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

IN=$DATA/data/ladder_shard_${SHARD}.jsonl
OUT=$DATA/data/ladder_results_${SHARD}.jsonl

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

# The smoke test also exercises the k=0 prompts, which have an empty profile and
# are the one prompt shape this pipeline has never scored before.
echo "=== smoke (20 instances) ==="
SMOKE=$DATA/data/ladder_smoke_${SHARD}.jsonl
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
assert all(r.get("option_n_tokens") for r in rows), \
    "option_n_tokens missing: the cluster copy of score_scaling.py is stale"
print("smoke OK:", len(rows), "instances,", sum(r["correct"] for r in rows), "correct")
PY

echo "=== full shard $SHARD (resumes from $OUT) ==="
python "$DATA/scripts/scaling/score_scaling.py" \
  --input "$IN" --out "$OUT" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --workers 32

echo "=== done $(date) ==="
wc -l "$OUT"
