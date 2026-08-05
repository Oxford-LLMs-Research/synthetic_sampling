#!/bin/bash
#SBATCH -J country-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/country_vllm_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/country_vllm_%j.err

# Country-injection experiment, three paired conditions per instance:
#   baseline              profile as sampled, country never stated
#   with_country          + "In which country do you live?" -> true country
#   with_country_placebo  same item, a country from a different region
#
# The placebo is the point. Naming any country appends a profile item and
# signals that location matters, and the temporal run showed that kind of
# effect can account for an entire apparent gain. true - placebo is the only
# contrast that isolates the country being the right one.
#
# 2,850 instances x 3 conditions x ~5.6 options = ~47.7k echo requests, the
# same size as the temporal job that finished inside this walltime.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
LOG=$DATA/vllm_serve_${SLURM_JOB_ID}.log
PORT=8000

mkdir -p "$DATA/logs" "$DATA/data"

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/vllm-env/bin/activate"

export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export NEBIUS_API_KEY=dummy

echo "=== $(date) node=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi -L

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
    echo "vLLM died during startup; last 80 log lines:"
    tail -n 80 "$LOG" || true
    exit 1
  fi
  sleep 20
done

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "vLLM did not become ready in time; last 80 log lines:"
  tail -n 80 "$LOG" || true
  exit 1
fi

echo "=== probe ==="
python "$DATA/scripts/nebius_probe.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model Qwen/Qwen3-32B

# Smoke test first: 20 instances x 3 conditions into a scratch file. If the
# placebo condition were missing or misnamed this fails here rather than after
# eleven hours of GPU time.
echo "=== smoke (20 instances) ==="
rm -f "$DATA/data/country_results_smoke.jsonl"
python "$DATA/scripts/nebius_run_experiments.py" --experiment country \
  --input "$DATA/data/country_injection_instances.jsonl" \
  --out  "$DATA/data/country_results_smoke.jsonl" \
  --scoring echo \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model Qwen/Qwen3-32B \
  --workers 32 --limit 20
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("/data/polf-sula/nuff1496/data/country_results_smoke.jsonl")]
c = collections.Counter(r["condition"] for r in rows)
print("smoke conditions:", dict(c))
assert set(c) == {"baseline", "with_country", "with_country_placebo"}, c
assert min(c.values()) == max(c.values()) == 20, c
print("smoke OK")
PY

echo "=== full country run (echo; resumes from --out) ==="
python "$DATA/scripts/nebius_run_experiments.py" --experiment country \
  --input "$DATA/data/country_injection_instances.jsonl" \
  --out  "$DATA/data/country_results.jsonl" \
  --scoring echo \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model Qwen/Qwen3-32B \
  --workers 32

echo "=== done $(date) ==="
wc -l "$DATA/data/country_results.jsonl"
