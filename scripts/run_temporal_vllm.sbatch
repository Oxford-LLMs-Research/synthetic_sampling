#!/bin/bash
#SBATCH -J temporal-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/temporal_vllm_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/temporal_vllm_%j.err

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

# Start vLLM in background
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

# Wait for server (model load from NFS can take a while)
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

echo "=== full temporal (echo; resumes from --out) ==="
python "$DATA/scripts/nebius_run_experiments.py" --experiment temporal \
  --input "$DATA/data/temporal_context_instances.jsonl" \
  --out  "$DATA/data/temporal_results.jsonl" \
  --scoring echo \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model Qwen/Qwen3-32B \
  --workers 32

echo "=== done $(date) ==="
wc -l "$DATA/data/temporal_results.jsonl"
