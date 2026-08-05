#!/bin/bash
#SBATCH -J probe-labels
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH -o /data/polf-sula/nuff1496/logs/probe_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/probe_%j.err

# Diagnose why the readout run's label arms returned no label log-probabilities.
#
# Job 8400296 aborted at its smoke test: label_num found no numeric token among
# the top 20 at the first generated position on 72 of 72 option slots, and
# label_alpha on 46 of 72, while greedy generation matched 19 of 24 on the same
# prompts. The model answers sensibly and simply does not put a bare label
# first. The standing hypothesis is that /completions applies no chat template,
# so nothing enforces "reply with only the option number" and the model
# continues with the answer text.
#
# This job spends about ten minutes of GPU to replace that guess with a
# transcript: four prompt endings and both endpoints, printing what the server
# actually returns at each of the first four positions. It scores nothing.
#
# Almost all of the walltime is model loading.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
PORT=8500
LOG=$DATA/vllm_probe_${SLURM_JOB_ID}.log
OUT=$DATA/data/probe_labels.txt

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

# Identical serving parameters to the run being diagnosed, so the transcript
# describes that run and not a different configuration.
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3-32B \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.92 --port "$PORT" \
  > "$LOG" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for vLLM on :$PORT ..."
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died:"; tail -n 60 "$LOG"; exit 1; }
  sleep 20
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || {
  echo "vLLM not ready:"; tail -n 60 "$LOG"; exit 1; }

python "$DATA/scripts/scaling/probe_labels.py" \
  --input "$DATA/data/readout_set.jsonl" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --n 3 --depth 4 | tee "$OUT"

echo "=== $(date) done; transcript at $OUT ==="
