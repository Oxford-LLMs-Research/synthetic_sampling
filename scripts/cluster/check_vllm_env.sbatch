#!/bin/bash
#SBATCH --job-name=checkvllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Verify the pinned vLLM venv on a real GPU: driver vs torch CUDA build, then
# an end-to-end serve of one small staged model with a label_num readout.
# Short wall on purpose so it backfills quickly. Added 11 Sep 2026.
#
# The serve check matters more than the import check: vLLM imports fine on a
# stack that cannot actually allocate, and the readout is what the whole grid
# depends on. Qwen3-4B is the cheapest staged model to prove it with.
#
#   sbatch --export=ALL scripts/cluster/check_vllm_env.sbatch
#   VENV=$DATA/vllm-029 MODEL=Qwen/Qwen3-4B sbatch --export=ALL ...
#
# Keep this file LF-only.

set -euo pipefail
VENV="${VENV:-$DATA/vllm-029}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.11.5-GCCcore-13.2.0
fi
source "$VENV/bin/activate"
export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "CHECKSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) venv=$VENV model=$MODEL"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import torch, vllm
print("VERSIONS vllm=%s torch=%s cuda_build=%s" % (vllm.__version__, torch.__version__, torch.version.cuda))
print("CUDA available=%s devices=%d" % (torch.cuda.is_available(), torch.cuda.device_count()))
assert torch.cuda.is_available(), "torch cannot see the GPU: driver vs cuda build mismatch"
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("MATMUL ok", float((x @ x).float().abs().mean()))
PY

echo "Serving $MODEL on port $PORT"
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  > "$DATA/vllm_check_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

elapsed=0
while ! curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -qF "$MODEL"; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "FATAL vLLM died during startup; tail of its log:"
    tail -40 "$DATA/vllm_check_${SLURM_JOB_ID:-local}.log"
    exit 1
  fi
  if [ "$elapsed" -ge 900 ]; then echo "FATAL vLLM not ready after 900s"; exit 1; fi
  sleep 5; elapsed=$((elapsed + 5))
done
echo "vLLM ready after ${elapsed}s"

# The readout the grid depends on: top-k logprobs at the answer position for a
# numbered option list, prompt ending in "Answer: " with the trailing space.
python - "$PORT" "$MODEL" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
prompt = ("Question: Which one would you consider more important?\n"
          "1. Freedom\n2. Equality\n"
          "Reply with only the option number.\n"
          "Answer: ")
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                     "temperature": 0.0, "logprobs": 20}).encode(),
    headers={"Content-Type": "application/json"})
top = json.load(urllib.request.urlopen(req))["choices"][0]["logprobs"]["top_logprobs"][0]
digits = {k.strip(): v for k, v in top.items() if k.strip() in ("1", "2")}
print("READOUT top5=%s" % sorted(top.items(), key=lambda kv: -kv[1])[:5])
print("READOUT digit_logprobs=%s" % digits)
assert digits, "no digit token in the top-20 logprobs: label_num readout would miss"
print("READOUT ok")
PY

echo "CHECK done"
