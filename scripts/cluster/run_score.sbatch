#!/bin/bash
#SBATCH --job-name=ss-score
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Minimal-core scoring job, ARC HTC pattern (docs/CLUSTER_HANDOFF.md in the
# paper workspace): wake env, serve vLLM offline, gate on a design-spanning
# smoke, then run. Resume-safe: the runner skips example_ids already in OUT.
# Keep this file LF-only (never Path.write_text on Windows).
# Required env: MODEL, INPUT, OUT. Optional: ARMS, REPLICATE_FRAC, DTYPE,
# MAX_MODEL_LEN, LIMIT, SHARD_INDEX, SHARD_COUNT, PORT, VLLM_READY_TIMEOUT_SEC,
# VENV, TP, EXTRA_VLLM_ARGS, GPU_MEM_UTIL.
#
# Resources are set at submit time, not here: the header is the single-GPU
# default (2 CPUs per GPU; the 30 Sep CPU test found throughput does not
# depend on the CPU count and ARC bills reserved CPUs). A multi-GPU serving
# is submitted with matching flags, which submit_phase2.sh does:
#   sbatch --gres=gpu:h100:$TP --cpus-per-task=$((2*TP)) --mem=... (TP=$TP)

set -euo pipefail

MODEL="${MODEL:?set MODEL}"
INPUT="${INPUT:?set INPUT}"
OUT="${OUT:?set OUT}"
ARMS="${ARMS:-label_num,echo_plain,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.1}"
# DTYPE=none omits --dtype (quantized checkpoints: gpt-oss MXFP4, FP8, NVFP4).
DTYPE="${DTYPE:-bfloat16}"
# Tensor-parallel size = GPUs in the allocation. It must match --gres.
TP="${TP:-1}"
# Per-model serve flags, word-split on purpose (e.g. Nemotron 3:
# "--trust-remote-code --mamba-ssm-cache-dtype float32").
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# Per-job port. A fixed default collides: HTC nodes host several single-GPU
# jobs, and two jobs sharing a port means the second one's `vllm serve` cannot
# bind while its readiness probe passes against the FIRST one's server (job
# 8511397, 9 Aug 2026). Derive it from the job id so a collision needs two jobs
# exactly 10000 apart in id on one node.
#
# Range 20000-29999 deliberately: it must stay BELOW the Linux ephemeral range
# (`net.ipv4.ip_local_port_range`, default 32768-60999). A listener port drawn
# from that range can lose the bind to a transient outgoing connection from
# torch/NCCL/curl on a busy node, which would fail intermittently and look
# like a vLLM problem.
PORT="${PORT:-$(( 20000 + (${SLURM_JOB_ID:-$$} % 10000) ))}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# Re-wake the ARC environment (guarded, so the script still runs elsewhere).
if command -v module >/dev/null 2>&1; then
  module load Python/3.11.5-GCCcore-13.2.0
fi
# One pinned serving stack for the whole Phase 2 grid (vLLM v0.29.0); the
# serving-stack confound is 36 percent of predictions, so a job that cannot
# find its venv stops instead of falling back to whatever is on PATH.
VENV="${VENV:-${DATA:-}/vllm-029}"
if [ -n "${DATA:-}" ]; then
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "FATAL venv not found: $VENV (set VENV, or build it with build_vllm_env.sbatch)"
    exit 1
  fi
  source "$VENV/bin/activate"
  export HF_HOME="$DATA/hf_cache"
fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$(dirname "$OUT")" logs

# Stamp the run into the job log. EXPERIMENT_REGISTRY.md needs the commit, the
# node and the arms; reconstructing them afterwards from job dates is guesswork,
# and a run whose repo state was never recorded cannot be reproduced. Uses
# SLURM_SUBMIT_DIR, not $0: Slurm runs a spool copy, so $0 is not the repo.
echo "RUNSTAMP repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse HEAD 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname) model=$MODEL"
echo "RUNSTAMP arms=$ARMS replicate=$REPLICATE_FRAC input=$INPUT out=$OUT"
echo "RUNSTAMP venv=$VENV vllm=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo NOT_RECORDED)"
echo "RUNSTAMP tp=$TP dtype=$DTYPE max_model_len=$MAX_MODEL_LEN extra_vllm_args=[${EXTRA_VLLM_ARGS}] shard=${SHARD_INDEX:-all}/${SHARD_COUNT:-1}"
echo "RUNSTAMP cpus=${SLURM_CPUS_PER_TASK:-NOT_RECORDED} gpus=${SLURM_GPUS_ON_NODE:-NOT_RECORDED}"

# The allocation must hold the GPUs the serving will ask for: vLLM with a
# tensor-parallel size above the visible GPU count dies only after loading.
if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE}" -lt "$TP" ]; then
  echo "FATAL TP=$TP but the allocation has ${SLURM_GPUS_ON_NODE} GPU(s); submit with --gres=gpu:h100:$TP"
  exit 1
fi

# Refuse to start if something already answers on this port. Attaching to
# another job's vLLM is the dangerous failure: if it happens to serve the same
# model name the run SUCCEEDS and writes scores from a serving stack this job
# does not own, which the reuse rule forbids and no downstream check would see.
if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "FATAL port $PORT on $(hostname) is already serving; refusing to attach"
  echo "      to another job's vLLM. Re-submit (PORT is derived from the job id)."
  exit 1
fi

SERVE_ARGS=()
[ "$DTYPE" != "none" ] && SERVE_ARGS+=(--dtype "$DTYPE")
[ "$TP" -gt 1 ] && SERVE_ARGS+=(--tensor-parallel-size "$TP")
# shellcheck disable=SC2206
[ -n "$EXTRA_VLLM_ARGS" ] && SERVE_ARGS+=($EXTRA_VLLM_ARGS)

echo "Starting vLLM for $MODEL on port $PORT (dtype $DTYPE, tp $TP)"
# --host 127.0.0.1: vLLM binds 0.0.0.0 by default, which on a shared node put
# the server on the cluster network under a now-predictable port. Nothing needs
# it off-node (ss-score talks to BASE_URL on loopback), and binding loopback is
# what makes the checks above authoritative rather than advisory.
# --served-model-name: with HF_HUB_OFFLINE=1 vLLM otherwise advertises the
# snapshot path as the API model id, so the readiness grep for $MODEL never
# matches and ss-score would 404 even against our own server.
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${SERVE_ARGS[@]}" \
  > "$(dirname "$OUT")/vllm_serve_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Wait until ready, or until vLLM dies. Slurm --time is the real backstop.
# Optional VLLM_READY_TIMEOUT_SEC caps the wait (unset = no curl timeout).
# Ready means "serving OUR model", not "something answers": a bare 200 is what
# let 8511397 attach to a neighbour.
elapsed=0
while ! curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -qF "$MODEL"; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM died during startup; tail of its log:"
    tail -40 "$(dirname "$OUT")/vllm_serve_${SLURM_JOB_ID:-local}.log" || true
    exit 1
  fi
  if [ -n "${VLLM_READY_TIMEOUT_SEC:-}" ] && [ "$elapsed" -ge "$VLLM_READY_TIMEOUT_SEC" ]; then
    echo "vLLM never came up within ${VLLM_READY_TIMEOUT_SEC}s"; exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

# Smoke gate spanning the design, not just its head. The stride is derived
# from the input, not fixed: a fixed 50 undershot MIN_ROWS=40 on 734-row
# (C3, jobs 8556542/8556543) and 1,568-row (A3) sets, aborting AFTER the
# model load — and an even stride on a paired-cell set samples only one
# cell (A3's pairs sit at odd/even lines). Target ~44 rows; step the
# stride down until coprime with 2 and 3 so neither pairs nor triples
# alias out (stepping DOWN only adds rows, never drops below target).
SMOKE_IN="$(dirname "$OUT")/smoke_in_${SLURM_JOB_ID:-local}.jsonl"
SMOKE_OUT="$(dirname "$OUT")/smoke_${SLURM_JOB_ID:-local}.jsonl"
N_INPUT=$(wc -l < "$INPUT")
STRIDE=$(( N_INPUT / 44 ))
while [ "$STRIDE" -gt 1 ] && { [ $((STRIDE % 2)) -eq 0 ] || [ $((STRIDE % 3)) -eq 0 ]; }; do
  STRIDE=$((STRIDE - 1))
done
if [ "$STRIDE" -le 1 ]; then
  # <88 input rows: NR % 1 == 1 never matches, so copy the whole file.
  cp "$INPUT" "$SMOKE_IN"
else
  awk -v s="$STRIDE" 'NR % s == 1' "$INPUT" > "$SMOKE_IN"
fi
N_SMOKE=$(wc -l < "$SMOKE_IN")
if [ "$N_SMOKE" -lt 40 ] && [ "$N_INPUT" -ge 40 ]; then
  echo "FATAL smoke_in has $N_SMOKE rows (need >=40) from $N_INPUT input rows"
  exit 1
fi
echo "smoke: $N_SMOKE rows from $N_INPUT (stride ${STRIDE:-copy})"
ss-score --input "$SMOKE_IN" --out "$SMOKE_OUT" \
  --base-url "$BASE_URL" --model "$MODEL" --arms "$ARMS" \
  --replicate-frac 0 --workers 16
ss-analyze smoke "$SMOKE_OUT" || { echo "SMOKE FAILED, aborting"; exit 1; }

ss-score \
  --input "$INPUT" \
  --out "$OUT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --arms "$ARMS" \
  --replicate-frac "$REPLICATE_FRAC" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${SHARD_INDEX:+--shard-index "$SHARD_INDEX"} \
  ${SHARD_COUNT:+--shard-count "$SHARD_COUNT"}

ss-analyze coverage "$OUT" "$INPUT" || true
echo "done"
