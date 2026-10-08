#!/bin/bash
# Submit Phase 2 scoring jobs from the roster, one job per (serving, shard).
#
#   INPUT=$DATA/phase2/instances_r0-50.jsonl WAVE=1 ./scripts/cluster/submit_phase2.sh
#   INPUT=... ONLY=Qwen/Qwen3-32B SHARDS=4 ./scripts/cluster/submit_phase2.sh
#   INPUT=... WAVE=3 DRY_RUN=1 ./scripts/cluster/submit_phase2.sh      # print, submit nothing
#
#   INPUT=... ONLY=google/gemma-4-31B SHARDS=3 ONLY_SHARD=2 ./scripts/...   # one shard
#
# Env: INPUT (required), WAVE (1-4) and/or ONLY (hf id), SHARDS (default 1),
# EXTRA_APPEND (serve flags added after the per-model ones, for tests),
# NODELIST (pin the job to a node, e.g. htc-g058 for a model that only fits
# its 96 GB cards, or a named node for a test),
# ONLY_SHARD (resubmit a single shard index; never resubmit a shard whose
# job is still running, two jobs would append to one output file),
# ARMS, REPLICATE_FRAC, TAG (results subfolder, default grid_r0-50),
# TIME (default 12:00:00), DRY_RUN=1, plus anything run_score.sbatch reads.
#
# Per serving it sets, from roster_phase2.tsv and the table below:
#   --gres=gpu:h100:<tp>  --cpus-per-task=<2 x tp>  --mem=<by tp>
#   TP, DTYPE, EXTRA_VLLM_ARGS, WORKERS, GPU_MEM_UTIL
# Every submit needs a CODE/EXPERIMENT_REGISTRY.md entry (status RUNNING) in
# the same sitting; the job ids this prints are what the entry records.
# Keep this file LF-only.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROSTER="$ROOT/scripts/cluster/roster_phase2.tsv"
INPUT="${INPUT:?set INPUT (the instance file on the cluster)}"
WAVE="${WAVE:-}"
ONLY="${ONLY:-}"
SHARDS="${SHARDS:-1}"
TAG="${TAG:-grid_r0-50}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-}"
ONLY_SHARD="${ONLY_SHARD:-}"
NODELIST="${NODELIST:-}"
# Phase 2 arms: label_num plus the two PMI premises. echo_plain is not part
# of this run (decided 2 Oct 2026); each serving's echo reading comes from
# its Phase 1 readout battery.
ARMS="${ARMS:-label_num,echo_qonly,echo_ctxfree}"
REPLICATE_FRAC="${REPLICATE_FRAC:-0.1}"
# Context window served. The longest label_num prompt in the grid input is
# about 1,520 tokens (measured on the Qwen, Gemma and Nemotron tokenizers,
# 7 Oct 2026). vLLM refuses to start unless one request of this length fits
# the KV cache, and at 16384 gemma-4-31B did not fit on every node (job
# 9012474: 13.76 GiB needed, 12.53 available).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
RESULTS="${RESULTS:-${DATA:?set DATA}/outputs/phase2/results}"

if [ -z "$WAVE" ] && [ -z "$ONLY" ]; then
  echo "set WAVE (1-4) and/or ONLY (hf id)"; exit 1
fi
if [ -z "$DRY_RUN" ] && [ ! -f "$INPUT" ]; then
  echo "INPUT not found: $INPUT"; exit 1
fi
mkdir -p "$ROOT/logs"

# Host memory by tensor-parallel size (ARC_BIG_MODEL_ENVELOPE.md: generous
# for big loads, 400G for TP8).
mem_for_tp() {
  case "$1" in
    1) echo 96G ;; 2) echo 192G ;; 4) echo 300G ;; 8) echo 400G ;;
    *) echo "unsupported tp $1" >&2; exit 1 ;;
  esac
}

n=0
while IFS=$'\t' read -r wave hf_id role precision tp est_gb notes; do
  [ "$wave" = "wave" ] && continue
  [ -z "${hf_id:-}" ] && continue
  [ -n "$WAVE" ] && [ "$wave" != "$WAVE" ] && continue
  [ -n "$ONLY" ] && [ "$hf_id" != "$ONLY" ] && continue

  # --dtype only for bf16 checkpoints; quantized ones carry their own.
  dtype=none
  extra=""
  case "$precision" in
    bf16) dtype=bfloat16 ;;
    "bf16->fp8") dtype=bfloat16; extra="--quantization fp8" ;;
  esac
  case "$hf_id" in
    nvidia/NVIDIA-Nemotron-3-*)
      extra="$extra --trust-remote-code --mamba-ssm-cache-dtype float32" ;;
  esac
  # Hybrid families hold one state-cache block per sequence, and vLLM's
  # default of 1024 sequences exceeds what fits beside the weights (983 on
  # Nemotron-3-Nano, 58 on Qwen3.5-35B-A3B-Base; jobs 9012460-67). The
  # client sends at most WORKERS (16) requests at once.
  case "$hf_id" in
    nvidia/NVIDIA-Nemotron-3-*|Qwen/Qwen3.5-*) extra="$extra --max-num-seqs 32" ;;
  esac
  # Qwen3.5 (qwen3_next) crashes in FlashAttention 3, the H100 default
  # (jobs 9012454-59: _vllm_fa3_C.fwd, aten::new_empty). Versions 4 and 2
  # both load and pass the smoke gate (jobs 9013033, 9013034); 4 is what
  # vLLM itself moves to on this GPU when 3 cannot serve a model (Gemma 4).
  case "$hf_id" in
    Qwen/Qwen3.5-*) extra="$extra --attention-config.flash_attn_version=4" ;;
  esac
  extra="$extra ${EXTRA_APPEND:-}"
  extra="${extra# }"; extra="${extra% }"

  # Client concurrency by memory headroom. Weights above 60 percent of the
  # allocation's GPU memory leave a KV cache too small for 32 workers: the
  # rotations of one instance stop finding their shared profile cached
  # (Qwen3-32B, 5-6 Oct: 32 workers at 0.92 gave a 2 percent prefix hit rate
  # and 1.09 inst/s; 8 workers at 0.92 gave 77 percent and 5.3 inst/s; 16
  # workers at 0.95 gave 77 percent and 6.7 inst/s).
  # WORKERS / GPU_MEM_UTIL in the environment override both.
  if [ $(( est_gb * 10 )) -gt $(( 480 * tp )) ]; then
    workers="${WORKERS:-16}"; mem_util="${GPU_MEM_UTIL:-0.95}"
  else
    workers="${WORKERS:-32}"; mem_util="${GPU_MEM_UTIL:-0.92}"
  fi

  slug="$(echo "$hf_id" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
  outdir="$RESULTS/$TAG/$slug"
  for (( i = 0; i < SHARDS; i++ )); do
    [ -n "$ONLY_SHARD" ] && [ "$i" != "$ONLY_SHARD" ] && continue
    out="$outdir/${slug}_shard${i}of${SHARDS}.jsonl"
    # Variables travel in the environment with --export=ALL, never inside
    # --export=: Slurm splits that list on commas, and ARMS contains commas.
    cmd=(env "MODEL=${hf_id}" "INPUT=${INPUT}" "OUT=${out}" "ARMS=${ARMS}"
         "REPLICATE_FRAC=${REPLICATE_FRAC}" "TP=${tp}" "DTYPE=${dtype}"
         "EXTRA_VLLM_ARGS=${extra}" "WORKERS=${workers}"
         "GPU_MEM_UTIL=${mem_util}" "MAX_MODEL_LEN=${MAX_MODEL_LEN}" "SHARD_INDEX=${i}" "SHARD_COUNT=${SHARDS}"
         sbatch --job-name="p2-${slug:0:24}"
         --gres="gpu:h100:${tp}" --cpus-per-task="$(( 2 * tp ))"
         --mem="$(mem_for_tp "$tp")" --time="$TIME" --export=ALL)
    [ -n "$NODELIST" ] && cmd+=(--nodelist="$NODELIST")
    cmd+=("$ROOT/scripts/cluster/run_score.sbatch")
    if [ -n "$DRY_RUN" ]; then
      printf 'DRY  wave=%s tp=%s %s shard %d/%d\n     %s\n' \
        "$wave" "$tp" "$hf_id" "$i" "$SHARDS" "${cmd[*]}"
    else
      mkdir -p "$outdir"
      job="$("${cmd[@]}")"
      echo "SUBMITTED wave=$wave tp=$tp model=$hf_id shard=$i/$SHARDS -> $job"
    fi
    n=$(( n + 1 ))
  done
done < "$ROSTER"

if [ "$n" -eq 0 ]; then
  echo "no roster row matched WAVE='$WAVE' ONLY='$ONLY'"; exit 1
fi
echo "$n job(s) $([ -n "$DRY_RUN" ] && echo 'listed (dry run)' || echo submitted). Add or update the registry entry now."
