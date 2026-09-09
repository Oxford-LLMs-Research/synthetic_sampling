#!/bin/bash
#SBATCH --job-name=stage
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CPU-only download job: stage one roster wave into $DATA/hf_cache from a
# compute node. Added 9 Sep 2026 after the login node SIGKILLed every
# multi-GB shard download (its memory cgroup counts the dirty page cache of
# writes to the NFS mount; a single-worker download died at the first 4GB
# file). Compute nodes carry a real memory allocation, so the same script
# runs unchanged there. Requires outbound network from compute nodes
# (verified with the curl probe below; the job aborts early if it fails).
#
#   export HF_TOKEN=...              # gated repos (meta-llama)
#   WAVE=1 sbatch --export=ALL scripts/cluster/stage_models.sbatch
#   WAVE=3 ONLY=meta-llama/Llama-3.1-70B sbatch --export=ALL scripts/cluster/stage_models.sbatch
#
# Resume-safe: re-submitting the same wave re-checks every file and fetches
# only what is missing. Keep this file LF-only.

set -euo pipefail
WAVE="${WAVE:?set WAVE (1-4)}"
ONLY="${ONLY:-}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.11.5-GCCcore-13.2.0
fi
source "$DATA/vllm-env/bin/activate"
export HF_HOME="$DATA/hf_cache"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE
export HF_WORKERS="${HF_WORKERS:-4}"

echo "STAGE job=${SLURM_JOB_ID} node=$(hostname) wave=$WAVE only=${ONLY:-all} workers=$HF_WORKERS"
if ! curl -sSI --max-time 30 https://huggingface.co | head -1; then
  echo "FATAL no outbound network to huggingface.co from $(hostname)"; exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$DATA/minimal-core}"
if [ -n "$ONLY" ]; then
  bash scripts/cluster/stage_models.sh "$WAVE" "$ONLY"
else
  bash scripts/cluster/stage_models.sh "$WAVE"
fi
echo "STAGE done wave=$WAVE"
