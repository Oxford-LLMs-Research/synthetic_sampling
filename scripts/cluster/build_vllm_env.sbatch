#!/bin/bash
#SBATCH --job-name=buildvllm
#SBATCH -p short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Build the pinned vLLM venv for the Phase 2 grid, CPU-only.
# Added 11 Sep 2026: the interactive `srun --gres=gpu:h100:1` route queues for
# hours behind GPU work, and nothing in a pip install needs a GPU. Wheels are
# downloaded and byte-compiled here; the CUDA check runs separately in
# check_vllm_env.sbatch, which needs a GPU for about a minute.
#
# One vLLM version serves the WHOLE grid. Scores are never compared across
# serving stacks (36 percent of predictions flip), so a mid-grid version
# bump would split the roster into two incomparable halves.
#
#   VLLM_VERSION=0.29.0 sbatch --export=ALL scripts/cluster/build_vllm_env.sbatch
#
# Keep this file LF-only.

set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
VENV="${VENV:-$DATA/vllm-029}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.11.5-GCCcore-13.2.0
fi

echo "BUILDSTAMP job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "BUILDSTAMP vllm=$VLLM_VERSION venv=$VENV python=$(python -V 2>&1)"

# Never build on top of a half-finished venv: a partial install is worse than
# no install, because it imports and then fails at serve time.
if [ -d "$VENV" ]; then
  echo "FATAL $VENV already exists. Remove it first (rm -rf) or set VENV to a new path."
  exit 1
fi

# Keep pip's cache and tmp on $DATA: /tmp on a compute node is small, and a
# multi-GB wheel set fills it (torch alone is ~2.5GB).
export PIP_CACHE_DIR="$DATA/.pip-cache"
export TMPDIR="$DATA/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "vllm==$VLLM_VERSION"

# The repo itself, so ss-score / ss-analyze exist in this venv.
cd "${SLURM_SUBMIT_DIR:-$DATA/minimal-core}"
python -m pip install -e .

# Import-only checks (no CUDA context, safe without a GPU).
python - <<'PY'
import vllm, torch, transformers, huggingface_hub
print("VERSIONS vllm=%s torch=%s cuda_build=%s transformers=%s hub=%s"
      % (vllm.__version__, torch.__version__, torch.version.cuda,
         transformers.__version__, huggingface_hub.__version__))
PY
which ss-score ss-analyze
python -m pip freeze > "$VENV/pip-freeze-$(date +%Y%m%d).txt"
echo "BUILD done venv=$VENV"
