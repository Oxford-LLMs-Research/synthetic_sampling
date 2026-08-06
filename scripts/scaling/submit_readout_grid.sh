#!/bin/bash
# Submit the readout grid, one sbatch per model (models need different GPU
# counts and walltimes, so a Slurm array is the wrong tool). Small models go
# first for fast feedback; the 70Bs and the 235B anchor queue longer.
#
# Run ON THE ARC LOGIN NODE:  bash $DATA/scripts/scaling/submit_readout_grid.sh
#
# Checkpoint identities and the TO VERIFY rows: docs/MODEL_CHECKPOINTS.md in
# the paper workspace. Do not submit the commented models until their ids are
# confirmed against $DATA/hf_cache/hub. DeepSeek is deliberately absent: the
# two-node attempt is its own exercise, not part of this loop.

set -euo pipefail
cd "$(dirname "$0")"

# --- single-GPU, ~2-4 h each -------------------------------------------------
sbatch --time=04:00:00 run_readout_grid.sbatch Qwen/Qwen3-4B 1
sbatch --time=04:00:00 run_readout_grid.sbatch meta-llama/Llama-3.1-8B-Instruct 1
sbatch --time=04:00:00 run_readout_grid.sbatch allenai/Olmo-3-1025-7B 1
sbatch --time=06:00:00 run_readout_grid.sbatch google/gemma-3-27b-it 1
sbatch --time=06:00:00 run_readout_grid.sbatch openai/gpt-oss-120b 1
sbatch --time=06:00:00 run_readout_grid.sbatch allenai/Olmo-3.1-32B-Instruct-DPO 1

# Qwen3-32B carries the extra matched-instruction pilot arm (echo scoring under
# the label arm's instruction line), pilot-only on this one model.
sbatch --time=06:00:00 \
  --export=ALL,ARMS=echo_plain,echo_listed,echo_listed_numinstr,label_num,label_num_natural,echo_qonly,echo_ctxfree \
  run_readout_grid.sbatch Qwen/Qwen3-32B 1

# --- base models: label-miss guard demoted to advisory ----------------------
# A base model that cannot emit a bare digit is a finding, not an abort.
sbatch --time=04:00:00 --export=ALL,READOUT_MAX_LABEL_MISS=1.0 \
  run_readout_grid.sbatch meta-llama/Llama-3.1-8B 1
# sbatch --time=06:00:00 --export=ALL,READOUT_MAX_LABEL_MISS=1.0 \
#   run_readout_grid.sbatch allenai/Olmo-3-1025-32B 1        # TO VERIFY id first
# sbatch --time=06:00:00 run_readout_grid.sbatch allenai/Olmo-3-7B-DPO 1   # TO VERIFY id first

# --- multi-GPU ---------------------------------------------------------------
sbatch --gres=gpu:h100:2 --time=08:00:00 --export=ALL,READOUT_MAX_LABEL_MISS=1.0 \
  run_readout_grid.sbatch meta-llama/Llama-3.1-70B 2
sbatch --gres=gpu:h100:2 --time=08:00:00 \
  run_readout_grid.sbatch meta-llama/Llama-3.1-70B-Instruct 2

# The 235B anchor: FP8 repack (~235 GB) on the 4x96GB node (htc-g058) if
# available, else TP8 on an 8x80GB node (htc-g[059-060]). Confirm with
# sinfo -p short -N -o "%N %G" before uncommenting one of these.
# sbatch --gres=gpu:h100:4 --time=10:00:00 run_readout_grid.sbatch Qwen/Qwen3-235B-A22B-FP8 4
# sbatch --gres=gpu:h100:8 --time=10:00:00 run_readout_grid.sbatch Qwen/Qwen3-235B-A22B-FP8 8
