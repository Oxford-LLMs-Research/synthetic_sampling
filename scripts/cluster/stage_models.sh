#!/bin/bash
# Stage Phase 2 roster weights into $DATA/hf_cache on ARC, one wave at a time.
# The roster is scripts/cluster/roster_phase2.tsv (set 9 Sep 2026; deliberation
# in the paper workspace, docs/MODEL_ROSTER.md, ids in MODEL_CHECKPOINTS.md).
#
#   ./scripts/cluster/stage_models.sh 1            # download wave 1
#   ./scripts/cluster/stage_models.sh 1 --dry-run  # list what wave 1 would fetch
#   ./scripts/cluster/stage_models.sh 3 meta-llama/Llama-3.1-70B   # one id only
#
# Run on the login node (it has the network; torch is not needed here). Gated
# repos (meta-llama) need HF_TOKEN exported first. Downloads resume, so a
# dropped connection is re-run, not a loss. Storage is 5 TB and not backed
# up: check `df -h $DATA` before a wave and delete weights whose grid run has
# landed AND been backed up locally (weights re-download, results do not).
# Keep this file LF-only.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TSV="$ROOT/scripts/cluster/roster_phase2.tsv"
WAVE="${1:?usage: stage_models.sh <wave> [--dry-run | <hf_id>]}"
ONLY="${2:-}"
# Parallel shard downloads on the login node get SIGKILLed by its memory cap
# (wave 1, 9 Sep: 8 workers died on the first 61GB model). Default low; raise
# with HF_WORKERS=n only if a wave proves it survives.
HF_WORKERS="${HF_WORKERS:-2}"

: "${DATA:?DATA is unset; run on ARC (DATA=/data/polf-sula/nuff1496)}"
export HF_HOME="${HF_HOME:-$DATA/hf_cache}"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE

if command -v hf >/dev/null 2>&1; then
  DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL=(huggingface-cli download)
else
  echo "no hf / huggingface-cli on PATH; activate a venv with huggingface_hub"; exit 1
fi

echo "HF_HOME=$HF_HOME"
df -h "$DATA" | tail -1

# Skip the header; fields are tab-separated: wave id role precision tp gb notes.
tail -n +2 "$TSV" | while IFS=$'\t' read -r wave id role precision tp gb notes; do
  [ "$wave" = "$WAVE" ] || continue
  if [ -n "$ONLY" ] && [ "$ONLY" != "--dry-run" ] && [ "$ONLY" != "$id" ]; then continue; fi
  echo "== wave $wave  $id  ($role, $precision, TP$tp, ~${gb}GB)  $notes"
  if [ "$ONLY" = "--dry-run" ]; then continue; fi
  # original/ holds Meta's consolidated .pth copies (doubles the Llama size);
  # GGUF and consolidated files are never served by vLLM.
  "${DL[@]}" "$id" \
    --exclude "original/*" --exclude "*.gguf" --exclude "consolidated*" \
    --max-workers "$HF_WORKERS"
  # Record what landed: snapshot hash and size, for MODEL_CHECKPOINTS.md.
  snap_dir="$HF_HOME/hub/models--${id//\//--}/snapshots"
  if [ -d "$snap_dir" ]; then
    for s in "$snap_dir"/*; do
      echo "STAGED id=$id snapshot=$(basename "$s") size=$(du -shL "$s" | cut -f1)"
    done
  fi
done

df -h "$DATA" | tail -1
echo "wave $WAVE done"
