#!/bin/bash
# Submit one scoring job. Example:
#   MODEL=Qwen/Qwen3-32B INPUT=instances.jsonl OUT=results.jsonl ./submit_grid.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"
sbatch --export=ALL "$ROOT/scripts/cluster/run_score.sbatch"
