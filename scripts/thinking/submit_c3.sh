#!/bin/bash
# Submit the C3 training-axis pair (RUN_CATALOGUE C3, pre-registered 12 Aug):
# Thinking-2507 (generate + inject + label_num score) and Instruct-2507
# (fresh direct scores) over the same 734-pair substrate. Two jobs, two
# servings — the contrast is cross-checkpoint, not within-serving.
#
#   ./scripts/thinking/submit_c3.sh
#   C3_LIMIT=40 ./scripts/thinking/submit_c3.sh   # canary both cells
#   (need >=40 pairs: thinking cell is 1 row/pair and smoke wants 40 rows)
#
# Read Thinking canary gates + *_trace.json before the full Thinking job.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/logs"

C3_LIMIT="${C3_LIMIT:-}"
export C3_LIMIT

echo "Submitting C3 Thinking (limit=${C3_LIMIT:-full})"
sbatch --export=ALL --job-name="c3-thinking" \
  "$ROOT/scripts/thinking/run_c3_thinking.sbatch"

echo "Submitting C3 Instruct (limit=${C3_LIMIT:-full})"
sbatch --export=ALL --job-name="c3-instruct" \
  "$ROOT/scripts/thinking/run_c3_instruct.sbatch"
