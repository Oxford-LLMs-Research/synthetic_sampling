#!/bin/bash
#SBATCH -J pmi-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/pmi_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/pmi_%j.err

# Was the paper's echo readout recoverable all along?
#
# Echo scoring ranks options by the mean log-probability of their own tokens,
# which mixes what the model believes about the person with how fluent each
# option's wording is. The standard correction is domain-conditional PMI
# (Holtzman et al. 2021): subtract each option's log-probability under a
# NEUTRAL premise, so the per-option fluency constant cancels. This job scores
# the two neutral premises alongside a fresh echo_plain control, all in one
# serving:
#
#   echo_plain    the paper's exact condition, fresh control for this serving
#   echo_qonly    the paper's template with the profile left EMPTY: the
#                 option's fit to the question alone
#   echo_ctxfree  a bare "Answer:" premise: the option's context-free fluency
#
# echo_plain - echo_ctxfree is the PMI-corrected rule; echo_plain - echo_qonly
# isolates the profile-driven component. If the corrected rule recovers the
# label readout's accuracy, the completion-probability assumption was sound and
# the flaw was normalization; if it does not, showing the options genuinely
# adds information and the hidden-options design cannot be repaired.
#
# The neutral scores are instance-independent, so the scorer caches them:
# ~157 unique option strings for ctxfree, one set per question for qonly.
# Cost is dominated by the echo_plain control (~23k requests); well under an
# hour of scoring after model load.
#
# The output filename is NEW (readout_pmi_*). score_formats.py resumes by
# example_id alone, so pointing --out at an existing readout_results_* file
# would mark every instance done and score nothing.

set -euo pipefail

DATA=/data/polf-sula/nuff1496

MODEL=${1:-Qwen/Qwen3-32B}
TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
PORT=$((8400 + SLURM_JOB_ID % 200))
LOG=$DATA/vllm_pmi_${SLURM_JOB_ID}.log

IN=$DATA/data/readout_set.jsonl
OUT=$DATA/data/readout_pmi_${TAG}.jsonl
ARMS=echo_plain,echo_qonly,echo_ctxfree

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
test -s "$IN" || { echo "FATAL: input missing or empty: $IN"; exit 1; }
wc -l "$IN"

# Same serving parameters as every other run in this project. Do not tune them.
HF_HUB_OFFLINE=1 vllm serve "$MODEL" \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.92 --port "$PORT" \
  > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM pid=$VLLM_PID log=$LOG"

cleanup() {
  echo "=== shutting down vLLM pid=$VLLM_PID ==="
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for vLLM on :$PORT ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM ready after ~$((i * 20))s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died during startup; last 80 log lines:"; tail -n 80 "$LOG" || true; exit 1
  fi
  sleep 20
done

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "vLLM did not become ready in time; last 80 log lines:"
  tail -n 80 "$LOG" || true; exit 1
fi

STATUS=$DATA/data/readout_pmi_STATUS_${TAG}.txt
: > "$STATUS"
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }
note "job $SLURM_JOB_ID on $(hostname), model $MODEL, arms $ARMS"

# One respondent from each of the 48 questions, exactly as in the readout run.
# The smoke runs the replicate at 1.0 so the cache-vs-replicate interaction is
# exercised: replicate neutral scores must be PRESENT (cache keyed on the set
# name, so the replicate re-requests) and echo_ctxfree scores must be finite.
echo "=== smoke (one respondent from each question) ==="
SMOKE=$DATA/data/pmi_smoke_${TAG}.jsonl
MINI=$DATA/data/pmi_smoke_in_${TAG}.jsonl
rm -f "$SMOKE"
awk 'NR % 100 == 1' "$IN" > "$MINI"
wc -l "$MINI"
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$MINI" --out "$SMOKE" --arms "$ARMS" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --workers 16 --replicate-frac 1.0
if ! python "$DATA/scripts/scaling/check_readout_smoke.py" "$SMOKE" >> "$STATUS" 2>&1
then
  note "FAILED at smoke test, see $STATUS and $LOG"
  exit 1
fi
note "smoke passed"

echo "=== full run ==="
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$IN" --out "$OUT" --arms "$ARMS" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --workers 32 --replicate-frac 0.25

python - "$OUT" "$IN" >> "$STATUS" 2>&1 <<'PY'
import collections, json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
want = sum(1 for _ in open(sys.argv[2], encoding="utf-8"))
print(f"RESULTS {len(rows)}/{want} instances written")
ok = collections.Counter(); tot = collections.Counter()
nonfinite = collections.Counter()
for r in rows:
    for key, d in r["results"].items():
        arm = key.split("|", 1)[1]
        tot[arm] += 1
        if "error" in d:
            continue
        if d.get("scores"):
            ok[arm] += 1
            nonfinite[arm] += sum(1 for v in d["scores"].values()
                                  if not math.isfinite(v))
for arm in sorted(tot):
    print(f"ARM {arm:<14} usable {ok[arm]:>6}/{tot[arm]:<6} "
          f"({ok[arm]/tot[arm]:.0%})  non-finite scores {nonfinite[arm]}")
low = [a for a in tot if ok[a] / tot[a] < 0.9]
print("VERDICT " + ("INCOMPLETE, low coverage in " + ", ".join(low) if low
                    else "all arms above 90% coverage"))
PY

note "run finished, $(wc -l < "$OUT") instances"
echo "=== $(date) done ==="
cat "$STATUS"
