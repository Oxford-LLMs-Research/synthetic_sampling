#!/bin/bash
#SBATCH -J readout-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/readout_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/readout_%j.err

# Does reading a LABEL token recover person-level signal that echo scoring
# destroys?
#
# The paper scores an option by the mean log-probability of its own tokens and
# takes the argmax. That ranking is contaminated by how fluent each option's
# wording is, independent of the respondent: a phrase can lose for everyone
# because it is unusual English here, not because the model believes it false of
# these people. Measured on the main grid, Qwen 3 32B scores 0.118 normalized
# accuracy while carrying discrimination AUC 0.593, with 83% of questions above
# a permutation null and 65 of the 85 below-chance questions still
# discriminating. So the readout is demonstrably throwing information away. The
# open question is how much of it a different readout gets back.
#
# 4,800 instances: 48 questions x 100 respondents at s6m4, drawn from the
# paper's OWN scored instances, stratified by survey and option count. 100
# respondents per question is what makes AUC and prior correction estimable;
# the surface-form set has nine and cannot support them.
#
# Scores are never reused across runs. Comparing a stored echo score against a
# fresh label score would confound the readout with the serving stack, and that
# confound is worth 36% of predictions (checks/control_check.py: the same model
# on the same instances under a different stack agrees on only 64.4%). Every arm
# is re-scored here, in one serving. Because the instances are the paper's own,
# the fresh echo_plain arm doubles as a check against the stored SGLang results.
#
# Arms, all through /completions so the endpoint is held fixed:
#   echo_plain         the paper's exact condition, options never shown
#   echo_listed        same scoring, options listed: separates seeing the
#                      choices from how the answer is read
#   label_num          numbered, cyclic Latin square over slots
#   label_num_natural  numbered, no rotation: separates position debiasing from
#                      the cost of showing an ordinal scale out of order
#   label_alpha        lettered, because label bias differs between alphabets
#   generate           greedy, strict exact match, non-matches scored incorrect
#   generate_sampled   8 draws at temperature 1, kept as a distribution, which
#                      is what a synthetic sample actually is
#
# score_formats.py resumes from --out, so requeue if it hits the walltime.

set -euo pipefail

DATA=/data/polf-sula/nuff1496

# Model is the first argument, so a second model is a resubmission rather than an
# edit:  sbatch run_readout_vllm.sbatch allenai/OLMo-3-32B-DPO
# Outputs and the port are keyed to a tag derived from it, so two models can run
# at once without colliding or overwriting each other's results.
MODEL=${1:-Qwen/Qwen3-32B}
TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
PORT=$((8400 + SLURM_JOB_ID % 200))
LOG=$DATA/vllm_readout_${SLURM_JOB_ID}.log

IN=$DATA/data/readout_set.jsonl
OUT=$DATA/data/readout_results_${TAG}.jsonl
# label_alpha dropped after probing the live server: asked for a letter this
# model answered with prose ("Based on the provided answers,") in four of six
# prompts and with a DIGIT in a fifth, so letters are not reliably emitted in a
# prompt whose profile is already rendered as "Q:/A:". Numbers are 229/229.
ARMS=echo_plain,echo_listed,label_num,label_num_natural,generate,generate_sampled

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

# Same serving parameters as every other run in this project. Do not tune them:
# results split across configurations are not comparable, which is the whole
# reason every arm is re-scored here rather than reusing stored scores.
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

# This runs unattended overnight, so nothing here may merely warn: a condition
# that invalidates an arm must abort before the GPU hours are spent, and the
# outcome must be readable in the morning without scrolling a log. The smoke
# test exits non-zero on anything fatal, and both stages append to STATUS.
STATUS=$DATA/data/readout_STATUS_${TAG}.txt
: > "$STATUS"
note() { echo "$(date +%H:%M) $*" | tee -a "$STATUS"; }
note "job $SLURM_JOB_ID on $(hostname), model $MODEL"

# The smoke must span QUESTIONS, not respondents. readout_set.jsonl is grouped
# 100 instances per question, so --limit 12 tested twelve respondents of a single
# question and passed a prompt format that failed on others. Every 100th line is
# one respondent from each of the 48 questions.
echo "=== smoke (one respondent from each question) ==="
SMOKE=$DATA/data/readout_smoke_${TAG}.jsonl
MINI=$DATA/data/readout_smoke_in_${TAG}.jsonl
rm -f "$SMOKE"
awk 'NR % 100 == 1' "$IN" > "$MINI"
wc -l "$MINI"
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$MINI" --out "$SMOKE" --arms "$ARMS" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --workers 16 --gen-draws 8 --replicate-frac 1.0
# The guard lives in its own file so its logic can be exercised against a
# stored run before it ever gets to abort a job here:
#   python check_readout_smoke.py --self-test <readout_results_*.jsonl>
# It aborts only when the run cannot produce what it exists to produce.
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
  --workers 32 --gen-draws 8 --replicate-frac 0.25

# A one-file verdict, so the morning check is `cat readout_STATUS.txt` rather
# than reading a log. Coverage per arm is what matters: an arm that silently
# returned nothing for most instances is the failure mode that survives a smoke
# test on twelve rows.
python - "$OUT" "$IN" >> "$STATUS" 2>&1 <<'PY'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
want = sum(1 for _ in open(sys.argv[2], encoding="utf-8"))
print(f"RESULTS {len(rows)}/{want} instances written")
ok = collections.Counter(); tot = collections.Counter(); unmatched = collections.Counter()
draws = collections.Counter(); matched = collections.Counter()
rel = collections.Counter(); matched_rel = collections.Counter()
for r in rows:
    for key, d in r["results"].items():
        arm = key.split("|", 1)[1]
        tot[arm] += 1
        if "error" in d:
            continue
        if arm.startswith("generate"):
            if d.get("predicted") is not None:
                ok[arm] += 1
            elif d.get("picked_relaxed") or d.get("counts_relaxed"):
                rel[arm] += 1
            else:
                unmatched[arm] += 1
            draws[arm] += d.get("n_draws", 0)
            matched[arm] += d.get("n_matched", 0)
            matched_rel[arm] += d.get("n_matched_relaxed", 0)
        elif d.get("scores"):
            ok[arm] += 1
for arm in sorted(tot):
    line = f"ARM {arm:<18} usable {ok[arm]:>6}/{tot[arm]:<6} ({ok[arm]/tot[arm]:.0%})"
    if arm.startswith("generate"):
        line += f"  +unique-prefix {rel[arm]}  unmatched {unmatched[arm]}"
        if draws[arm]:
            line += (f"  draw match strict {matched[arm]/draws[arm]:.0%}"
                     f" relaxed {matched_rel[arm]/draws[arm]:.0%}")
    print(line)
low = [a for a in tot if ok[a] / tot[a] < 0.9 and not a.startswith("generate")]
print("VERDICT " + ("INCOMPLETE, low coverage in " + ", ".join(low) if low
                    else "all scoring arms above 90% coverage"))
PY

note "run finished, $(wc -l < "$OUT") instances"
echo "=== $(date) done ==="
cat "$STATUS"
