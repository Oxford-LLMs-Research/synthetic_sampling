#!/bin/bash
#SBATCH -J formats-vllm
#SBATCH -p short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o /data/polf-sula/nuff1496/logs/formats_%j.out
#SBATCH -e /data/polf-sula/nuff1496/logs/formats_%j.err

# Is the wording instability in the scoring RULE or in the MODEL?
#
# Paraphrasing the answer options while preserving their meaning changes the
# paper's prediction for 63% of respondents (36.2% agreement against 22.9%
# chance), measured on results that already existed for four models. Reversing
# the option ORDER changes only 17%. So echo scoring was chosen for the
# invariance that turned out not to matter.
#
# Two explanations remain, and they call for opposite responses: the readout is
# ranking phrases by fluency, in which case reading a label token fixes it and
# the method should change; or the model's belief is itself unstable under
# rephrasing, in which case nothing fixes it and the instability is a finding
# about synthetic respondents.
#
# Four readouts on the SAME 1,252 instances under both option sets, paired at
# the instance level: echo with options shown, numbered labels, lettered
# labels, and free generation matched verbatim. Position bias is removed by a
# cyclic Latin square, not by randomisation, because randomising converts it to
# noise and that noise lands squarely on the per-instance agreement rate this
# run exists to measure.
#
# One GPU, ~30k requests. score_formats.py resumes from --out, so requeue if it
# hits the walltime.

set -euo pipefail

DATA=/data/polf-sula/nuff1496
PORT=8300
LOG=$DATA/vllm_formats_${SLURM_JOB_ID}.log

IN=$DATA/data/surface_form_test.jsonl
OUT=$DATA/data/formats_results.jsonl

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
# two servings of this model differ by ~0.9 nats in mean option logprob and pick
# the same option only 64% of the time, so results split across configurations
# would not be comparable.
HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen3-32B \
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

# The smoke test checks the one thing that cannot be verified offline: that the
# server actually returns the label token among its top logprobs. If the
# tokeniser emits labels in a form the matcher misses, every label arm silently
# collapses to -inf and the run is wasted.
echo "=== smoke (15 instances) ==="
SMOKE=$DATA/data/formats_smoke.jsonl
rm -f "$SMOKE"
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$IN" --out "$SMOKE" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --workers 16 --limit 15
python - "$SMOKE" <<'PY'
import json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
assert len(rows) == 15, f"expected 15 scored, got {len(rows)}"
bad = [k for r in rows for k, v in r["results"].items() if "error" in v]
assert not bad, f"errors in arms: {sorted(set(bad))[:5]}"
for arm in ("label_num", "label_alpha"):
    keys = [k for k in rows[0]["results"] if k.endswith(arm)]
    assert keys, f"no {arm} results"
    vals = [v for r in rows for k in keys
            for v in r["results"][k]["scores"].values()]
    finite = [v for v in vals if math.isfinite(v)]
    assert len(finite) > 0.9 * len(vals), (
        f"{arm}: {len(vals) - len(finite)}/{len(vals)} options never received a "
        "label logprob; the tokeniser is emitting labels the matcher misses")
gen = [r["results"][k] for r in rows for k in r["results"] if k.endswith("generate")]
matched = sum(1 for g in gen if g.get("predicted") is not None)
print(f"smoke OK: {len(rows)} instances, label arms finite, "
      f"generate matched {matched}/{len(gen)}")
if matched < 0.5 * len(gen):
    print("WARNING: generate arm matching under 50%; inspect 'raw' fields")
PY

echo "=== full run ==="
python "$DATA/scripts/scaling/score_formats.py" \
  --input "$IN" --out "$OUT" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen3-32B \
  --workers 32

echo "=== $(date) done ==="
wc -l "$OUT"
