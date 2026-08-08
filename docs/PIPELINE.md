# Pipeline

## 1. Load surveys

`SurveyLoader` reads microdata from `DataPaths.raw_data_dir` and bundled JSON
from `surveys/metadata/`. `apply_harmonisation` is called on every load; Phase 0
fills option-set hygiene and interview-date population. ESS configs already
declare `interview_date_col=inwds`.

## 2. Generate instances

`DatasetBuilder` (profiles) samples respondents and targets, builds profiles
with leakage exclusions, and resolves ESS country-specific **concepts** to
per-respondent variables via `CountrySpecificHandler`. Output: JSONL with
`example_id`, `questions`, `target_question`, `options` / `option_sets`.

```
ss-generate --config configs/local.yaml \
  --out outputs/<experiment>/inputs/instances.jsonl
```

## 3. Score

`ss-score` posts to an OpenAI-compatible `/completions` endpoint.

Default arms: `label_num`, `echo_plain`, `echo_qonly`, `echo_ctxfree`.
Replicate: hash-stable fraction of instances also scored as
`original_replicate`. Resume by `example_id`. Optional sharding via
`--shard-index` / `--shard-count`.

`label_num` uses a cyclic Latin square and a trailing space after `Answer:`.
See [ELICITATIONS.md](ELICITATIONS.md).

```
ss-score --input outputs/<experiment>/inputs/instances.jsonl \
  --out outputs/<experiment>/results/results.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-32B \
  --replicate-frac 0.1
```

## 4. Check

```
ss-analyze smoke outputs/<experiment>/results/smoke.jsonl
ss-analyze coverage outputs/<experiment>/results/results.jsonl \
  outputs/<experiment>/inputs/instances.jsonl
```

Smoke aborts only on fatal label-readout failure. Coverage fails if any arm is
below 90% usable.

## 5. Analyse

Importable kernels in `synthetic_sampling.analysis`:

- `normalized_accuracy` (stated-M convention)
- discrimination `auc` / `weighted_auc`
- `prior_corrected`
- `clustered_bootstrap_ci` / `question_mean`
- `replicate_agreement` / `summarize_controls`

```
ss-analyze replicate outputs/<experiment>/results/results.jsonl --arm label_num
```

## Outputs layout

One folder per experiment under `outputs/`, split by role:

```
outputs/<experiment>/inputs/    instance files fed to the models
outputs/<experiment>/results/   scored outputs, smoke and status artifacts
```

No loose files at the `outputs/` root. `outputs/` is gitignored and must stay
regenerable (inputs from `scripts/` generators or converters, results from
scoring runs); anything pulled from the cluster that is NOT regenerable goes to
the outer workspace `../outputs_recovered/`, outside the repo. Converters and
launchers default their paths to this layout
(`scripts/convert_injection_instances.py` is the pattern).

## Cluster

`scripts/cluster/run_score.sbatch` starts vLLM then `ss-score`. Keep LF line
endings. Full grid launchers for Phase 1 remain on branch `make-package` until
that phase moves over.
