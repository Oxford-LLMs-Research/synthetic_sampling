# Synthetic Sampling (minimal core)

Rebuild branch `rebuild/minimal-core`: the smallest pipeline that supports the
new study (primary elicitation `label_num`).

## Pipeline

```
surveys  ->  profiles  ->  instances.jsonl  ->  scoring  ->  results.jsonl
                              |                    |
                           checks.smoke      analysis + checks.coverage
```

Install editable:

```
pip install -e ".[dev]"
```

Generate instances:

```
ss-generate --config configs/local.yaml --survey wvs \
  --out outputs/<experiment>/inputs/instances.jsonl
```

Score (default arms: `label_num`, `echo_plain`, `echo_qonly`, `echo_ctxfree`):

```
ss-score --input outputs/<experiment>/inputs/instances.jsonl \
  --out outputs/<experiment>/results/results.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-32B
```

Checks:

```
ss-analyze smoke outputs/<experiment>/results/smoke.jsonl
ss-analyze coverage outputs/<experiment>/results/results.jsonl \
  outputs/<experiment>/inputs/instances.jsonl
ss-analyze replicate outputs/<experiment>/results/results.jsonl --arm label_num
```

## Outputs layout

One folder per experiment under `outputs/`, split by role; no loose files at
the `outputs/` root:

```
outputs/
  <experiment>/            e.g. country_injection, temporal_context
    inputs/                instance files fed to the models
    results/               scored outputs, smoke and status artifacts
```

`outputs/` is gitignored and regenerable: inputs come from generators or
converters in `scripts/`, results from scoring runs. Run artifacts pulled back
from the cluster that are NOT regenerable live in the outer workspace
(`../outputs_recovered/`), outside the repo, so repo tidying cannot delete
them.

## Package layout

| Module | Role |
|--------|------|
| `surveys/` | Registry, loaders, hygiene hooks, bundled metadata |
| `profiles/` | Generator, ESS concept resolve, DatasetBuilder, leakage |
| `scoring/` | /completions client, prompts, arms, runner |
| `analysis/` | Normalized accuracy, AUC, prior-correct, bootstrap, replicate |
| `checks/` | Smoke, coverage, number verification |

## Standing rules

See [CLAUDE.md](CLAUDE.md). Arms contract: [docs/ELICITATIONS.md](docs/ELICITATIONS.md).
Full pipeline notes: [docs/PIPELINE.md](docs/PIPELINE.md).

Paper workspace (`synthetic_sampling_aaai`) stays on `--scoring echo` until
Phase 1 confirms `label_num` across the roster.
