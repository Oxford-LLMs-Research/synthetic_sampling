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
ss-generate --config configs/local.yaml --survey wvs --out outputs/instances.jsonl
```

Score (default arms: `label_num`, `echo_plain`, `echo_qonly`, `echo_ctxfree`):

```
ss-score --input outputs/instances.jsonl --out outputs/results.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-32B
```

Checks:

```
ss-analyze smoke outputs/smoke.jsonl
ss-analyze coverage outputs/results.jsonl outputs/instances.jsonl
ss-analyze replicate outputs/results.jsonl --arm label_num
```

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
