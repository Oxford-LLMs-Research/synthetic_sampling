# CLAUDE.md (code repo)

Standing rules for `synthetic_sampling` on branch `rebuild/minimal-core`.
Paper/build workspace rules live in `synthetic_sampling_aaai/CLAUDE.md`.

## Repo map

- This repo is the experiment code (git-tracked, remote
  `Oxford-LLMs-Research/synthetic_sampling`).
- Survey microdata and result dumps live in the sibling workspace
  `../data`, `../results`, `../analysis` (not in git).
- Paper workspace: `C:\Users\murrn\cursor\synthetic_sampling_aaai\`.

## Pipeline modules

`surveys` -> `profiles` -> `scoring` -> `analysis` / `checks`.
Thin CLIs: `ss-generate`, `ss-score`, `ss-analyze`.

## Experiment rules

- Reuse rule: instances yes, scores never across servings (serving-stack
  confound is worth 36 percent of predictions). Replicate arm is mandatory;
  agreement is read against 64 percent (cross-serving floor), not 100 percent.
- Default scoring arm is `label_num`. `echo_plain` is retained as the paper
  method / serving control; PMI premises `echo_qonly` / `echo_ctxfree` are kept.
- Dated note (6 Aug 2026): the paper-workspace rule remains `--scoring echo`
  until Phase 1 certifies `label_num` across the model roster. Both arms exist;
  do not flip the paper CLAUDE rule from this repo.
- `.sbatch` files must be LF; never patch them through `Path.write_text` on
  Windows.
- Cluster operations: paper workspace `docs/CLUSTER_HANDOFF.md`. Cluster
  storage is not backed up.

## Verification

- Every reported number is verified against its source CSV by script
  (`checks.number_verify`), never by eye.
- Smoke gate and coverage report must pass before a full allocation runs.

## Phase 0 hooks

`surveys.harmonise` and `SurveyConfig.interview_date_col` are stubs until the
hygiene census lands. Do not invent exclusions without the census.
