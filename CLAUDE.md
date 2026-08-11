# CLAUDE.md (code repo)

Standing rules for `synthetic_sampling` on branch `rebuild/minimal-core`.
Paper/build workspace rules live in `synthetic_sampling_aaai/CLAUDE.md`.

## Repo map

- This repo is the experiment code (git-tracked, remote
  `Oxford-LLMs-Research/synthetic_sampling`).
- Survey microdata and result dumps live in the sibling workspace
  `../data`, `../results`, `../analysis` (not in git).
- Paper workspace: `C:\Users\murrn\cursor\synthetic_sampling_aaai\`.
- Outputs layout (dated 8 Aug 2026): one folder per experiment,
  `outputs/<experiment>/inputs/` (instance files fed to the models) and
  `outputs/<experiment>/results/` (scored outputs, smoke/status artifacts).
  No loose files at the `outputs/` root. `outputs/` is gitignored and must
  stay regenerable; ANYTHING non-regenerable goes to the outer
  `../outputs_recovered/`, never inside the repo. Widened 9 Aug 2026: that
  directory was for cluster pulls, but the hazard is the same whichever
  machine made the file — `outputs/` is documented as safe to clear and
  rebuild, so a non-regenerable artifact left there is one cleanup away from
  gone. Sampled LLM generation is non-regenerable by this test: re-running it
  yields different text, which is a different substrate, not a restored one.
  Back it up as a self-contained folder with a README stating why it cannot
  be rebuilt, plus `SHA256SUMS.json` so the copy can be re-verified without
  the source (`narrative_substrate_b3/` is the pattern). `scripts/` mirrors the
  same layout (one folder per experiment; generic wrappers and `cluster/`
  at top level); code needed by two experiments graduates to `src/`,
  never copied between script folders.

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

## Experiment registry

`EXPERIMENT_REGISTRY.md` (repo root, tracked) is the run record: every
experiment, past and future, gets an entry. Add it when the run is SUBMITTED
(status PLANNED / RUNNING) and complete it when the run lands; an experiment
that ran without an entry did not happen as far as this project is concerned.
Rationale and Result are capped at three sentences each — anything longer
belongs in the paper workspace's `PAPER_STATE.md`, which the entry links to.
Never invent a field: `NOT RECORDED` is a valid value, a plausible-looking
commit id is not. `run_score.sbatch` prints `RUNSTAMP` lines (commit, job,
node, model, arms) at job start; read an entry's provenance off those, not off
job dates.

Held in two places on purpose (decided 9 Aug 2026): git gives it a revision
history, and `../outputs_recovered/experiment_registry/` keeps a copy that
survives losing the checkout — it is not regenerable, no script produces its
prose, and for runs before 9 Aug it is the only surviving statement of what
produced those scores. **The repo copy is canonical; the mirror is never
hand-edited.** Run `python scripts/sync_registry.py` after every registry edit,
in the same commit, and `--check` to test for drift. Paths inside the file use
`CODE/` `WORK/` `PAPER/` prefixes, not repo-relative links, so the two copies
stay byte-identical.

## Verification

- Every reported number is verified against its source CSV by script
  (`checks.number_verify`), never by eye.
- Smoke gate and coverage report must pass before a full allocation runs.

## Phase 0 hooks

`surveys.harmonise` and `SurveyConfig.interview_date_col` are stubs until the
hygiene census lands. Do not invent exclusions without the census.
