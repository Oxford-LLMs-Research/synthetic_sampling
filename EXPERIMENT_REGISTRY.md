# EXPERIMENT_REGISTRY.md

The run record for `synthetic_sampling`. **Every experiment, past and future,
gets an entry here** — one entry per scoring run family, added when the run is
submitted and completed when it lands.

This file is the index: what was asked, what came back, and enough provenance
to rebuild the run. The narrative record lives in the paper workspace
(`PAPER/docs/EXPERIMENTS.md` for what ran, `RUN_CATALOGUE.md` for what is
decided but unrun, `PAPER_STATE.md` for pre-registrations and full verdicts).
Where the two disagree about a number, the source CSV named under
**Verified by** wins.

## Paths in this file

Three roots, so entries stay readable and survive this folder being moved:

| Prefix | Location |
|---|---|
| `CODE/` | `C:\Users\murrn\cursor\synthetic_sampling\synthetic_sampling\` (git repo, remote `Oxford-LLMs-Research/synthetic_sampling`) |
| `WORK/` | `C:\Users\murrn\cursor\synthetic_sampling\` (outer workspace: `data`, `results`, `analysis`, `outputs_recovered`) |
| `PAPER/` | `C:\Users\murrn\cursor\synthetic_sampling_aaai\` |

## How to add an entry

Copy the template. Two hard limits, both load-bearing: **Rationale and Result
are three sentences each, maximum.** If a result needs more, the extra belongs
in `PAPER/docs/PAPER_STATE.md` and the entry links to it.

```markdown
### <ID> — <short title>
- **Status:** PLANNED | RUNNING | LANDED | BLOCKED | ABANDONED
- **Rationale:** Test whether X improves / changes Y. (<= 3 sentences)
- **Result:** X does not improve Y. Caveats: ... (<= 3 sentences; PENDING if unrun)
- **Ran:** <date>, job(s) <slurm ids>
- **Hardware:** <partition, GPU, node>
- **Code:** <CODE/ paths> @ <commit>
- **Inputs:** <path>
- **Outputs:** <path>
- **Verified by:** <script> against <csv>
```

Rules that keep the entries fillable:

- **Record the commit at submit time.** `CODE/scripts/cluster/run_score.sbatch`
  prints `RUNSTAMP` lines (commit, job, node, model, arms, input, output) at job
  start — read an entry's provenance off those, never off job dates. Runs before
  9 Aug 2026 predate the stamp and carry `NOT RECORDED`.
- **Never invent a field.** `NOT RECORDED` is a valid value and an honest one;
  a plausible-looking commit id is not.
- Scores are never reused across servings (see `CODE/CLAUDE.md`), so a re-run on
  a new serving is a **new entry**, not an edit to the old one.
- After editing, regenerate `SHA256SUMS.json` — see this folder's `README.md`.

## Index

| ID | Status | Date | Model(s) | One line |
|---|---|---|---|---|
| [A1-COUNTRY](#a1-country--country-injection-under-label_num) | LANDED | 8 Aug 2026 | Qwen3-32B | True country does nothing; a false one costs 2.1 points |
| [A1-TEMPORAL](#a1-temporal--temporal-context-under-label_num) | LANDED | 8 Aug 2026 | Qwen3-32B | No temporal conditioning; adding any context line costs ~1 point |
| [A1-OLMO](#a1-olmo--a1-roster-completion-on-olmo-3132b) | LANDED | 9 Aug 2026 | Olmo-3.1-32B | Placebo penalty replicates; Qwen's context-line cost does not |
| [A6-MOE](#a6-moe--moe-readout-battery-and-speed-benchmark) | LANDED | 8 Aug 2026 | Qwen3-30B-A3B | `label_num` certifies on an MoE; ~4.0x faster than dense 32B |
| [B3-NARRATIVE](#b3-narrative--validated-narrative-presentation-battery) | LANDED | 9 Aug 2026 | 3-model roster | Presentation null holds; form flips predictions, not accuracy |
| [EXPA-PARAPHRASE](#expa-paraphrase--format-stability-under-validated-paraphrase) | LANDED | 6 Aug 2026 | Qwen3-32B, Olmo-3.1-32B | Instability was mostly the scoring rule, not the model |
| [LADDER-READOUT](#ladder-readout--feature-ladder-x-elicitation) | LANDED | 6–7 Aug 2026 | Qwen3-4B/32B, Olmo-3.1-32B | Accuracy saturates on the first informative feature |
| [READOUT-GRID](#readout-grid--six-arm-readout-grid-on-olmo) | LANDED | 6 Aug 2026 | Olmo-3.1-32B | Six-arm elicitation sweep; `label_num_natural` fails on Olmo |
| [READOUT-PMI](#readout-pmi--pmi--fluency-decomposition) | LANDED | 5 Aug 2026 | Qwen3-32B | Isolates echo's option-constant fluency term |
| [READOUT-BATTERY](#readout-battery--baseline-readout-battery) | LANDED | 3–4 Aug 2026 | Qwen3-4B/32B, Olmo-3.1-32B | The substrate T0.1 and the roster comparisons run on |
| [T0.1-CALIBRATION](#t01-calibration--label-distribution-calibration-reanalysis) | LANDED | 8 Aug 2026 | reanalysis, no GPU | Rank-faithful but overconfident; one temperature repairs it |

---

## Landed

### A1-COUNTRY — country injection under `label_num`

- **Status:** LANDED
- **Rationale:** Test whether naming the respondent's country in the profile
  improves prediction accuracy. The original null was measured under echo
  scoring, whose option-constant fluency term attenuates any context effect
  toward zero, so the null may be an instrument artefact rather than a finding.
- **Result:** Injecting the true country does nothing (-0.003, CI -0.015 to
  +0.010), but injecting a *false* country costs 2.1 points (-0.021, CI -0.035
  to -0.006) — the model can be misled by country but not informed by it.
  Same-serving `echo_plain` shows neither effect, so echo did mask something
  real; caveats are that this is one model (Olmo pending, see A1-OLMO) and the
  placebo penalty just exceeds the pre-registered -0.02 band.
- **Ran:** 8 Aug 2026, job 8498891 (rounds 1–2 failed: 8492735, 8498369)
- **Hardware:** ARC HTC `short`, 1x H100, node `htc-g058`, 8 CPU / 96G; 130.4 min
- **Code:** `CODE/scripts/injection/convert_injection_instances.py`,
  `CODE/scripts/cluster/submit_a1.sh`, `CODE/scripts/cluster/run_score.sbatch`
  @ 56cf61a or later (exact cluster HEAD NOT RECORDED; rounds 1–2 prove it
  postdates bf5e760 and 56cf61a)
- **Inputs:** `CODE/outputs/country_injection/inputs/country_injection_label_set.jsonl`
  (8,550 = 2,850 x 3 conditions), from
  `WORK/outputs_recovered/country_injection_instances.jsonl`
- **Outputs:** `CODE/outputs/country_injection/results/`; job log
  `WORK/outputs_recovered/a1_logs/a1-country-qwen3-32b-8498891.out`
- **Verified by:** `CODE/scripts/injection/verify_a1_numbers.py` against
  `WORK/analysis/injection/a1_{levels,deltas}_qwen_qwen3-32b.csv`;
  coverage 8,550/8,550, all 4 arms 100% usable, replicate 99.95% on 2,098 pairs
- **Pre-registration:** `PAPER/docs/PAPER_STATE.md` 8 Aug (predictions 1, 2, 4, 5, 6)

### A1-TEMPORAL — temporal context under `label_num`

- **Status:** LANDED
- **Rationale:** Test whether telling the model when the interview happened
  (year, exact date, or country-and-year together) improves accuracy. The
  same instrument-attenuation worry as A1-COUNTRY applies, and the original
  design never tested country and time jointly, so the rebuilt file adds a
  combined cell completing a 2x2 factorial.
- **Result:** No temporal conditioning — the true year and a placebo year are
  indistinguishable (+0.001, CI -0.003 to +0.006; only 2.1% of predictions
  flip between them, against 7–10% for any injected-vs-baseline pair), and the
  2x2 interaction is +0.007, inside its +-0.02 band. All five injected
  conditions sit uniformly 0.7–1.2 points *below* baseline, so what costs
  accuracy is adding a context line at all, not what it says; caveats are one
  model only and `with_date` covering 2,083 of 2,100 base_ids.
- **Ran:** 8 Aug 2026, job 8498370 (round 1 failed: 8492736)
- **Hardware:** ARC HTC `short`, 1x H100, node `htc-g058`, 8 CPU / 96G; 185.0 min
- **Code:** as A1-COUNTRY
- **Inputs:** `CODE/outputs/temporal_context/inputs/temporal_context_label_set.jsonl`
  (12,583 = 2,100 x 5 conditions + 2,083 `with_date`)
- **Outputs:** `CODE/outputs/temporal_context/results/`; job log
  `WORK/outputs_recovered/a1_logs/a1-temporal-qwen3-32b-8498370.out`
- **Verified by:** as A1-COUNTRY; coverage 12,583/12,583, replicate 99.94% on 3,141 pairs
- **Known gap:** the pre-registered "232 of 2,100 profiles already disclose
  country" split records no rule and is not reproducible — candidate rules give
  159/240/251/273/401. `CODE/scripts/injection/analyze_a1.py` uses a stated rule
  (own country name appears verbatim as a profile answer, n=159) and the split
  is uninformative under it. Needs a dated note in `PAPER/docs/PAPER_STATE.md`.

### A6-MOE — MoE readout battery and speed benchmark

- **Status:** LANDED
- **Rationale:** Test whether `label_num` certifies on a mixture-of-experts
  model and whether its calibration matches the dense 32B pattern, while
  measuring throughput on the same serving stack. With the 235B-A22B anchor
  this separates architecture from scale (4B dense / 32B dense / 30B MoE).
- **Result:** `label_num` certifies — 0 label-logprob misses, 100% usable on
  all four arms, replicate 99.75%; calibration is the T0.1 pattern (raw ECE
  0.344 at fitted T 4.08, the 4B's is 4.03, so overconfidence tracks *active*
  parameters, falling to 0.021 after scaling) and accuracy sits between dense
  4B and dense 32B at norm 0.235. Serving is **~4.0x faster than dense 32B**
  on the same node, against a ~9.7x active-parameter ratio; the caveat is that
  its absolute accuracy is below the dense 32Bs, so it enters cross-model
  comparisons only with that level caveat.
- **Ran:** 8 Aug 2026, job 8498233
- **Hardware:** ARC HTC, 1x H100, node `htc-g058`; 17.8 min for 4,800 x 4 arms
- **Code:** `CODE/scripts/cluster/run_score.sbatch` @ commit NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/readout_set.jsonl` (4,800 instances)
- **Outputs:** `CODE/outputs/readout_moe/results/`; backup + log in
  `WORK/outputs_recovered/readout_moe/`
- **Verified by:** `WORK/analysis/calibration/verify_a6_numbers.py` (27 checks)
  and `CODE/scripts/injection/verify_a1_numbers.py` for the throughput rows
  against `WORK/analysis/injection/a1_throughput.csv`
- **Note:** the fidelity, calibration and latency findings are three readings of
  *one* serving, not three experiments. Not yet moved into
  `PAPER/docs/EXPERIMENTS.md`.

### EXPA-PARAPHRASE — format stability under validated paraphrase

- **Status:** LANDED
- **Rationale:** Test whether the historical ~36% paraphrase agreement is the
  model's own wording sensitivity or an artefact of echo scoring. The old
  synonym arm was withdrawn as invalid, so this uses meaning-preserving set-level
  paraphrases with a blind round-trip validation gate.
- **Result:** The instability was mostly the scoring rule — `echo_plain`
  agreement 39.6% (Qwen) / 46.4% (Olmo) with accuracy collapsing to chance,
  versus `label_num` 82.9% / 79.3% with accuracy essentially preserved. The
  residual 17–21% of label-readout flips is the model's own wording
  sensitivity; one question (set041) was excluded for a source-data typo
  duplicate that no paraphrase pair can blindly distinguish.
- **Ran:** 6 Aug 2026, jobs 8444993 (Qwen3-32B), 8445096 (Olmo-3.1-32B)
- **Hardware:** ARC HTC, 1x H100, nodes `htc-g053` (Qwen) / `htc-g058` (Olmo)
- **Code:** pre-rebuild package, @ 96ba37a or later ("Add the validated-paraphrase
  pipeline", 6 Aug); exact HEAD NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/paraphrase_set.jsonl` (4,699 instances, 47 questions)
- **Outputs:** `WORK/outputs_recovered/paraphrase_results_*.jsonl`, `paraphrase_STATUS_*.txt`
- **Verified by:** `WORK/analysis/readout/paraphrase_agreement_<tag>.csv`,
  `paraphrase_per_question_<tag>.csv`
- **Full entry:** `PAPER/docs/EXPERIMENTS.md` "Experiment A", `PAPER_STATE.md` 6 Aug

### LADDER-READOUT — feature ladder x elicitation

- **Status:** LANDED
- **Rationale:** Test how accuracy scales with the number of profile features
  fed to the model, and whether informative features beat random ones, under
  each elicitation arm. This fixes the scale against which every later
  presentation effect is read.
- **Result:** Accuracy saturates at ~0.35 on the first informative feature and
  the informative-vs-random gate at k=24 is only +0.04 to +0.06. AUC rises
  0.67 to 0.73 while accuracy stays flat — the model holds signal it does not
  convert into its top choice, which is the opening every later experiment
  targets.
- **Ran:** 6–7 Aug 2026, jobs 8450394 (Qwen3-32B), 8450395 (Qwen3-4B),
  8452571 (Olmo-3.1-32B)
- **Hardware:** ARC HTC, 1x H100, nodes `htc-g060` (Qwen x2) / `htc-g058` (Olmo)
- **Code:** pre-rebuild package @ f9a96e6 or later ("Add the ladder-x-elicitation
  rider", 6 Aug); exact HEAD NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/ladder_readout_set.jsonl` (17,918 instances)
- **Outputs:** `WORK/outputs_recovered/ladder_readout_results_*.jsonl`,
  `ladder_readout_STATUS_*.txt`
- **Verified by:** `WORK/analysis/ladder/ladder_readout_curve_*.csv`
- **Note:** Olmo's `label_num` shows 1,590 non-finite scores (label-tokenisation
  misses); its smoke gate flagged 8/602 options with no label logprob.

### READOUT-GRID — six-arm readout grid on Olmo

- **Status:** LANDED
- **Rationale:** Test six elicitation arms side by side on one serving to
  choose the paper's primary readout, including the `label_num_natural` and
  `echo_listed` variants not carried elsewhere.
- **Result:** `label_num` is the arm that survives; `label_num_natural` fails
  on Olmo with 2,591 non-finite scores against `label_num`'s 270, and the
  natural-label variant was dropped from the default arm set. Caveat: single
  model, so the arm choice is corroborated by, not established from, this run.
- **Ran:** 6 Aug 2026, job 8445095
- **Hardware:** ARC HTC, 1x H100, node `htc-g058`, tp 1
- **Code:** pre-rebuild package; commit NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/readout_set.jsonl` (4,800 instances)
- **Outputs:** `WORK/outputs_recovered/readout_grid_allenai_olmo-3.1-32b-instruct-dpo.jsonl`,
  `readout_grid_STATUS_*.txt`
- **Verified by:** `WORK/analysis/readout/readout_contrasts_*_grid.csv`,
  `readout_replicate_*_grid.csv`

### READOUT-PMI — PMI / fluency decomposition

- **Status:** LANDED
- **Rationale:** Test how much of echo scoring is an option-constant fluency
  term rather than a response to the profile, by scoring question-only and
  context-free premises alongside the full prompt.
- **Result:** The premises are retained as standing arms (`echo_qonly`,
  `echo_ctxfree`) in the default arm set, which is what made the A1
  attenuation hypothesis testable. Full decomposition verdict: see
  `PAPER/docs/PAPER_STATE.md`.
- **Ran:** 5 Aug 2026, job 8442517
- **Hardware:** ARC HTC, 1x H100, node `htc-g058`
- **Code:** pre-rebuild package @ 349e02b or later ("Add PMI/fluency
  decomposition arms and analysis"); exact HEAD NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/readout_set.jsonl`
- **Outputs:** `WORK/outputs_recovered/readout_pmi_qwen_qwen3-32b.jsonl`,
  `readout_pmi_STATUS_*.txt`
- **Verified by:** `WORK/analysis/readout/readout_contrasts.csv`

### READOUT-BATTERY — baseline readout battery

- **Status:** LANDED
- **Rationale:** Establish the standard 4,800-instance readout substrate and
  score it across the model roster, as the common ground every later
  experiment is read against.
- **Result:** Produced the roster baselines still in use — `label_num` norm
  accuracy 0.257 (Qwen3-32B), 0.216 (Qwen3-4B), 0.276 (Olmo-3.1-32B) — and
  the scores T0.1 reanalysed. Caveat: the generate/sampled arms were flagged
  unreliable by the smoke gate in the earliest run and are not used.
- **Ran:** 3–4 Aug 2026, jobs 8401729, 8404317 (Qwen3-4B), 8419856 (Olmo-3.1-32B)
- **Hardware:** ARC HTC, 1x H100, nodes `htc-g058` / `htc-g053`
- **Code:** pre-rebuild package; commit NOT RECORDED
- **Inputs:** `WORK/outputs_recovered/readout_set.jsonl` (built 2 Aug 2026)
- **Outputs:** `WORK/outputs_recovered/readout_results_*.jsonl`, `readout_STATUS*.txt`
- **Verified by:** `WORK/analysis/readout/readout_summary_*.csv`,
  `WORK/analysis/calibration/calibration_summary.csv`

### T0.1-CALIBRATION — label distribution calibration reanalysis

- **Status:** LANDED
- **Rationale:** Test whether the `label_num` probability distribution can be
  used as a distribution-valued estimand, rather than only its argmax. Zero
  GPU: a reanalysis of the existing readout and ladder scores.
- **Result:** Rank-faithful but overconfident — one cross-fitted temperature
  repairs top-label calibration (ECE 0.02 to 0.04), and the argmax ceiling
  matches the same-features XGBoost ceiling under GroupKFold-on-country.
  Marginals still need per-question base-rate correction (model TV ~0.27
  versus XGBoost 0.084), so temperature fixes probabilities but not shares.
- **Ran:** 8 Aug 2026 (no GPU)
- **Hardware:** local
- **Code:** `WORK/analysis/calibration/analyze_label_calibration.py`
- **Inputs:** READOUT-BATTERY and LADDER-READOUT score files
- **Outputs:** `WORK/analysis/calibration/calibration_summary.csv`,
  `temperature_scaling.csv`, `reliability_readout_*.csv`
- **Verified by:** `WORK/analysis/calibration/verify_a6_numbers.py` (shared table)

### A1-OLMO — A1 roster completion on Olmo-3.1-32B

- **Status:** LANDED
- **Rationale:** Complete the pre-registered A1 roster by running both
  injection experiments on the second model family, so the country and
  temporal verdicts rest on two architectures rather than one.
- **Result:** The country verdict replicates: the true country does nothing
  (-0.008, CI -0.020 to +0.003) while the false-country placebo again costs
  2.1 points (-0.021, CI -0.032 to -0.008, vs Qwen's -0.021), and
  year-vs-placebo stays indistinguishable (+0.002, 2.6% flips). The Qwen
  finding that adding any context line costs ~1 point does NOT replicate —
  all five injected temporal cells sit 0.6–1.2 points *above* baseline with
  CIs spanning zero — so that cost is model-specific, not a law. Caveats:
  `label_num` miss rates 1.8–5.0% (the known Olmo label-tokenisation cost;
  passes the 90% gate), and the run predates the port-isolation fix (see
  B3-NARRATIVE), though a cross-serve would have 404'd since its only
  same-model neighbour was cancelled — full job logs pending pull.
- **Ran:** 9 Aug 2026, jobs 8510995 (country) and 8510996 (temporal);
  results landed locally 21:34. Earlier rounds: 8492739/8492740,
  8498371/8498372.
- **Hardware:** ARC HTC `short`, 1x H100, nodes `htc-g058` (country) /
  `htc-g053` (temporal); runtimes NOT RECORDED (full logs pending pull)
- **Failure history:**
  - Round 1 — `ModuleNotFoundError: No module named 'pandas'` in `ss-score`
    at import. Fixed by bf5e760.
  - Round 2 — `vLLM never came up` under the old fixed-poll-budget wait; Olmo
    loads slower than Qwen3-32B and fell the wrong side of it twice, while
    `a1-temporal-qwen3-32b` passed in the same round. Fixed by 56cf61a.
  - Round 3 — abandoned; superseded by round 4 before it was diagnosed.
  - Round 4 (current) — both cells resubmitted and serving. An earlier read of
    8510995's job log showed only `Starting vLLM` and was misread here as a
    silent death; the job was still loading. Job logs are not a liveness
    signal — use `squeue` / `sacct`.
- **Code:** `CODE/scripts/cluster/submit_a1.sh` (submitted with
  `A1_MODELS="allenai/Olmo-3.1-32B-Instruct-DPO"`) @ commit NOT RECORDED
  (RUNSTAMP in the full logs, pending pull); analysis
  `CODE/scripts/injection/analyze_a1.py` @ this commit
- **Inputs:** as A1-COUNTRY / A1-TEMPORAL (same instance files, new serving)
- **Outputs:** `CODE/outputs/{country_injection,temporal_context}/results/`
  (`*olmo*.jsonl`); backup `WORK/outputs_recovered/a1_scores/`; job logs in
  `WORK/outputs_recovered/a1_logs/` (round-4 logs are stale snapshots,
  full copies pending pull)
- **Verified by:** `CODE/scripts/injection/verify_a1_numbers.py` against
  `WORK/analysis/injection/a1_{levels,deltas}_allenai_olmo-3.1-32b-instruct-dpo.csv`;
  coverage 8,550/8,550 and 12,583/12,583, replicate 100.0% on 2,098 pairs
  (country) / 99.97% on 3,141 (temporal)

### B3-NARRATIVE — validated narrative presentation battery

- **Status:** LANDED
- **Rationale:** Test whether presenting the profile as a flowing narrative
  rather than q:a pairs changes accuracy, against the wording-variance ceiling
  set by two independent narratives of the same profile. Both contrasts are
  within-pair in a single serving, so the reuse rule is satisfied by
  construction.
- **Result:** The powered null holds on all three models — no narrative-vs-qa
  delta approaches the +0.04 falsifier (largest is Olmo's narrative2-qa at
  -0.025, CI -0.073 to +0.020) and narrative1-vs-narrative2 agreement lands
  inside the predicted 75–90% band (86.0% Qwen / 81.5% Olmo / 85.6% MoE).
  The registered form-beyond-wording signature IS present on all three:
  narrative-vs-qa flips exceed narrative-vs-narrative flips (17.4–17.8% vs
  14.0% on Qwen), so form changes which predictions flip without moving
  accuracy. Since neither A1's injection effect nor B3's format effect
  cleared its +0.04 falsifier, the pre-registered A1xB3 crossed rider does
  not run; caveat: Olmo carries 105 non-finite `label_num` scores (miss
  2.2–3.2%, passes the gate).
- **Ran:** 9 Aug 2026, resubmission round on the port-isolated harness;
  job ids NOT RECORDED (logs pending pull); results landed locally 21:34.
  Failed round: 8511397 (Qwen3-32B) FAILED 15:45:10, 8511398 (Olmo)
  cancelled, 8511399 (MoE) cancelled — no scores existed from it.
- **Hardware:** ARC HTC `short`, 1x H100 per model, ~8h wall budget;
  nodes NOT RECORDED (RUNSTAMP pending pull)
- **Blocker of the failed round (diagnosed, fixed):** nothing to do with B3. Every
  request returned `HTTP 404: The model 'Qwen/Qwen3-32B' does not exist` — the
  job attached to a **different job's vLLM**. `run_score.sbatch` hardcoded
  `PORT=8000`, HTC nodes host several single-GPU jobs, this job's own
  `vllm serve` could not bind, and the readiness probe passed instantly against
  the neighbour's server. The second FATAL (`label_num: 0/0 options ... the
  tokeniser emits labels the matcher misses`) is a divide-by-zero artefact of
  an empty score set, not a cause.
- **Why it mattered beyond this job:** 8511398 was co-scheduled on `htc-g058`
  with A1-OLMO's 8510995, **both Olmo-3.1-32B-Instruct-DPO**. A matching model
  name produces no 404, so it would have passed its gate and scored the whole
  battery against the A1 job's serving — a silent violation of the one-serving
  rule that no downstream check could detect. Cancelled before that happened.
- **Harness fix (9 Aug 2026):** `PORT` now derives from `SLURM_JOB_ID`; the job
  aborts if anything already answers on that port; and readiness requires the
  served model to match `$MODEL`, not merely a 200.
- **Gate credit:** caught in 11 s against an 8 h allocation, because the smoke
  spans the design rather than the head of the file.
- **Code:** `CODE/scripts/narrative/`, `CODE/scripts/cluster/submit_b3.sh` @
  1f72154 or later (the port-isolation commit; exact cluster HEAD in the
  RUNSTAMP, pending pull); analysis `CODE/scripts/narrative/analyze_b3.py`
  @ this commit
- **Inputs:** `CODE/outputs/narrative/inputs/narrative_label_set.jsonl` — 723 of
  734 pairs x 3 arms (qa / narrative1 / narrative2); 11 pairs excluded as data
  after blind fact-recovery validation. Backup:
  `WORK/outputs_recovered/narrative_substrate_b3/`
- **Outputs:** `CODE/outputs/narrative/results/` (3 files, one per model);
  backup `WORK/outputs_recovered/narrative_scores_b3/`
- **Verified by:** substrate assembly by script at 4c1b7aa; numbers by
  `CODE/scripts/narrative/verify_b3_numbers.py` against
  `WORK/analysis/narrative/b3_{levels,contrasts}_<tag>.csv`; coverage
  2,169/2,169 per model, all 4 arms 100% usable, replicate 100.0% on 514
  pairs per model
- **Pre-registration:** `PAPER/docs/PAPER_STATE.md` 8 Aug (predictions 1–3
  and the conditional A1xB3 cross rule)
- **Roster:** Qwen3-32B, Olmo-3.1-32B, Qwen3-30B-A3B (the MoE joins on A6
  certification; read within model, never as a level shift)
- **Constraint:** never split the arms across jobs or shard by arm — the three
  arms must score in one serving.

---

## Not yet registered

Catalogue entries A2–A5, B1, B2, C1 and T0.2–T0.5 are decided but unrun; they
enter this file when submitted. Historical pre-rebuild runs (`control_*`,
`scaling_*`, `readout_mini`) exist in `WORK/outputs_recovered/` without STATUS
files and are unregistered — their rationale and results need recovering from
`PAPER/docs/EXPERIMENTS.md` before entries can be written honestly.
