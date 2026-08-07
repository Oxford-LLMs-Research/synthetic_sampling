# Elicitation arms: the reference card

One instance runs through every example: WVS Q149, "Most people consider both
freedom and equality to be important, but if you had to choose between them,
which one would you consider more important?", options `Freedom`, `Equality`,
`Don't know`. All arms share the paper's prompt skeleton (preamble, Profile
block, Question, Instructions line, trailing `Answer:`); only the option block
and the emission target vary. All arms go through /completions. Implementation (minimal-core):
`synthetic_sampling.scoring` (`build_prompt`, `score_echo`, `score_labels`).
Ancestor on `make-package`: `scripts/scaling/score_formats.py`.

## Scored arms

| arm | options in prompt | instruction line | requests per instance | what is read |
|---|---|---|---|---|
| `echo_plain` | none | "Reply with a short concise answer." | M (one per option) | mean token log-prob of the option's OWN words appended after `Answer:` |
| `echo_listed` | bulleted `- Freedom` | same | M | same echo rule, options visible |
| `echo_listed_numinstr` | numbered `1. Freedom` | "Reply with only the option number." | M | same echo rule; pilot-only control for the instruction-wording confound |
| `label_num` | numbered, rotated M times (cyclic Latin square) | "Reply with only the option number." | M (one per rotation) | log-prob of each option's DIGIT token at the answer slot, averaged over positions; prompt must end `Answer: ` with the trailing space |
| `label_num_natural` | numbered, natural order, once | same | 1 | same digit readout, no rotation; vs `label_num` separates position debiasing from ordinal-order disruption |
| `echo_qonly` | none; profile EMPTIED, question kept | same as echo_plain | M, cached per question | echo rule under the question-only premise: fluency + question-fit, the population prior in string form |
| `echo_ctxfree` | none; entire prompt is `Answer:` | none | M, cached per option set (~157 strings total) | echo rule under no context: pure wording fluency in answer position |

Dropped arms: `label_alpha` (letters not reliably emitted; itself a finding),
`generate`/`generate_sampled` (self-agreement below the cross-stack floor;
unusable as measurement).

## Derived rules (arithmetic on stored scores, no requests)

| rule | formula | cancels | leaves |
|---|---|---|---|
| `pmi_free` | echo_plain − echo_ctxfree | wording fluency (Holtzman et al. 2021 domain-conditional PMI) | question-fit + profile-driven |
| `pmi_q` | echo_plain − echo_qonly | fluency AND question-fit | profile-driven only |
| ladder contrastive | score(k) − score(k=0), per rung | same as pmi_q, inside the ladder design | feature-driven component per rung |

## Controls

- `original_replicate`: 25% of instances re-scored on unchanged input, fresh
  requests, same serving. Agreement ~99.9-100% is the ceiling every agreement
  rate is read against (the cross-serving floor is 64.4%).
- Every run scores a fresh `echo_plain` in its own serving; stored scores are
  never reused across servings (36% of predictions flip between stacks).

## Reference levels (Qwen 3 32B, 4,800 readout instances, one serving)

| rule | norm acc | note |
|---|---|---|
| echo_ctxfree as predictor | −0.101 | the fluency default; anti-predicts |
| echo_qonly as predictor | 0.013 | question prior in string form |
| echo_plain (the paper) | 0.115 | AUC 0.586 |
| pmi_q | 0.059 | stripping the question prior hurts the argmax |
| pmi_free | 0.149 | +0.033 vs echo_plain, n.s.; a fifth of the gap to label_num |
| echo_listed | 0.195 | seeing the options |
| label_num | 0.257 | AUC 0.603; the repaired readout |
| label_num_natural | 0.267 | rotation costs ~1 point here |
| majority class | 0.337 | model-free |

Reading down the scored rows: accuracy rises as the score depends less on how
the answer is worded and more on what the model chose. The derived rows show
the rise is not recoverable by arithmetic on hidden-options scores: the PMI
verdict (PAPER_STATE.md, 5 Aug) is that what the label readout adds is the
option list itself, not better normalization.
