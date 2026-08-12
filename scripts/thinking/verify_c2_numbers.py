"""Pin every C2 number quoted anywhere against the analysis CSVs.

Run after `analyze_c2.py`. Nothing about C2 gets written into PAPER_STATE
or the paper until this exits 0.

Verdicts these rows carry, against the 12 Aug pre-registration:
- Endpoint (thinking-minus-direct within +-0.02, falsifier +0.04): the
  falsifier is untouched and the sign runs the other way. Raw readout
  -0.0200 (CI -0.057..+0.015, spanning zero, at the band edge); chat
  readout -0.0400 (CI -0.079..-0.007, EXCLUDING zero) — properly-elicited
  native thinking significantly hurts against the direct chat readout
  (itself the best cell, 0.575 raw acc, consistent with A4's template
  gain). C1's null was not an artifact of bad elicitation.
- Instrument claim (native templating cures the C1 pathology): empty rate
  0.0% PASSES; block presence and closure 100.0%; parse failure 0.0%
  (734/734 bare-digit); truncation 0. Loop census 6.1% vs the <5%
  target — a MARGINAL MISS on a noisy lower bound, against C1's 95%.
- Calibration: the confidence-inflation signature replicates under the
  clean instrument — mean conf 0.805 -> 0.957 (raw), 0.907 -> 0.982
  (chat); ECE 0.253 -> 0.425 and 0.332 -> 0.450. Thinking makes the
  model more wrong about how right it is, with zero pathology to blame.
- Stated answers agree with the injected label readouts at 99.3%/98.6%
  and their accuracy (0.531) sits BELOW the direct chat readout (0.575):
  the model's own post-thinking answers lose to just asking directly.
- Replicates 100.0% on 365 pairs, both arms.

    python scripts/thinking/verify_c2_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "thinking"

TAG = "qwen_qwen3-32b"

LEVELS = [
    dict(arm="label_num", condition="direct", n=734, miss_rate=0.0000,
         norm_acc=0.3372, tv=0.2834, mean_conf=0.8045, ece=0.2525),
    dict(arm="label_num", condition="thinking", n=734, miss_rate=0.0000,
         norm_acc=0.3086, tv=0.2634, mean_conf=0.9573, ece=0.4246),
    dict(arm="chat_label_num", condition="direct", n=734, miss_rate=0.0000,
         norm_acc=0.3728, tv=0.2674, mean_conf=0.9069, ece=0.3319),
    dict(arm="chat_label_num", condition="thinking", n=734, miss_rate=0.0000,
         norm_acc=0.3124, tv=0.2621, mean_conf=0.9823, ece=0.4497),
]

CONTRASTS = [
    dict(arm="label_num", contrast="thinking-direct", n_pairs=734,
         delta_acc=-0.0200, ci_lo=-0.0573, ci_hi=0.0147,
         flip_rate=0.2207, agree_rate=0.7793),
    dict(arm="label_num", contrast="replicate", n_pairs=365,
         delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
         flip_rate=0.0000, agree_rate=1.0000),
    dict(arm="chat_label_num", contrast="thinking-direct", n_pairs=734,
         delta_acc=-0.0400, ci_lo=-0.0787, ci_hi=-0.0067,
         flip_rate=0.2343, agree_rate=0.7657),
    dict(arm="chat_label_num", contrast="replicate", n_pairs=365,
         delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
         flip_rate=0.0000, agree_rate=1.0000),
]

ELICITATION = [
    dict(model=TAG, n_transcripts=734, n_generation_error=0,
         block_rate=1.0000, closed_rate=1.0000, empty_rate=0.0000,
         loop_rate=0.0613, parse_fail_rate=0.0000,
         median_think_words=467.5, stated_acc=0.5313,
         stated_vs_label_agree=0.9932, stated_vs_chat_label_agree=0.9864,
         n_parsed=734),
]


def main() -> int:
    bad: list[str] = []
    n = 0
    bad += verify_table_against_csv(
        LEVELS, ANALYSIS / f"c2_levels_{TAG}.csv",
        join_keys=("arm", "condition"),
        value_cols=("n", "miss_rate", "norm_acc", "tv",
                    "mean_conf", "ece"), tol=5e-5)
    n += len(LEVELS) * 6
    bad += verify_table_against_csv(
        CONTRASTS, ANALYSIS / f"c2_contrasts_{TAG}.csv",
        join_keys=("arm", "contrast"),
        value_cols=("n_pairs", "delta_acc", "ci_lo", "ci_hi",
                    "flip_rate", "agree_rate"), tol=5e-5)
    n += len(CONTRASTS) * 6
    bad += verify_table_against_csv(
        ELICITATION, ANALYSIS / f"c2_elicitation_{TAG}.csv",
        join_keys=("model",),
        value_cols=("n_transcripts", "n_generation_error", "block_rate",
                    "closed_rate", "empty_rate", "loop_rate",
                    "parse_fail_rate", "median_think_words", "stated_acc",
                    "stated_vs_label_agree", "stated_vs_chat_label_agree",
                    "n_parsed"), tol=5e-5)
    n += len(ELICITATION) * 12

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
