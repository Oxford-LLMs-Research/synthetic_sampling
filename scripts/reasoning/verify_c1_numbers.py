"""Pin every C1 number quoted anywhere against the analysis CSVs.

Run after `analyze_c1.py` for all three roster models. Nothing about C1 gets
written into PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the 8 Aug pre-registration:
- Prediction 1 (reasoned-minus-qa within +-0.02, falsifier +0.04): holds on
  all three models — -0.0116 (Qwen3-32B), -0.0158 (Olmo), -0.0063 (MoE),
  every CI spanning zero and none approaching +0.04. Reason-then-answer does
  not move the accuracy ceiling; every point estimate is slightly negative.
- Prediction 2 (parse failure <5% on instruction-tuned models): PASSES for
  both Qwens (0.27% / 0.14% / 0.00%) and FAILS for Olmo (35.2% qa, 41.9%
  narrative) — almost entirely EMPTY completions (immediate stop, zero
  text) on the raw /completions continuation, not format misses. A datum,
  recorded, never repaired; Olmo's reasoned arms are diluted accordingly.
- Prediction 3 (interaction beyond +-0.02 = non-additivity): Qwen3-32B's
  point estimate is -0.0289 (CI -0.064 to +0.007, spanning zero), driven by
  narrative_reasoned-narrative_direct = -0.0402 (CI -0.084 to -0.002, the
  one primary CI excluding zero). Olmo 0.0000 and MoE -0.0072 are null.
- Calibration (reported either way): reasoning inflates label_num mean
  confidence on every model (e.g. Qwen3-32B 0.805 -> 0.950) with no accuracy
  gain, so ECE worsens everywhere (+0.16 / +0.16 / +0.09 qa->reasoned).
- Stated-vs-label agreement 93.7-98.9%: the pre-commitment label readout
  and the stated final answer are near-interchangeable readouts.
- Replicate ceiling 100.0% on 744 pairs for every arm and model, read
  against the 64.4% cross-serving floor.

    python scripts/reasoning/verify_c1_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "reasoning"

# tag -> (levels rows, contrast rows, elicitation rows), all read off the
# analyze_c1 CSVs.
TABLES = {
    "qwen_qwen3-32b": (
        [
            dict(arm="label_num", condition="qa", n=734,
                 miss_rate=0.0000, norm_acc=0.3374, tv=0.2821,
                 mean_conf=0.8046, ece=0.2554),
            dict(arm="label_num", condition="reasoned", n=734,
                 miss_rate=0.0000, norm_acc=0.3152, tv=0.2510,
                 mean_conf=0.9501, ece=0.4119),
            dict(arm="label_num", condition="narrative_direct", n=723,
                 miss_rate=0.0000, norm_acc=0.3306, tv=0.2746,
                 mean_conf=0.7926, ece=0.2490),
            dict(arm="label_num", condition="narrative_reasoned", n=723,
                 miss_rate=0.0000, norm_acc=0.2584, tv=0.2975,
                 mean_conf=0.9437, ece=0.4402),
            dict(arm="echo_plain", condition="qa", n=734,
                 miss_rate=0.0000, norm_acc=0.2468, tv=0.3610,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="reasoned", n=734,
                 miss_rate=0.0000, norm_acc=0.2322, tv=0.3261,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_direct", n=723,
                 miss_rate=0.0000, norm_acc=0.2079, tv=0.3976,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_reasoned", n=723,
                 miss_rate=0.0000, norm_acc=0.1215, tv=0.3942,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=-0.0116, ci_lo=-0.0450, ci_hi=0.0221,
                 flip_rate=0.2139, agree_rate=0.7861),
            dict(arm="label_num",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0402, ci_lo=-0.0841, ci_hi=-0.0015,
                 flip_rate=0.2503, agree_rate=0.7497),
            dict(arm="label_num", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0064, ci_lo=-0.0336, ci_hi=0.0219,
                 flip_rate=0.1798, agree_rate=0.8202),
            dict(arm="label_num", contrast="interaction", n_pairs=723,
                 delta_acc=-0.0289, ci_lo=-0.0637, ci_hi=0.0074,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="label_num", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=-0.0078, ci_lo=-0.0690, ci_hi=0.0657,
                 flip_rate=0.3651, agree_rate=0.6349),
            dict(arm="echo_plain",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0512, ci_lo=-0.1419, ci_hi=0.0365,
                 flip_rate=0.4675, agree_rate=0.5325),
            dict(arm="echo_plain", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0304, ci_lo=-0.0884, ci_hi=0.0296,
                 flip_rate=0.2974, agree_rate=0.7026),
            dict(arm="echo_plain", contrast="interaction", n_pairs=723,
                 delta_acc=-0.0416, ci_lo=-0.1201, ci_hi=0.0331,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="echo_plain", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
        [
            dict(cell="qa", n_transcripts=734,
                 parse_fail_rate=0.0027, empty_rate=0.0000,
                 stated_acc=0.5423, stated_vs_label_agree=0.9891,
                 n_parsed=732, n_no_marker=1),
            dict(cell="narrative", n_transcripts=723,
                 parse_fail_rate=0.0014, empty_rate=0.0000,
                 stated_acc=0.5014, stated_vs_label_agree=0.9778,
                 n_parsed=722, n_no_marker=0),
        ],
    ),
    "allenai_olmo-3.1-32b-instruct-dpo": (
        [
            dict(arm="label_num", condition="qa", n=734,
                 miss_rate=0.0218, norm_acc=0.3278, tv=0.2882,
                 mean_conf=0.7573, ece=0.2183),
            dict(arm="label_num", condition="reasoned", n=734,
                 miss_rate=0.0967, norm_acc=0.3023, tv=0.2861,
                 mean_conf=0.8856, ece=0.3758),
            dict(arm="label_num", condition="narrative_direct", n=723,
                 miss_rate=0.0304, norm_acc=0.3026, tv=0.3046,
                 mean_conf=0.7954, ece=0.2804),
            dict(arm="label_num", condition="narrative_reasoned", n=723,
                 miss_rate=0.1342, norm_acc=0.2719, tv=0.2924,
                 mean_conf=0.8857, ece=0.4120),
            dict(arm="echo_plain", condition="qa", n=734,
                 miss_rate=0.0000, norm_acc=0.2426, tv=0.3674,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="reasoned", n=734,
                 miss_rate=0.0000, norm_acc=0.2419, tv=0.3248,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_direct", n=723,
                 miss_rate=0.0000, norm_acc=0.2250, tv=0.3822,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_reasoned", n=723,
                 miss_rate=0.0000, norm_acc=0.2136, tv=0.3289,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=-0.0158, ci_lo=-0.0505, ci_hi=0.0177,
                 flip_rate=0.2016, agree_rate=0.7984),
            dict(arm="label_num",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0176, ci_lo=-0.0462, ci_hi=0.0112,
                 flip_rate=0.1826, agree_rate=0.8174),
            dict(arm="label_num", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0102, ci_lo=-0.0639, ci_hi=0.0423,
                 flip_rate=0.2697, agree_rate=0.7303),
            dict(arm="label_num", contrast="interaction", n_pairs=723,
                 delta_acc=0.0000, ci_lo=-0.0333, ci_hi=0.0335,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="label_num", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=-0.0010, ci_lo=-0.0440, ci_hi=0.0471,
                 flip_rate=0.2698, agree_rate=0.7302),
            dict(arm="echo_plain",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0038, ci_lo=-0.0428, ci_hi=0.0417,
                 flip_rate=0.2351, agree_rate=0.7649),
            dict(arm="echo_plain", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0121, ci_lo=-0.0487, ci_hi=0.0249,
                 flip_rate=0.3347, agree_rate=0.6653),
            dict(arm="echo_plain", contrast="interaction", n_pairs=723,
                 delta_acc=-0.0013, ci_lo=-0.0388, ci_hi=0.0371,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="echo_plain", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
        [
            dict(cell="qa", n_transcripts=734,
                 parse_fail_rate=0.3515, empty_rate=0.3488,
                 stated_acc=0.5189, stated_vs_label_agree=0.9370,
                 n_parsed=476, n_no_marker=257),
            dict(cell="narrative", n_transcripts=723,
                 parse_fail_rate=0.4191, empty_rate=0.4080,
                 stated_acc=0.4762, stated_vs_label_agree=0.9500,
                 n_parsed=420, n_no_marker=302),
        ],
    ),
    "qwen_qwen3-30b-a3b-instruct-2507": (
        [
            dict(arm="label_num", condition="qa", n=734,
                 miss_rate=0.0000, norm_acc=0.3529, tv=0.2558,
                 mean_conf=0.8867, ece=0.3295),
            dict(arm="label_num", condition="reasoned", n=734,
                 miss_rate=0.0000, norm_acc=0.3436, tv=0.2272,
                 mean_conf=0.9676, ece=0.4201),
            dict(arm="label_num", condition="narrative_direct", n=723,
                 miss_rate=0.0000, norm_acc=0.3358, tv=0.2160,
                 mean_conf=0.8772, ece=0.3322),
            dict(arm="label_num", condition="narrative_reasoned", n=723,
                 miss_rate=0.0000, norm_acc=0.3177, tv=0.2532,
                 mean_conf=0.9725, ece=0.4400),
            dict(arm="echo_plain", condition="qa", n=734,
                 miss_rate=0.0000, norm_acc=0.3137, tv=0.3084,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="reasoned", n=734,
                 miss_rate=0.0000, norm_acc=0.3446, tv=0.2440,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_direct", n=723,
                 miss_rate=0.0000, norm_acc=0.3046, tv=0.3505,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative_reasoned", n=723,
                 miss_rate=0.0000, norm_acc=0.2731, tv=0.2850,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=-0.0063, ci_lo=-0.0440, ci_hi=0.0274,
                 flip_rate=0.2262, agree_rate=0.7738),
            dict(arm="label_num",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0123, ci_lo=-0.0480, ci_hi=0.0243,
                 flip_rate=0.2476, agree_rate=0.7524),
            dict(arm="label_num", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0093, ci_lo=-0.0400, ci_hi=0.0173,
                 flip_rate=0.2185, agree_rate=0.7815),
            dict(arm="label_num", contrast="interaction", n_pairs=723,
                 delta_acc=-0.0072, ci_lo=-0.0429, ci_hi=0.0281,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="label_num", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="reasoned-qa", n_pairs=734,
                 delta_acc=0.0194, ci_lo=-0.0400, ci_hi=0.0851,
                 flip_rate=0.2888, agree_rate=0.7112),
            dict(arm="echo_plain",
                 contrast="narrative_reasoned-narrative_direct", n_pairs=723,
                 delta_acc=-0.0133, ci_lo=-0.0736, ci_hi=0.0432,
                 flip_rate=0.3264, agree_rate=0.6736),
            dict(arm="echo_plain", contrast="narrative_direct-qa", n_pairs=723,
                 delta_acc=-0.0081, ci_lo=-0.0381, ci_hi=0.0202,
                 flip_rate=0.2089, agree_rate=0.7911),
            dict(arm="echo_plain", contrast="interaction", n_pairs=723,
                 delta_acc=-0.0325, ci_lo=-0.0983, ci_hi=0.0254,
                 flip_rate=float("nan"), agree_rate=float("nan")),
            dict(arm="echo_plain", contrast="replicate", n_pairs=744,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
        [
            dict(cell="qa", n_transcripts=734,
                 parse_fail_rate=0.0000, empty_rate=0.0000,
                 stated_acc=0.5504, stated_vs_label_agree=0.9482,
                 n_parsed=734, n_no_marker=0),
            dict(cell="narrative", n_transcripts=723,
                 parse_fail_rate=0.0000, empty_rate=0.0000,
                 stated_acc=0.5297, stated_vs_label_agree=0.9502,
                 n_parsed=723, n_no_marker=0),
        ],
    ),
}


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, (levels, contrasts, elic) in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            levels, ANALYSIS / f"c1_levels_{tag}.csv",
            join_keys=("arm", "condition"),
            value_cols=("n", "miss_rate", "norm_acc", "tv",
                        "mean_conf", "ece"), tol=5e-5)]
        n += len(levels) * 6
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            contrasts, ANALYSIS / f"c1_contrasts_{tag}.csv",
            join_keys=("arm", "contrast"),
            value_cols=("n_pairs", "delta_acc", "ci_lo", "ci_hi",
                        "flip_rate", "agree_rate"), tol=5e-5)]
        n += len(contrasts) * 6
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            elic, ANALYSIS / f"c1_elicitation_{tag}.csv",
            join_keys=("cell",),
            value_cols=("n_transcripts", "parse_fail_rate", "empty_rate",
                        "stated_acc", "stated_vs_label_agree",
                        "n_parsed", "n_no_marker"), tol=5e-5)]
        n += len(elic) * 7

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
