"""Pin every B3 number quoted anywhere against the analysis CSVs.

Run after `analyze_b3.py` for all three roster models. Nothing about B3 gets
written into PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the 8 Aug pre-registration:
- Prediction 1 (narrative-minus-qa within +-0.02, falsifier +0.04): no cell
  approaches +0.04 on any model; the largest magnitude is Olmo's
  narrative2-qa at -0.0249 (CI -0.073 to +0.020, spanning zero). The powered
  null holds: presentation does not move the ceiling.
- Prediction 2 (narrative1-vs-2 agreement in the 75-90% band): 86.0% (Qwen),
  81.5% (Olmo), 85.6% (MoE) — all inside the band.
- Prediction 3 (form-beyond-wording = narrative-vs-qa flips exceed
  narrative-vs-narrative flips): present on ALL three models under label_num
  (e.g. Qwen 17.4/17.8% vs 14.0%). Form changes which predictions flip
  without moving accuracy.
- Replicate ceiling 100.0% on 514 pairs for every arm and model, read
  against the 64.4% cross-serving floor.

    python scripts/narrative/verify_b3_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "narrative"

# tag -> (levels rows, contrast rows), all read off the analyze_b3 CSVs.
TABLES = {
    "qwen_qwen3-32b": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3384, tv=0.2871),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3359, tv=0.2746),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3276, tv=0.2789),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.2397, tv=0.3654),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.2142, tv=0.3949),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.2185, tv=0.3893),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0051, ci_lo=-0.0314, ci_hi=0.0221,
                 flip_rate=0.1743, agree_rate=0.8257),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0081, ci_lo=-0.0272, ci_hi=0.0112,
                 flip_rate=0.1784, agree_rate=0.8216),
            dict(arm="label_num", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=0.0030, ci_lo=-0.0147, ci_hi=0.0194,
                 flip_rate=0.1397, agree_rate=0.8603),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0223, ci_lo=-0.0765, ci_hi=0.0351,
                 flip_rate=0.2960, agree_rate=0.7040),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0167, ci_lo=-0.0720, ci_hi=0.0382,
                 flip_rate=0.2960, agree_rate=0.7040),
            dict(arm="echo_plain", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=-0.0056, ci_lo=-0.0358, ci_hi=0.0222,
                 flip_rate=0.1812, agree_rate=0.8188),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
    "allenai_olmo-3.1-32b-instruct-dpo": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0221, norm_acc=0.3289, tv=0.2913),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0290, norm_acc=0.3043, tv=0.3032),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0318, norm_acc=0.2849, tv=0.2974),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.2371, tv=0.3681),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.2196, tv=0.3834),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.2157, tv=0.3819),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0116, ci_lo=-0.0682, ci_hi=0.0441,
                 flip_rate=0.2780, agree_rate=0.7220),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0249, ci_lo=-0.0733, ci_hi=0.0203,
                 flip_rate=0.2586, agree_rate=0.7414),
            dict(arm="label_num", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=0.0133, ci_lo=-0.0161, ci_hi=0.0422,
                 flip_rate=0.1853, agree_rate=0.8147),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0135, ci_lo=-0.0510, ci_hi=0.0248,
                 flip_rate=0.3430, agree_rate=0.6570),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0136, ci_lo=-0.0582, ci_hi=0.0302,
                 flip_rate=0.3140, agree_rate=0.6860),
            dict(arm="echo_plain", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=0.0000, ci_lo=-0.0216, ci_hi=0.0227,
                 flip_rate=0.1812, agree_rate=0.8188),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
    "qwen_qwen3-30b-a3b-instruct-2507": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3482, tv=0.2604),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3358, tv=0.2173),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3746, tv=0.2109),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3089, tv=0.3094),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3046, tv=0.3505),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3209, tv=0.3357),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0093, ci_lo=-0.0400, ci_hi=0.0173,
                 flip_rate=0.2199, agree_rate=0.7801),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=0.0125, ci_lo=-0.0149, ci_hi=0.0418,
                 flip_rate=0.1950, agree_rate=0.8050),
            # The one CI that excludes zero: narrative2 beats narrative1 on
            # the MoE by 2.2 points (ci_hi -0.0001). A wording-variance
            # datum, read within model, at the band edge.
            dict(arm="label_num", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=-0.0218, ci_lo=-0.0449, ci_hi=-0.0001,
                 flip_rate=0.1438, agree_rate=0.8562),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0081, ci_lo=-0.0381, ci_hi=0.0202,
                 flip_rate=0.2089, agree_rate=0.7911),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=0.0043, ci_lo=-0.0232, ci_hi=0.0318,
                 flip_rate=0.1978, agree_rate=0.8022),
            dict(arm="echo_plain", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=-0.0125, ci_lo=-0.0391, ci_hi=0.0131,
                 flip_rate=0.1438, agree_rate=0.8562),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
}


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, (levels, contrasts) in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            levels, ANALYSIS / f"b3_levels_{tag}.csv",
            join_keys=("arm", "condition"),
            value_cols=("n", "miss_rate", "norm_acc", "tv"), tol=5e-5)]
        n += len(levels) * 4
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            contrasts, ANALYSIS / f"b3_contrasts_{tag}.csv",
            join_keys=("arm", "contrast"),
            value_cols=("n_pairs", "delta_acc", "ci_lo", "ci_hi",
                        "flip_rate", "agree_rate"), tol=5e-5)]
        n += len(contrasts) * 6

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
