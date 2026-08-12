"""Pin every A4 number quoted anywhere against the analysis CSVs.

Run after `analyze_a4.py` for all three roster models. Nothing about A4
gets written into PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the 12 Aug pre-registration:
- Instrument validation: NO contrast flips beyond +-0.04 anywhere — the
  raw-completion instrument stands and every landed null keeps its
  standing. Olmo chat narrative1-qa -0.0234 (CI spans zero) and MoE
  +0.0040 replicate B3's null under the template.
- The exception pattern, third sighting: Qwen3-32B under its own chat
  template pays for prose narratives — narrative1-qa = -0.0333 (CI
  -0.064..-0.004) and narrative2-qa = -0.0307 (CI -0.060..-0.005), both
  CIs excluding zero, where the raw readout in the SAME serving is null
  (-0.0024 / -0.0053). Same model that paid the C1 narrative_reasoned
  penalty; raw completion is the one regime where it tolerates prose.
- Template tax (descriptive): small and positive on qa — +0.0200 (CI
  +0.003..+0.040, excluding zero) on Qwen3-32B, +0.0109 MoE, +0.0069
  Olmo; mixed on narratives (MoE narrative1 +0.0229, CI excluding zero).
  The template changes 9-15% of individual predictions.
- Calibration (pinned distributional outcome): the chat readout inflates
  mean confidence on every model (e.g. Qwen3-32B qa 0.804 -> 0.908) and
  worsens ECE (0.256 -> 0.336) — the tuned format buys a little accuracy
  and costs calibration.
- Instrument facts: Olmo's label-tokenisation misses VANISH under chat
  (miss_rate 0.0000 vs 0.0207-0.0332 raw); replicates 100.0% on
  chat_label_num for all models, >=99.81% on the raw arms (514 pairs).

    python scripts/chat_template/verify_a4_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "chat_template"

# tag -> (levels rows, contrast rows), all read off the analyze_a4 CSVs.
TABLES = {
    "qwen_qwen3-32b": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3341, tv=0.2858,
                 mean_conf=0.8043, ece=0.2558),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3359, tv=0.2746,
                 mean_conf=0.7919, ece=0.2442),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3268, tv=0.2817,
                 mean_conf=0.7965, ece=0.2516),
            dict(arm="chat_label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3672, tv=0.2739,
                 mean_conf=0.9077, ece=0.3364),
            dict(arm="chat_label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3226, tv=0.2668,
                 mean_conf=0.9194, ece=0.3861),
            dict(arm="chat_label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3278, tv=0.2598,
                 mean_conf=0.9180, ece=0.3786),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.2412, tv=0.3627,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.2069, tv=0.3976,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.2235, tv=0.3906,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0024, ci_lo=-0.0288, ci_hi=0.0252,
                 flip_rate=0.1743, agree_rate=0.8257),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0053, ci_lo=-0.0244, ci_hi=0.0151,
                 flip_rate=0.1798, agree_rate=0.8202),
            dict(arm="label_num", contrast="narrative1-narrative2", n_pairs=723,
                 delta_acc=0.0029, ci_lo=-0.0121, ci_hi=0.0179,
                 flip_rate=0.1314, agree_rate=0.8686),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0019, agree_rate=0.9981),
            dict(arm="chat_label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0333, ci_lo=-0.0642, ci_hi=-0.0038,
                 flip_rate=0.1729, agree_rate=0.8271),
            dict(arm="chat_label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0307, ci_lo=-0.0603, ci_hi=-0.0046,
                 flip_rate=0.1826, agree_rate=0.8174),
            dict(arm="chat_label_num", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=-0.0027, ci_lo=-0.0229, ci_hi=0.0159,
                 flip_rate=0.1383, agree_rate=0.8617),
            dict(arm="chat_label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0277, ci_lo=-0.0850, ci_hi=0.0319,
                 flip_rate=0.2974, agree_rate=0.7026),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0152, ci_lo=-0.0738, ci_hi=0.0412,
                 flip_rate=0.2932, agree_rate=0.7068),
            dict(arm="echo_plain", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=-0.0124, ci_lo=-0.0413, ci_hi=0.0171,
                 flip_rate=0.1798, agree_rate=0.8202),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="chat-vs-raw", contrast="template|qa", n_pairs=723,
                 delta_acc=0.0200, ci_lo=0.0032, ci_hi=0.0400,
                 flip_rate=0.0899, agree_rate=0.9101),
            dict(arm="chat-vs-raw", contrast="template|narrative1",
                 n_pairs=723,
                 delta_acc=-0.0109, ci_lo=-0.0339, ci_hi=0.0152,
                 flip_rate=0.1189, agree_rate=0.8811),
            dict(arm="chat-vs-raw", contrast="template|narrative2",
                 n_pairs=723,
                 delta_acc=-0.0053, ci_lo=-0.0348, ci_hi=0.0269,
                 flip_rate=0.1120, agree_rate=0.8880),
        ],
    ),
    "allenai_olmo-3.1-32b-instruct-dpo": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0207, norm_acc=0.3250, tv=0.2873,
                 mean_conf=0.7570, ece=0.2203),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0290, norm_acc=0.3029, tv=0.3045,
                 mean_conf=0.7955, ece=0.2813),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0332, norm_acc=0.2892, tv=0.2947,
                 mean_conf=0.8034, ece=0.3013),
            dict(arm="chat_label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3342, tv=0.2570,
                 mean_conf=0.7899, ece=0.2511),
            dict(arm="chat_label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.2909, tv=0.2979,
                 mean_conf=0.8372, ece=0.3158),
            dict(arm="chat_label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3096, tv=0.2991,
                 mean_conf=0.8389, ece=0.3147),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.2376, tv=0.3669,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.2162, tv=0.3862,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.2095, tv=0.3804,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0103, ci_lo=-0.0657, ci_hi=0.0425,
                 flip_rate=0.2739, agree_rate=0.7261),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0197, ci_lo=-0.0671, ci_hi=0.0263,
                 flip_rate=0.2600, agree_rate=0.7400),
            dict(arm="label_num", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=0.0094, ci_lo=-0.0197, ci_hi=0.0387,
                 flip_rate=0.1840, agree_rate=0.8160),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="chat_label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0234, ci_lo=-0.0670, ci_hi=0.0207,
                 flip_rate=0.2711, agree_rate=0.7289),
            dict(arm="chat_label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0119, ci_lo=-0.0475, ci_hi=0.0222,
                 flip_rate=0.2476, agree_rate=0.7524),
            dict(arm="chat_label_num", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=-0.0115, ci_lo=-0.0370, ci_hi=0.0146,
                 flip_rate=0.1881, agree_rate=0.8119),
            dict(arm="chat_label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0174, ci_lo=-0.0545, ci_hi=0.0189,
                 flip_rate=0.3389, agree_rate=0.6611),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0189, ci_lo=-0.0655, ci_hi=0.0270,
                 flip_rate=0.3167, agree_rate=0.6833),
            dict(arm="echo_plain", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=0.0015, ci_lo=-0.0238, ci_hi=0.0260,
                 flip_rate=0.1770, agree_rate=0.8230),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="chat-vs-raw", contrast="template|qa", n_pairs=723,
                 delta_acc=0.0069, ci_lo=-0.0189, ci_hi=0.0304,
                 flip_rate=0.1494, agree_rate=0.8506),
            dict(arm="chat-vs-raw", contrast="template|narrative1",
                 n_pairs=723,
                 delta_acc=-0.0061, ci_lo=-0.0345, ci_hi=0.0226,
                 flip_rate=0.1660, agree_rate=0.8340),
            dict(arm="chat-vs-raw", contrast="template|narrative2",
                 n_pairs=723,
                 delta_acc=0.0148, ci_lo=-0.0094, ci_hi=0.0390,
                 flip_rate=0.1328, agree_rate=0.8672),
        ],
    ),
    "qwen_qwen3-30b-a3b-instruct-2507": (
        [
            dict(arm="label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3465, tv=0.2604,
                 mean_conf=0.8869, ece=0.3351),
            dict(arm="label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3358, tv=0.2160,
                 mean_conf=0.8772, ece=0.3322),
            dict(arm="label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3757, tv=0.2108,
                 mean_conf=0.8810, ece=0.3186),
            dict(arm="chat_label_num", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3613, tv=0.2608,
                 mean_conf=0.9185, ece=0.3542),
            dict(arm="chat_label_num", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3727, tv=0.2431,
                 mean_conf=0.9129, ece=0.3444),
            dict(arm="chat_label_num", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3573, tv=0.2405,
                 mean_conf=0.9140, ece=0.3538),
            dict(arm="echo_plain", condition="qa", n=723,
                 miss_rate=0.0000, norm_acc=0.3089, tv=0.3094,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative1", n=723,
                 miss_rate=0.0000, norm_acc=0.3046, tv=0.3505,
                 mean_conf=float("nan"), ece=float("nan")),
            dict(arm="echo_plain", condition="narrative2", n=723,
                 miss_rate=0.0000, norm_acc=0.3209, tv=0.3357,
                 mean_conf=float("nan"), ece=float("nan")),
        ],
        [
            dict(arm="label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0080, ci_lo=-0.0376, ci_hi=0.0178,
                 flip_rate=0.2199, agree_rate=0.7801),
            dict(arm="label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=0.0139, ci_lo=-0.0136, ci_hi=0.0436,
                 flip_rate=0.1964, agree_rate=0.8036),
            dict(arm="label_num", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=-0.0219, ci_lo=-0.0457, ci_hi=0.0012,
                 flip_rate=0.1452, agree_rate=0.8548),
            dict(arm="label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0019, agree_rate=0.9981),
            dict(arm="chat_label_num", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=0.0040, ci_lo=-0.0203, ci_hi=0.0273,
                 flip_rate=0.2033, agree_rate=0.7967),
            dict(arm="chat_label_num", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=-0.0073, ci_lo=-0.0352, ci_hi=0.0230,
                 flip_rate=0.1812, agree_rate=0.8188),
            dict(arm="chat_label_num", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=0.0113, ci_lo=-0.0093, ci_hi=0.0323,
                 flip_rate=0.1438, agree_rate=0.8562),
            dict(arm="chat_label_num", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="echo_plain", contrast="narrative1-qa", n_pairs=723,
                 delta_acc=-0.0081, ci_lo=-0.0381, ci_hi=0.0202,
                 flip_rate=0.2089, agree_rate=0.7911),
            dict(arm="echo_plain", contrast="narrative2-qa", n_pairs=723,
                 delta_acc=0.0043, ci_lo=-0.0232, ci_hi=0.0318,
                 flip_rate=0.1978, agree_rate=0.8022),
            dict(arm="echo_plain", contrast="narrative1-narrative2",
                 n_pairs=723,
                 delta_acc=-0.0125, ci_lo=-0.0391, ci_hi=0.0131,
                 flip_rate=0.1438, agree_rate=0.8562),
            dict(arm="echo_plain", contrast="replicate", n_pairs=514,
                 delta_acc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 flip_rate=0.0000, agree_rate=1.0000),
            dict(arm="chat-vs-raw", contrast="template|qa", n_pairs=723,
                 delta_acc=0.0109, ci_lo=-0.0099, ci_hi=0.0329,
                 flip_rate=0.0899, agree_rate=0.9101),
            dict(arm="chat-vs-raw", contrast="template|narrative1",
                 n_pairs=723,
                 delta_acc=0.0229, ci_lo=0.0039, ci_hi=0.0457,
                 flip_rate=0.1024, agree_rate=0.8976),
            dict(arm="chat-vs-raw", contrast="template|narrative2",
                 n_pairs=723,
                 delta_acc=-0.0103, ci_lo=-0.0341, ci_hi=0.0106,
                 flip_rate=0.0996, agree_rate=0.9004),
        ],
    ),
}


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, (levels, contrasts) in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            levels, ANALYSIS / f"a4_levels_{tag}.csv",
            join_keys=("arm", "condition"),
            value_cols=("n", "miss_rate", "norm_acc", "tv",
                        "mean_conf", "ece"), tol=5e-5)]
        n += len(levels) * 6
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            contrasts, ANALYSIS / f"a4_contrasts_{tag}.csv",
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
