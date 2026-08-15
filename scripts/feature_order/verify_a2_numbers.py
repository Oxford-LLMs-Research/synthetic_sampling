"""Pin every A2 number quoted anywhere against the analysis CSVs.

Run after `analyze_a2.py` for all three roster models. Nothing about A2
gets written into PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the 13 Aug pre-registration:
- Accuracy (co-primary): no label_num last-first or shuf-first contrast
  hits the +-0.04 falsifier with a CI excluding zero. Qwen and Olmo
  point estimates sit inside +-0.02; MoE shuf-first is +0.0274 (outside
  the predicted band, CI spans zero). Order does not become a grid
  convention on accuracy; the ladder default (informative-first) stays.
- Flip rates (co-primary): 11.6-16.2% last/shuf-vs-first against a 0.0%
  same-serving replicate ceiling (579 pairs, all models). Order changes
  which prediction without reliably moving accuracy (B3's signature).
- Instrument: label_num miss 0 on both Qwens; Olmo 1.2-2.2% (known
  pattern, passes the 90% gate). Replicates 100% on both arms.

    python scripts/feature_order/verify_a2_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "feature_order"

TABLES = {
    "qwen_qwen3-32b": (
        [
            dict(arm="label_num", cell="informative_first", n=734,
                 miss_rate=0.0000, norm_acc=0.3417),
            dict(arm="label_num", cell="informative_last", n=734,
                 miss_rate=0.0000, norm_acc=0.3142),
            dict(arm="label_num", cell="shuffled", n=734,
                 miss_rate=0.0000, norm_acc=0.3329),
        ],
        [
            dict(arm="label_num", contrast="last-first", n_pairs=734,
                 delta_acc=-0.0171, ci_lo=-0.0440, ci_hi=0.0084,
                 flip_rate=0.1199, agree_rate=0.8801),
            dict(arm="label_num", contrast="shuf-first", n_pairs=734,
                 delta_acc=-0.0040, ci_lo=-0.0227, ci_hi=0.0147,
                 flip_rate=0.1158, agree_rate=0.8842),
            dict(arm="label_num", contrast="replicate", n_pairs=579,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
    "allenai_olmo-3.1-32b-instruct-dpo": (
        [
            dict(arm="label_num", cell="informative_first", n=734,
                 miss_rate=0.0218, norm_acc=0.3278),
            dict(arm="label_num", cell="informative_last", n=734,
                 miss_rate=0.0136, norm_acc=0.3528),
            dict(arm="label_num", cell="shuffled", n=734,
                 miss_rate=0.0123, norm_acc=0.3512),
        ],
        [
            dict(arm="label_num", contrast="last-first", n_pairs=734,
                 delta_acc=0.0149, ci_lo=-0.0120, ci_hi=0.0442,
                 flip_rate=0.1621, agree_rate=0.8379),
            dict(arm="label_num", contrast="shuf-first", n_pairs=734,
                 delta_acc=0.0166, ci_lo=-0.0080, ci_hi=0.0446,
                 flip_rate=0.1499, agree_rate=0.8501),
            dict(arm="label_num", contrast="replicate", n_pairs=579,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
    "qwen_qwen3-30b-a3b-instruct-2507": (
        [
            dict(arm="label_num", cell="informative_first", n=734,
                 miss_rate=0.0000, norm_acc=0.3529),
            dict(arm="label_num", cell="informative_last", n=734,
                 miss_rate=0.0000, norm_acc=0.3656),
            dict(arm="label_num", cell="shuffled", n=734,
                 miss_rate=0.0000, norm_acc=0.3917),
        ],
        [
            dict(arm="label_num", contrast="last-first", n_pairs=734,
                 delta_acc=0.0086, ci_lo=-0.0133, ci_hi=0.0330,
                 flip_rate=0.1308, agree_rate=0.8692),
            dict(arm="label_num", contrast="shuf-first", n_pairs=734,
                 delta_acc=0.0274, ci_lo=-0.0013, ci_hi=0.0598,
                 flip_rate=0.1417, agree_rate=0.8583),
            dict(arm="label_num", contrast="replicate", n_pairs=579,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    ),
}


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, (levels, contrasts) in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            levels, ANALYSIS / f"a2_levels_{tag}.csv",
            join_keys=("arm", "cell"),
            value_cols=("n", "miss_rate", "norm_acc"), tol=5e-5)]
        n += len(levels) * 3
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            contrasts, ANALYSIS / f"a2_contrasts_{tag}.csv",
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
