"""Pin every A3 number quoted anywhere against the analysis CSVs.

Run after `analyze_a3.py` for all three roster models. Nothing about A3
gets written into PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the 13 Aug pre-registration as
amended 15 Aug (DK/Refusal stay in Phase 2):
- DK census (descriptive): label_num does not inflate. Mean per-target
  inflation is -0.17pp (Qwen), -1.36pp (Olmo), -1.24pp (MoE); 0 of 16
  targets over 10pp on every model. The 5pp / 2-of-16 band holds. This
  is a model finding, not a convention switch.
- Substantive stability: present-minus-absent on substantive-truth rows
  is +0.0047 / +0.0017 / +0.0068, every CI inside +-0.03, none
  approaching the +-0.04 falsifier. Hiding DK does not move accuracy.
  Flips 5.8-10.1% against a 0% replicate ceiling.
- echo_plain still carries a residual DK artifact on isolated targets
  (Olmo QLEB7 +47pp; Qwen sclact +13pp) but mean inflation stays inside
  the band. label_num does not inherit it.
- Refusal: on the four ESS dual-carriers, label_num almost never picks
  Refusal (0 on all four x three models). The small DK hits are Don't
  know, not Refusal.

    python scripts/default_options/verify_a3_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "default_options"

TABLES = {
    "qwen_qwen3-32b": {
        "verdict": [
            dict(arm="label_num", n_targets=16, mean_inflation=-0.0017,
                 n_targets_over_cap=0),
            dict(arm="echo_plain", n_targets=16, mean_inflation=0.0107,
                 n_targets_over_cap=1),
        ],
        "contrasts": [
            dict(arm="label_num", contrast="present-absent", n_pairs=771,
                 delta_acc=0.0047, ci_lo=-0.0086, ci_hi=0.0195,
                 flip_rate=0.0584, agree_rate=0.9416),
            dict(arm="label_num", contrast="replicate", n_pairs=406,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    },
    "allenai_olmo-3.1-32b-instruct-dpo": {
        "verdict": [
            dict(arm="label_num", n_targets=16, mean_inflation=-0.0136,
                 n_targets_over_cap=0),
            dict(arm="echo_plain", n_targets=16, mean_inflation=0.0329,
                 n_targets_over_cap=1),
        ],
        "contrasts": [
            dict(arm="label_num", contrast="present-absent", n_pairs=760,
                 delta_acc=0.0017, ci_lo=-0.0167, ci_hi=0.0186,
                 flip_rate=0.1013, agree_rate=0.8987),
            dict(arm="label_num", contrast="replicate", n_pairs=406,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    },
    "qwen_qwen3-30b-a3b-instruct-2507": {
        "verdict": [
            dict(arm="label_num", n_targets=16, mean_inflation=-0.0124,
                 n_targets_over_cap=0),
            dict(arm="echo_plain", n_targets=16, mean_inflation=-0.0074,
                 n_targets_over_cap=0),
        ],
        "contrasts": [
            dict(arm="label_num", contrast="present-absent", n_pairs=771,
                 delta_acc=0.0068, ci_lo=-0.0128, ci_hi=0.0265,
                 flip_rate=0.0597, agree_rate=0.9403),
            dict(arm="label_num", contrast="replicate", n_pairs=406,
                 delta_acc=float("nan"), ci_lo=float("nan"),
                 ci_hi=float("nan"), flip_rate=0.0000, agree_rate=1.0000),
        ],
    },
}


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, tables in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            tables["verdict"], ANALYSIS / f"a3_dk_verdict_{tag}.csv",
            join_keys=("arm",),
            value_cols=("n_targets", "mean_inflation",
                        "n_targets_over_cap"), tol=5e-5)]
        n += len(tables["verdict"]) * 3
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            tables["contrasts"], ANALYSIS / f"a3_contrasts_{tag}.csv",
            join_keys=("arm", "contrast"),
            value_cols=("n_pairs", "delta_acc", "ci_lo", "ci_hi",
                        "flip_rate", "agree_rate"), tol=5e-5)]
        n += len(tables["contrasts"]) * 6

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
