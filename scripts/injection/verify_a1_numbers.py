"""Pin every A1 number quoted anywhere against the analysis CSVs.

Run after `analyze_a1.py`. Nothing about A1 gets written into PAPER_STATE or
the paper until this exits 0. One pinned table per model tag; the Qwen tables
are the 8 Aug landing, the Olmo tables the 9 Aug roster completion.

    python scripts/injection/verify_a1_numbers.py
"""

from __future__ import annotations

from pathlib import Path

from synthetic_sampling.checks.number_verify import verify_table_against_csv

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "injection"

# (experiment, arm, condition) -> the level numbers quoted.
QWEN_LEVELS = [
    dict(experiment="country_injection", arm="label_num", condition="baseline",
         n=2850, miss_rate=0.0, norm_acc=0.1468, tv=0.3870),
    dict(experiment="country_injection", arm="label_num", condition="with_country",
         n=2850, miss_rate=0.0, norm_acc=0.1371, tv=0.3833),
    dict(experiment="country_injection", arm="label_num", condition="with_country_placebo",
         n=2850, miss_rate=0.0, norm_acc=0.1015, tv=0.3961),
    dict(experiment="country_injection", arm="echo_plain", condition="baseline",
         n=2850, miss_rate=0.0, norm_acc=-0.0337, tv=0.5351),
    dict(experiment="temporal_context", arm="label_num", condition="baseline",
         n=2100, miss_rate=0.0, norm_acc=0.0848, tv=0.3885),
    dict(experiment="temporal_context", arm="label_num", condition="with_year",
         n=2100, miss_rate=0.0, norm_acc=0.0714, tv=0.3957),
    dict(experiment="temporal_context", arm="label_num", condition="with_year_placebo",
         n=2100, miss_rate=0.0, norm_acc=0.0681, tv=0.3997),
    dict(experiment="temporal_context", arm="label_num", condition="with_country",
         n=2100, miss_rate=0.0, norm_acc=0.0774, tv=0.3977),
    dict(experiment="temporal_context", arm="label_num", condition="with_country_and_year",
         n=2100, miss_rate=0.0, norm_acc=0.0715, tv=0.3959),
    dict(experiment="temporal_context", arm="label_num", condition="with_date",
         n=2083, miss_rate=0.0, norm_acc=0.0735, tv=0.4001),
]

QWEN_DELTAS = [
    # Prediction 1: with_country - baseline, at most +0.02, falsifier +0.04.
    dict(experiment="country_injection", arm="label_num",
         contrast="with_country-baseline",
         delta_acc=-0.0029, ci_lo=-0.0150, ci_hi=0.0103, flip_rate=0.0993),
    # Prediction 2: placebo zero or negative, within -0.02.
    dict(experiment="country_injection", arm="label_num",
         contrast="with_country_placebo-baseline",
         delta_acc=-0.0210, ci_lo=-0.0350, ci_hi=-0.0058, flip_rate=0.1137),
    # Prediction 4: same contrasts under same-serving echo_plain.
    dict(experiment="country_injection", arm="echo_plain",
         contrast="with_country-baseline",
         delta_acc=0.0036, ci_lo=-0.0063, ci_hi=0.0135, flip_rate=0.1053),
    dict(experiment="country_injection", arm="echo_plain",
         contrast="with_country_placebo-baseline",
         delta_acc=0.0018, ci_lo=-0.0072, ci_hi=0.0117, flip_rate=0.1074),
    # Prediction 3: temporal cells within +-0.01; with_year vs its placebo.
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year-baseline",
         delta_acc=-0.0106, ci_lo=-0.0225, ci_hi=-0.0009, flip_rate=0.0681),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year_placebo-baseline",
         delta_acc=-0.0119, ci_lo=-0.0246, ci_hi=-0.0016, flip_rate=0.0752),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year-with_year_placebo",
         delta_acc=0.0014, ci_lo=-0.0033, ci_hi=0.0062, flip_rate=0.0210),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_date-baseline",
         delta_acc=-0.0098, ci_lo=-0.0230, ci_hi=0.0020, flip_rate=0.0768),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline",
         delta_acc=-0.0069, ci_lo=-0.0218, ci_hi=0.0059, flip_rate=0.1038),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country_and_year-baseline",
         delta_acc=-0.0101, ci_lo=-0.0244, ci_hi=0.0028, flip_rate=0.1043),
    # Prediction 7: 2x2 interaction within +-0.02.
    dict(experiment="temporal_context", arm="label_num",
         contrast="interaction_2x2", delta_acc=0.0074),
    dict(experiment="temporal_context", arm="echo_plain",
         contrast="interaction_2x2", delta_acc=-0.0014),
    # Disclosure split (stated rule, see analyze_a1.py).
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline|disclosed",
         n_pairs=159, delta_acc=-0.0207, ci_lo=-0.0829, ci_hi=0.0374),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline|undisclosed",
         n_pairs=1941, delta_acc=-0.0049, ci_lo=-0.0178, ci_hi=0.0081),
    dict(experiment="temporal_context", arm="echo_plain",
         contrast="with_country-baseline|disclosed",
         n_pairs=159, delta_acc=0.0452, ci_lo=0.0090, ci_hi=0.0904),
]

# Olmo-3.1-32B-Instruct-DPO, the A1-OLMO roster completion (landed 9 Aug,
# analyzed 11 Aug). Headlines: the placebo-country penalty REPLICATES
# (-0.021, CI excludes zero) and the true-country null replicates; the Qwen
# "any context line costs ~1 point" does NOT replicate — every injected
# temporal cell sits 0.6-1.1 points ABOVE baseline with CIs spanning zero.
# label_num miss rates are 1.8-5.0% (the known Olmo label-tokenisation cost).
OLMO_LEVELS = [
    dict(experiment="country_injection", arm="label_num", condition="baseline",
         n=2850, miss_rate=0.0175, norm_acc=0.1230, tv=0.3922),
    dict(experiment="country_injection", arm="label_num", condition="with_country",
         n=2850, miss_rate=0.0239, norm_acc=0.1186, tv=0.4040),
    dict(experiment="country_injection", arm="label_num", condition="with_country_placebo",
         n=2850, miss_rate=0.0207, norm_acc=0.0946, tv=0.4125),
    dict(experiment="country_injection", arm="echo_plain", condition="baseline",
         n=2850, miss_rate=0.0, norm_acc=-0.0456, tv=0.5592),
    dict(experiment="temporal_context", arm="label_num", condition="baseline",
         n=2100, miss_rate=0.0252, norm_acc=0.0666, tv=0.4353),
    dict(experiment="temporal_context", arm="label_num", condition="with_year",
         n=2100, miss_rate=0.0429, norm_acc=0.0878, tv=0.4312),
    dict(experiment="temporal_context", arm="label_num", condition="with_year_placebo",
         n=2100, miss_rate=0.0410, norm_acc=0.0869, tv=0.4337),
    dict(experiment="temporal_context", arm="label_num", condition="with_country",
         n=2100, miss_rate=0.0314, norm_acc=0.0816, tv=0.4381),
    dict(experiment="temporal_context", arm="label_num", condition="with_country_and_year",
         n=2100, miss_rate=0.0505, norm_acc=0.0902, tv=0.4376),
    dict(experiment="temporal_context", arm="label_num", condition="with_date",
         n=2083, miss_rate=0.0422, norm_acc=0.0839, tv=0.4381),
]

OLMO_DELTAS = [
    # Prediction 1: true country, null again (-0.008, CI spans zero).
    dict(experiment="country_injection", arm="label_num",
         contrast="with_country-baseline",
         delta_acc=-0.0079, ci_lo=-0.0198, ci_hi=0.0035, flip_rate=0.1196),
    # Prediction 2: the placebo penalty replicates almost exactly (-0.0208
    # here vs Qwen's -0.0210), again just past the -0.02 band edge.
    dict(experiment="country_injection", arm="label_num",
         contrast="with_country_placebo-baseline",
         delta_acc=-0.0208, ci_lo=-0.0322, ci_hi=-0.0083, flip_rate=0.1330),
    # Prediction 4: echo_plain shows neither effect, as on Qwen.
    dict(experiment="country_injection", arm="echo_plain",
         contrast="with_country-baseline",
         delta_acc=-0.0003, ci_lo=-0.0089, ci_hi=0.0084, flip_rate=0.0779),
    dict(experiment="country_injection", arm="echo_plain",
         contrast="with_country_placebo-baseline",
         delta_acc=-0.0060, ci_lo=-0.0147, ci_hi=0.0023, flip_rate=0.0786),
    # Prediction 3: with_year vs its placebo indistinguishable (+0.0017,
    # 2.6% flips) — no temporal conditioning on Olmo either.
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year-baseline",
         delta_acc=0.0111, ci_lo=-0.0020, ci_hi=0.0247, flip_rate=0.1162),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year_placebo-baseline",
         delta_acc=0.0094, ci_lo=-0.0042, ci_hi=0.0231, flip_rate=0.1152),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_year-with_year_placebo",
         delta_acc=0.0017, ci_lo=-0.0025, ci_hi=0.0059, flip_rate=0.0262),
    # NOTE the sign: all injected cells sit ABOVE baseline on Olmo (n.s.),
    # so the Qwen context-line cost is model-specific, not a law.
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_date-baseline",
         delta_acc=0.0076, ci_lo=-0.0050, ci_hi=0.0200, flip_rate=0.0994),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline",
         delta_acc=0.0058, ci_lo=-0.0057, ci_hi=0.0178, flip_rate=0.1019),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country_and_year-baseline",
         delta_acc=0.0103, ci_lo=-0.0031, ci_hi=0.0241, flip_rate=0.1424),
    # Prediction 7: 2x2 interaction within +-0.02 on both arms.
    dict(experiment="temporal_context", arm="label_num",
         contrast="interaction_2x2", delta_acc=-0.0066),
    dict(experiment="temporal_context", arm="echo_plain",
         contrast="interaction_2x2", delta_acc=-0.0046),
    # Disclosure split (same stated rule; n=159 stays uninformative).
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline|disclosed",
         n_pairs=159, delta_acc=0.0211, ci_lo=0.0000, ci_hi=0.0512),
    dict(experiment="temporal_context", arm="label_num",
         contrast="with_country-baseline|undisclosed",
         n_pairs=1941, delta_acc=0.0050, ci_lo=-0.0066, ci_hi=0.0169),
    dict(experiment="temporal_context", arm="echo_plain",
         contrast="with_country-baseline|disclosed",
         n_pairs=159, delta_acc=-0.0060, ci_lo=-0.0442, ci_hi=0.0321),
]

TABLES = {
    "qwen_qwen3-32b": (QWEN_LEVELS, QWEN_DELTAS),
    "allenai_olmo-3.1-32b-instruct-dpo": (OLMO_LEVELS, OLMO_DELTAS),
}

# A6's speed benchmark, closed off the A1 logs. All three jobs on htc-g058
# (sacct 9 Aug), per-instance load matched to within 0.02 scored cells.
THROUGHPUT = [
    dict(job="8498891", instances=8550, minutes=130.4, inst_per_s=1.0928,
         speedup_vs_this_row=4.1128),
    dict(job="8498370", instances=12583, minutes=185.0, inst_per_s=1.1336,
         speedup_vs_this_row=3.9647),
    dict(job="8498233", instances=4800, minutes=17.8, inst_per_s=4.4944,
         speedup_vs_this_row=1.0),
]


def main() -> int:
    bad: list[str] = []
    n = 0
    for tag, (levels, deltas) in TABLES.items():
        bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
            levels, ANALYSIS / f"a1_levels_{tag}.csv",
            join_keys=("experiment", "arm", "condition"),
            value_cols=("n", "miss_rate", "norm_acc", "tv"), tol=5e-5)]
        n += len(levels) * 4
        for row in deltas:
            cols = tuple(k for k in ("n_pairs", "delta_acc", "ci_lo", "ci_hi", "flip_rate")
                         if k in row)
            bad += [f"[{tag}] {m}" for m in verify_table_against_csv(
                [row], ANALYSIS / f"a1_deltas_{tag}.csv",
                join_keys=("experiment", "arm", "contrast"),
                value_cols=cols, tol=5e-5)]
            n += len(cols)

    bad += verify_table_against_csv(
        THROUGHPUT, ANALYSIS / "a1_throughput.csv", join_keys=("job",),
        value_cols=("instances", "minutes", "inst_per_s", "speedup_vs_this_row"),
        tol=5e-5)
    n += len(THROUGHPUT) * 4

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
