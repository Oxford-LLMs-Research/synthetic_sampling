"""Marginal recovery split by country-information presence in the profile.

Robustness check for analyze_marginal_recovery.py: the headline result there
is that model-predicted answer shares track country-level opinion worse than
the pooled cross-country distribution. A natural objection is that many
profiles never reveal the respondent's country, so the model cannot be
expected to produce country-specific marginals. This script splits every
instance by whether the randomly sampled profile contains a country or region
question (same wording-matching logic as the conditional-stereotyping
analysis) and recomputes the comparison within each subset.

Matched design: a (survey, target, country) cell enters the comparison only
if BOTH subsets have >= MIN_N respondents in it, so the two conditions are
evaluated on identical cells. TV distances are computed against each
subset's own ground-truth distribution.

Outputs to analysis/marginal_recovery/:
  example_country_flag.csv      cached example_id -> country_in_profile flag
  country_split_per_cell.csv    per (model, cell): TVs in both subsets
  country_split_summary.csv     per model: mean TVs and win shares by subset
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from analyze_disaggregated import (  # noqa: E402
    _normalize_question_text,
    get_country_region_question_wordings,
)

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
ANALYSIS = ROOT / "analysis"
MAIN_DATA = ROOT / "synthetic_sampling" / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
METADATA = SCRIPTS.parent / "src" / "synthetic_sampling" / "profiles" / "metadata"
OUT = ANALYSIS / "marginal_recovery"

MIN_N = 20
PROFILE = "s6m4"

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]


def _matches(question_text: str, targets: list[str]) -> bool:
    if not question_text or not targets:
        return False
    q_lower = question_text.lower().strip()
    q_norm = _normalize_question_text(question_text)
    for t in targets:
        if not t:
            continue
        t_lower = t.lower().strip()
        t_norm = _normalize_question_text(t)
        if q_lower == t_lower or q_norm == t_norm:
            return True
        if t_norm in q_norm or q_norm in t_norm:
            return True
    return False


def build_example_flags() -> pd.DataFrame:
    cache = OUT / "example_country_flag.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"example_id": str, "country_in_profile": bool})

    wordings_by_survey: dict[str, dict[str, list[str]]] = {}
    rows = []
    for f in sorted(MAIN_DATA.glob("*_instances.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                survey = d["survey"]
                if survey not in wordings_by_survey:
                    wordings_by_survey[survey] = get_country_region_question_wordings(
                        survey, METADATA
                    )
                w = wordings_by_survey[survey]
                country_w = w.get("country", [])
                region_w = w.get("region", [])
                flag = any(
                    _matches(q, country_w) or _matches(q, region_w)
                    for q in d.get("questions", {})
                )
                rows.append((d["example_id"], flag))
    df = pd.DataFrame(rows, columns=["example_id", "country_in_profile"])
    df.to_csv(cache, index=False)
    return df


def tv_distance(counts_a: Counter, counts_b: Counter) -> float:
    labels = set(counts_a) | set(counts_b)
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    return 0.5 * sum(abs(counts_a[l] / na - counts_b[l] / nb) for l in labels)


def cell_stats(g: pd.DataFrame, global_true: Counter) -> dict:
    true_c = Counter(g["ground_truth"])
    pred_c = Counter(g["predicted"])
    return {
        "n": len(g),
        "tv_model": tv_distance(pred_c, true_c),
        "tv_global": tv_distance(global_true, true_c),
    }


def main() -> None:
    flags = build_example_flags()
    share = flags["country_in_profile"].mean()
    print(f"instances flagged country-in-profile: {share:.3f}")

    resp_country = pd.read_csv(OUT / "respondent_country.csv", dtype=str, keep_default_na=False)

    all_rows = []
    for model in MODELS:
        df = pd.read_csv(
            ANALYSIS / model / "results_data.csv",
            usecols=["example_id", "survey", "respondent_id", "target_code",
                     "profile_type", "ground_truth", "predicted"],
            dtype=str,
        )
        df = df[df["profile_type"] == PROFILE]
        df = df.merge(flags, on="example_id", how="left")
        df = df.merge(resp_country, on=["survey", "respondent_id"], how="left")

        global_true_by_q = {
            key: Counter(g["ground_truth"])
            for key, g in df.groupby(["survey", "target_code"], sort=False)
        }
        for (survey, target, country), g in df.groupby(
            ["survey", "target_code", "country"], sort=False
        ):
            if not isinstance(country, str):
                continue
            g_in = g[g["country_in_profile"] == True]   # noqa: E712
            g_out = g[g["country_in_profile"] == False]  # noqa: E712
            if len(g_in) < MIN_N or len(g_out) < MIN_N:
                continue
            global_true = global_true_by_q[(survey, target)]
            s_in = cell_stats(g_in, global_true)
            s_out = cell_stats(g_out, global_true)
            all_rows.append({
                "model": model, "survey": survey, "target_code": target,
                "country": country,
                "n_in": s_in["n"], "tv_model_in": s_in["tv_model"],
                "tv_global_in": s_in["tv_global"],
                "n_out": s_out["n"], "tv_model_out": s_out["tv_model"],
                "tv_global_out": s_out["tv_global"],
            })
        print(f"{model}: done")

    cells = pd.DataFrame(all_rows)
    cells.to_csv(OUT / "country_split_per_cell.csv", index=False)

    summary = (
        cells.assign(
            beats_global_in=cells["tv_model_in"] < cells["tv_global_in"],
            beats_global_out=cells["tv_model_out"] < cells["tv_global_out"],
        )
        .groupby("model")
        .agg(
            n_cells=("tv_model_in", "size"),
            tv_model_country_in=("tv_model_in", "mean"),
            tv_model_country_absent=("tv_model_out", "mean"),
            tv_global_country_in=("tv_global_in", "mean"),
            tv_global_country_absent=("tv_global_out", "mean"),
            beats_global_in=("beats_global_in", "mean"),
            beats_global_absent=("beats_global_out", "mean"),
        )
        .sort_values("tv_model_country_in")
    )
    summary.to_csv(OUT / "country_split_summary.csv")
    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
