"""Aggregate marginal recovery: do models track country-level answer shares?

The paper shows models fail at individual-level prediction. The deployment
pitch, however, is often aggregate: "simulate a population's answer
distribution." This script tests that directly.

For each (survey, target question, country) cell with >= MIN_N respondents:
  - true distribution  = shares of recorded answers in the cell
  - model distribution = shares of the model's predicted answers in the cell
  - TV(model, true)    = total variation distance between the two
  - TV(global, true)   = distance between the cell's true distribution and the
    question's pooled cross-country distribution: the accuracy of a "one
    global distribution fits every country" predictor that uses no model at
    all, only the population margin.

If TV(model, true) >= TV(global, true) in most cells, the model's synthetic
sample tracks cross-country variation in public opinion no better than
ignoring the country entirely.

Outputs to analysis/marginal_recovery/:
  respondent_country.csv   cached (survey, respondent_id) -> ISO2 map
  per_cell_<model>.csv     one row per (survey, target, country, profile)
  summary.csv              per model x profile: mean TVs, win share, n cells
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from repair_ids import repair

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
ANALYSIS = ROOT / "analysis"
MAIN_DATA = ROOT / "synthetic_sampling" / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
SCRIPTS = Path(__file__).resolve().parent
OUT = ANALYSIS / "marginal_recovery"
OUT.mkdir(exist_ok=True)

MIN_N = 30

MODELS = [
    "deepseek", "gemma3-27b", "gpt-oss",
    "llama3.1_70b_base", "llama3.1_70b_instruct",
    "llama3.1_8b_base", "llama3.1_8b_instruct",
    "olmo3_32b_base", "olmo3_32b_dpo",
    "olmo3_7b_base", "olmo3_7b_dpo",
    "qwen3-32b", "qwen3-4b",
]


def _norm_raw_country(v: str) -> str:
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_respondent_country() -> pd.DataFrame:
    cache = OUT / "respondent_country.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype=str)

    canonical = json.load(open(SCRIPTS / "country_canonical_mapping.json"))
    by_survey = canonical["by_survey"]
    iso_numeric = canonical["iso_numeric"]

    seen: dict[tuple[str, str], str] = {}
    for f in sorted(MAIN_DATA.glob("*_instances.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                key = (d["survey"], str(d["id"]))
                if key in seen:
                    continue
                raw = _norm_raw_country(d.get("country", ""))
                iso2 = by_survey.get(d["survey"], {}).get(raw) or iso_numeric.get(raw) or raw
                seen[key] = iso2
    df = pd.DataFrame(
        [(s, r, c) for (s, r), c in seen.items()],
        columns=["survey", "respondent_id", "country"],
    )
    df.to_csv(cache, index=False)
    return df


def tv_distance(counts_a: Counter, counts_b: Counter) -> float:
    labels = set(counts_a) | set(counts_b)
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    return 0.5 * sum(abs(counts_a[l] / na - counts_b[l] / nb) for l in labels)


def js_divergence(counts_a: Counter, counts_b: Counter) -> float:
    """Jensen-Shannon divergence (natural log), matching the main-text metric."""
    labels = sorted(set(counts_a) | set(counts_b))
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    p = np.array([counts_a[l] / na for l in labels])
    q = np.array([counts_b[l] / nb for l in labels])
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def process_model(model: str, resp_country: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(
        ANALYSIS / model / "results_data.csv",
        usecols=["example_id", "survey", "respondent_id", "target_code",
                 "profile_type", "ground_truth", "predicted"],
        dtype=str,
    )
    df = repair(df, verbose=False)
    df = df.merge(resp_country, on=["survey", "respondent_id"], how="left")

    rows = []
    for (survey, target, profile), g in df.groupby(
        ["survey", "target_code", "profile_type"], sort=False
    ):
        global_true = Counter(g["ground_truth"])
        for country, cg in g.groupby("country"):
            n = len(cg)
            if n < MIN_N or not isinstance(country, str):
                continue
            true_c = Counter(cg["ground_truth"])
            pred_c = Counter(cg["predicted"])
            rows.append({
                "survey": survey,
                "target_code": target,
                "profile_type": profile,
                "country": country,
                "n": n,
                "tv_model": tv_distance(pred_c, true_c),
                "tv_global": tv_distance(global_true, true_c),
                "jsd_model": js_divergence(pred_c, true_c),
                "jsd_global": js_divergence(global_true, true_c),
            })
    out = pd.DataFrame(rows)
    out["model_beats_global"] = out["tv_model"] < out["tv_global"]
    out["model_beats_global_jsd"] = out["jsd_model"] < out["jsd_global"]
    out.to_csv(OUT / f"per_cell_{model}.csv", index=False)
    return out


def main() -> None:
    resp_country = build_respondent_country()
    print(f"respondent-country map: {len(resp_country)} respondents, "
          f"{resp_country['country'].nunique()} countries")

    summaries = []
    for model in MODELS:
        cells = process_model(model, resp_country)
        s = (
            cells.groupby("profile_type")
            .agg(
                n_cells=("tv_model", "size"),
                mean_tv_model=("tv_model", "mean"),
                mean_tv_global=("tv_global", "mean"),
                beats_global_share=("model_beats_global", "mean"),
                mean_jsd_model=("jsd_model", "mean"),
                mean_jsd_global=("jsd_global", "mean"),
                beats_global_share_jsd=("model_beats_global_jsd", "mean"),
            )
            .reset_index()
        )
        s.insert(0, "model", model)
        summaries.append(s)
        print(f"{model}: done ({len(cells)} cells)")

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 140):
        print(summary[summary["profile_type"] == "s6m4"]
              .sort_values("mean_tv_model").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
