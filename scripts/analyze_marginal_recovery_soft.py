"""Does the aggregate-recovery failure survive using probabilities, not argmax?

The marginal recovery result compares a synthetic sample's answer shares with
the true country shares, where the synthetic shares come from *tallying one
predicted answer per respondent*. That is what a practitioner querying a
synthetic panel obtains, but a reader can object that the argmax throws away
the model's uncertainty: perhaps the model represents the country distribution
well and we destroyed it by taking each respondent's mode.

This script tests that objection directly, mirroring the hard/soft check
already done for the heterogeneity metrics. For every respondent in a cell we
convert the per-option mean token logprobs into a distribution with a softmax,
average those distributions over the cell, and recompute the distance to the
truth. The comparison baseline is unchanged: the question's pooled
cross-country margin, which uses no model and no country information.

For each (survey, target, country) cell with >= MIN_N respondents at rich
profiles we report
    tv_hard    TV(argmax tally, truth)
    tv_soft    TV(mean predicted distribution, truth)
    tv_global  TV(pooled cross-country margin, truth)

The hard column is computed here from the raw JSONL rather than copied, so
agreement with analyze_marginal_recovery.py is a genuine cross-check of both
pipelines.

Outputs to analysis/marginal_recovery/:
  soft_per_cell_<model>.csv
  soft_summary.csv
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from repair_ids import load_valid_targets

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
RESULTS = ROOT / "results"
OUT = ROOT / "analysis" / "marginal_recovery"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_SUFFIX = "_s6m4"
MIN_N = 30

# results/ directory name -> the model key used everywhere else
RESULT_DIR_TO_MODEL = {
    "deepseek-v3p1-terminus": "deepseek",
    "gemma-3-27b-instruct": "gemma3-27b",
    "gpt_oss": "gpt-oss",
    "llama3.1-70b-base": "llama3.1_70b_base",
    "llama3.1-70b-instruct": "llama3.1_70b_instruct",
    "llama3.1-8b-base": "llama3.1_8b_base",
    "llama3.1-8b-instruct": "llama3.1_8b_instruct",
    "olmo3-32b-base": "olmo3_32b_base",
    "olmo3-32b-dpo": "olmo3_32b_dpo",
    "olmo3-7b-base": "olmo3_7b_base",
    "olmo3-7b-dpo": "olmo3_7b_dpo",
    "qwen3-32b": "qwen3-32b",
    "qwen3-4b": "qwen3-4b",
}


def softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def split_example_id(eid: str, survey: str, valid: dict[str, list[str]]
                     ) -> tuple[str, str] | None:
    """example_id -> (respondent_id, target_code).

    Same resolution rule as repair_ids: match the longest known target code
    that the stem ends with. Splitting on underscores at a fixed position is
    wrong whenever the target code itself contains one.
    """
    if not eid.endswith(PROFILE_SUFFIX):
        return None
    stem = eid[: -len(PROFILE_SUFFIX)]
    for code in valid.get(survey, []):
        suffix = f"_{code}"
        if stem.endswith(suffix):
            respondent = stem[: -len(suffix)]
            if respondent.startswith(f"{survey}_"):
                respondent = respondent[len(survey) + 1:]
            return respondent, code
    return None


def tv_from_vectors(p: dict[str, float], q: dict[str, float]) -> float:
    labels = set(p) | set(q)
    sp = sum(p.values()) or 1.0
    sq = sum(q.values()) or 1.0
    return 0.5 * sum(abs(p.get(l, 0.0) / sp - q.get(l, 0.0) / sq) for l in labels)


def process_model(result_dir: Path, model: str,
                  country_of: dict[tuple[str, str], str],
                  valid: dict[str, list[str]]) -> pd.DataFrame:
    # (survey, target, country) -> counters / accumulated distributions
    true_c: dict[tuple, Counter] = defaultdict(Counter)
    hard_c: dict[tuple, Counter] = defaultdict(Counter)
    soft_c: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n_resp: dict[tuple, int] = defaultdict(int)
    # (survey, target) -> pooled truth, for the no-model baseline
    pooled: dict[tuple, Counter] = defaultdict(Counter)

    unresolved = 0
    for f in sorted(result_dir.glob("*_results.jsonl")):
        survey = f.stem.split("_survey_")[1].rsplit("_results", 1)[0]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                eid = d.get("example_id", "")
                parts = split_example_id(eid, survey, valid)
                if parts is None:
                    unresolved += 1
                    continue
                respondent, target = parts
                lp = d.get("option_logprobs") or {}
                if len(lp) < 2:
                    continue

                gt = d["ground_truth"]
                pooled[(survey, target)][gt] += 1

                country = country_of.get((survey, respondent))
                if not country:
                    continue
                key = (survey, target, country)
                true_c[key][gt] += 1
                hard_c[key][d["predicted"]] += 1
                opts = list(lp)
                probs = softmax(np.array([lp[o] for o in opts], dtype=float))
                for o, p in zip(opts, probs):
                    soft_c[key][o] += float(p)
                n_resp[key] += 1

    rows = []
    for key, n in n_resp.items():
        if n < MIN_N:
            continue
        survey, target, country = key
        t = {k: float(v) for k, v in true_c[key].items()}
        h = {k: float(v) for k, v in hard_c[key].items()}
        g = {k: float(v) for k, v in pooled[(survey, target)].items()}
        rows.append({
            "model": model,
            "survey": survey,
            "target_code": target,
            "country": country,
            "n": n,
            "tv_hard": tv_from_vectors(h, t),
            "tv_soft": tv_from_vectors(soft_c[key], t),
            "tv_global": tv_from_vectors(g, t),
        })

    out = pd.DataFrame(rows)
    out["hard_beats_global"] = out["tv_hard"] < out["tv_global"]
    out["soft_beats_global"] = out["tv_soft"] < out["tv_global"]
    out["soft_beats_hard"] = out["tv_soft"] < out["tv_hard"]
    out.to_csv(OUT / f"soft_per_cell_{model}.csv", index=False)
    if unresolved:
        print(f"    ({unresolved:,} rows skipped: not rich profile or unparsed)")
    return out


def main() -> None:
    country_map = pd.read_csv(OUT / "respondent_country.csv", dtype=str, keep_default_na=False)
    country_of = {
        (r.survey, r.respondent_id): r.country
        for r in country_map.itertuples(index=False)
    }
    valid = load_valid_targets()
    print(f"respondent-country map: {len(country_of):,} respondents")

    summaries = []
    for result_dir_name, model in RESULT_DIR_TO_MODEL.items():
        d = RESULTS / result_dir_name
        if not d.exists():
            print(f"{model}: MISSING {d}")
            continue
        cells = process_model(d, model, country_of, valid)
        if cells.empty:
            print(f"{model}: no cells")
            continue
        summaries.append({
            "model": model,
            "n_cells": len(cells),
            "tv_hard": cells["tv_hard"].mean(),
            "tv_soft": cells["tv_soft"].mean(),
            "tv_global": cells["tv_global"].mean(),
            "hard_beats_global": cells["hard_beats_global"].mean(),
            "soft_beats_global": cells["soft_beats_global"].mean(),
            "soft_beats_hard": cells["soft_beats_hard"].mean(),
        })
        print(f"{model}: {len(cells)} cells  "
              f"hard {summaries[-1]['tv_hard']:.3f}  "
              f"soft {summaries[-1]['tv_soft']:.3f}  "
              f"global {summaries[-1]['tv_global']:.3f}")

    summary = pd.DataFrame(summaries).sort_values("tv_soft")
    summary.to_csv(OUT / "soft_summary.csv", index=False)

    print("\n=== soft vs hard marginal recovery, rich profiles ===")
    with pd.option_context("display.width", 160):
        print(summary.round(3).to_string(index=False))

    print(f"\nmean over models: hard {summary['tv_hard'].mean():.3f}  "
          f"soft {summary['tv_soft'].mean():.3f}  "
          f"global {summary['tv_global'].mean():.3f}")
    print(f"cells where soft beats the pooled margin: "
          f"{summary['soft_beats_global'].min():.1%} to "
          f"{summary['soft_beats_global'].max():.1%}")


if __name__ == "__main__":
    main()
