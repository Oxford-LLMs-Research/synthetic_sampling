"""Sample-decision cell census: usable respondents per survey x country.

Phase 0 input for the stratified-draw design (sample decision, gate 3).
Counts respondents per country cell in each source's microdata, after the
same duplicate-id hygiene the XGB full fit applied, and reports what a
uniform draw vs a per-country floor would leave in each cell.

Outputs (WORK/analysis/sample_decision/):
  cell_census.csv    survey, country, n_respondents
  survey_summary.csv survey totals, country counts, min/median/max cell
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
RAW_DATA = WORK / "data"
OUT_DIR = WORK / "analysis" / "sample_decision"

sys.path.insert(0, str(REPO / "src"))

SURVEYS = [
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_10",
    "ess_wave_11",
]


def main() -> None:
    from synthetic_sampling.surveys import DataPaths
    from synthetic_sampling.surveys.loaders import SurveyLoader
    from synthetic_sampling.surveys.registry import get_survey_config

    paths = DataPaths.default_bundled(RAW_DATA, "./outputs")
    loader = SurveyLoader(paths, verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cell_rows: list[dict] = []
    summary_rows: list[dict] = []

    for survey in SURVEYS:
        df, _meta = loader.load_survey(survey)
        cfg = get_survey_config(survey)

        ids = df[cfg.respondent_id_col].astype(str)
        dup = ids.duplicated(keep=False)
        if dup.any():
            # same convention as run_xgb_full.py: ambiguous ids leave the pool
            df = df.loc[~dup]

        counts = (
            df[cfg.country_col]
            .astype(str)
            .value_counts()
            .sort_index()
        )
        for country, n in counts.items():
            cell_rows.append(
                {"survey": survey, "country": country, "n_respondents": int(n)}
            )

        summary_rows.append(
            {
                "survey": survey,
                "n_respondents": int(counts.sum()),
                "n_countries": int(len(counts)),
                "cell_min": int(counts.min()),
                "cell_median": float(counts.median()),
                "cell_max": int(counts.max()),
                "dropped_dup_ids": int(dup.sum()),
            }
        )
        print(
            f"{survey:16s} n={counts.sum():>7,} countries={len(counts):>2} "
            f"cell min/med/max = {counts.min()}/{counts.median():.0f}/{counts.max()}"
            + (f"  (dropped {dup.sum()} dup-id rows)" if dup.any() else "")
        )

    pd.DataFrame(cell_rows).to_csv(OUT_DIR / "cell_census.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "survey_summary.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'cell_census.csv'}")
    print(f"wrote {OUT_DIR / 'survey_summary.csv'}")


if __name__ == "__main__":
    main()
