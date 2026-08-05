#!/usr/bin/env python
r"""List the (target, country) cells the oracle should be fitted on.

The ladder's informative and anti arms order a respondent's features by how well
each predicts the target IN THAT RESPONDENT'S COUNTRY. Pooling countries would
let a feature look predictive merely by proxying for country, which is the one
thing a within-respondent profile experiment must not confound.

So the oracle runs per (target, country) cell, and this writes the cell list.
Targets are the ladder's own set, so the mechanical run and the three-arm run
sit on the same questions.

Countries are ranked by how many respondents answered the target, which is a
power criterion rather than an outcome one, and the top N are taken. N trades
oracle compute against how much country variation the analysis can see. Their
existing audit covers 89 cells, so N=3 reproduces that scale and N=5 is about
1.7 times it.

    python .../make_oracle_cells.py --per-target 5
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from synthetic_sampling.config import DataPaths                        # noqa: E402
from synthetic_sampling.config.surveys import get_survey_config        # noqa: E402
from synthetic_sampling.loaders.survey_loader import SurveyLoader      # noqa: E402

SCALE = REPO / "outputs" / "scaling_experiment"
MIN_RESPONDENTS = 300     # below this an oracle fit is not worth doing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target", type=int, default=5)
    ap.add_argument("--targets", type=Path, default=SCALE / "ladder_targets.csv")
    ap.add_argument("--out", type=Path, default=SCALE / "oracle_cells.csv")
    args = ap.parse_args()

    want = pd.read_csv(args.targets)
    paths = DataPaths.from_yaml(str(REPO / "configs" / "local.yaml"))
    loader = SurveyLoader(paths, verbose=False)

    rows = []
    for survey, grp in want.groupby("survey"):
        cfg = get_survey_config(survey)
        df, _ = loader.load_survey(survey)
        cc = cfg.country_col
        if cc not in df.columns:
            print(f"  {survey}: no country column {cc!r}, skipped")
            continue
        for target in sorted(grp["target_code"]):
            if target not in df.columns:
                print(f"  {survey}/{target}: not in the data file, skipped")
                continue
            sub = df.loc[df[target].notna(), [cc, target]]
            counts = sub.groupby(cc).size().sort_values(ascending=False)
            counts = counts[counts >= MIN_RESPONDENTS]
            for country, n in counts.head(args.per_target).items():
                rows.append({"survey": survey, "target_code": target,
                             "country": str(country), "n_respondents": int(n),
                             "n_countries_available": int(len(counts))})
        print(f"  {survey}: done")

    if not rows:
        raise SystemExit("no cells produced")
    cells = pd.DataFrame(rows)
    cells.to_csv(args.out, index=False)

    print(f"\n{len(cells)} cells over "
          f"{cells.groupby(['survey','target_code']).ngroups} targets")
    print(f"respondents per cell: median {cells['n_respondents'].median():,.0f}, "
          f"min {cells['n_respondents'].min():,}, "
          f"max {cells['n_respondents'].max():,}")
    print("\ncells per survey:")
    for s, n in cells["survey"].value_counts().sort_index().items():
        print(f"  {s:<18}{n:>4}")
    short = (cells.groupby(["survey", "target_code"])["country"].size()
             < args.per_target)
    if short.any():
        print(f"\n{int(short.sum())} targets have fewer than {args.per_target} "
              f"countries with >= {MIN_RESPONDENTS} respondents:")
        for (s, t), n in (cells.groupby(["survey", "target_code"])["country"]
                          .size()[short].items()):
            print(f"  {s:<18}{t:<12}{n} countries")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
