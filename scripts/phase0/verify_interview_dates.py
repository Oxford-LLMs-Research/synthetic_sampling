"""Verify SurveyConfig interview-timing fields against the microdata.

For every survey in the registry: the configured columns must exist in the
file we hold, dates must parse under the configured format, and the observed
interview-year range must match the recorded `field_period`. Exit 0 or the
config is wrong.

    python scripts/phase0/verify_interview_dates.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from synthetic_sampling.surveys.registry import SURVEY_REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data"

FILES = {
    "wvs": (DATA / "WVS" / "WVS_2017_22.csv", "csv"),
    "afrobarometer": (DATA / "Afrobarometer" / "afrobarometer_r9_converted.csv", "csv"),
    "arabbarometer": (DATA / "Arabbarometer" / "ArabBarometer_WaveVIII_English_v3.csv", "csv"),
    "asianbarometer": (DATA / "Asianbarometer" / "asiabarom_combined_datasets.csv", "csv"),
    "latinobarometer": (DATA / "Latinobarometro" / "Latinobarometro_2023_Eng_Spss_v1_0.sav", "sav"),
    "ess_wave_10": (DATA / "ESS" / "wave_10" / "ESS10_with_consolidations.csv", "csv"),
    "ess_wave_11": (DATA / "ESS" / "wave_11" / "ESS11_with_consolidations.csv", "csv"),
}


def load(path: Path, kind: str, cols: list[str]) -> pd.DataFrame:
    if kind == "sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(str(path), usecols=cols)
        return df
    return pd.read_csv(path, usecols=cols, low_memory=False)


def years_from_period(period: str) -> tuple[int, int]:
    ys = [int(y) for y in re.findall(r"(20\d\d)", period)]
    return min(ys), max(ys)


def main() -> int:
    bad: list[str] = []
    for survey_id, cfg in SURVEY_REGISTRY.items():
        path, kind = FILES[survey_id]
        cols = [c for c in (
            cfg.interview_date_col, cfg.interview_year_col,
            *(cfg.interview_date_parts or ())) if c]
        cols = list(dict.fromkeys(cols))
        if not cols:
            bad.append(f"{survey_id}: no timing columns configured")
            continue
        if not cfg.wave_label or not cfg.field_period:
            bad.append(f"{survey_id}: wave_label/field_period missing")
        df = load(path, kind, cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            bad.append(f"{survey_id}: columns not in file: {missing}")
            continue

        observed_years: set[int] = set()
        if cfg.interview_date_col:
            raw = df[cfg.interview_date_col].dropna()
            parsed = 0
            for v in raw:
                s = str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
                try:
                    dt = datetime.strptime(s, cfg.interview_date_format)
                except ValueError:
                    continue
                parsed += 1
                observed_years.add(dt.year)
            share = parsed / len(raw) if len(raw) else 0
            if share < 0.90:
                bad.append(f"{survey_id}: only {share:.1%} of "
                           f"{cfg.interview_date_col} parse as "
                           f"{cfg.interview_date_format}")
            print(f"{survey_id:16s} {cfg.interview_date_col}: "
                  f"{share:.1%} parse, years {min(observed_years)}-"
                  f"{max(observed_years)}" if observed_years else
                  f"{survey_id:16s} {cfg.interview_date_col}: nothing parsed")
        if cfg.interview_year_col:
            ys = pd.to_numeric(df[cfg.interview_year_col], errors="coerce").dropna()
            ys = ys[(ys > 2000) & (ys < 2030)].astype(int)
            observed_years.update(ys.unique())
            print(f"{survey_id:16s} {cfg.interview_year_col}: "
                  f"years {ys.min()}-{ys.max()}")
        if not observed_years and cfg.interview_date_parts:
            # Parts-only source (Latino): year is constant, from field_period.
            lo, hi = years_from_period(cfg.field_period)
            observed_years = set(range(lo, hi + 1))
            for c in cfg.interview_date_parts:
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                print(f"{survey_id:16s} {c}: range "
                      f"{int(vals.min())}-{int(vals.max())}")

        lo, hi = years_from_period(cfg.field_period)
        if observed_years and not (min(observed_years) >= lo - 0
                                   and max(observed_years) <= hi):
            bad.append(f"{survey_id}: observed years "
                       f"{min(observed_years)}-{max(observed_years)} outside "
                       f"field_period {cfg.field_period!r}")

    for m in bad:
        print("FAIL", m)
    print(f"\n{len(bad)} failures across {len(SURVEY_REGISTRY)} surveys")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
