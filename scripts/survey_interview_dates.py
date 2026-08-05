"""Load per-respondent interview year / fine date from raw survey files.

Maps respondent_id strings in the same format as the main JSONL `id` field
(see extract_respondent_demographics.py / surveys.py).

Returns dict[respondent_id] -> {
    "survey_year": int,
    "interview_date": str | None,   # human-readable cue for with_date
    "date_precision": "year" | "month" | "day",
}

Latino year is wave-level 2023 (no within-survey year variance); fine dates
are still built from DIAREAL + MESREAL.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(r"C:\Users\murrn\cursor\synthetic_sampling\data")

DateRec = dict  # survey_year, interview_date, date_precision


def _year_ok(y) -> bool:
    try:
        yi = int(float(y))
    except (TypeError, ValueError):
        return False
    return 1990 <= yi <= 2035


def _parse_iso_date(val) -> tuple[int | None, str | None]:
    """Return (year, YYYY-MM-DD) from an ISO-ish date string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None, None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return None, None
    # ESS inwds is "yyyy-mm-dd hh:mm:ss"
    date_part = s.split()[0] if " " in s else s
    try:
        ts = pd.to_datetime(date_part, errors="coerce")
    except Exception:
        return None, None
    if pd.isna(ts):
        return None, None
    return int(ts.year), ts.strftime("%Y-%m-%d")


def load_wvs() -> dict[str, DateRec]:
    df = pd.read_csv(
        DATA / "WVS" / "WVS_2017_22.csv",
        usecols=["D_INTERVIEW", "A_YEAR", "J_INTDATE"],
        low_memory=False,
    )
    out: dict[str, DateRec] = {}
    for row in df.itertuples(index=False):
        rid = str(row.D_INTERVIEW)
        year = None
        try:
            y = int(float(row.A_YEAR))
            if _year_ok(y):
                year = y
        except (TypeError, ValueError):
            pass
        interview_date = None
        precision = "year"
        try:
            jd = int(float(row.J_INTDATE))
        except (TypeError, ValueError):
            jd = 0
        if jd > 20000101:
            s = str(jd)
            if len(s) == 8:
                interview_date = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                precision = "day"
                if year is None and _year_ok(int(s[:4])):
                    year = int(s[:4])
        if year is None:
            continue
        out[rid] = {
            "survey_year": year,
            "interview_date": interview_date,
            "date_precision": precision,
        }
    return out


def load_afrobarometer() -> dict[str, DateRec]:
    df = pd.read_csv(
        DATA / "Afrobarometer" / "afrobarometer_r9_converted.csv",
        usecols=["RESPNO", "DATEINTR"],
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    out: dict[str, DateRec] = {}
    for row in df.itertuples(index=False):
        year, iso = _parse_iso_date(row.DATEINTR)
        if year is None:
            continue
        out[str(row.RESPNO)] = {
            "survey_year": year,
            "interview_date": iso,
            "date_precision": "day" if iso else "year",
        }
    return out


def load_arabbarometer() -> dict[str, DateRec]:
    df = pd.read_csv(
        DATA / "Arabbarometer" / "ArabBarometer_WaveVIII_English_v3.csv",
        usecols=["ID", "DATE"],
        low_memory=False,
    )
    out: dict[str, DateRec] = {}
    for row in df.itertuples(index=False):
        year, iso = _parse_iso_date(row.DATE)
        if year is None:
            continue
        out[str(row.ID)] = {
            "survey_year": year,
            "interview_date": iso,
            "date_precision": "day" if iso else "year",
        }
    return out


def load_asianbarometer() -> dict[str, DateRec]:
    df = pd.read_csv(
        DATA / "Asianbarometer" / "asiabarom_combined_datasets.csv",
        usecols=["country", "idnumber", "year", "month"],
        low_memory=False,
    )
    out: dict[str, DateRec] = {}
    for row in df.itertuples(index=False):
        rid = f"{row.country}_{row.idnumber}"
        try:
            year = int(float(row.year))
        except (TypeError, ValueError):
            continue
        if not _year_ok(year):
            continue
        month = row.month
        interview_date = None
        precision = "year"
        if isinstance(month, str) and month.strip() and month.strip().lower() != "nan":
            interview_date = f"{month.strip()} {year}"
            precision = "month"
        out[rid] = {
            "survey_year": year,
            "interview_date": interview_date,
            "date_precision": precision,
        }
    return out


def load_ess(wave: str) -> dict[str, DateRec]:
    """wave: 'wave_10' or 'wave_11' -> survey_id ess_wave_10 / ess_wave_11."""
    f = next((DATA / "ESS" / wave).glob("*consolidations*.csv"))
    df = pd.read_csv(f, usecols=["cntry", "idno", "inwds"], low_memory=False)
    out: dict[str, DateRec] = {}
    for row in df.itertuples(index=False):
        rid = f"{row.cntry}_{row.idno}"
        year, iso = _parse_iso_date(row.inwds)
        if year is None:
            continue
        out[rid] = {
            "survey_year": year,
            "interview_date": iso,
            "date_precision": "day" if iso else "year",
        }
    return out


def load_latinobarometer() -> dict[str, DateRec]:
    """Year is wave-level 2023; fine date from day+month when valid."""
    import pyreadstat

    df, _ = pyreadstat.read_sav(
        str(DATA / "Latinobarometro" / "Latinobarometro_2023_Eng_Spss_v1_0.sav"),
        usecols=["IDENPA", "NUMENTRE", "DIAREAL", "MESREAL"],
    )
    out: dict[str, DateRec] = {}
    year = 2023
    for row in df.itertuples(index=False):
        rid = f"{row.IDENPA}_{row.NUMENTRE}"
        interview_date = None
        precision = "year"
        try:
            day = int(float(row.DIAREAL))
            month = int(float(row.MESREAL))
            if 1 <= month <= 12 and 1 <= day <= 31:
                interview_date = f"{year:04d}-{month:02d}-{day:02d}"
                # validate calendar date
                ts = pd.to_datetime(interview_date, errors="coerce")
                if pd.isna(ts):
                    interview_date = None
                else:
                    precision = "day"
        except (TypeError, ValueError):
            pass
        out[rid] = {
            "survey_year": year,
            "interview_date": interview_date,
            "date_precision": precision,
        }
    return out


SURVEY_LOADERS = {
    "wvs": load_wvs,
    "afrobarometer": load_afrobarometer,
    "arabbarometer": load_arabbarometer,
    "asianbarometer": load_asianbarometer,
    "ess_wave_10": lambda: load_ess("wave_10"),
    "ess_wave_11": lambda: load_ess("wave_11"),
    "latinobarometer": load_latinobarometer,
}

SURVEY_INSTANCE_FILES = {
    "wvs": "wvs_instances.jsonl",
    "afrobarometer": "afrobarometer_instances.jsonl",
    "arabbarometer": "arabbarometer_instances.jsonl",
    "asianbarometer": "asianbarometer_instances.jsonl",
    "ess_wave_10": "ess_wave_10_instances.jsonl",
    "ess_wave_11": "ess_wave_11_instances.jsonl",
    "latinobarometer": "latinobarometer_instances.jsonl",
}


def load_all_date_maps() -> dict[str, dict[str, DateRec]]:
    maps = {}
    for survey, loader in SURVEY_LOADERS.items():
        print(f"  loading dates: {survey} ...", flush=True)
        m = loader()
        n_fine = sum(1 for v in m.values() if v.get("interview_date"))
        years = sorted({v["survey_year"] for v in m.values()})
        print(f"    {len(m):,} respondents, fine_date={n_fine:,}, years={years}")
        maps[survey] = m
    return maps


if __name__ == "__main__":
    load_all_date_maps()
