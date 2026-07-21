"""Extract harmonized respondent demographics from the raw survey files.

Builds analysis/equity_audit/respondent_demographics.csv with one row per
(survey, respondent_id) and four harmonized dimensions:

  gender     male / female
  age_band   18-29 / 30-49 / 50+
  education  low (up to primary / lower secondary) /
             mid (secondary) / high (post-secondary)
  urban      urban / rural

respondent_id reproduces the JSONL id format of the main dataset so the
table joins directly onto results_data.csv (see implementation reference,
section 3, for the id mappings).

Harmonization choices (documented for the appendix):
  WVS Q275 (ISCED 0-8):       0-2 low, 3-4 mid, 5-8 high
  ESS eisced (ES-ISCED I-V2): 1-2 low, 3-4 mid, 5-7 high (55/77/88/99 NA)
  Afrobarometer Q94 (0-9):    0-3 low, 4-5 mid, 6-9 high; URBRUR semi-urban -> urban
  Arab Barometer Q1003 (1-7): 1-3 low, 4-5 mid, 6-7 high
  Asian Barometer SE5 (text): primary or less -> low, secondary -> mid,
                              university/college/post-graduate -> high
  Latinobarometro S11:        1-7 (<=6 yrs) low, 8-13 (7-12 yrs) mid,
                              14-17 (technical/university) high
  Latinobarometro TAMCIUD:    <=2 (<=10,000 inhabitants) rural, else urban
  ESS domicil:                1-3 urban, 4-5 rural
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"C:\Users\murrn\cursor\synthetic_sampling\data")
OUT = Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis\equity_audit")
OUT.mkdir(exist_ok=True)


def age_band(age: pd.Series) -> pd.Series:
    a = pd.to_numeric(age, errors="coerce")
    a = a.where((a >= 16) & (a <= 105))
    return pd.cut(a, [16, 30, 50, 106], labels=["18-29", "30-49", "50+"],
                  right=False).astype(object)


def code_map(series: pd.Series, mapping: dict) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=object)
    for (lo, hi), label in mapping.items():
        out[(s >= lo) & (s <= hi)] = label
    return out


def wvs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "WVS" / "WVS_2017_22.csv",
                     usecols=["D_INTERVIEW", "Q260", "Q262", "Q275", "H_URBRURAL"],
                     low_memory=False)
    return pd.DataFrame({
        "survey": "wvs",
        "respondent_id": df["D_INTERVIEW"].astype(str),
        "gender": code_map(df["Q260"], {(1, 1): "male", (2, 2): "female"}),
        "age_band": age_band(df["Q262"]),
        "education": code_map(df["Q275"], {(0, 2): "low", (3, 4): "mid", (5, 8): "high"}),
        "urban": code_map(df["H_URBRURAL"], {(1, 1): "urban", (2, 2): "rural"}),
    })


def ess(wave: str) -> pd.DataFrame:
    f = next((DATA / "ESS" / wave).glob("*.csv"))
    df = pd.read_csv(f, usecols=["cntry", "idno", "gndr", "agea", "eisced", "domicil"],
                     low_memory=False)
    return pd.DataFrame({
        "survey": wave.replace("wave_", "ess_wave_"),
        "respondent_id": df["cntry"].astype(str) + "_" + df["idno"].astype(str),
        "gender": code_map(df["gndr"], {(1, 1): "male", (2, 2): "female"}),
        "age_band": age_band(df["agea"].replace(999, np.nan)),
        "education": code_map(df["eisced"], {(1, 2): "low", (3, 4): "mid", (5, 7): "high"}),
        "urban": code_map(df["domicil"], {(1, 3): "urban", (4, 5): "rural"}),
    })


def afrobarometer() -> pd.DataFrame:
    df = pd.read_csv(DATA / "Afrobarometer" / "afrobarometer_r9_converted.csv",
                     usecols=["RESPNO", "URBRUR", "Q1", "Q94", "Q100"],
                     encoding="utf-8", encoding_errors="replace", low_memory=False)
    return pd.DataFrame({
        "survey": "afrobarometer",
        "respondent_id": df["RESPNO"].astype(str),
        "gender": code_map(df["Q100"], {(1, 1): "male", (2, 2): "female"}),
        "age_band": age_band(df["Q1"].where(df["Q1"] < 130)),
        "education": code_map(df["Q94"], {(0, 3): "low", (4, 5): "mid", (6, 9): "high"}),
        "urban": code_map(df["URBRUR"], {(1, 1): "urban", (2, 2): "rural", (3, 3): "urban"}),
    })


def arabbarometer() -> pd.DataFrame:
    df = pd.read_csv(DATA / "Arabbarometer" / "ArabBarometer_WaveVIII_English_v3.csv",
                     usecols=["ID", "Q13", "Q1001", "Q1002", "Q1003"], low_memory=False)
    return pd.DataFrame({
        "survey": "arabbarometer",
        "respondent_id": df["ID"].astype(str),
        "gender": code_map(df["Q1002"], {(1, 1): "male", (2, 2): "female"}),
        "age_band": age_band(df["Q1001"]),
        "education": code_map(df["Q1003"], {(1, 3): "low", (4, 5): "mid", (6, 7): "high"}),
        "urban": code_map(df["Q13"], {(1, 1): "urban", (2, 2): "rural"}),
    })


def asianbarometer() -> pd.DataFrame:
    df = pd.read_csv(DATA / "Asianbarometer" / "asiabarom_combined_datasets.csv",
                     usecols=["country", "idnumber", "level", "SE2", "SE3_1", "SE5"],
                     low_memory=False)

    def edu(text):
        if not isinstance(text, str):
            return np.nan
        t = text.lower()
        if "university" in t or "college" in t or "post-graduate" in t:
            return "high"
        if "secondary" in t or "high school" in t:
            return "mid"
        if "primary" in t or "no formal" in t or "informal" in t:
            return "low"
        return np.nan

    return pd.DataFrame({
        "survey": "asianbarometer",
        "respondent_id": df["country"].astype(str) + "_" + df["idnumber"].astype(str),
        "gender": df["SE2"].map({"Male": "male", "Female": "female"}),
        "age_band": age_band(df["SE3_1"].where(df["SE3_1"] < 98)),
        "education": df["SE5"].map(edu),
        "urban": df["level"].map({"Urban": "urban", "Rural": "rural"}),
    })


def latinobarometer() -> pd.DataFrame:
    import pyreadstat
    df, _ = pyreadstat.read_sav(
        str(DATA / "Latinobarometro" / "Latinobarometro_2023_Eng_Spss_v1_0.sav"),
        usecols=["IDENPA", "NUMENTRE", "SEXO", "EDAD", "S11", "TAMCIUD"])
    return pd.DataFrame({
        "survey": "latinobarometer",
        "respondent_id": df["IDENPA"].astype(str) + "_" + df["NUMENTRE"].astype(str),
        "gender": code_map(df["SEXO"], {(1, 1): "male", (2, 2): "female"}),
        "age_band": age_band(df["EDAD"]),
        "education": code_map(df["S11"], {(1, 7): "low", (8, 13): "mid", (14, 17): "high"}),
        "urban": code_map(df["TAMCIUD"], {(1, 2): "rural", (3, 8): "urban"}),
    })


def main() -> None:
    frames = [wvs(), ess("wave_10"), ess("wave_11"), afrobarometer(),
              arabbarometer(), asianbarometer(), latinobarometer()]
    demo = pd.concat(frames, ignore_index=True)
    demo.to_csv(OUT / "respondent_demographics.csv", index=False)

    print(f"{len(demo):,} respondents extracted")
    for col in ["gender", "age_band", "education", "urban"]:
        cov = demo.groupby("survey")[col].apply(lambda s: s.notna().mean())
        print(f"\ncoverage {col}:")
        print(cov.round(3).to_string())

    # Join check against the evaluation sample
    rc = pd.read_csv(Path(r"C:\Users\murrn\cursor\synthetic_sampling\analysis")
                     / "marginal_recovery" / "respondent_country.csv", dtype=str)
    merged = rc.merge(demo, on=["survey", "respondent_id"], how="left")
    print("\njoin rate onto evaluation respondents:")
    print(merged.groupby("survey")["gender"].apply(lambda s: s.notna().mean())
          .round(3).to_string())


if __name__ == "__main__":
    main()
