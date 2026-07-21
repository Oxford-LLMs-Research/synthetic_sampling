"""Build paired instance files for the two Colab GPU experiments.

Experiment A — temporal context (reviewer iYUR):
    ~2,000 WVS rich-profile instances, each joined with the respondent's
    fieldwork year (A_YEAR from the raw WVS file). The runner scores each
    instance twice: with and without a "The survey was conducted in {year}."
    line in the prompt.

Experiment B — controlled country injection:
    Rich-profile instances whose randomly sampled profile contains NO
    geographic item (the implicit condition of the observational split),
    stratified by world region. The runner scores each instance twice: with
    the original profile, and with one appended feature
    "In which country do you live?" -> country name. Unlike the
    observational split, this holds the rest of the profile fixed, giving a
    within-instance causal estimate of the country cue's effect.

Both experiments use the same model and scoring pipeline (colab_run_experiments.py),
so each condition pair differs only in the manipulated element.

Outputs (to synthetic_sampling/outputs/colab_experiments/):
    temporal_context_instances.jsonl
    country_injection_instances.jsonl
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
MAIN_DATA = ROOT / "synthetic_sampling" / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
MR = ROOT / "analysis" / "marginal_recovery"
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / "synthetic_sampling" / "outputs" / "colab_experiments"
OUT.mkdir(exist_ok=True)

SEED = 42
PROFILE = "s6m4"
N_TEMPORAL = 2000
N_PER_REGION = 150

COUNTRY_QUESTION = "In which country do you live?"

NAME_OVERRIDES = {
    "TW": "Taiwan", "KR": "South Korea", "RU": "Russia", "BO": "Bolivia",
    "VE": "Venezuela", "IR": "Iran", "TZ": "Tanzania", "MD": "Moldova",
    "VN": "Vietnam", "LA": "Laos", "SY": "Syria", "PS": "Palestine",
    "CZ": "Czech Republic", "GB": "Great Britain", "US": "United States",
    "MK": "North Macedonia", "CD": "Democratic Republic of the Congo",
    "CI": "Ivory Coast", "SZ": "Eswatini", "TR": "Turkey",
}


def country_name(iso2: str) -> str | None:
    if iso2 in NAME_OVERRIDES:
        return NAME_OVERRIDES[iso2]
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=iso2)
        if c is None:
            return None
        return getattr(c, "common_name", None) or c.name
    except ImportError:
        return None


def load_instances(survey_file: str, profile: str) -> list[dict]:
    out = []
    with open(MAIN_DATA / survey_file, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["profile_type"] == profile:
                out.append(d)
    return out


def build_temporal() -> None:
    rng = random.Random(SEED)
    inst = load_instances("wvs_instances.jsonl", PROFILE)

    wvs = pd.read_csv(ROOT / "data" / "WVS" / "WVS_2017_22.csv",
                      usecols=["D_INTERVIEW", "A_YEAR"], dtype=str)
    year_map = dict(zip(wvs["D_INTERVIEW"], wvs["A_YEAR"]))

    by_target = defaultdict(list)
    for d in inst:
        year = year_map.get(str(d["id"]))
        if year and year.strip() and int(float(year)) > 2000:
            d["survey_year"] = int(float(year))
            by_target[d["target_code"]].append(d)

    total = sum(len(v) for v in by_target.values())
    sampled = []
    for target, items in sorted(by_target.items()):
        k = max(1, round(N_TEMPORAL * len(items) / total))
        rng.shuffle(items)
        sampled.extend(items[:k])
    rng.shuffle(sampled)
    sampled = sampled[:N_TEMPORAL]

    path = OUT / "temporal_context_instances.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for d in sampled:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    years = pd.Series([d["survey_year"] for d in sampled])
    print(f"temporal: {len(sampled)} instances, {len({d['target_code'] for d in sampled})} targets, "
          f"years {years.min()}-{years.max()}")


def build_country_injection() -> None:
    rng = random.Random(SEED)
    flags = pd.read_csv(MR / "example_country_flag.csv")
    implicit_ids = set(flags.loc[~flags["country_in_profile"], "example_id"])
    resp_country = pd.read_csv(MR / "respondent_country.csv", dtype=str)
    country_map = {(r.survey, r.respondent_id): r.country
                   for r in resp_country.itertuples()}
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))

    by_region = defaultdict(list)
    skipped_name = 0
    for f in sorted(MAIN_DATA.glob("*_instances.jsonl")):
        for d in load_instances(f.name, PROFILE):
            if d["example_id"] not in implicit_ids:
                continue
            # The wording-based flag covers residence questions only; birth-country
            # items ("In which country were you born / was your mother born?")
            # also reveal the country and must not leak into the implicit baseline.
            if any("in which country" in q.lower() for q in d["questions"]):
                continue
            iso2 = country_map.get((d["survey"], str(d["id"])))
            region = region_map.get(iso2) if iso2 else None
            if not region or region == "Unknown":
                continue
            name = country_name(iso2)
            if not name:
                skipped_name += 1
                continue
            d["country_iso2"] = iso2
            d["country_name"] = name
            d["region"] = region
            by_region[region].append(d)

    sampled = []
    for region, items in sorted(by_region.items()):
        rng.shuffle(items)
        sampled.extend(items[:N_PER_REGION])
    rng.shuffle(sampled)

    path = OUT / "country_injection_instances.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for d in sampled:
            d["country_question"] = COUNTRY_QUESTION
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    counts = pd.Series({r: min(len(v), N_PER_REGION) for r, v in by_region.items()})
    print(f"country injection: {len(sampled)} instances across {len(by_region)} regions "
          f"(skipped {skipped_name} for missing country name)")
    print(counts.sort_values().to_string())


if __name__ == "__main__":
    build_temporal()
    build_country_injection()
