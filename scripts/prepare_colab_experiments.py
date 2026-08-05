"""Build paired instance files for the two Colab GPU experiments.

Experiment A — temporal context (reviewer iYUR):
    Multi-survey rich-profile (s6m4) instances joined to each respondent's
    fieldwork year (and fine interview date where available). The runner
    scores each instance under:
      baseline | with_year | with_year_placebo | with_date (if fine date)
    Primary isolation: true year vs placebo year vs no year.
    Secondary ablation: finer interview date where available.

Experiment B — controlled country injection:
    Rich-profile instances whose randomly sampled profile contains NO
    geographic item and whose profile and target question never name the
    respondent's country, stratified by world region. The runner scores:
      baseline | with_country | with_country_placebo
    where the placebo appends the same item carrying a country from a
    different region. Primary isolation: true country vs placebo country,
    since appending any country changes the prompt in ways that can move
    accuracy on their own.

Usage:
    python prepare_colab_experiments.py                  # temporal only (default)
    python prepare_colab_experiments.py --experiment country
    python prepare_colab_experiments.py --experiment all
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from survey_interview_dates import SURVEY_INSTANCE_FILES, load_all_date_maps

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
MAIN_DATA = ROOT / "synthetic_sampling" / "outputs" / "main_data_smaller_20_jan_26" / "main_data"
MR = ROOT / "analysis" / "marginal_recovery"
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / "synthetic_sampling" / "outputs" / "colab_experiments"
OUT.mkdir(exist_ok=True)

SEED = 42
PROFILE = "s6m4"
N_PER_SURVEY = 300
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


def _sample_stratified_by_target(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Proportional sample across target_code, up to k items."""
    if k >= len(items):
        rng.shuffle(items)
        return list(items)
    by_target: dict[str, list] = defaultdict(list)
    for d in items:
        by_target[d["target_code"]].append(d)
    total = len(items)
    sampled: list[dict] = []
    for target, group in sorted(by_target.items()):
        n = max(1, round(k * len(group) / total))
        rng.shuffle(group)
        sampled.extend(group[:n])
    rng.shuffle(sampled)
    return sampled[:k]


def _sample_survey(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Within-survey sample: stratify by survey_year when ≥2 years, else by target."""
    by_year: dict[int, list] = defaultdict(list)
    for d in items:
        by_year[d["survey_year"]].append(d)
    if len(by_year) < 2:
        return _sample_stratified_by_target(items, k, rng)

    total = len(items)
    sampled: list[dict] = []
    seen: set[str] = set()
    for year, group in sorted(by_year.items()):
        n = max(1, round(k * len(group) / total))
        chunk = _sample_stratified_by_target(group, min(n, len(group)), rng)
        for d in chunk:
            if d["example_id"] not in seen:
                sampled.append(d)
                seen.add(d["example_id"])
    if len(sampled) < k:
        remainder = [d for d in items if d["example_id"] not in seen]
        rng.shuffle(remainder)
        for d in remainder:
            if len(sampled) >= k:
                break
            sampled.append(d)
            seen.add(d["example_id"])
    rng.shuffle(sampled)
    return sampled[:k]


def _assign_placebos(sampled: list[dict], rng: random.Random) -> None:
    """Global-pool placebo years: sample from other instances' years ≠ true year."""
    year_pool = [d["survey_year"] for d in sampled]
    distinct = sorted(set(year_pool))
    if len(distinct) < 2:
        raise RuntimeError(
            "need ≥2 distinct survey years in the temporal sample for placebos"
        )
    for d in sampled:
        true = d["survey_year"]
        candidates = [y for y in year_pool if y != true]
        if not candidates:
            # true year is the only year present for this row's pool edge case
            candidates = [y for y in distinct if y != true]
        d["survey_year_placebo"] = rng.choice(candidates)


def build_temporal() -> None:
    rng = random.Random(SEED)
    print("Loading interview date maps from raw survey files...")
    date_maps = load_all_date_maps()

    by_survey: dict[str, list] = {}
    skipped_no_date = 0
    for survey, fname in SURVEY_INSTANCE_FILES.items():
        date_map = date_maps[survey]
        enriched = []
        n_profile = 0
        for d in load_instances(fname, PROFILE):
            n_profile += 1
            rec = date_map.get(str(d["id"]))
            if not rec:
                skipped_no_date += 1
                continue
            row = dict(d)
            row["survey_year"] = int(rec["survey_year"])
            row["interview_date"] = rec.get("interview_date")
            row["date_precision"] = rec.get("date_precision", "year")
            enriched.append(row)
        by_survey[survey] = enriched
        print(f"  {survey}: {len(enriched):,} s6m4 instances with year "
              f"(of {n_profile:,} profile rows)")

    sampled: list[dict] = []
    for survey in sorted(by_survey):
        items = by_survey[survey]
        take = _sample_survey(items, N_PER_SURVEY, rng)
        sampled.extend(take)
        years = sorted({d["survey_year"] for d in take})
        n_fine = sum(1 for d in take if d.get("interview_date"))
        print(f"  sampled {survey}: {len(take)}  years={years}  fine_date={n_fine}")

    rng.shuffle(sampled)
    _assign_placebos(sampled, rng)

    # Safety: placebos must differ
    bad = sum(1 for d in sampled if d["survey_year"] == d["survey_year_placebo"])
    if bad:
        raise RuntimeError(f"{bad} rows still have placebo == true year")

    path = OUT / "temporal_context_instances.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for d in sampled:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    surveys = pd.Series([d["survey"] for d in sampled])
    years = pd.Series([d["survey_year"] for d in sampled])
    n_fine = sum(1 for d in sampled if d.get("interview_date"))
    print(
        f"temporal: {len(sampled)} instances across {surveys.nunique()} surveys, "
        f"years {years.min()}-{years.max()}, fine_date={n_fine} "
        f"(skipped_no_date={skipped_no_date})"
    )
    print(surveys.value_counts().sort_index().to_string())
    print(f"wrote {path}")


def _country_leak_pattern(name: str) -> re.Pattern:
    """Match a country's name or its usual adjectival forms as whole words.

    The wording-based `country_in_profile` flag only covers residence items, so
    it misses the other way a profile gives the country away: many survey
    questions are localised ("Which political party in Serbia...", "the most
    recent national election in Germany"), and some answers name the country's
    language or the respondent's ancestry ("Slovak", "Ukrainian"). For those
    instances the baseline already states the country and the injection adds
    nothing, so they cannot serve as an implicit baseline.

    Matching the bare name misses demonyms and matching a short stem invents
    false positives ("Oman" inside "woman", "Mali" inside "malicious"), so we
    take both a whole-word match on the name and a stem plus the common
    adjectival endings, each anchored at word boundaries.
    """
    base = re.escape(name.split(",")[0].strip())
    stem = re.sub(r"(a|e|y|ia)$", "", base, flags=re.I)
    return re.compile(
        r"\b(" + base + r"|" + stem + r"(a|ia|y|an|ian|ish|ese|i|n|ic))\b", re.I
    )


def _reveals_country(instance: dict, name: str) -> bool:
    parts = [f"{q} {v}" for q, v in instance["questions"].items()]
    parts.append(instance.get("target_question", ""))
    return bool(_country_leak_pattern(name).search(" ".join(parts)))


def build_country_injection() -> None:
    rng = random.Random(SEED)
    flags = pd.read_csv(MR / "example_country_flag.csv")
    implicit_ids = set(flags.loc[~flags["country_in_profile"], "example_id"])
    resp_country = pd.read_csv(MR / "respondent_country.csv", dtype=str, keep_default_na=False)
    country_map = {(r.survey, r.respondent_id): r.country
                   for r in resp_country.itertuples()}
    region_map = json.load(open(SCRIPTS / "country_to_region.json"))

    by_region = defaultdict(list)
    skipped_name = 0
    skipped_leak = 0
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
            if _reveals_country(d, name):
                skipped_leak += 1
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

    # Placebo country, by the same logic as the placebo year in the temporal
    # experiment. Naming any country appends one profile item and signals that
    # location is relevant, and either can move accuracy with no national
    # knowledge involved; the temporal run showed that effect accounting for the
    # whole of an apparent gain. Contrasting the true country against a country
    # from a different region holds the added item fixed and leaves only the
    # contribution of the country being the right one.
    pool = sorted({(d["country_iso2"], d["country_name"], d["region"])
                   for d in sampled})
    by_region_pool = defaultdict(list)
    for iso2, name, region in pool:
        by_region_pool[region].append((iso2, name))
    for d in sampled:
        other_regions = [r for r in by_region_pool if r != d["region"]]
        pr = rng.choice(other_regions)
        iso2, name = rng.choice(by_region_pool[pr])
        d["country_placebo_iso2"] = iso2
        d["country_placebo_name"] = name
        d["country_placebo_region"] = pr
    bad = sum(1 for d in sampled
              if d["country_placebo_iso2"] == d["country_iso2"]
              or d["country_placebo_region"] == d["region"])
    if bad:
        raise RuntimeError(f"{bad} rows have a placebo in the true region")

    path = OUT / "country_injection_instances.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for d in sampled:
            d["country_question"] = COUNTRY_QUESTION
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    counts = pd.Series({r: min(len(v), N_PER_REGION) for r, v in by_region.items()})
    print(f"country injection: {len(sampled)} instances across {len(by_region)} regions "
          f"(skipped {skipped_name} for missing country name, "
          f"{skipped_leak} whose profile or target already names the country)")
    print(counts.sort_values().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        choices=["temporal", "country", "all"],
        default="temporal",
        help="which experiment instance file(s) to rebuild (default: temporal)",
    )
    args = ap.parse_args()
    if args.experiment in ("temporal", "all"):
        build_temporal()
    if args.experiment in ("country", "all"):
        build_country_injection()


if __name__ == "__main__":
    main()
