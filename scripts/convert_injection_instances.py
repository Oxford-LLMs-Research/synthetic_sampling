"""A1 converter: injection instance files -> minimal-core runner format.

The country and temporal injection nulls (RUN_CATALOGUE A1) were measured
under old echo scoring by ``nebius_run_experiments.py`` (commit 6ae112f),
serving Qwen3-32B with vLLM on ARC. The retest rescoreS the EXISTING instances
under ``label_num`` plus replicate (reuse rule: instances yes, scores never).
This converter expands each source instance into one runner instance per
condition, reproducing the original constructions exactly:

  country   conditions baseline / with_country / with_country_placebo; the
            country question is APPENDED to the profile as a final q:a pair
            ("In which country do you live?": <real or placebo name>).
  temporal  conditions baseline / with_year / with_year_placebo / with_date
            (with_date only where an interview date exists); a context line
            goes between profile and target question via the instance
            ``extra`` field, which ``scoring.prompts.build_prompt`` renders
            and the PMI premises exclude.

The temporal substrate additionally carries the previously missing combined
cell (added 8 Aug, pre-registration addendum): with_country and
with_country_and_year complete a 2x2 factorial with baseline and with_year on
ONE substrate, so the joint effect decomposes within a single serving. Country
names for temporal instances join from the country file on (survey, country
code); five WVS codes absent there are resolved by a fixed table cross-checked
against the historical ``country_canonical_mapping.json`` ISO entries.

example_id gains the condition suffix, so resume-by-example_id and the
hash-based replicate draw stay deterministic and every condition of a source
instance lands in the same output file (paired same-serving contrasts).
base_id is set to the SOURCE example_id: the pairing key across conditions.

Usage (from the repo root; defaults match the recovered files):
    python scripts/convert_injection_instances.py
    python scripts/convert_injection_instances.py --experiment country
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTER = REPO.parent
DEFAULT_IN = {
    "country": OUTER / "outputs_recovered" / "country_injection_instances.jsonl",
    "temporal": OUTER / "outputs_recovered" / "temporal_context_instances.jsonl",
}
DEFAULT_OUT = {
    "country": REPO / "outputs" / "country_injection" / "inputs"
    / "country_injection_label_set.jsonl",
    "temporal": REPO / "outputs" / "temporal_context" / "inputs"
    / "temporal_context_label_set.jsonl",
}

CARRY = ("survey", "target_code", "id", "country", "target_question")
COUNTRY_QUESTION = "In which country do you live?"

# WVS codes missing from the country file's (survey, code) -> name map;
# ISO-3166 numeric, cross-checked against country_canonical_mapping.json.
WVS_NAME_FALLBACK = {
    "196": "Cyprus", "417": "Kyrgyzstan", "446": "Macau",
    "462": "Maldives", "788": "Tunisia",
}


def load_country_names(country_path: Path) -> dict[tuple[str, str], str]:
    """(survey, country code) -> country name, from the country file."""
    names: dict[tuple[str, str], str] = {}
    with open(country_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            names[(r["survey"], str(r["country"]))] = r["country_name"]
    for code, name in WVS_NAME_FALLBACK.items():
        names.setdefault(("wvs", code), name)
    return names


def conditions(experiment: str, src: dict) -> list[str]:
    if experiment == "country":
        return ["baseline", "with_country", "with_country_placebo"]
    conds = ["baseline", "with_year", "with_year_placebo"]
    if src.get("interview_date"):
        conds.append("with_date")
    # The combined cell: 2x2 factorial with baseline / with_year.
    if src.get("country_name"):
        conds += ["with_country", "with_country_and_year"]
    return conds


def convert_one(experiment: str, src: dict, cond: str) -> dict:
    """One runner instance; mirrors nebius_run_experiments.build_prompt.

    The country injection appends the country q:a pair as the profile's last
    entry; the temporal injection is an ``extra`` line between profile and
    question. with_country_and_year composes the two constructions unchanged.
    """
    questions = dict(src["questions"])
    extra = ""
    if cond in ("with_country", "with_country_and_year"):
        questions[src["country_question"]] = src["country_name"]
    elif cond == "with_country_placebo":
        questions[src["country_question"]] = src["country_placebo_name"]
    if cond in ("with_year", "with_country_and_year"):
        extra = f"The survey was conducted in {src['survey_year']}.\n\n"
    elif cond == "with_year_placebo":
        extra = f"The survey was conducted in {src['survey_year_placebo']}.\n\n"
    elif cond == "with_date":
        extra = f"The interview took place on {src['interview_date']}.\n\n"

    options = list(src["options"])
    inst = {
        "example_id": f"{src['example_id']}_{cond}",
        "base_id": src["example_id"],
        "condition": cond,
        **{k: src.get(k) for k in CARRY},
        "questions": questions,
        "option_sets": {"original": options},
        "ground_truth": src["answer"],
        "ground_truth_index": options.index(src["answer"]),
    }
    if extra:
        inst["extra"] = extra
    return inst


def convert_file(experiment: str, in_path: Path, out_path: Path,
                 country_names: dict | None = None) -> Counter:
    stats: Counter = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            src = json.loads(line)
            stats["source"] += 1
            if src["answer"] not in src["options"]:
                stats["skipped_answer_not_in_options"] += 1
                continue
            if experiment == "temporal" and country_names is not None:
                name = country_names.get((src["survey"], str(src["country"])))
                if name:
                    src["country_question"] = COUNTRY_QUESTION
                    src["country_name"] = name
                else:
                    stats["no_country_name"] += 1
            for cond in conditions(experiment, src):
                inst = convert_one(experiment, src, cond)
                fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                stats[cond] += 1
                stats["written"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", choices=("country", "temporal", "both"),
                    default="both")
    ap.add_argument("--in-country", type=Path, default=DEFAULT_IN["country"])
    ap.add_argument("--in-temporal", type=Path, default=DEFAULT_IN["temporal"])
    ap.add_argument("--out-country", type=Path, default=DEFAULT_OUT["country"])
    ap.add_argument("--out-temporal", type=Path,
                    default=DEFAULT_OUT["temporal"])
    args = ap.parse_args(argv)

    todo = ("country", "temporal") if args.experiment == "both" \
        else (args.experiment,)
    country_names = load_country_names(args.in_country)
    for exp in todo:
        in_path = getattr(args, f"in_{exp}")
        out_path = getattr(args, f"out_{exp}")
        stats = convert_file(exp, in_path, out_path,
                             country_names=country_names)
        conds = {k: v for k, v in stats.items()
                 if k not in ("source", "written", "no_country_name",
                              "skipped_answer_not_in_options")}
        print(f"{exp}: {stats['source']} source instances -> "
              f"{stats['written']} runner instances {dict(conds)} "
              f"skipped={stats['skipped_answer_not_in_options']} "
              f"no_country_name={stats['no_country_name']} "
              f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
