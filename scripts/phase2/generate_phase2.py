"""Second main run: draw respondents and build the instance files.

One profile depth (36 features), run-1's question list (old_targets.csv, all
267 kept), a nested per-country draw. Profile construction and its run-1
settings live in synthetic_sampling.profiles.phase2; the draw in
synthetic_sampling.surveys.draw.

    python scripts/phase2/generate_phase2.py --all --floor 50
    python scripts/phase2/generate_phase2.py --all --floor 100 --start 50   # extension tranche
    python scripts/phase2/generate_phase2.py --survey wvs --floor 2          # smoke

Outputs (outputs/phase2/inputs/, regenerable: everything is seeded):
    draw_<survey>.csv                       every ranked respondent up to --floor
    <survey>_instances_r<start>-<floor>.jsonl
    coverage_<survey>_r<start>-<floor>.json  counts the coverage report reads
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from synthetic_sampling.profiles import phase2                          # noqa: E402
from synthetic_sampling.profiles.generator import (                     # noqa: E402
    RespondentProfileGenerator)
from synthetic_sampling.profiles.targets import detect_response_format  # noqa: E402
from synthetic_sampling.surveys import DataPaths                        # noqa: E402
from synthetic_sampling.surveys.draw import nested_cell_ranks, take_floor  # noqa: E402
from synthetic_sampling.surveys.loaders import SurveyLoader             # noqa: E402
from synthetic_sampling.surveys.registry import get_survey_config       # noqa: E402

TARGETS_CSV = Path(__file__).with_name("old_targets.csv")
OUT_DIR = REPO / "outputs" / "phase2" / "inputs"
SURVEYS = ["wvs", "afrobarometer", "arabbarometer", "asianbarometer",
           "latinobarometer", "ess_wave_10", "ess_wave_11"]


def native(obj):
    """JSON-safe copy (numpy scalars, NaN) of a value."""
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def read_targets(survey_id: str) -> list[str]:
    with open(TARGETS_CSV, encoding="utf-8") as fh:
        return [r["target_code"] for r in csv.DictReader(fh)
                if r["survey"] == survey_id]


def process_survey(survey_id: str, loader: SurveyLoader, floor: int,
                   start: int, out_dir: Path) -> dict:
    cfg = get_survey_config(survey_id)
    df, metadata = loader.load_survey(survey_id)
    var_meta = {v: m for block in metadata.values() if isinstance(block, dict)
                for v, m in block.items() if isinstance(m, dict)}

    wanted = read_targets(survey_id)
    targets = [t for t in wanted if t in var_meta and t in df.columns]
    absent = sorted(set(wanted) - set(targets))
    print(f"  {len(targets)}/{len(wanted)} targets present"
          + (f"  ABSENT: {absent}" if absent else ""))

    ranks = nested_cell_ranks(df[cfg.respondent_id_col], df[cfg.country_col],
                              survey_id, phase2.BASE_SEED)
    country_labels = (var_meta.get(cfg.country_col, {}).get("values") or {})
    ranks["country_label"] = [
        country_labels.get(native(c) if not isinstance(c, str) else c)
        or country_labels.get(str(c).split(".")[0]) or str(c)
        for c in ranks["country"]]
    tranche = take_floor(ranks, floor, start)
    out_dir.mkdir(parents=True, exist_ok=True)
    take_floor(ranks, floor).to_csv(out_dir / f"draw_{survey_id}.csv",
                                    index=False, lineterminator="\n")
    cells = tranche.groupby("country").size()
    print(f"  drew {len(tranche)} respondents in {len(cells)} cells "
          f"(ranks {start}-{floor - 1}; smallest cell {cells.min()})")

    gen = RespondentProfileGenerator(
        survey_data=df, metadata=metadata,
        respondent_id_col=cfg.respondent_id_col,
        country_col=cfg.country_col, survey=survey_id,
        missing_value_labels=phase2.MISSING_VALUE_LABELS,
        missing_value_patterns=phase2.MISSING_VALUE_PATTERNS,
        similarity_model=phase2.SIMILARITY_MODEL,
        similarity_threshold=phase2.SIMILARITY_THRESHOLD)
    # On the FULL data: country-specific option sets are read off every
    # respondent in a country, not off the drawn ones.
    gen.set_target_questions(targets)
    # Respondent lookup scans the frame; keep only the drawn rows from here.
    drawn_ids = set(tranche["respondent_id"])
    gen.survey_data = df[df[cfg.respondent_id_col].isin(drawn_ids)]

    tag = phase2.profile_tag()
    span = f"r{start}-{floor}"
    out_path = out_dir / f"{survey_id}_instances_{span}.jsonl"
    stats = Counter()
    n_features = Counter()
    per_target = Counter()
    per_cell = defaultdict(set)
    label_of = dict(zip(tranche["respondent_id"], tranche["country_label"]))
    rank_of = dict(zip(tranche["respondent_id"], tranche["rank"]))

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh, \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n, resp_id in enumerate(tranche["respondent_id"]):
            for target_code in targets:
                stats["pairs"] += 1
                try:
                    profile = phase2.build_profile(gen, resp_id, target_code)
                except (ValueError, KeyError) as exc:
                    stats["core_failed"] += 1
                    if stats["core_failed"] == 1:
                        print(f"    first core failure: "
                              f"{type(exc).__name__}: {exc}")
                    continue
                inst = gen.generate_prediction_instance_from_profile(
                    profile=profile, target_code=target_code,
                    skip_missing_targets=True)
                if inst is None:
                    stats["target_missing"] += 1
                    continue
                options = native(inst.options)
                if inst.answer not in options:
                    stats["answer_not_in_options"] += 1
                    continue
                if len(options) < 2:
                    stats["single_option"] += 1
                    continue
                rid = str(native(resp_id))
                meta = var_meta[target_code]
                rec = {
                    "example_id": f"{survey_id}_{rid}_{target_code}_{tag}",
                    "base_id": f"{survey_id}_{rid}_{target_code}",
                    "survey": survey_id,
                    "id": str(inst.id),
                    "country": (str(inst.country)
                                if inst.country is not None else None),
                    "country_label": label_of[resp_id],
                    "draw_rank": int(rank_of[resp_id]),
                    "questions": inst.features,
                    "target_question": inst.target_question,
                    "target_code": target_code,
                    "target_section": inst.target_section,
                    "target_topic_tag": meta.get("topic_tag"),
                    "target_response_format": detect_response_format(
                        meta.get("values") or {}),
                    "answer": inst.answer,
                    "options": options,
                    "option_sets": {"original": options},
                    "ground_truth": inst.answer,
                    "ground_truth_index": options.index(inst.answer),
                    "profile_type": tag,
                    "n_features": profile.n_features,
                }
                fh.write(json.dumps(native(rec), ensure_ascii=False) + "\n")
                stats["instances"] += 1
                n_features[profile.n_features] += 1
                per_target[target_code] += 1
                per_cell[label_of[resp_id]].add(rid)
            if (n + 1) % 500 == 0:
                print(f"    {n + 1}/{len(tranche)} respondents, "
                      f"{stats['instances']} instances")

    short = sum(v for k, v in n_features.items() if k < phase2.N_FEATURES)
    cell_sizes = {c: len(v) for c, v in per_cell.items()}
    report = {
        "survey": survey_id, "floor": floor, "start": start,
        "targets_wanted": len(wanted), "targets_used": len(targets),
        "targets_absent": absent,
        "respondents_drawn": int(len(tranche)), "cells": int(len(cells)),
        **stats,
        "short_profiles": short,
        "n_features_hist": {str(k): v for k, v in sorted(n_features.items())},
        "instances_per_target_min": min(per_target.values(), default=0),
        "targets_with_no_instances": sorted(set(targets) - set(per_target)),
        "respondents_with_instances_per_cell_min":
            min(cell_sizes.values(), default=0),
        "instances_per_target": dict(sorted(per_target.items())),
    }
    (out_dir / f"coverage_{survey_id}_{span}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8",
        newline="\n")
    print(f"  wrote {out_path.name}: {stats['instances']} instances from "
          f"{stats['pairs']} pairs (target missing {stats['target_missing']}, "
          f"core failed {stats['core_failed']}, short of "
          f"{phase2.N_FEATURES}: {short})")
    return report


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--survey", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--floor", type=int, default=50,
                    help="respondents per source-country cell (rank < floor)")
    ap.add_argument("--start", type=int, default=0,
                    help="first rank to generate (extension tranche)")
    ap.add_argument("--raw-data-dir", type=Path, default=WORK / "data")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if not args.all and not args.survey:
        ap.error("--survey or --all")

    loader = SurveyLoader(
        DataPaths.default_bundled(args.raw_data_dir, "./outputs"),
        verbose=False)
    for survey_id in (SURVEYS if args.all else args.survey):
        print(f"\n=== {survey_id} ===")
        process_survey(survey_id, loader, args.floor, args.start, args.out_dir)
        loader.clear_cache(survey_id)


if __name__ == "__main__":
    main()
