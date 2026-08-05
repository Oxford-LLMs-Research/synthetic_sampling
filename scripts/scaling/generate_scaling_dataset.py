#!/usr/bin/env python
"""Profile-scaling experiment: build k = 48 and k = 96 profiles.

Tests whether the k^0.28 information-scaling curve fitted on 6/12/24 features
continues or bends. Nothing the main experiment reads or writes is modified:
this imports from generate_main_dataset.py and generator.py, and writes to its
own output directory.

WHY NOT expand_profile(add_features_per_section=N)
--------------------------------------------------
The main experiment allocates an equal quota m to each of 6 sections, and
generate_profile enforces m <= the SMALLEST section in the pool, across all
eight sections. preflight_pool_check.py measured that ceiling:

    Arab Barometer  min section 16  -> 6x16 = 96 legal
    WVS             min section 13  -> 6x8  = 48 legal
    ESS 11          min section  9  -> 6x8  = 48 legal
    ESS 10          min section  6  -> only 24
    Afrobarometer   min section  5  -> only 24
    Asian Barometer min section  5  -> only 24
    Latinobarometro min section  4  -> only 24, and it is exactly at the limit

So equal quota reaches 48 on three surveys and 96 on one. The pools themselves
are not the problem: they hold 238-600 features. One small section caps every
profile.

WHAT THIS DOES INSTEAD
----------------------
Keeps the main experiment's 24-feature profile as an untouched core, then tops
up round-robin across that profile's own six sections, one feature at a time,
skipping sections as they are exhausted. This is the most even allocation the
instrument permits, and it degrades gracefully instead of raising.

Guarantees:
  - the 24-feature core is bit-identical to the main experiment's rich profile
    (verified against the shipped instances by --verify),
  - 96 superset 48 superset 24, by construction,
  - per-target semantic-similarity exclusions apply to every added feature,
    so leakage filtering is not weakened at larger k,
  - deterministic given (base_seed, respondent_id, target_code).

Respondents and targets are read from the existing instance files rather than
re-sampled, so the two experiments cannot drift apart.

Usage:
    python .../generate_scaling_dataset.py --survey wvs --verify --limit 20
    python .../generate_scaling_dataset.py --all
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from synthetic_sampling.config import load_config                          # noqa: E402
from synthetic_sampling.config.surveys import (                            # noqa: E402
    ALL_SURVEYS, ESS_SURVEYS, get_survey_config,
)
from synthetic_sampling.profiles.generator import RespondentProfileGenerator  # noqa: E402

# Imported, never modified.
from generate_main_dataset import (                                        # noqa: E402
    RICHNESS_LEVELS, convert_to_native_types, get_respondent_target_seed,
    load_survey_data,
)

MAIN_INSTANCES = (REPO / "outputs" / "main_data_smaller_20_jan_26" / "main_data")
OUT_DIR = REPO / "outputs" / "scaling_experiment"

SCALING_LEVELS = [48, 96]
BASE_SEED = 42
CORE_K = 24
CORE_TAG = "s6m4"

MISSING_VALUE_PATTERNS = [
    "missing", "refused", "no answer", "not asked",
    "not applicable", "decline", "can't", "do not understand",
    "not available", "no response", "nan", "na", "n/a",
]


# ---------------------------------------------------------------------------
# Reading the main experiment's respondent x target grid
# ---------------------------------------------------------------------------

def read_main_grid(survey_id: str) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """(respondent_id, target_code) pairs and the shipped rich profiles.

    The rich profiles come back so the core can be checked against them rather
    than assumed to reproduce.
    """
    path = MAIN_INSTANCES / f"{survey_id}_instances.jsonl"
    pairs, rich = [], {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            d = json.loads(line)
            if d["profile_type"] != CORE_TAG:
                continue
            key = (d["id"], d["target_code"])
            pairs.append(key)
            rich[f"{d['id']}|{d['target_code']}"] = d
    return pairs, rich


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------

def build_core(gen, resp_id, target_code, seed):
    """Reproduce the main experiment's rich profile: sparse -> medium -> rich.

    The main script never generates 6x4 directly; it expands through the three
    configured levels, and the expansion path determines which features land in
    the profile. Generating 6x4 in one call would give a different, equally
    valid profile that is not the one the published results were computed on.
    """
    profile = None
    for level in RICHNESS_LEVELS.values():
        n_sec, m_feat = level["n_sections"], level["m_features"]
        if profile is None:
            profile = gen.generate_profile(
                respondent_id=resp_id, n_sections=n_sec,
                m_features_per_section=m_feat, seed=seed,
                shuffle_features=False, target_code=target_code)
        else:
            profile = gen.expand_profile(
                profile=profile,
                add_sections=n_sec - profile.config.n_sections,
                add_features_per_section=m_feat - profile.config.m_features_per_section,
                target_code=target_code)
    return profile


def top_up(gen, profile, target_code, k, seed):
    """Extend a profile to k features, as evenly across its sections as possible.

    Round-robin one feature at a time over the profile's own sampled sections,
    dropping a section once it has nothing left the respondent answered. Returns
    (profile, reached_k). Never raises: a respondent who cannot reach k comes
    back short and is filtered out downstream.
    """
    pool = gen.get_available_pool_for_target(target_code)
    resp = gen._get_respondent_data(profile.respondent_id)
    chosen = set(profile.feature_codes)

    rng = np.random.RandomState(seed % (2 ** 31))
    queues = {}
    for section in profile.sections_sampled:
        cand = [c for c in pool.get(section, []) if c not in chosen]
        rng.shuffle(cand)
        queues[section] = cand

    features = dict(profile.features)
    order = list(profile.sections_sampled)
    while len(features) < k:
        progressed = False
        for section in order:
            if len(features) >= k:
                break
            q = queues.get(section)
            while q:
                code = q.pop()
                if code in chosen:
                    continue
                if gen._respondent_has_valid_value(code, resp):
                    features[code] = gen._build_feature_info(code, resp)
                    chosen.add(code)
                    progressed = True
                    break
        if not progressed:
            break

    out = deepcopy(profile)
    out.features = features
    return out, len(features) == k


# ---------------------------------------------------------------------------
# Per-survey driver
# ---------------------------------------------------------------------------

def process_survey(survey_id, raw_data_dir, metadata_dir, verify, limit,
                   similarity_model="all-MiniLM-L6-v2", similarity_threshold=0.85):
    survey_config = get_survey_config(survey_id)
    metadata, data = load_survey_data(survey_config, raw_data_dir, metadata_dir)
    pairs, rich = read_main_grid(survey_id)
    target_codes = sorted({t for _, t in pairs})
    print(f"  {len(pairs)} respondent x target pairs, {len(target_codes)} targets")

    gen = RespondentProfileGenerator(
        survey_data=data, metadata=metadata,
        respondent_id_col=survey_config.respondent_id_col,
        country_col=survey_config.country_col, survey=survey_id,
        missing_value_labels=[], missing_value_patterns=MISSING_VALUE_PATTERNS,
        similarity_model=similarity_model,
        similarity_threshold=similarity_threshold)
    print("  computing semantic exclusions ...")
    gen.set_target_questions(target_codes)

    # The instance files store str(instance.id); the generator indexes by the
    # raw column value, which is numeric for some surveys and composite for
    # others. Map one to the other rather than guessing the round trip.
    id_col = survey_config.respondent_id_col
    raw_ids = (data[id_col] if id_col in data.columns else data.index.to_series())
    id_lookup = {}
    for raw in raw_ids:
        native = convert_to_native_types(raw)
        id_lookup.setdefault(str(native), native)
    missing = {r for r, _ in pairs if r not in id_lookup}
    if missing:
        print(f"    WARNING: {len(missing)} respondent ids not resolvable, "
              f"e.g. {sorted(missing)[:3]}")
    pairs = [(r, t) for r, t in pairs if r in id_lookup]

    if limit:
        # Deterministic respondent subsample, all of a respondent's targets kept
        # together so the per-question means stay balanced.
        respondents = sorted({r for r, _ in pairs})
        rng = np.random.RandomState(BASE_SEED)
        rng.shuffle(respondents)
        keep = set(respondents[:limit])
        pairs = [p for p in pairs if p[0] in keep]
        print(f"  subsampled to {len(keep)} respondents, {len(pairs)} pairs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{survey_id}_scaling_instances.jsonl"
    stats = defaultdict(int)
    core_mismatch = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for n, (resp_id_str, target_code) in enumerate(pairs):
            resp_id = id_lookup[resp_id_str]
            # The main script seeds from the native id, before stringification.
            seed = get_respondent_target_seed(BASE_SEED, resp_id, target_code)
            try:
                core = build_core(gen, resp_id, target_code, seed)
            except (ValueError, KeyError) as exc:
                stats["core_failed"] += 1
                if stats["core_failed"] == 1:
                    print(f"    first core failure: {type(exc).__name__}: {exc}")
                continue

            if verify:
                stats["core_checked"] += 1
                want = rich[f"{resp_id_str}|{target_code}"]["questions"]
                got = {i.get("question", i.get("description", c)): i["value_label"]
                       for c, i in core.features.items()}
                if got != want:
                    core_mismatch += 1

            profile = core
            for k in SCALING_LEVELS:
                profile, full = top_up(gen, profile, target_code, k, seed + k)
                stats[f"k{k}_full" if full else f"k{k}_short"] += 1
                if not full:
                    continue
                inst = gen.generate_prediction_instance_from_profile(
                    profile=profile, target_code=target_code,
                    skip_missing_targets=True)
                if inst is None:
                    continue
                tag = f"s6m4x{k}"
                rec = {
                    "example_id": f"{survey_id}_{resp_id_str}_{target_code}_{tag}",
                    "base_id": f"{survey_id}_{resp_id_str}_{target_code}",
                    "survey": survey_id, "id": str(inst.id),
                    "country": str(inst.country) if inst.country is not None else None,
                    "questions": inst.features,
                    "target_question": inst.target_question,
                    "target_code": target_code,
                    "target_section": inst.target_section,
                    "answer": inst.answer,
                    "options": convert_to_native_types(inst.options),
                    "profile_type": tag,
                    "n_features": len(profile.features),
                }
                fh.write(json.dumps(convert_to_native_types(rec),
                                    ensure_ascii=False) + "\n")
            if (n + 1) % 2000 == 0:
                print(f"    {n + 1}/{len(pairs)} pairs")

    print(f"  wrote {out_path}")
    print(f"    pairs {len(pairs)}, core built {len(pairs) - stats['core_failed']}, "
          f"core failed {stats['core_failed']}")
    for k in SCALING_LEVELS:
        full, short = stats[f"k{k}_full"], stats[f"k{k}_short"]
        tot = full + short
        print(f"    k={k:<3} full {full} / {tot} ({100 * full / max(tot, 1):.1f}%)")
    if verify:
        checked = stats["core_checked"]
        print(f"    core profiles matching the shipped rich profile: "
              f"{checked - core_mismatch}/{checked}"
              f"{'  <-- MISMATCH' if core_mismatch else '  OK'}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check the 24-feature core against the shipped instances")
    ap.add_argument("--limit", type=int, default=None,
                    help="respondents per survey, for a smoke test")
    ap.add_argument("--config", default="configs/local.yaml")
    args = ap.parse_args()

    cfg_path = REPO / args.config
    if not cfg_path.exists():
        cfg_path = REPO.parent / args.config
    cfg = load_config(cfg_path)
    raw_data_dir, metadata_dir = cfg["paths"].raw_data_dir, cfg["paths"].metadata_dir

    surveys = list(ALL_SURVEYS) if args.all else [args.survey]
    for s in surveys:
        print(f"\n=== {s} ===")
        process_survey(s, raw_data_dir, metadata_dir, args.verify, args.limit)


if __name__ == "__main__":
    main()
