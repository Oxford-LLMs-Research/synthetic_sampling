#!/usr/bin/env python
"""Pre-flight for the profile-scaling experiment: how deep can a profile go?

READ-ONLY. Touches nothing the main experiment uses. It answers the one
question that decides whether k = 48 and k = 96 are legal at all.

RespondentProfileGenerator.generate_profile() enforces

    m_features_per_section <= min(len(f) for f in pool.values())

over ALL sections in the pool, not just the n_sections it goes on to sample.
So a single small section caps the depth of every profile. The main experiment
runs 6 sections x 4; this reports, per survey, how far that can be pushed.

It also reports the per-respondent picture, because the pool being large enough
does not mean a given respondent answered enough of it: generate_profile falls
back to a SHORT profile in that case, silently apart from a warning. That is
already happening at m = 4 (90.4% of rich profiles reach a full 24 features).

Usage:
    python synthetic_sampling/scripts/scaling/preflight_pool_check.py
    python synthetic_sampling/scripts/scaling/preflight_pool_check.py --survey wvs
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from synthetic_sampling.config import load_config                      # noqa: E402
from synthetic_sampling.config.surveys import ALL_SURVEYS, get_survey_config  # noqa: E402
from synthetic_sampling.profiles.generator import RespondentProfileGenerator  # noqa: E402

# Imported, never modified: the main generator script owns these definitions.
from generate_main_dataset import load_survey_data                     # noqa: E402

# The depths a 6-section profile would need for each target size.
TARGETS = {24: 4, 48: 8, 96: 16}

MISSING_VALUE_PATTERNS = [
    "missing", "refused", "no answer", "not asked",
    "not applicable", "decline", "can't", "do not understand",
    "not available", "no response", "nan", "na", "n/a",
]


def check(survey_id: str, raw_data_dir: Path, metadata_dir: Path) -> dict:
    survey_config = get_survey_config(survey_id)
    metadata, data = load_survey_data(survey_config, raw_data_dir, metadata_dir)

    gen = RespondentProfileGenerator(
        survey_data=data,
        metadata=metadata,
        respondent_id_col=survey_config.respondent_id_col,
        country_col=survey_config.country_col,
        survey=survey_id,
        missing_value_labels=[],
        missing_value_patterns=MISSING_VALUE_PATTERNS,
    )
    # No set_target_questions(): that triggers embedding of every item to build
    # the similarity exclusions, which costs minutes and only shrinks the pool.
    # This reports the ceiling; per-target pools are this or smaller.
    pool = gen.get_available_pool()
    sizes = {s: len(f) for s, f in sorted(pool.items(), key=lambda kv: len(kv[1]))}
    return {"survey": survey_id, "sizes": sizes,
            "n_sections": len(sizes), "min": min(sizes.values()),
            "total": sum(sizes.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=None)
    ap.add_argument("--config", default="configs/local.yaml")
    args = ap.parse_args()

    cfg = load_config(REPO.parent / args.config if not (REPO / args.config).exists()
                      else REPO / args.config)
    raw_data_dir = cfg["paths"].raw_data_dir
    metadata_dir = cfg["paths"].metadata_dir

    surveys = [args.survey] if args.survey else list(ALL_SURVEYS)
    rows = []
    for s in surveys:
        try:
            rows.append(check(s, raw_data_dir, metadata_dir))
        except Exception as exc:                                   # noqa: BLE001
            print(f"{s:<18} ERROR {type(exc).__name__}: {exc}")

    print()
    print("%-18s %8s %7s %7s | %s" % ("survey", "sections", "min", "total",
                                      "max feasible 6xM profile"))
    print("-" * 92)
    for r in rows:
        feasible = [k for k, m in TARGETS.items() if m <= r["min"]]
        print("%-18s %8d %7d %7d | %s"
              % (r["survey"], r["n_sections"], r["min"], r["total"],
                 ", ".join(str(k) for k in feasible) or "NONE (even 24 fails)"))

    print()
    print("smallest sections per survey (these set the ceiling):")
    for r in rows:
        head = list(r["sizes"].items())[:3]
        print("  %-18s %s" % (r["survey"],
                              ", ".join(f"{s}={n}" for s, n in head)))


if __name__ == "__main__":
    main()
