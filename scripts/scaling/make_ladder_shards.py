#!/usr/bin/env python
r"""Build the by-feature ladder: three orderings of the same features, level by level.

For each (respondent, target) the model answers the same question many times,
seeing 0, 1, 2, 4, 8, 16, 32, 64, 96 features and finally everything the
respondent answered. The features are identical across arms; only the ORDER in
which they arrive differs:

    informative   the oracle's most predictive first
    random        a seeded permutation, the paper's own design
    anti          the same ranking read from the bottom

Two levels are shared rather than repeated. k=0 is the same prompt in every arm,
so it is emitted once; it is the template with an empty profile, which is the
control for what the profile CONTENT adds rather than the model's unconditional
prior, since the instruction still refers to prior answers. The final level uses
every usable feature, so all three arms coincide there by construction, which
gives the analysis its own measurement-noise floor at no extra cost.

Three things this gets right that earlier attempts did not.

Profiles come from the project's own RespondentProfileGenerator, configured
exactly as generate_scaling_dataset.py configures it, including
MISSING_VALUE_PATTERNS. Verified against the shipped instances: the option sets
it produces match the paper's for every Latinobarometro target. Building option
lists from raw metadata instead would not: metadata labels reproduce the paper's
options on 1 of 40 WVS targets and 1 of 40 Asian Barometer targets, because the
pipeline harmonises response categories and drops non-substantive codes.

Respondents come only from the oracle's reserved split. The informative arm is
ordered by a ranking fitted on other people, so it is never scored on the
respondents that chose it.

The usable feature set is discovered, not assumed. A respondent has no valid
value for some ranked features, and set_always_include silently skips those, so
asking for k would otherwise yield fewer than k. The all-features level is
generated first and its features define the universe; every shorter level is a
prefix of that, so level sizes are exact and the arms are nested by construction.

    python .../make_ladder_shards.py --per-target 100 --shards 4
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "scaling"))
from synthetic_sampling.config import DataPaths                        # noqa: E402
from synthetic_sampling.config.surveys import get_survey_config        # noqa: E402
from synthetic_sampling.loaders.survey_loader import SurveyLoader      # noqa: E402
from synthetic_sampling.profiles.generator import (                    # noqa: E402
    RespondentProfileGenerator)
from generate_scaling_dataset import (                                 # noqa: E402
    MISSING_VALUE_PATTERNS, convert_to_native_types)

SCALE = REPO / "outputs" / "scaling_experiment"
IMP = REPO.parent / "analysis" / "feature_importance"
OUT = SCALE / "ladder_shards"
LEVELS = [0, 1, 2, 4, 8, 16, 32, 64, 96]
ARMS = ("informative", "random", "anti")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target", type=int, default=100)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=Path, default=SCALE / "ladder_targets.csv")
    args = ap.parse_args()

    # generate_scaling_dataset installs its own UTF-8 stdout wrapper at import.
    # Wrapping again at module level closed the underlying buffer when the first
    # wrapper was collected, so only re-wrap if the import did not.
    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")

    targets = pd.read_csv(args.targets)
    imp = pd.read_csv(IMP / "feature_importance.csv", dtype={"country": str})
    res = pd.read_csv(IMP / "reserved_respondents.csv", dtype=str)
    print(f"{len(targets)} targets | {imp.groupby(['survey','target_code','country']).ngroups}"
          f" ranked cells | {len(res):,} reserved records", flush=True)

    # Every AAAI target per survey, for the leakage exclusion above.
    q = pd.read_csv(REPO.parent / "analysis" / "normalized_accuracy" /
                    "per_question_norm_acc.csv")
    all_targets = {s: set(g.target_code) for s, g in q.groupby("survey")}

    paths = DataPaths.from_yaml(str(REPO / "configs" / "local.yaml"))
    loader = SurveyLoader(paths, verbose=False)
    rng = random.Random(args.seed)
    out: list[dict] = []
    stats: collections.Counter = collections.Counter()

    for survey, grp in targets.groupby("survey"):
        cfg = get_survey_config(survey)
        data, meta = loader.load_survey(survey)
        code2text = {c: v.get("question", "")
                     for sec in meta.values() if isinstance(sec, dict)
                     for c, v in sec.items() if isinstance(v, dict)}
        text2code = {t: c for c, t in code2text.items() if t}
        codes = sorted(grp.target_code)
        gen = RespondentProfileGenerator(
            survey_data=data, metadata=meta,
            respondent_id_col=cfg.respondent_id_col, country_col=cfg.country_col,
            survey=survey, missing_value_labels=[],
            missing_value_patterns=MISSING_VALUE_PATTERNS, similarity_model=None)
        # Register every AAAI target for this survey, not just the ladder's.
        # The paper's leakage prevention keeps any target question out of every
        # profile, and a ladder profile that could contain one would not be
        # comparable. The oracle ranked only the current target out, so its
        # rankings do list other targets; those are filtered below.
        gen.set_target_questions(sorted(all_targets.get(survey, set()) | set(codes)))

        # The reserved list stores str(id); the generator indexes by the raw
        # column value, numeric for some surveys and composite for others, so
        # a string lookup raises KeyError. Map one to the other rather than
        # guessing the round trip (generate_scaling_dataset.py hits this too).
        id_col = cfg.respondent_id_col
        raw_ids = (data[id_col] if id_col in data.columns
                   else data.index.to_series())
        id_lookup: dict[str, object] = {}
        for raw in raw_ids:
            native = convert_to_native_types(raw)
            id_lookup.setdefault(str(native), native)

        for target in codes:
            cells = imp[(imp.survey == survey) & (imp.target_code == target)]
            if cells.empty:
                stats["target: no ranking"] += 1
                continue
            countries = sorted(cells.country.unique())
            quota = max(1, args.per_target // len(countries))
            n_target = 0
            for country in countries:
                order = (cells[cells.country == country].sort_values("rank")
                         .feature.tolist())
                # Drop anything the generator will not place: features absent
                # from the metadata, and target questions excluded by the
                # leakage rule. Asking for an excluded feature raises.
                order = [c for c in order if c in gen._all_features
                         and c not in gen._excluded_features]
                pool = sorted(res[(res.survey == survey) &
                                  (res.target_code == target) &
                                  (res.country == country)].respondent_id.unique())
                if not order or not pool:
                    stats["cell: empty order or pool"] += 1
                    continue
                unresolved = [r for r in pool if r not in id_lookup]
                if unresolved:
                    stats["respondent id unresolvable"] += len(unresolved)
                    pool = [r for r in pool if r in id_lookup]
                if not pool:
                    stats["cell: no resolvable respondents"] += 1
                    continue
                if len(pool) > quota:
                    pool = rng.sample(pool, quota)

                for rid in pool:
                    # Discover the usable set: generating with the full ranking
                    # keeps only the features this respondent actually answered,
                    # in rank order. Every shorter level is then a true prefix.
                    gen.set_always_include(order)
                    prof = gen.generate_profile(
                        respondent_id=id_lookup[rid], n_sections=0,
                        m_features_per_section=0, seed=args.seed,
                        target_code=target)
                    usable = list(prof.features) if prof is not None else []
                    # PredictionInstance re-keys features by question TEXT, so
                    # two codes with identical wording collapse into one entry
                    # and a k-feature request yields fewer than k. Country
                    # variants cluster at the bottom of the ranking, which is
                    # why the anti arm suffered most. Dropping the duplicates
                    # also stops the same question appearing twice in a prompt.
                    seen_text, dedup = set(), []
                    for c in usable:
                        t = code2text.get(c, c)
                        if t in seen_text:
                            stats["feature dropped: duplicate wording"] += 1
                            continue
                        seen_text.add(t)
                        dedup.append(c)
                    usable = dedup
                    if len(usable) < 8:
                        stats["respondent: fewer than 8 usable features"] += 1
                        continue
                    gen.set_always_include(usable)
                    full = gen.generate_prediction_instance(
                        respondent_id=id_lookup[rid], target_code=target,
                        n_sections=0, m_features_per_section=0, seed=args.seed)
                    if full is None or not full.features:
                        stats["respondent: no instance"] += 1
                        continue

                    base = {"survey": survey, "id": str(rid), "country": country,
                            "target_code": target,
                            "target_question": full.target_question,
                            "options": list(full.options), "answer": full.answer}
                    stem = f"{survey}_{rid}_{target}_ladder"

                    def emit(arm, k, features):
                        out.append({**base, "questions": features,
                                    "profile_type": f"ladder_{arm}_k{k:03d}",
                                    "n_features": len(features), "arm": arm,
                                    "example_id": f"{stem}_{arm}_k{k:03d}"})
                        stats[f"{arm} k={k:03d}"] += 1

                    emit("all", len(usable), dict(full.features))   # arms coincide
                    gen.set_always_include([])
                    zero = gen.generate_prediction_instance(
                        respondent_id=id_lookup[rid], target_code=target, n_sections=0,
                        m_features_per_section=0, seed=args.seed)
                    if zero is not None:
                        emit("all", 0, dict(zero.features))

                    orders = {"informative": usable,
                              "anti": list(reversed(usable))}
                    shuf = list(usable)
                    random.Random(f"{stem}|{args.seed}").shuffle(shuf)
                    orders["random"] = shuf

                    for arm, seq in orders.items():
                        for k in LEVELS:
                            if k == 0 or k >= len(usable):
                                continue
                            gen.set_always_include(seq[:k])
                            inst = gen.generate_prediction_instance(
                                respondent_id=id_lookup[rid], target_code=target,
                                n_sections=0, m_features_per_section=0,
                                seed=args.seed)
                            if inst is None or len(inst.features) != k:
                                stats[f"level mismatch {arm} k={k}"] += 1
                                continue
                            emit(arm, k, dict(inst.features))
                    n_target += 1
            stats["pairs"] += n_target
            print(f"  {survey:<16}{target:<12}{n_target:>4} respondents", flush=True)

    if not out:
        raise SystemExit("nothing generated")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("ladder_shard_*.jsonl"):
        old.unlink()
    out.sort(key=lambda r: (r["example_id"], r["n_features"]))
    handles = [open(OUT / f"ladder_shard_{i:02d}.jsonl", "w", encoding="utf-8")
               for i in range(args.shards)]
    try:
        for i, r in enumerate(out):
            handles[i % args.shards].write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for h in handles:
            h.close()

    pairs = len({r["example_id"].split("_ladder_")[0] for r in out})
    opts = sum(len(set(r["options"])) for r in out) / len(out)
    print(f"\n{len(out):,} prompts over {pairs:,} respondent-target pairs")
    print(f"  ~{len(out) * opts:,.0f} scoring requests at {opts:.1f} options per prompt")
    for k, v in sorted(stats.items()):
        if not k.startswith(("informative", "random", "anti", "all")):
            print(f"  {k}: {v:,}")
    for i in range(args.shards):
        p = OUT / f"ladder_shard_{i:02d}.jsonl"
        print(f"  {p.name}  {sum(1 for _ in open(p, encoding='utf-8')):>7,} prompts"
              f"  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
