#!/usr/bin/env python
r"""Build the ladder-x-elicitation input: the ladder's own feature sequences,
new rungs, scored under multiple readouts.

The paraphrase and PMI results scoped the paper's central negative claim to
RANDOMLY drawn profile features: whether the model's ceiling rises under
INFORMATIVE features plus the label readout is untested, and this experiment
tests it. Rungs k = 0/1/8/24/48/96 plus the full-feature endpoint, informative
and random orderings, one instance file consumable by score_formats.py
unchanged.

Everything is reconstructed from the existing ladder shards so the feature
SEQUENCES are identical to job 8396440's:

  informative order   the "all" endpoint record's questions dict, whose
                      insertion order is the oracle ranking (every shorter
                      level in the shards is a prefix of it by construction)
  random order        reproduced with the shards' own seed recipe,
                      random.Random(f"{stem}|42").shuffle(copy_of_usable),
                      and ASSERTED against every stored random-arm record's
                      prefix before anything is written
  new rungs 24, 48    truncations of those same sequences, which the original
                      LEVELS (powers of two) skipped; 24 matters because it is
                      s6m4, the operating point of every headline number

The anti ordering is deliberately absent (the deliberation kept two orderings);
the full-set endpoint is emitted once per pair in informative order, as in the
original design where the arms coincide there.

    python make_ladder_readout_set.py --pairs 1500
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
SHARDS = SCALE / "ladder_shards"
OUT = SCALE / "ladder_readout_set.jsonl"

RUNGS = [1, 8, 24, 48, 96]
ORDERINGS = ("informative", "random")
SEED = 42                     # make_ladder_shards.py's default, part of the recipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500,
                    help="respondent-target pairs to keep, stratified by target")
    ap.add_argument("--seed", type=int, default=20260806,
                    help="subsampling seed (NOT the shards' ordering seed)")
    args = ap.parse_args()

    # ---- gather the shards, grouped by pair -------------------------------
    by_pair: dict[str, dict] = {}
    n_rows = 0
    for shard in sorted(SHARDS.glob("ladder_shard_*.jsonl")):
        for line in open(shard, encoding="utf-8"):
            r = json.loads(line)
            n_rows += 1
            stem = r["example_id"].split("_ladder_")[0] + "_ladder"
            p = by_pair.setdefault(stem, {"base": None, "full": None,
                                          "k0": None, "random_records": []})
            if p["base"] is None:
                p["base"] = {k: r[k] for k in ("survey", "id", "country",
                                               "target_code", "target_question",
                                               "options", "answer")}
            if r["arm"] == "all" and r["n_features"] > 0:
                p["full"] = r
            elif r["arm"] == "all" and r["n_features"] == 0:
                p["k0"] = r
            elif r["arm"] == "random":
                p["random_records"].append(r)
    print(f"{n_rows:,} shard rows, {len(by_pair):,} pairs")

    # ---- reconstruct and VERIFY the orderings -----------------------------
    pairs = {}
    dropped = collections.Counter()
    for stem, p in by_pair.items():
        if p["full"] is None or p["k0"] is None:
            dropped["missing endpoint or k0"] += 1
            continue
        usable = list(p["full"]["questions"])          # informative order
        answers = p["full"]["questions"]               # text -> answer, complete
        shuf = list(usable)
        random.Random(f"{stem}|{SEED}").shuffle(shuf)
        ok = True
        for rec in p["random_records"]:
            k = rec["n_features"]
            if list(rec["questions"]) != shuf[:k]:
                ok = False
                break
        if not ok:
            dropped["random order irreproducible"] += 1
            continue
        pairs[stem] = {"base": p["base"], "usable": usable,
                       "answers": answers, "random": shuf,
                       "n_random_checked": len(p["random_records"])}
    print(f"{len(pairs):,} pairs with verified orderings "
          f"({sum(v['n_random_checked'] for v in pairs.values()):,} stored "
          f"random records matched); dropped: {dict(dropped) or 'none'}")
    if dropped.get("random order irreproducible"):
        sys.exit("the seed recipe failed to reproduce stored orderings; "
                 "do not proceed")

    # ---- stratified subsample by target -----------------------------------
    rng = random.Random(args.seed)
    by_target = collections.defaultdict(list)
    for stem, v in pairs.items():
        by_target[(v["base"]["survey"], v["base"]["target_code"])].append(stem)
    frac = min(1.0, args.pairs / len(pairs))
    keep = []
    for tgt in sorted(by_target):
        stems = sorted(by_target[tgt])
        n = max(1, round(frac * len(stems)))
        keep.extend(rng.sample(stems, min(n, len(stems))))
    print(f"kept {len(keep):,} pairs across {len(by_target)} targets "
          f"(requested ~{args.pairs})")

    # ---- emit -------------------------------------------------------------
    n_inst = 0
    n_req_echo = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for stem in sorted(keep):
            v = pairs[stem]
            base = v["base"]
            opts = base["options"]
            gt_idx = opts.index(base["answer"]) if base["answer"] in opts else None

            def emit(arm: str, k: int, feature_seq: list[str]) -> None:
                nonlocal n_inst, n_req_echo
                rec = {"example_id": f"{stem}_{arm}_k{k:03d}",
                       "survey": base["survey"], "id": base["id"],
                       "country": base["country"],
                       "target_code": base["target_code"],
                       "target_question": base["target_question"],
                       "questions": {q: v["answers"][q] for q in feature_seq},
                       "option_sets": {"original": list(opts)},
                       "ground_truth": base["answer"],
                       "ground_truth_index": gt_idx,
                       "arm": arm, "n_features": k}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_inst += 1
                n_req_echo += len(opts)

            emit("shared", 0, [])
            n_usable = len(v["usable"])
            for arm in ORDERINGS:
                seq = v["usable"] if arm == "informative" else v["random"]
                for k in RUNGS:
                    if k < n_usable:
                        emit(arm, k, seq[:k])
            emit("shared", n_usable, v["usable"])      # full endpoint, once

    # Cost: echo_plain + echo_listed + label_num each cost sum-M requests;
    # ctxfree is cached per option set; replicate re-runs 25% of the original.
    total = 3 * n_req_echo + int(0.25 * 3 * n_req_echo)
    print(f"wrote {n_inst:,} instances to {OUT.name}")
    print(f"~{n_req_echo:,} requests per echo/label arm; "
          f"~{total:,} total for arms echo_plain,echo_listed,label_num "
          f"+ 25% replicate (ctxfree ~free, cached)")


if __name__ == "__main__":
    main()
