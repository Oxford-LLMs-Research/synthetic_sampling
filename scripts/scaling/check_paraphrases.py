#!/usr/bin/env python
r"""Automatic gates on the paraphrased option sets, and the round-trip inputs.

The January synonym arm was withdrawn because word-level substitution produced
7.4% artefacts. This pipeline replaces it. Generation (Claude Fable 5,
5 Aug 2026, set-level with explicit constraints) is only trusted as far as
these gates and the two checks that follow them: a blind round-trip mapping by
a DIFFERENT model, and a full human review of all 276 pairs (56 sets is small
enough that the review is a census, not a sample).

Gates enforced here, per set:
  alignment      same length as the original set
  verbatim       keep_verbatim options copied exactly
  duplicates     no repeated string within the paraphrased set
  collision      no paraphrase equal to a DIFFERENT original option of its set
  length         within +-30% of the original word count (1-2 words allowed
                 for one-word originals); violations are FLAGGED, not fatal,
                 because the recorded delta is the analysis covariate

Outputs (to outputs/scaling_experiment/paraphrase/):
  paraphrase_validation.csv   one row per option: original, paraphrase, word
                              and char deltas, content-word Jaccard, flags
  roundtrip_input.json        per set: independently shuffled originals and
                              paraphrases, NO alignment information; the true
                              permutations go to roundtrip_key.json, which the
                              validating model must never see
  roundtrip_key.json          the answer key for score_roundtrip

    python check_paraphrases.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PARA = REPO / "outputs" / "scaling_experiment" / "paraphrase"

STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "not",
        "no", "yes", "is", "it", "this", "that", "i", "my", "with", "at"}


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def content_words(s: str) -> set[str]:
    return {w for w in words(s) if w not in STOP}


def main() -> None:
    inp = json.load(open(PARA / "paraphrase_input.json", encoding="utf-8"))
    raw = json.load(open(PARA / "paraphrase_raw.json", encoding="utf-8"))
    by_id = {r["set_id"]: r for r in raw}

    rows, fatal = [], []
    rng = np.random.default_rng(20260805)
    rt_in, rt_key = [], {}
    for s in inp:
        sid = s["set_id"]
        r = by_id.get(sid)
        if r is None:
            fatal.append(f"{sid}: missing from generation output")
            continue
        orig, para = s["options"], r["paraphrases"]
        if len(para) != len(orig):
            fatal.append(f"{sid}: {len(para)} paraphrases for {len(orig)} options")
            continue
        if len(set(para)) != len(para):
            fatal.append(f"{sid}: duplicate strings in paraphrased set")
        for i, (o, p, keep) in enumerate(zip(orig, para, s["keep_verbatim"])):
            if keep and p != o:
                fatal.append(f"{sid}[{i}]: default option not verbatim: "
                             f"{o!r} -> {p!r}")
            others = set(orig) - {o}
            if p in others:
                fatal.append(f"{sid}[{i}]: paraphrase equals a DIFFERENT "
                             f"original option: {p!r}")
            wo, wp = len(words(o)), len(words(p))
            len_ok = (wp <= 2 if wo == 1 else abs(wp - wo) <= 0.3 * wo + 1e-9)
            co, cp = content_words(o), content_words(p)
            jac = len(co & cp) / len(co | cp) if (co | cp) else 1.0
            rows.append({"set_id": sid, "idx": i, "survey": s["survey"],
                         "target_code": s["target_code"],
                         "original": o, "paraphrase": p,
                         "kept_verbatim": bool(keep),
                         "anchor_word_kept": bool(r["anchor_word_kept"][i]),
                         "words_original": wo, "words_paraphrase": wp,
                         "chars_original": len(o), "chars_paraphrase": len(p),
                         "len_within_30pct": len_ok,
                         "content_jaccard": round(jac, 3)})
        # Blind round-trip material: two independent shuffles, key kept apart.
        po = rng.permutation(len(orig)).tolist()
        pp = rng.permutation(len(orig)).tolist()
        rt_in.append({"set_id": sid, "question": s["question"],
                      "list_a": [orig[i] for i in po],
                      "list_b": [para[i] for i in pp]})
        rt_key[sid] = {"perm_a": po, "perm_b": pp}

    d = pd.DataFrame(rows)
    d.to_csv(PARA / "paraphrase_validation.csv", index=False,
             encoding="utf-8")
    json.dump(rt_in, open(PARA / "roundtrip_input.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(rt_key, open(PARA / "roundtrip_key.json", "w",
                           encoding="utf-8"), indent=1)

    free = d[~d.kept_verbatim]
    print(f"{d.set_id.nunique()} sets, {len(d)} options "
          f"({len(free)} paraphrased, {int(d.kept_verbatim.sum())} verbatim defaults)")
    print(f"mean |word delta| {abs(free.words_paraphrase - free.words_original).mean():.2f}, "
          f"length gate violations {int((~free.len_within_30pct).sum())}")
    print(f"content-word Jaccard: mean {free.content_jaccard.mean():.3f}, "
          f"share fully disjoint {(free.content_jaccard == 0).mean():.1%}, "
          f"share with anchor kept {free.anchor_word_kept.mean():.1%}")
    if fatal:
        print(f"\n{len(fatal)} FATAL gate failures:")
        for f in fatal:
            print(f"  {f}")
        sys.exit(1)
    print("\nall gates pass; round-trip inputs written "
          "(roundtrip_key.json must never reach the validating model)")


if __name__ == "__main__":
    main()
