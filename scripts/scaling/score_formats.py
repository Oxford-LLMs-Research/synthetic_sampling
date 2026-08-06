#!/usr/bin/env python
r"""Is the wording instability in the SCORING RULE or in the MODEL?

Paraphrasing the answer options, preserving their meaning, changes the paper's
prediction for 63% of respondents. That is measured with the paper's rule: echo
scoring, options hidden, rank by mean token log-probability. Two explanations
survive that measurement and they call for opposite responses:

    the rule    ranking whole phrases by likelihood is dominated by how fluent
                each phrase is, so a paraphrase reshuffles the ranking. Reading
                a LABEL token instead removes that, because "B" is no more
                fluent than "C". The paper's method should then change.

    the model   the model's belief about this person is itself unstable under
                rephrasing, in which case no readout fixes it, and the
                instability is a finding about synthetic respondents rather
                than about measurement.

Readouts are run on the SAME instances, so every comparison is paired at the
instance level. Select with --arms; not all are wanted in every run:

    echo_plain        the paper's exact condition, options never shown
    echo_listed       the same scoring with the options listed, which separates
                      "sees the choices" from "how the answer is read"
    label_num         options numbered, rank by log-probability of the number
    label_num_natural the same without the Latin square, which separates
                      position debiasing from the disruption of showing an
                      ordinal scale out of order
    label_alpha       lettered, because label bias differs between alphabets
                      and neither is obviously the right choice
    echo_qonly        echo scoring under the paper's template with the profile
                      left empty: the option's fit to the QUESTION alone
    echo_ctxfree      echo scoring under a bare "Answer:" premise: the option's
                      context-free fluency. echo_plain minus these two is the
                      domain-conditional PMI correction (Holtzman et al. 2021),
                      and the decomposition fluency / question-fit /
                      profile-driven is what analyze_pmi.py reads off them
    generate          greedy generation, exact match to an option after
                      stripping formatting; non-matches are recorded, never
                      dropped, and scored as incorrect
    generate_sampled  k draws at temperature 1, kept as an empirical
                      distribution over options, which is what a synthetic
                      sample actually is and what makes this arm comparable on
                      the distributional metrics

Position bias is removed by a cyclic Latin square rather than by randomising:
each instance is presented M times with the options rotated, and each option's
label log-probability is averaged over the presentations in which it occupied a
different slot. Randomising per instance would convert position bias into noise,
which is fine for an aggregate but degrades exactly the per-instance agreement
this script measures. The Latin square costs M requests, the same as echo.

Everything goes through /completions, the endpoint the paper used, so the chat
template is not an additional moving part and base models remain scoreable.

    python .../score_formats.py --input surface_form_test.jsonl \
        --out formats_results.jsonl --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-32B --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# parents[1] is the scripts/ directory both here and on the cluster, where this
# file lives at $DATA/scripts/scaling/. score_scaling.py resolves it the same
# way and is known to import cleanly there.
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from nebius_run_experiments import (                                  # noqa: E402
    PROMPT_TEMPLATE, render_profile, _post_with_retries)

# echo_plain is the paper's exact condition and must be measured inside this run:
# the pre-existing surface-form results are on Llama and OLMo, so they cannot be
# compared to a Qwen run. label_num_natural is label_num without the Latin
# square, which separates position debiasing from the ordinal disruption that
# rotating a Likert scale causes.
# generate_sampled draws k times at temperature 1 and keeps the empirical
# distribution over options. Greedy generation yields only a hard label, so it
# cannot take part in the distributional comparisons; sampling can, and it is
# also what a synthetic-sample product actually does.
ARMS = ("echo_plain", "echo_listed", "echo_listed_numinstr",
        "label_num", "label_num_natural",
        "label_alpha", "echo_qonly", "echo_ctxfree",
        "generate", "generate_sampled")

# Scored twice on the unchanged option set, in separate requests, to establish
# the measurement's own reproducibility. Without it an agreement rate has no
# ceiling to be read against, and we know the ceiling is far below 1: the same
# model on the same instances under a different serving stack agrees on only
# 64.4% of predictions (checks/control_check.py).
REPLICATE = "original"
LETTERS = [chr(ord("A") + i) for i in range(26)]

# The neutral arms' scores depend on (prompt, options) alone, never on the
# respondent: echo_ctxfree collapses 4,800 instances to ~157 unique option
# strings and echo_qonly to one request set per (question, option-set variant).
# The cache key MUST include the option-set name: without it the replicate pass
# would read the cache and report a fake 100% agreement, when the replicate
# exists precisely to measure the serving's own nondeterminism.
_NEUTRAL_CACHE: dict[tuple, dict] = {}
_NEUTRAL_LOCK = threading.Lock()


def _cached_echo(session, url, headers, model, prompt, options,
                 set_name: str) -> dict:
    key = (set_name, prompt, tuple(options))
    with _NEUTRAL_LOCK:
        cached = _NEUTRAL_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    sc = score_echo(session, url, headers, model, prompt, options)
    with _NEUTRAL_LOCK:
        _NEUTRAL_CACHE[key] = dict(sc)
    return sc


def build_prompt(inst: dict, options: list[str], arm: str) -> str:
    """The paper's template, varied only in how the options are presented.

    The profile, the question and the instruction line are untouched, so the
    only thing that differs between arms is the option block and what the model
    is asked to emit.
    """
    profile = render_profile(dict(inst["questions"]))
    base = PROMPT_TEMPLATE.format(profile=profile, extra="",
                                  question=inst["target_question"])
    if arm == "echo_plain":
        return base
    if arm == "echo_ctxfree":
        # The context-free premise of the PMI correction. NOT the empty string:
        # /completions returns no log-probability for the first token after
        # BOS, so an empty prompt degrades the shortest options most. "Answer:"
        # is the minimal domain premise and it matches the boundary convention
        # of every echo arm, since score_echo scores f"{prompt} {o}".
        # Independent of the instance, so the scorer caches it per option set.
        return "Answer:"
    if arm == "echo_qonly":
        # The paper's template, byte-identical except the profile is EMPTY, so
        # echo_plain minus echo_qonly isolates the profile's contribution and
        # nothing else. The dangling "Profile:" header this produces is
        # deliberate; stripping the block would change two things at once.
        return PROMPT_TEMPLATE.format(profile="", extra="",
                                      question=inst["target_question"])
    if arm == "echo_listed":
        block = "\n".join(f"- {o}" for o in options)
        instr = "Reply with a short concise answer."
    elif arm == "echo_listed_numinstr":
        # Matched-instruction control: the label arm's exact option block and
        # instruction line, but scored by echoing the option TEXT, so the only
        # difference from label_num is which token is read. Separates the
        # instruction-wording confound from the readout itself. Pilot-only:
        # if it moves echo_listed by less than replicate noise, cite and drop.
        block = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        instr = "Reply with only the option number."
    elif arm in ("label_num", "label_num_natural"):
        block = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        instr = "Reply with only the option number."
    elif arm == "label_alpha":
        block = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        instr = "Reply with only the option letter."
    elif arm in ("generate", "generate_sampled"):
        block = "\n".join(f"- {o}" for o in options)
        # The template's own tail ("Reply with a short concise answer ... No
        # extra text") pulls toward brevity, which fights a verbatim copy of a
        # long option: with options like "No, this information should only be
        # for government use" the model would emit "No". So this arm replaces
        # the whole instruction rather than only its first sentence.
        head, _, _ = base.partition("\nInstructions:")
        return (f"{head.rstrip()}\n\nOptions:\n{block}\n\nInstructions: Copy "
                "one of the options above exactly as written, word for word. "
                "Do not shorten, paraphrase or explain it. Output only that "
                "option.\n\nAnswer:")
    else:
        raise ValueError(arm)
    # Splice the option block in before the instruction line, keeping the
    # trailing "Answer:" that the paper's template ends with.
    head, _, _ = base.partition("\nInstructions:")
    tail = (f"{head.rstrip()}\n\nOptions:\n{block}\n\nInstructions: {instr} "
            "No reasoning. No explanation. No extra text.\n\nAnswer:")
    # The label arms end with the space. Probed on the live server: without it
    # this model emits a standalone space token first and the label only at the
    # next position, which returned nothing on 72 of 72 option slots and aborted
    # a run. With the space, every label appears at position 0.
    return tail + " " if arm.startswith("label_") else tail


def _label_logprobs(session, url, headers, model, prompt, labels,
                    depth: int = 3) -> dict:
    """Distribution over label tokens, at the first position where one appears.

    Measured, not assumed: with the prompt ending "Answer:" this model emits a
    standalone space token at position 0 and the digit only at position 1, so
    reading position 0 alone returned nothing on 72 of 72 option slots and
    aborted a run. The prompt now ends with the space, which puts the label at
    position 0; scanning a few positions as well makes this robust to
    tokenisers that split the boundary differently, at no extra request cost.

    Returns {} when no label appears anywhere in the scanned window, so the
    caller can record a miss rather than silently score one.
    """
    payload = {"model": model, "prompt": prompt, "max_tokens": depth,
               "temperature": 0, "logprobs": 20}
    r = _post_with_retries(session, url, headers, payload)
    lp = r.json()["choices"][0]["logprobs"]
    want = {l.upper() for l in labels}
    for top in (lp.get("top_logprobs") or []):
        # Tokenisers differ on whether the label carries a leading space, and
        # some emit it with punctuation attached, so normalise before matching.
        norm: dict[str, float] = {}
        for tok, v in top.items():
            key = tok.strip().strip(".):").upper()
            if key and (key not in norm or v > norm[key]):
                norm[key] = v
        if want & set(norm):
            return norm
    return {}


def score_echo(session, url, headers, model, prompt, options) -> dict:
    """Mean token log-probability of each option, the paper's rule."""
    cut = len(prompt)
    scores = {}
    for o in options:
        full = f"{prompt} {o}"
        payload = {"model": model, "prompt": full, "max_tokens": 1,
                   "temperature": 0, "echo": True, "logprobs": 1}
        r = _post_with_retries(session, url, headers, payload)
        lp = r.json()["choices"][0]["logprobs"]
        end = len(full)
        vals = [t for t, off in zip(lp["token_logprobs"], lp["text_offset"])
                if t is not None and cut <= off < end]
        scores[o] = sum(vals) / len(vals) if vals else float("-inf")
    return scores


def score_labels(session, url, headers, model, inst, options, arm) -> dict:
    """Latin square over slots: every option is scored in every position.

    label_num_natural skips the rotation and presents the options once, in their
    given order. Comparing the two isolates how much of any difference is
    position debiasing and how much is the cost of showing an ordinal scale
    out of order, which the rotation unavoidably does.
    """
    M = len(options)
    labels = ([str(i + 1) for i in range(M)]
              if arm.startswith("label_num") else LETTERS[:M])
    totals = {o: [] for o in options}
    for shift in (0,) if arm == "label_num_natural" else range(M):
        shown = [options[(i + shift) % M] for i in range(M)]
        prompt = build_prompt(inst, shown, arm)
        top = _label_logprobs(session, url, headers, model, prompt, labels)
        for slot, o in enumerate(shown):
            v = top.get(labels[slot].upper())
            if v is not None:
                totals[o].append(v)
    return {o: (sum(v) / len(v) if v else float("-inf"))
            for o, v in totals.items()}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


_SCAFFOLD = re.compile(r"^(?:answer\s*[:.\-]?\s*|[-*]\s+|\d+[.)]\s*|[a-z][.)]\s+)",
                       re.IGNORECASE)


def _clean_generation(text: str) -> str:
    """Strip formatting scaffolding without touching the answer's wording."""
    t = (text or "").strip()
    t = t.split("\n")[0].strip()
    for _ in range(3):                       # e.g. 'Answer: - "Very safe"'
        new = _SCAFFOLD.sub("", t).strip()
        new = new.strip('"“”\'`').strip()
        if new == t:
            break
        t = new
    return t.rstrip(".!,;: ").strip()


def score_generate(session, url, headers, model, inst, options,
                   n: int = 1, temperature: float = 0.0) -> dict:
    """Generate, then require an EXACT match to an option after cleanup.

    No prefix or substring fallback. On 38% of these questions one option is a
    substring of another ("Agree" inside "Strongly agree", "Safe" inside "Very
    safe"), so those fallbacks either resolve ambiguity by preferring the
    shorter string, reintroducing the length bias this experiment exists to
    escape, or refuse and silently discard a correct answer.

    Anything that does not match exactly is recorded as unmatched. The analysis
    scores it INCORRECT rather than dropping it: non-matches concentrate on hard
    items, so dropping them flatters the model, and a synthetic respondent that
    cannot emit a valid option has failed at the task it is sold to do. The
    unmatched rate is reported in its own right.

    With n > 1 and temperature > 0 the arm returns an empirical distribution
    over options rather than one label, which is both what a synthetic-sample
    product actually does and what makes it comparable on the distributional
    metrics.
    """
    prompt = build_prompt(inst, options, "generate")
    # Long options must fit, or a correct answer is truncated into a non-match.
    budget = min(48, 8 + int(1.8 * max(len(o.split()) for o in options)))
    payload = {"model": model, "prompt": prompt, "max_tokens": budget,
               "temperature": temperature, "n": n}
    r = _post_with_retries(session, url, headers, payload)

    by_norm = {_norm(o): o for o in options}
    norms = list(by_norm)
    # A set is prefix-safe when no option's wording begins another's. There, and
    # only there, an abbreviated reply ("No" for "No, this information should
    # only be for government use") identifies one option unambiguously, so
    # accepting it recovers a real answer without the length bias that a general
    # prefix rule would introduce. Both tallies are returned so the analysis can
    # report the strict result and the relaxed one side by side.
    prefix_safe = not any(a != b and b.startswith(a) for a in norms for b in norms)

    strict: dict[str, int] = {}
    relaxed: dict[str, int] = {}
    raws = []
    for choice in r.json()["choices"]:
        raw = choice.get("text") or ""
        raws.append(raw.strip()[:80])
        t = _norm(_clean_generation(raw))
        hit = by_norm.get(t)
        if hit is not None:
            strict[hit] = strict.get(hit, 0) + 1
            relaxed[hit] = relaxed.get(hit, 0) + 1
            continue
        if prefix_safe and t:
            cand = [o for o in norms if o.startswith(t)]
            if len(cand) == 1:
                o = by_norm[cand[0]]
                relaxed[o] = relaxed.get(o, 0) + 1

    top = max(strict, key=strict.get) if strict else None
    top_rel = max(relaxed, key=relaxed.get) if relaxed else None
    return {"picked": top, "picked_relaxed": top_rel,
            "match": "exact" if top is not None else
                     ("unique_prefix" if top_rel is not None else "unmatched"),
            "prefix_safe": prefix_safe, "n_draws": len(raws),
            "n_matched": sum(strict.values()),
            "n_matched_relaxed": sum(relaxed.values()),
            "counts": strict, "counts_relaxed": relaxed,
            "raw": raws[0] if raws else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen-draws", type=int, default=8,
                    help="draws for generate_sampled, at temperature 1")
    ap.add_argument("--replicate-frac", type=float, default=1.0,
                    help="share of instances also scored twice on the "
                         "unchanged options, to measure readout noise")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    url = f"{args.base_url.rstrip('/')}/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resuming: {len(done)} already scored", flush=True)

    insts = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sets = r.get("option_sets") or {}
            usable = {k: v for k, v in sets.items()
                      if v and all(isinstance(x, str) and x for x in v)}
            # "original" is required; further sets are optional. The paraphrase
            # experiment supplies a second set, the readout experiment does not,
            # and both run through this scorer unchanged.
            if "original" not in usable:
                continue
            if len({len(v) for v in usable.values()}) != 1:
                continue
            if r["example_id"] in done:
                continue
            # A second pass over the UNCHANGED options. Its disagreement with
            # the first is the measurement's own noise, and every other
            # agreement rate has to be read against it rather than against 1.
            # It is a variance estimate, not a contrast, so it does not need the
            # full sample; a fraction keeps the cost of the readout experiment
            # from doubling. Chosen by a hash of the id so the subset is stable
            # across resumed runs.
            h = int(hashlib.sha256(r["example_id"].encode()).hexdigest()[:8], 16)
            if (h % 10000) / 10000 < args.replicate_frac:
                usable[f"{REPLICATE}_replicate"] = list(usable[REPLICATE])
            r["_sets"] = usable
            insts.append(r)
    if args.limit:
        insts = insts[:args.limit]
    print(f"{len(insts)} instances x {len(arms)} arms x "
          f"{len(insts[0]['_sets']) if insts else 0} option sets", flush=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers,
                                            pool_maxsize=args.workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def work(inst: dict) -> dict:
        out = {"example_id": inst["example_id"], "base_id": inst.get("base_id"),
               "survey": inst.get("survey"), "target_code": inst.get("target_code"),
               "ground_truth_index": inst.get("ground_truth_index"),
               "results": {}}
        for set_name, options in inst["_sets"].items():
            for arm in arms:
                try:
                    if arm in ("echo_plain", "echo_listed",
                               "echo_listed_numinstr"):
                        sc = score_echo(session, url, headers, args.model,
                                        build_prompt(inst, options, arm), options)
                        best = max(sc, key=sc.get)
                        rec = {"scores": sc, "predicted": best,
                               "predicted_index": options.index(best)}
                    elif arm in ("echo_qonly", "echo_ctxfree"):
                        # Cached: the score is instance-independent. `predicted`
                        # is still recorded; for echo_ctxfree it is the fluency
                        # default, itself a quantity the analysis reads.
                        sc = _cached_echo(session, url, headers, args.model,
                                          build_prompt(inst, options, arm),
                                          options, set_name)
                        best = max(sc, key=sc.get)
                        rec = {"scores": sc, "predicted": best,
                               "predicted_index": options.index(best)}
                    elif arm in ("label_num", "label_num_natural", "label_alpha"):
                        sc = score_labels(session, url, headers, args.model,
                                          inst, options, arm)
                        best = max(sc, key=sc.get)
                        rec = {"scores": sc, "predicted": best,
                               "predicted_index": options.index(best)}
                    else:
                        g = score_generate(
                            session, url, headers, args.model, inst, options,
                            n=args.gen_draws if arm == "generate_sampled" else 1,
                            temperature=1.0 if arm == "generate_sampled" else 0.0)
                        # Persist the relaxed tallies too. An earlier version
                        # dropped them, which silently discarded the
                        # unique-prefix recovery: only `raw` survived, and that
                        # holds the first draw alone, so nothing could be
                        # recomputed for the sampled arm.
                        rec = {"predicted": g["picked"], "match": g["match"],
                               "raw": g["raw"], "counts": g["counts"],
                               "picked_relaxed": g["picked_relaxed"],
                               "counts_relaxed": g["counts_relaxed"],
                               "prefix_safe": g["prefix_safe"],
                               "n_draws": g["n_draws"],
                               "n_matched": g["n_matched"],
                               "n_matched_relaxed": g["n_matched_relaxed"],
                               "predicted_index": (options.index(g["picked"])
                                                   if g["picked"] else None)}
                except Exception as exc:                      # noqa: BLE001
                    rec = {"error": f"{type(exc).__name__}: {exc}"}
                out["results"][f"{set_name}|{arm}"] = rec
        return out

    t0, n = time.time(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(work, insts):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                fh.flush()
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(insts)}  {rate:.2f} inst/s  "
                      f"eta {(len(insts) - n) / max(rate, 1e-9) / 60:.0f} min",
                      flush=True)
    print(f"done: {n} instances in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
