#!/usr/bin/env python
"""Score the scaling instances with the paper's echo method.

Imports the prompt template and the scorer from nebius_run_experiments.py
rather than copying them, so this cannot drift from the method that produced
the published temporal and country results. That file is never modified.

Differences from nebius_run_experiments.py, all because this experiment has no
conditions:
  - one prompt per instance instead of three,
  - output written in the MAIN results schema (option_logprobs,
    option_perplexities, predicted, correct), so the existing normalized
    accuracy machinery reads it without a translation step.

Verified equivalent to the main run's scoring before writing this:
  - PROMPT_TEMPLATE and render_profile are byte-identical between
    nebius_run_experiments.py and colab_run_experiments.py, and match the
    template printed in the supplement;
  - the main results satisfy perplexity == exp(-logprob) exactly, so their
    option_logprobs is the MEAN token logprob, which is what
    score_options_echo returns (sum(vals)/len(vals)).

Usage on the cluster, one shard per GPU:
    python score_scaling.py \
      --input  $DATA/data/scaling_shard_00.jsonl \
      --out    $DATA/data/scaling_results_00.jsonl \
      --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-32B --workers 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

# Imported, never modified.
from nebius_run_experiments import (                                   # noqa: E402
    PROMPT_TEMPLATE, render_profile, score_options_echo, _post_with_retries,
)


def score_options_echo_detailed(session, base_url, key, model, prompt, options):
    """score_options_echo, plus the token count behind each mean.

    option_logprobs is the MEAN token logprob, sum(vals)/len(vals). That
    normalisation is not neutral: measured on the main run, longer options score
    systematically higher (correlation +0.37 between option length and logprob,
    within question), because later tokens in a phrase are predictable given the
    earlier ones. Recovering the unnormalised sum after the fact needs len(vals),
    which the published scorer discards.

    Storing the count costs nothing at run time and makes both scoring rules
    computable from the same results, so a length-sensitivity check never
    requires re-running the model.

    Returns (scores, n_tokens). scores is byte-for-byte what score_options_echo
    returns; verify_against_published() asserts that on real prompts at startup,
    so this copy cannot drift from the published method silently.
    """
    scores, n_tokens = {}, {}
    url = f"{base_url.rstrip('/')}/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    cut = len(prompt)
    for option in options:
        full = f"{prompt} {option}"
        payload = {"model": model, "prompt": full, "max_tokens": 1,
                   "temperature": 0, "echo": True, "logprobs": 1}
        r = _post_with_retries(session, url, headers, payload)
        lp = r.json()["choices"][0]["logprobs"]
        end = len(full)
        vals = [t for t, o in zip(lp["token_logprobs"], lp["text_offset"])
                if t is not None and cut <= o < end]
        scores[option] = sum(vals) / len(vals) if vals else float("-inf")
        n_tokens[option] = len(vals)
    return scores, n_tokens


def verify_against_published(session, base_url, key, model, instances, n=3):
    """Assert the detailed scorer reproduces the published one exactly.

    Runs both on a few real prompts, once per job rather than per instance, so
    the cost is a handful of requests. A mismatch means the copy above has
    drifted from nebius_run_experiments.score_options_echo and the run must not
    proceed, because its results would not be comparable to the published ones.
    """
    for inst in instances[:n]:
        prompt, options = build_prompt(inst), list(inst["options"])
        want = score_options_echo(session, base_url, key, model, prompt, options)
        got, ntok = score_options_echo_detailed(
            session, base_url, key, model, prompt, options)
        if want != got:
            bad = {o: (want.get(o), got.get(o)) for o in want if want[o] != got.get(o)}
            raise SystemExit(
                "detailed scorer disagrees with the published one on "
                f"{inst['example_id']}: {bad}")
        if any(v < 1 for v in ntok.values()):
            raise SystemExit(f"zero-token option span on {inst['example_id']}")
    print(f"scorer verified against nebius_run_experiments on {n} prompts",
          flush=True)


def build_prompt(inst: dict) -> str:
    """The baseline prompt: no condition, no injected item."""
    return PROMPT_TEMPLATE.format(
        profile=render_profile(dict(inst["questions"])),
        extra="",
        question=inst["target_question"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("NEBIUS_API_KEY", "dummy"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Resume: an interrupted shard picks up where it stopped.
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resuming: {len(done)} instances already scored", flush=True)

    instances = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        instances = instances[:args.limit]
    todo = [i for i in instances if i["example_id"] not in done]
    print(f"{len(instances)} instances, {len(todo)} to score, "
          f"workers={args.workers}", flush=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if todo:
        verify_against_published(session, args.base_url, args.api_key,
                                 args.model, todo)

    def work(inst: dict) -> dict | None:
        prompt = build_prompt(inst)
        options = list(inst["options"])
        # Serial over options on purpose: the first request populates the
        # prefix cache and the rest hit it. Parallelising them would make all
        # of them miss at once.
        scores, n_tokens = score_options_echo_detailed(
            session, args.base_url, args.api_key, args.model, prompt, options)
        if not scores:
            return None
        predicted = max(scores, key=scores.get)
        gt = inst.get("answer") or inst.get("ground_truth")
        return {
            "example_id": inst["example_id"],
            "ground_truth": gt,
            "options": options,
            "option_logprobs": scores,
            # Token count behind each mean, so the unnormalised sum logprob is
            # recoverable and length sensitivity can be checked without re-running.
            "option_n_tokens": n_tokens,
            "option_perplexities": {o: math.exp(-lp) if lp != float("-inf")
                                    else float("inf") for o, lp in scores.items()},
            "predicted": predicted,
            "correct": bool(predicted == gt),
            "profile_type": inst.get("profile_type"),
            "n_features": inst.get("n_features"),
        }

    t0 = time.time()
    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(work, todo):
                if rec is None:
                    continue
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                n += 1
                if n % 200 == 0:
                    rate = n / (time.time() - t0)
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"  {n}/{len(todo)}  {rate:.1f} inst/s  "
                          f"eta {eta:.0f} min", flush=True)

    print(f"done: {n} instances in {(time.time() - t0) / 60:.1f} min "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
