"""Probe whether Nebius can score a SUPPLIED continuation (echo / prompt-logprobs).

The perplexity method needs the log-probabilities of answer-option tokens that
we provide, conditioned on the prompt. That requires either:

  (1) the legacy /v1/completions endpoint with echo=true + logprobs, which
      returns per-token logprobs for the prompt tokens themselves; or
  (2) a vLLM-style prompt_logprobs field.

Standard /v1/chat/completions with logprobs=true returns logprobs only for
GENERATED tokens, which is NOT sufficient: you cannot read the logprob of a
fixed option you did not let the model generate.

This script:
  Test 1  confirms /v1/completions echoes prompt-token logprobs.
  Test 2  scores two candidate answers end to end (mean per-token logprob of
          each option) and checks the model prefers the correct one, which is
          exactly what colab_run_experiments.py does.

Run (nothing is billed beyond a few hundred tokens):

    export NEBIUS_API_KEY=...        # your key
    python nebius_probe.py                       # defaults to Qwen/Qwen3-32B
    python nebius_probe.py --model Qwen/Qwen3-32B --base-url https://api.studio.nebius.com/v1

If Test 1 fails, Nebius cannot run this method and the experiments must go to
the GPU cluster instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

PROMPT = "Question: The capital of France is\nAnswer:"
OPTIONS = ["Paris", "London"]        # Paris should win
CORRECT = "Paris"


def completions(base_url: str, key: str, model: str, text: str,
                echo: bool, logprobs: int = 1, max_tokens: int = 1) -> dict:
    r = requests.post(
        f"{base_url.rstrip('/')}/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": text,
            "max_tokens": max_tokens,
            "temperature": 0,
            "echo": echo,
            "logprobs": logprobs,
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def option_mean_logprob(base_url: str, key: str, model: str,
                        prompt: str, option: str) -> float:
    """Mean per-token logprob of `option` conditioned on `prompt`, via echo."""
    full = prompt + " " + option
    resp = completions(base_url, key, model, full, echo=True)
    lp = resp["choices"][0]["logprobs"]
    toks = lp["token_logprobs"]
    offsets = lp["text_offset"]
    cut = len(prompt)                # option tokens are those beyond the prompt
    opt_lps = [t for t, off in zip(toks, offsets)
               if off >= cut and t is not None]
    if not opt_lps:
        raise RuntimeError("no option-token logprobs recovered from echo")
    return sum(opt_lps) / len(opt_lps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--base-url", default=os.environ.get(
        "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1"))
    args = ap.parse_args()

    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("FAIL: set NEBIUS_API_KEY in your environment first.")
        return 2

    print(f"endpoint: {args.base_url}")
    print(f"model:    {args.model}\n")

    # --- Test 1: does /v1/completions echo prompt-token logprobs? ----------
    print("Test 1: echo / prompt-logprobs on /v1/completions ...")
    try:
        resp = completions(args.base_url, key, args.model, PROMPT, echo=True)
        lp = resp["choices"][0]["logprobs"]
        toks, tlp = lp["tokens"], lp["token_logprobs"]
        # with echo, the prompt tokens appear; the first has null logprob (no
        # left context) and the rest are real numbers
        echoed = sum(1 for t in tlp if t is not None)
        if echoed >= max(2, len(toks) - 2):
            print(f"  PASS: {len(toks)} tokens returned, {echoed} with logprobs "
                  f"(prompt tokens are scored).")
        else:
            print(f"  FAIL: only {echoed} logprobs over {len(toks)} tokens; "
                  "prompt tokens are not being scored.")
            _chat_hint(args.base_url, key, args.model)
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {e}")
        print("  /v1/completions with echo is unavailable on this deployment.")
        _chat_hint(args.base_url, key, args.model)
        return 1

    # --- Test 2: end-to-end option scoring --------------------------------
    print("\nTest 2: perplexity-style option scoring ...")
    try:
        scored = {o: option_mean_logprob(args.base_url, key, args.model,
                                         PROMPT, o) for o in OPTIONS}
        pick = max(scored, key=scored.get)
        for o, s in sorted(scored.items(), key=lambda kv: -kv[1]):
            print(f"  {o:<8} mean logprob {s:+.3f}")
        if pick == CORRECT:
            print(f"  PASS: picks '{pick}', matching the perplexity method.\n")
            print("RESULT: Nebius supports the method. Safe to run the "
                  "experiments here.")
            return 0
        print(f"  WARN: picked '{pick}', expected '{CORRECT}'. Echo works but "
              "check tokenization/offset handling before a full run.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {e}")
        return 1


def _chat_hint(base_url: str, key: str, model: str) -> None:
    """Report whether chat logprobs exist, and note they are insufficient."""
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "logprobs": True, "top_logprobs": 1},
            timeout=60,
        )
        has = r.status_code == 200 and "logprobs" in r.text
        print(f"  note: chat/completions logprobs present={has}. Even if present, "
              "these cover only GENERATED tokens and cannot score fixed options.")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
