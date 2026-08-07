"""Scoring arms: label_num (primary), echo_plain, PMI premises, replicate."""

from __future__ import annotations

import threading

from .client import post_with_retries
from .prompts import build_prompt

# Default package arms. Dead arms (generate*, label_alpha, echo_listed*) stay
# off this list until a claim needs them; they remain on make-package.
DEFAULT_ARMS = ("label_num", "echo_plain", "echo_qonly", "echo_ctxfree")
KEPT_ARMS = DEFAULT_ARMS + ("label_num_natural",)
REPLICATE = "original"

_NEUTRAL_CACHE: dict[tuple, dict] = {}
_NEUTRAL_LOCK = threading.Lock()


def score_echo(session, url, headers, model, prompt, options) -> dict:
    """Mean token log-probability of each option (paper echo rule)."""
    cut = len(prompt)
    scores = {}
    for o in options:
        full = f"{prompt} {o}"
        payload = {
            "model": model, "prompt": full, "max_tokens": 1,
            "temperature": 0, "echo": True, "logprobs": 1,
        }
        r = post_with_retries(session, url, headers, payload)
        lp = r.json()["choices"][0]["logprobs"]
        end = len(full)
        vals = [
            t for t, off in zip(lp["token_logprobs"], lp["text_offset"])
            if t is not None and cut <= off < end
        ]
        scores[o] = sum(vals) / len(vals) if vals else float("-inf")
    return scores


def _cached_echo(session, url, headers, model, prompt, options,
                 set_name: str) -> dict:
    """Cache PMI premises; key includes set_name so replicates stay real."""
    key = (set_name, prompt, tuple(options))
    with _NEUTRAL_LOCK:
        cached = _NEUTRAL_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    sc = score_echo(session, url, headers, model, prompt, options)
    with _NEUTRAL_LOCK:
        _NEUTRAL_CACHE[key] = dict(sc)
    return sc


def _label_logprobs(session, url, headers, model, prompt, labels,
                    depth: int = 3) -> dict:
    """Distribution over label tokens at the first position where one appears."""
    payload = {
        "model": model, "prompt": prompt, "max_tokens": depth,
        "temperature": 0, "logprobs": 20,
    }
    r = post_with_retries(session, url, headers, payload)
    lp = r.json()["choices"][0]["logprobs"]
    want = {l.upper() for l in labels}
    for top in (lp.get("top_logprobs") or []):
        norm: dict[str, float] = {}
        for tok, v in top.items():
            key = tok.strip().strip(".):").upper()
            if key and (key not in norm or v > norm[key]):
                norm[key] = v
        if want & set(norm):
            return norm
    return {}


def score_labels(session, url, headers, model, inst, options, arm) -> dict:
    """Latin-square (or natural-order) digit readout."""
    m = len(options)
    labels = [str(i + 1) for i in range(m)]
    totals = {o: [] for o in options}
    shifts = (0,) if arm == "label_num_natural" else range(m)
    for shift in shifts:
        shown = [options[(i + shift) % m] for i in range(m)]
        prompt = build_prompt(inst, shown, arm)
        top = _label_logprobs(session, url, headers, model, prompt, labels)
        for slot, o in enumerate(shown):
            v = top.get(labels[slot].upper())
            if v is not None:
                totals[o].append(v)
    return {
        o: (sum(v) / len(v) if v else float("-inf"))
        for o, v in totals.items()
    }


def score_arm(session, url, headers, model, inst, options, arm,
              set_name: str = "original") -> dict:
    """Dispatch one arm; return scores + predicted option."""
    if arm == "echo_plain":
        sc = score_echo(
            session, url, headers, model,
            build_prompt(inst, options, arm), options)
    elif arm in ("echo_qonly", "echo_ctxfree"):
        sc = _cached_echo(
            session, url, headers, model,
            build_prompt(inst, options, arm), options, set_name)
    elif arm in ("label_num", "label_num_natural"):
        sc = score_labels(session, url, headers, model, inst, options, arm)
    else:
        raise ValueError(f"unsupported arm: {arm}")
    best = max(sc, key=sc.get)
    return {
        "scores": sc,
        "predicted": best,
        "predicted_index": options.index(best),
    }
