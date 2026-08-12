"""Scoring arms: label_num (primary), echo_plain, PMI premises, replicate."""

from __future__ import annotations

import threading

from .client import post_with_retries
from .prompts import build_chat_messages, build_prompt

# Default package arms. Dead arms (generate*, label_alpha, echo_listed*) stay
# off this list until a claim needs them; they remain on make-package.
DEFAULT_ARMS = ("label_num", "echo_plain", "echo_qonly", "echo_ctxfree")
# chat_label_num (RUN_CATALOGUE A4): the same label readout through the
# model's own chat template via /chat/completions, so template-vs-raw is a
# paired within-serving contrast. NOTE its name ends with "label_num" on
# purpose — the smoke gate's critical-arm check pools both label readouts.
KEPT_ARMS = DEFAULT_ARMS + ("label_num_natural", "chat_label_num")
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


def _label_logprobs_chat(session, url, headers, model, messages, labels,
                         depth: int = 3,
                         chat_template_kwargs: dict | None = None) -> dict:
    """Chat twin of ``_label_logprobs``: first generated position (within
    ``depth``) whose top-20 contains a label token."""
    payload = {
        "model": model, "messages": messages, "max_tokens": depth,
        "temperature": 0, "logprobs": True, "top_logprobs": 20,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    r = post_with_retries(session, url, headers, payload)
    content = (r.json()["choices"][0].get("logprobs") or {}).get("content") or []
    want = {l.upper() for l in labels}
    for pos in content:
        norm: dict[str, float] = {}
        for entry in (pos.get("top_logprobs") or []):
            key = entry["token"].strip().strip(".):").upper()
            v = entry["logprob"]
            if key and (key not in norm or v > norm[key]):
                norm[key] = v
        if want & set(norm):
            return norm
    return {}


def score_labels_chat(session, url, headers, model, inst, options,
                      chat_template_kwargs: dict | None = None) -> dict:
    """Latin-square digit readout through the model's own chat template."""
    m = len(options)
    labels = [str(i + 1) for i in range(m)]
    totals = {o: [] for o in options}
    for shift in range(m):
        shown = [options[(i + shift) % m] for i in range(m)]
        messages = build_chat_messages(inst, shown, "chat_label_num")
        top = _label_logprobs_chat(
            session, url, headers, model, messages, labels,
            chat_template_kwargs=chat_template_kwargs)
        for slot, o in enumerate(shown):
            v = top.get(labels[slot].upper())
            if v is not None:
                totals[o].append(v)
    return {
        o: (sum(v) / len(v) if v else float("-inf"))
        for o, v in totals.items()
    }


def score_arm(session, urls, headers, model, inst, options, arm,
              set_name: str = "original",
              chat_template_kwargs: dict | None = None) -> dict:
    """Dispatch one arm; return scores + predicted option.

    ``urls`` maps endpoint kind to URL: ``{"completions": ..., "chat": ...}``
    (a bare string is accepted for backward compatibility with raw arms).
    """
    if isinstance(urls, str):
        urls = {"completions": urls}
    if arm == "echo_plain":
        sc = score_echo(
            session, urls["completions"], headers, model,
            build_prompt(inst, options, arm), options)
    elif arm in ("echo_qonly", "echo_ctxfree"):
        sc = _cached_echo(
            session, urls["completions"], headers, model,
            build_prompt(inst, options, arm), options, set_name)
    elif arm in ("label_num", "label_num_natural"):
        sc = score_labels(
            session, urls["completions"], headers, model, inst, options, arm)
    elif arm == "chat_label_num":
        sc = score_labels_chat(
            session, urls["chat"], headers, model, inst, options,
            chat_template_kwargs=chat_template_kwargs)
    else:
        raise ValueError(f"unsupported arm: {arm}")
    best = max(sc, key=sc.get)
    return {
        "scores": sc,
        "predicted": best,
        "predicted_index": options.index(best),
    }
