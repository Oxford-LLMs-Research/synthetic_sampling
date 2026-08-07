"""HTTP client for OpenAI-compatible /completions endpoints."""

from __future__ import annotations

import random
import time

import requests


def post_with_retries(session, url, headers, payload, retries: int = 12):
    """POST with backoff on 429 / 5xx / nan-400; raise otherwise."""
    for attempt in range(retries):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 20) + random.random())
                continue
            if r.status_code in (500, 502, 503, 504) or (
                    r.status_code == 400 and "nan" in r.text.lower()):
                time.sleep(0.5 + random.random())
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 + random.random())
    raise RuntimeError("exhausted retries (persistent 5xx/429)")


def make_session(workers: int = 32) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
