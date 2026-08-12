"""Scoring runner: resume by example_id, sharding, workers, replicate arm."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .arms import DEFAULT_ARMS, REPLICATE, score_arm
from .client import make_session


def _load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _select_instances(
    input_path: Path,
    done: set[str],
    *,
    limit: Optional[int] = None,
    replicate_frac: float = 0.1,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
) -> list[dict]:
    insts: list[dict] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sets = r.get("option_sets") or {}
            usable = {
                k: v for k, v in sets.items()
                if v and all(isinstance(x, str) and x for x in v)
            }
            # Instances without option_sets: fall back to top-level options.
            if "original" not in usable and r.get("options"):
                usable = {"original": list(r["options"])}
            if "original" not in usable:
                continue
            if len({len(v) for v in usable.values()}) != 1:
                continue
            eid = r["example_id"]
            if eid in done:
                continue
            if shard_count is not None and shard_index is not None:
                h = int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)
                if h % shard_count != shard_index:
                    continue
            h = int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)
            if (h % 10000) / 10000 < replicate_frac:
                usable[f"{REPLICATE}_replicate"] = list(usable[REPLICATE])
            r["_sets"] = usable
            insts.append(r)
    if limit is not None:
        insts = insts[:limit]
    return insts


def run_scoring(
    *,
    input_path: Path,
    out_path: Path,
    base_url: str,
    model: str,
    arms: Sequence[str] = DEFAULT_ARMS,
    workers: int = 32,
    limit: Optional[int] = None,
    replicate_frac: float = 0.1,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    api_key: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
) -> int:
    """Score instances to JSONL. Returns number of instances written this call."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    urls = {
        "completions": f"{base_url.rstrip('/')}/completions",
        "chat": f"{base_url.rstrip('/')}/chat/completions",
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    done = _load_done(out_path)
    if done:
        print(f"resuming: {len(done)} already scored", flush=True)

    insts = _select_instances(
        input_path, done, limit=limit, replicate_frac=replicate_frac,
        shard_index=shard_index, shard_count=shard_count)
    n_sets = len(insts[0]["_sets"]) if insts else 0
    print(f"{len(insts)} instances x {len(arms)} arms x {n_sets} option sets",
          flush=True)

    session = make_session(workers)

    def work(inst: dict) -> dict:
        out = {
            "example_id": inst["example_id"],
            "base_id": inst.get("base_id"),
            "survey": inst.get("survey"),
            "target_code": inst.get("target_code"),
            "ground_truth_index": inst.get("ground_truth_index"),
            "results": {},
        }
        for set_name, options in inst["_sets"].items():
            for arm in arms:
                try:
                    rec = score_arm(
                        session, urls, headers, model, inst, options, arm,
                        set_name=set_name,
                        chat_template_kwargs=chat_template_kwargs)
                except Exception as exc:  # noqa: BLE001
                    rec = {"error": f"{type(exc).__name__}: {exc}"}
                out["results"][f"{set_name}|{arm}"] = rec
        return out

    t0, n = time.time(), 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
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
    return n


def parse_arms(spec: str) -> list[str]:
    return [a.strip() for a in spec.split(",") if a.strip()]
