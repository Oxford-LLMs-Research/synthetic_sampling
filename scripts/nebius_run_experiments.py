"""Nebius API runner for the two paired follow-up experiments.

Two scoring modes:

  chat_number (default, fast)
      One /chat/completions call per (instance, condition). Options are listed
      as 1..K; first-token logprobs over the digit tokens rank the options.
      ~1 cell/s at 16 workers on Studio — hours not days.

  echo (paper method, slow on Studio)
      One /completions echo call per option (mean token logprob of the answer
      string). Prefer Colab GPU (colab_run_experiments.py) for this method.

Temporal conditions:
  baseline | with_year | with_year_placebo | with_date (unless --skip-date)

Usage:

    python nebius_run_experiments.py --experiment temporal \\
        --input ../outputs/colab_experiments/temporal_context_instances.jsonl \\
        --out ../outputs/colab_experiments/temporal_results_fast.jsonl \\
        --scoring chat_number --skip-date --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Predict how the respondent would answer "
    "the target question using their prior answers.\n\n"
    "Profile: {profile}\n\n"
    "{extra}"
    "Question: {question}\n\n"
    "Instructions: Reply with a short concise answer. No reasoning. "
    "No explanation. No extra text.\n\n"
    "Answer:"
)

ENV_FILE = Path(r"C:\Users\murrn\cursor\features_project\.env")


def load_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    if key:
        return key.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("No NEBIUS_API_KEY in env and no LLM_API_KEY in .env")


def render_profile(questions: dict) -> str:
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in questions.items())


def temporal_conditions(instance: dict, skip_date: bool) -> list[str]:
    conds = ["baseline", "with_year", "with_year_placebo"]
    if not skip_date and instance.get("interview_date"):
        conds.append("with_date")
    return conds


def build_prompt(instance: dict, condition: str, experiment: str) -> str:
    questions = dict(instance["questions"])
    extra = ""
    if experiment == "temporal":
        if condition == "with_year":
            extra = f"The survey was conducted in {instance['survey_year']}.\n\n"
        elif condition == "with_year_placebo":
            extra = (
                f"The survey was conducted in {instance['survey_year_placebo']}.\n\n"
            )
        elif condition == "with_date":
            extra = (
                f"The interview took place on {instance['interview_date']}.\n\n"
            )
    if experiment == "country" and condition == "with_country":
        questions[instance["country_question"]] = instance["country_name"]
    if experiment == "country" and condition == "with_country_placebo":
        questions[instance["country_question"]] = instance["country_placebo_name"]
    return PROMPT_TEMPLATE.format(
        profile=render_profile(questions),
        extra=extra,
        question=instance["target_question"],
    )


def _post_with_retries(session, url, headers, payload, retries=12):
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


def score_options_echo(session, base_url, key, model, prompt, options) -> dict:
    """Paper method: one echo request per option."""
    scores = {}
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
    return scores


def _number_logprobs_from_choice(choice: dict, n_opts: int) -> dict[int, float]:
    """Map 1-based option index -> logprob from the first generated token."""
    content = (choice.get("message") or {}).get("content") or ""
    logprobs = choice.get("logprobs") or {}
    content_list = logprobs.get("content") or []
    out: dict[int, float] = {}
    if content_list:
        top = content_list[0].get("top_logprobs") or []
        # also include the chosen token
        chosen = content_list[0]
        candidates = list(top)
        if chosen.get("token") is not None:
            candidates = [{
                "token": chosen["token"],
                "logprob": chosen["logprob"],
            }] + candidates
        for item in candidates:
            tok = str(item.get("token", "")).strip()
            # Qwen may emit "1", "1.", " 1", "▁1"
            digits = "".join(ch for ch in tok if ch.isdigit())
            if not digits:
                continue
            idx = int(digits)
            if 1 <= idx <= n_opts and idx not in out:
                out[idx] = float(item["logprob"])
    # fallback: if model replied with a number but it wasn't in top_logprobs
    if content.strip().isdigit():
        idx = int(content.strip())
        if 1 <= idx <= n_opts and idx not in out:
            out[idx] = 0.0  # known chosen; relative rank still usable via missing=-inf
    return out


def score_options_chat_number(session, base_url, key, model, prompt,
                              options) -> dict:
    """Fast path: one chat call; rank by first-token logprob of option numbers."""
    numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    content = (
        f"{prompt}\n\nOptions:\n{numbered}\n\n"
        "Reply with only the option number."
    )
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": min(20, max(len(options), 5)),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = _post_with_retries(session, url, headers, payload)
    choice = r.json()["choices"][0]
    idx_lp = _number_logprobs_from_choice(choice, len(options))
    scores = {}
    for i, opt in enumerate(options, start=1):
        scores[opt] = idx_lp.get(i, float("-inf"))
    # if nothing recovered, fall back to the generated content as hard pick
    if all(math.isinf(v) and v < 0 for v in scores.values()):
        content = (choice.get("message") or {}).get("content") or ""
        digits = "".join(ch for ch in content if ch.isdigit())
        if digits:
            idx = int(digits)
            if 1 <= idx <= len(options):
                scores[options[idx - 1]] = 0.0
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["temporal", "country"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--base-url", default="https://api.studio.nebius.com/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scoring", choices=["chat_number", "echo"], default="chat_number",
                    help="chat_number=1 req/cell (fast); echo=paper perplexity (slow)")
    ap.add_argument("--skip-date", action="store_true",
                    help="omit with_date condition (run fine-date ablation later)")
    args = ap.parse_args()

    api_key = load_key()
    score_fn = (score_options_chat_number if args.scoring == "chat_number"
                else score_options_echo)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["example_id"], r["condition"]))
                except json.JSONDecodeError:
                    pass
        print(f"resuming: {len(done)} (instance, condition) pairs already scored",
              flush=True)

    instances = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        instances = instances[:args.limit]

    tasks = []  # (example_id, condition, prompt, options, gt)
    for inst in instances:
        gt = inst.get("answer") or inst.get("ground_truth")
        conditions = (
            temporal_conditions(inst, args.skip_date)
            if args.experiment == "temporal"
            else ["baseline", "with_country", "with_country_placebo"]
        )
        for cond in conditions:
            key = (inst["example_id"], cond)
            if key in done:
                continue
            prompt = build_prompt(inst, cond, args.experiment)
            tasks.append((inst["example_id"], cond, prompt, list(inst["options"]), gt))

    print(f"{len(instances)} instances, {len(tasks)} cells to do, "
          f"scoring={args.scoring}, workers={args.workers}, "
          f"skip_date={args.skip_date}", flush=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)  # local vLLM; without this, http pool stays tiny

    def work(task):
        eid, cond, prompt, options, gt = task
        scores = score_fn(session, args.base_url, api_key, args.model,
                          prompt, options)
        predicted = max(scores, key=scores.get)
        return {
            "example_id": eid,
            "condition": cond,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": predicted == gt,
            "option_logprobs": scores,
            "scoring": args.scoring,
        }

    n_cells = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out_fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR cell scoring: {e}", flush=True)
                continue
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_cells += 1
            if n_cells % 50 == 0:
                rate = n_cells / (time.time() - t0)
                eta = (len(tasks) - n_cells) / rate / 60 if rate else 0
                print(f"  {n_cells}/{len(tasks)} cells  {rate:.2f} cells/s  "
                      f"eta {eta:.1f} min", flush=True)

    print(f"done: {n_cells} cells written to {args.out} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
