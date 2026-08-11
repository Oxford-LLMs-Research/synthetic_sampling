"""C1 stage 1: generate reasoning transcripts (RUN_CATALOGUE C1).

Substrate: the same 734 sampled pairs as the narrative experiment
(narrative_tasks.jsonl defines them), so B3 and C1 share one substrate and
one qa baseline scale. For each pair the model sees the k=24 profile, the
target question, and the options in NATURAL order (no rotation at the
reasoning stage; the label readout afterwards rotates as always), and is
asked to think step by step, ending with "Final answer: <option number>".

Sampled generation at the model card's recommended settings (temperature
0.6, top-p 0.95, top-k 20) with a FIXED per-request seed, no verbal length
cap (max_tokens 2048 is the hard stop; a transcript cut off before its
final-answer line is recorded data), against the same vLLM
/completions serving the scoring uses. Greedy decoding is deliberately not
the default: model cards warn it degenerates into repetition over long
generations, which would handicap exactly the hypothesis C1 tests; the
fixed seed recovers reproducibility on the same serving, the same grade the
rest of the pipeline has. Sampling parameters are recorded in every output
row. Transcripts are stored verbatim, resume by example_id; the transcript
is DATA: parse failures and missing final-answer markers are recorded
downstream (make_reasoned_set.py), never repaired.

Usage (inside a cluster job with vLLM up):
  python scripts/reasoning/generate_reasoning.py --base-url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen3-32B --out outputs/reasoning/generated/reasoning_qwen_qwen3-32b.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from synthetic_sampling.scoring.client import make_session, post_with_retries
from synthetic_sampling.scoring.prompts import PROMPT_TEMPLATE, render_profile

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"

REASONING_INSTRUCTION = (
    "Think step by step about how this respondent would answer the target "
    "question, using their prior answers as evidence. End with a line of "
    "the form 'Final answer: <option number>'."
)


def build_reasoning_prompt(inst: dict) -> str:
    """Same profile conventions as scoring: prose `profile_text` wins."""
    profile = (inst.get("profile_text")
               or render_profile(dict(inst["questions"])))
    base = PROMPT_TEMPLATE.format(
        profile=profile, extra="",
        question=inst["target_question"])
    head, _, _ = base.partition("\nInstructions:")
    options = inst["option_sets"]["original"]
    block = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    return (f"{head.rstrip()}\n\nOptions:\n{block}\n\n"
            f"Instructions: {REASONING_INSTRUCTION}\n\nReasoning:")


def load_substrate(tasks_path: Path, ladder_path: Path) -> list[dict]:
    eids = set()
    with open(tasks_path, encoding="utf-8") as fh:
        for line in fh:
            eids.add(json.loads(line)["example_id"])
    out = []
    with open(ladder_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in eids:
                out.append(r)
    return sorted(out, key=lambda r: r["example_id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--ladder-set", type=Path, default=LADDER_SET)
    ap.add_argument("--input", type=Path, default=None,
                    help="Runner-format instance file to reason over instead "
                         "of the default substrate (e.g. the validated "
                         "narrative arm for the presentation x elicitation "
                         "2x2); prose profile_text is honoured.")
    ap.add_argument("--input-arm-label", default=None,
                    help="With --input: keep only rows whose arm_label "
                         "matches (e.g. narrative1, so the 2x2 reasons over "
                         "draft 1 only, per the 8 Aug pre-registration).")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if args.input is not None:
        with open(args.input, encoding="utf-8") as fh:
            pool = [json.loads(line) for line in fh]
        if args.input_arm_label:
            pool = [r for r in pool
                    if r.get("arm_label") == args.input_arm_label]
        pool.sort(key=lambda r: r["example_id"])
    else:
        pool = load_substrate(args.tasks, args.ladder_set)
    insts = [r for r in pool if r["example_id"] not in done]
    if args.limit:
        insts = insts[:args.limit]
    print(f"{len(insts)} transcripts to generate "
          f"({len(done)} already done)")

    url = f"{args.base_url.rstrip('/')}/completions"
    headers = {"Authorization": "Bearer EMPTY",
               "Content-Type": "application/json"}
    session = make_session(args.workers)

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "seed": args.seed}

    def work(inst: dict) -> dict:
        payload = {
            "model": args.model, "prompt": build_reasoning_prompt(inst),
            "max_tokens": args.max_tokens, **sampling,
        }
        try:
            r = post_with_retries(session, url, headers, payload)
            choice = r.json()["choices"][0]
            return {"example_id": inst["example_id"],
                    "reasoning_raw": choice["text"],
                    "finish_reason": choice.get("finish_reason"),
                    "sampling": sampling}
        except Exception as exc:  # noqa: BLE001
            return {"example_id": inst["example_id"],
                    "error": f"{type(exc).__name__}: {exc}"}

    t0, n = time.time(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(work, insts):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                fh.flush()
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(insts)}  {rate:.2f} inst/s", flush=True)
    print(f"done: {n} transcripts in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
