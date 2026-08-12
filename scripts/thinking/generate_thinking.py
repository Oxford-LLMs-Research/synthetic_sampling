"""C2 stage 1: native-thinking transcripts through the chat template.

The elicitation axis rerun with the instrument C1 lacked (RUN_CATALOGUE C2,
pre-registered 12 Aug): Qwen3-32B generates through /chat/completions with
``enable_thinking: true`` — its own trained reasoning mode — over the same
734-pair qa substrate as C1. The user message is IDENTICAL to the one the
direct (thinking-off) cell is scored with, so the toggle is the only
difference in the request.

Sampled at the model card's thinking-mode settings (t=0.6, top-p 0.95,
top-k 20) with a fixed seed; max_tokens 4096 (thinking runs longer than
C1's scaffolded reasoning; a generation cut inside its think block is
recorded data, never repaired). Transcripts stored verbatim, resume by
example_id. NEVER route this through C1's raw /completions scaffold — the
raw regime is the pathology C2 exists to remove.

--trace N dumps the first N request/response pairs verbatim to a JSON file
for eyeball verification before the full run is trusted.

Usage (inside a cluster job with vLLM up):
  python scripts/thinking/generate_thinking.py --base-url http://127.0.0.1:PORT/v1 \
      --model Qwen/Qwen3-32B --out outputs/thinking/generated/thinking_qwen_qwen3-32b.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from synthetic_sampling.scoring.client import make_session, post_with_retries
from synthetic_sampling.scoring.prompts import build_chat_messages

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"


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
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trace", type=int, default=0,
                    help="Dump the first N request/response pairs verbatim "
                         "to <out stem>_trace.json for inspection.")
    args = ap.parse_args(argv)

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    pool_all = load_substrate(args.tasks, args.ladder_set)
    insts = [r for r in pool_all if r["example_id"] not in done]
    if args.limit:
        insts = insts[:args.limit]
    print(f"{len(insts)} thinking transcripts to generate "
          f"({len(done)} already done)")

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": "Bearer EMPTY",
               "Content-Type": "application/json"}
    session = make_session(args.workers)

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "seed": args.seed}
    trace_rows: list[dict] = []

    def work(inst: dict) -> dict:
        messages = build_chat_messages(
            inst, inst["option_sets"]["original"], "chat_label_num")
        payload = {
            "model": args.model, "messages": messages,
            "max_tokens": args.max_tokens, **sampling,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        try:
            r = post_with_retries(session, url, headers, payload)
            choice = r.json()["choices"][0]
            rec = {"example_id": inst["example_id"],
                   "thinking_raw": choice["message"]["content"],
                   "finish_reason": choice.get("finish_reason"),
                   "sampling": sampling}
        except Exception as exc:  # noqa: BLE001
            rec = {"example_id": inst["example_id"],
                   "error": f"{type(exc).__name__}: {exc}"}
        if len(trace_rows) < args.trace:
            trace_rows.append({"request": payload, "response": rec})
        return rec

    t0, n = time.time(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(work, insts):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                fh.flush()
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(insts)}  {rate:.2f} inst/s", flush=True)
    print(f"done: {n} transcripts in {(time.time() - t0) / 60:.1f} min")
    if args.trace:
        trace_path = args.out.with_name(args.out.stem + "_trace.json")
        trace_path.write_text(
            json.dumps(trace_rows, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"trace ({len(trace_rows)} pairs) -> {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
