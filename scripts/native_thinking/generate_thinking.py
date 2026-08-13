"""C2 stage 1: native-thinking transcripts through the chat template.

The elicitation axis rerun with the instrument C1 lacked (RUN_CATALOGUE C2,
pre-registered 12 Aug): Qwen3-32B generates through /chat/completions with
``enable_thinking: true`` — its own trained reasoning mode — over the same
734-pair qa substrate as C1.

Generation uses its OWN instruction (think, then reply with the option
number). It deliberately does NOT reuse the scoring ``chat_label_num``
user message, which ends in ``No reasoning`` — that line is a readout
format lock, and sending it while the template forces a think block is a
contradiction. Profile / question / options stay the same content as the
scoring cells; only the instruction differs. Scoring still uses the
standard label template (``No reasoning`` = emit the digit only) with
thinking OFF and the think-block content injected on the ``_ton`` cell.

Sampled at the model card's thinking-mode settings (t=0.6, top-p 0.95,
top-k 20) with a fixed seed; max_tokens 4096 (thinking runs longer than
C1's scaffolded reasoning; a generation cut inside its think block is
recorded data, never repaired). Transcripts stored verbatim, resume by
example_id. NEVER route this through C1's raw /completions scaffold — the
raw regime is the pathology C2 exists to remove.

--trace N dumps the first N request/response pairs verbatim to a JSON file
for eyeball verification before the full run is trusted.

Usage (inside a cluster job with vLLM up):
  python scripts/native_thinking/generate_thinking.py --base-url http://127.0.0.1:PORT/v1 \
      --model Qwen/Qwen3-32B --out outputs/native_thinking/generated/thinking_qwen_qwen3-32b.jsonl
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from synthetic_sampling.scoring.client import make_session, post_with_retries
from synthetic_sampling.scoring.prompts import render_profile

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
TASKS = REPO / "outputs" / "narrative" / "inputs" / "narrative_tasks.jsonl"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"

# Visible-answer format only — the think block is supplied by the template
# hard switch (enable_thinking=true), not by forbidding reasoning in text.
THINKING_INSTRUCTION = (
    "Think carefully about how this respondent would answer the target "
    "question, using their prior answers as evidence. After thinking, "
    "reply with only the option number."
)


def build_thinking_messages(inst: dict) -> list[dict]:
    """Chat user turn for native-thinking generation (not the score template)."""
    profile = (inst.get("profile_text")
               or render_profile(dict(inst["questions"])))
    options = inst["option_sets"]["original"]
    block = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    content = (
        "You are a helpful assistant. Predict how the respondent would "
        "answer the target question using their prior answers.\n\n"
        f"Profile: {profile}\n\n"
        f"Question: {inst['target_question']}\n\n"
        f"Options:\n{block}\n\n"
        f"Instructions: {THINKING_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


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
    ap.add_argument("--chat-template-kwargs",
                    default='{"enable_thinking": true}',
                    help="JSON object forwarded to the chat template. "
                         "Empty string omits the field (Thinking-2507: "
                         "thinking is always on; the kwarg is unnecessary).")
    args = ap.parse_args(argv)
    chat_kwargs = None
    if args.chat_template_kwargs.strip():
        chat_kwargs = json.loads(args.chat_template_kwargs)

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
    trace_lock = threading.Lock()

    def work(inst: dict) -> dict:
        messages = build_thinking_messages(inst)
        payload = {
            "model": args.model, "messages": messages,
            "max_tokens": args.max_tokens, **sampling,
        }
        if chat_kwargs is not None:
            payload["chat_template_kwargs"] = chat_kwargs
        # Thinking-2507 may emit only </think> (template pre-opened the
        # block). Prepend <think> when the open tag is missing so
        # split_think and traces share one canonical string shape.
        # (Stitch happens after the response; see below.)
        try:
            r = post_with_retries(session, url, headers, payload)
            body = r.json()
            choice = body["choices"][0]
            msg = choice.get("message") or {}
            # Prefer raw content (tags in text when no reasoning-parser).
            # If a parser split the block out, stitch it back so split_think
            # and the canary trace still see one string.
            content = msg.get("content")
            reasoning = (msg.get("reasoning")
                         or msg.get("reasoning_content") or "")
            if content is None:
                content = ""
            # Parser-split shape: reasoning in a side field, answer in content.
            if reasoning and "<think>" not in content and "</think>" not in content:
                content = f"<think>{reasoning}</think>{content}"
            # Thinking-2507 close-only content (no open tag) is left as-is;
            # split_think accepts that shape.
            rec = {"example_id": inst["example_id"],
                   "thinking_raw": content,
                   "finish_reason": choice.get("finish_reason"),
                   "sampling": sampling}
            raw_for_trace = body
        except Exception as exc:  # noqa: BLE001
            rec = {"example_id": inst["example_id"],
                   "error": f"{type(exc).__name__}: {exc}"}
            raw_for_trace = None
        if args.trace:
            with trace_lock:
                if len(trace_rows) < args.trace:
                    trace_rows.append({
                        "request": payload,
                        "response": raw_for_trace,
                        "record": rec,
                    })
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
