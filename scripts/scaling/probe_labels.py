#!/usr/bin/env python
r"""Why did the label arms return no label log-probabilities?

The readout run aborted at its smoke test: label_num found no numeric token in
the top-k at the first generated position on 72 of 72 option slots, and
label_alpha on 46 of 72, while greedy generation matched 19 of 24. So the model
answers sensibly and simply does not place a bare label first.

The standing hypothesis is that /completions applies no chat template, so an
instruction-tuned model has nothing enforcing "reply with only the option
number" and continues with the answer text instead. That is a guess. This script
replaces it with evidence: it posts the same instance under several prompt
endings and both endpoints, prints what the server actually returns at each of
the first few positions, and says for each variant whether a label is present
and with what probability mass.

Nothing here is scored or kept. It exists to choose the fix.

    python .../probe_labels.py --base-url http://127.0.0.1:8400/v1 \
        --model Qwen/Qwen3-32B --n 3
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import requests

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from nebius_run_experiments import (                                  # noqa: E402
    PROMPT_TEMPLATE, render_profile, _post_with_retries)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def variants(profile: str, question: str, options: list[str]) -> dict[str, str]:
    """Prompt endings that differ only in how hard they force a label first."""
    numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    head = PROMPT_TEMPLATE.format(profile=profile, extra="", question=question)
    head, _, _ = head.partition("\nInstructions:")
    head = head.rstrip()
    body = f"{head}\n\nOptions:\n{numbered}\n\n"
    return {
        # What the failing run used.
        "as_run": body + "Instructions: Reply with only the option number. "
                         "No reasoning. No explanation. No extra text.\n\nAnswer:",
        # Same, with the space the label would follow. Some tokenisers emit
        # " 1" as one token and some emit " " then "1"; this distinguishes them.
        "trailing_space": body + "Instructions: Reply with only the option "
                                 "number.\n\nAnswer: ",
        # A stem that is syntactically incomplete without a number, which does
        # not rely on instruction-following at all.
        "forced_stem": body + "The respondent would choose option number",
        # Same idea, answer-shaped.
        "answer_stem": body + "Answer: The respondent would choose option number",
    }


def probe_completions(session, url, headers, model, prompt, labels, depth):
    payload = {"model": model, "prompt": prompt, "max_tokens": depth,
               "temperature": 0, "logprobs": 20}
    r = _post_with_retries(session, url, headers, payload)
    lp = r.json()["choices"][0]["logprobs"]
    out = []
    for i, top in enumerate(lp.get("top_logprobs") or []):
        norm = {t.strip().strip(".):").upper(): v for t, v in top.items()}
        hit = [l for l in labels if l in norm]
        out.append({"pos": i, "top1": (lp["tokens"][i] if lp.get("tokens") else "?"),
                    "labels_present": hit,
                    "sample": sorted(top.items(), key=lambda kv: -kv[1])[:6]})
    return out


def probe_chat(session, base_url, headers, model, content, labels):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0, "logprobs": True,
               "top_logprobs": 20,
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = _post_with_retries(session, url, headers, payload)
        c = r.json()["choices"][0]
        tops = (c.get("logprobs") or {}).get("content") or []
        if not tops:
            return None
        top = {d["token"]: d["logprob"] for d in tops[0].get("top_logprobs", [])}
        norm = {t.strip().strip(".):").upper(): v for t, v in top.items()}
        return {"top1": tops[0].get("token"),
                "labels_present": [l for l in labels if l in norm],
                "sample": sorted(top.items(), key=lambda kv: -kv[1])[:6]}
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("/data/polf-sula/nuff1496/data/readout_set.jsonl"))
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    url = f"{args.base_url.rstrip('/')}/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    session = requests.Session()

    insts = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            insts.append(json.loads(line))
            if len(insts) >= args.n:
                break

    for inst in insts:
        options = inst["option_sets"]["original"]
        labels = [str(i + 1) for i in range(len(options))]
        profile = render_profile(dict(inst["questions"]))
        print("=" * 78)
        print(f"{inst['survey']} {inst['target_code']}  {len(options)} options")
        print(f"  labels sought: {labels}")
        for name, prompt in variants(profile, inst["target_question"], options).items():
            print(f"\n--- /completions, prompt variant: {name}")
            print(f"    ends: ...{prompt[-60:]!r}")
            for row in probe_completions(session, url, headers, args.model,
                                         prompt, labels, args.depth):
                pretty = ", ".join(f"{t!r}:{v:.2f}" for t, v in row["sample"])
                print(f"    pos {row['pos']}  generated {row['top1']!r}  "
                      f"labels here: {row['labels_present'] or 'NONE'}")
                print(f"          top: {pretty}")
        chat = probe_chat(session, args.base_url, headers, args.model,
                          variants(profile, inst["target_question"], options)["as_run"],
                          labels)
        print(f"\n--- /chat/completions (applies the chat template)")
        if chat is None:
            print("    no logprobs returned")
        elif "error" in chat:
            print(f"    {chat['error']}")
        else:
            pretty = ", ".join(f"{t!r}:{v:.2f}" for t, v in chat["sample"])
            print(f"    generated {chat['top1']!r}  "
                  f"labels here: {chat['labels_present'] or 'NONE'}")
            print(f"          top: {pretty}")
    print("\nPick the first variant that shows the sought labels at some "
          "position with real mass.")


if __name__ == "__main__":
    main()
