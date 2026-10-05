"""Which option numbers can the label_num readout see, per roster tokenizer?

The readout matches the first generated token against the option numbers.
A tokenizer that writes "12" as "1" + "2" makes option 12 unreadable and
credits its mass to option 1. This prints, for every roster model, the
highest option number that is a single token after the prompt's "Answer: "
tail. CPU only; reads tokenizer.json from the local HF cache with the
`tokenizers` library, not `transformers`: importing torch fails under the
ARC login node's memory limit.

    HF_HOME=$DATA/hf_cache python scripts/phase2/check_label_tokens.py
    ... check_label_tokens.py --models Qwen/Qwen3-4B --max-label 37
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TAIL = ("Instructions: Reply with only the option number. No reasoning. "
        "No explanation. No extra text.\n\nAnswer: ")
LOCAL = True
ROSTER = Path(__file__).resolve().parents[1] / "cluster" / "roster_phase2.tsv"


def continuation(tok, label: str) -> list[int]:
    """Token ids the label adds after the prompt tail."""
    base = tok.encode(TAIL, add_special_tokens=False).ids
    full = tok.encode(TAIL + label, add_special_tokens=False).ids
    k = 0
    while k < min(len(base), len(full)) and base[k] == full[k]:
        k += 1
    return full[k:]


def check(model: str, max_label: int) -> str:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        hf_hub_download(model, "tokenizer.json", local_files_only=LOCAL))
    split = {}
    for n in range(1, max_label + 1):
        ids = continuation(tok, str(n))
        if len(ids) != 1:
            split[n] = [tok.decode([i]) for i in ids]
    if not split:
        return f"OK     all of 1-{max_label} are single tokens"
    first = min(split)
    return (f"SPLIT  single tokens up to {first - 1}; "
            f"{first} -> {split[first]}; {len(split)} of {max_label} split")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--max-label", type=int, default=37)
    ap.add_argument("--download", action="store_true",
                    help="fetch tokenizer.json if it is not in the cache")
    args = ap.parse_args()
    global LOCAL
    LOCAL = not args.download
    models = args.models
    if not models:
        with open(ROSTER, encoding="utf-8") as fh:
            models = [r["hf_id"] for r in csv.DictReader(fh, delimiter="\t")]
    for m in models:
        try:
            line = check(m, args.max_label)
        except Exception as exc:  # noqa: BLE001
            line = f"ERROR  {type(exc).__name__}: {str(exc)[:120]}"
        print(f"{m:55s} {line}", flush=True)


if __name__ == "__main__":
    main()
