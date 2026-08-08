"""B3 stage 2: automatic gates on generated narratives + validation packaging.

Consumes the generated drafts (JSONL rows {task_id, narrative}) and applies
the hard gates that need no LLM:

  length        120-450 words
  form          no q:a scaffolding, no bullet/numbered lists, no headings
  person        third person (no first/second-person portrait)
  distinctness  the two drafts of a pair are not near-verbatim copies

Drafts passing the gates are packaged into BLIND round-trip validation tasks
for the validator model (a different model than the generator): narrative +
the profile questions, each with a candidate option list built from the
answers observed across the full ladder set for that question text (the
source answers themselves are withheld). `score_narrative_roundtrip.py`
scores the validator's extractions against the withheld answers.

Usage:
  python scripts/narrative/check_narratives.py --narratives outputs/narrative/generated/narratives.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
IN_DIR = REPO / "outputs" / "narrative" / "inputs"
LADDER_SET = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"

BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
QA_SCAFFOLD = re.compile(r"(?m)^\s*(?:Q|A|Question|Answer)\s*:")
HEADING = re.compile(r"(?m)^\s*#{1,6}\s+")
FIRST_SECOND = re.compile(
    r"(?i)\b(?:I am|I'm|I have|you are|you're|my answers?|your answers?)\b")


def word_count(text: str) -> int:
    return len(text.split())


def trigram_jaccard(a: str, b: str) -> float:
    def grams(t):
        w = t.lower().split()
        return {" ".join(w[i:i + 3]) for i in range(max(len(w) - 2, 0))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 1.0
    return len(ga & gb) / len(ga | gb)


def gate_draft(narrative: str) -> list[str]:
    """Hard-gate one draft; returns failure reasons (empty = pass)."""
    fails = []
    n = word_count(narrative)
    if not 120 <= n <= 450:
        fails.append(f"length_{n}_words")
    if QA_SCAFFOLD.search(narrative):
        fails.append("qa_scaffolding")
    if BULLET.search(narrative):
        fails.append("list_formatting")
    if HEADING.search(narrative):
        fails.append("heading")
    if FIRST_SECOND.search(narrative):
        fails.append("not_third_person")
    return fails


def answer_vocab(ladder_set: Path) -> dict[str, list[str]]:
    """Question text -> observed answers across the full ladder set."""
    vocab: dict[str, set[str]] = defaultdict(set)
    with open(ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for q, a in r["questions"].items():
                vocab[q].add(a)
    return {q: sorted(v) for q, v in vocab.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--narratives", type=Path, required=True,
                    help="JSONL rows {task_id, narrative}")
    ap.add_argument("--tasks", type=Path,
                    default=IN_DIR / "narrative_tasks.jsonl")
    ap.add_argument("--ladder-set", type=Path, default=LADDER_SET)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: directory of --narratives")
    ap.add_argument("--near-copy-jaccard", type=float, default=0.5)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.narratives.parent

    tasks = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = t
    drafts = {}
    with open(args.narratives, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            drafts[r["task_id"]] = r["narrative"].strip()

    rows, by_pair = [], defaultdict(dict)
    for task_id, t in sorted(tasks.items()):
        narrative = drafts.get(task_id)
        if narrative is None:
            rows.append({"task_id": task_id, "status": "missing",
                         "failures": "not_generated"})
            continue
        fails = gate_draft(narrative)
        by_pair[t["example_id"]][t["draft"]] = (task_id, narrative, fails)
        rows.append({"task_id": task_id,
                     "status": "fail" if fails else "pass",
                     "failures": ";".join(fails),
                     "words": word_count(narrative)})
    # Distinctness gate between the drafts of a pair.
    for eid, dd in by_pair.items():
        if len(dd) == 2 and not dd[1][2] and not dd[2][2]:
            j = trigram_jaccard(dd[1][1], dd[2][1])
            if j > args.near_copy_jaccard:
                for d in (1, 2):
                    for row in rows:
                        if row["task_id"] == dd[d][0]:
                            row["status"] = "fail"
                            row["failures"] = f"near_copy_jaccard_{j:.2f}"

    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / "narrative_gate_census.csv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task_id", "status",
                                           "failures", "words"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    status = {r["task_id"]: r["status"] for r in rows}
    n_pass = sum(1 for s in status.values() if s == "pass")
    print(f"gates: {n_pass}/{len(tasks)} drafts pass "
          f"({sum(1 for s in status.values() if s == 'fail')} fail, "
          f"{sum(1 for s in status.values() if s == 'missing')} missing)")

    # Blind validation tasks for passing drafts.
    vocab = answer_vocab(args.ladder_set)
    n_tasks = 0
    with open(out_dir / "narrative_validation_tasks.jsonl", "w",
              encoding="utf-8", newline="\n") as fh:
        for task_id, t in sorted(tasks.items()):
            if status.get(task_id) != "pass":
                continue
            qs = [{"n": i + 1, "question": q,
                   "options": vocab.get(q, [])}
                  for i, q in enumerate(t["questions"])]
            fh.write(json.dumps({
                "task_id": task_id, "narrative": drafts[task_id],
                "questions": qs}, ensure_ascii=False) + "\n")
            n_tasks += 1
    print(f"wrote {n_tasks} blind validation tasks -> "
          f"{out_dir / 'narrative_validation_tasks.jsonl'}")
    return 0 if n_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
