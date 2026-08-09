"""B3 stage 3: score the blind round-trip extractions (the hard coverage gate).

The validator model returns, per validation task, its extraction of the
respondent's answer for every profile question (or NOT_STATED). A draft
passes only if EVERY question's answer is recovered exactly (option-level
string match). First failure -> the draft goes on the regeneration list;
a pair whose draft fails twice is excluded, and the exclusion is a datum.

Inputs:
  --tasks        narrative_tasks.jsonl (carries the withheld true answers)
  --extractions  JSONL rows {task_id, answers: {"1": "<option or NOT_STATED>", ...}}

Outputs (next to --extractions):
  narrative_roundtrip.csv     per draft: n_questions, n_exact, n_not_stated,
                              n_wrong, pass, wrong question numbers
  narrative_regen_list.jsonl  failed drafts, ready for one regeneration round
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "outputs" / "narrative" / "inputs"


def norm(s: str) -> str:
    """Mojibake-tolerant: the source data carries corrupted apostrophes
    ('Can�t choose', a Phase 0 hygiene item); the replacement char and
    curly-apostrophe variants collapse so a clean extraction can match."""
    s = str(s).replace("’", "'").replace("�", "'")
    return " ".join(s.split()).strip().lower()


def score_extraction(true_answers: list[str], extracted: dict) -> dict:
    n_exact = n_not_stated = n_wrong = 0
    wrong = []
    for i, truth in enumerate(true_answers, start=1):
        got = extracted.get(str(i), "NOT_STATED")
        if norm(got) == norm(truth):
            n_exact += 1
        elif norm(got) in ("not_stated", "notstated", "not stated"):
            n_not_stated += 1
            wrong.append(i)
        else:
            n_wrong += 1
            wrong.append(i)
    return {"n_questions": len(true_answers), "n_exact": n_exact,
            "n_not_stated": n_not_stated, "n_wrong": n_wrong,
            "wrong_questions": wrong,
            "pass": n_exact == len(true_answers)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", type=Path,
                    default=IN_DIR / "narrative_tasks.jsonl")
    ap.add_argument("--extractions", type=Path, required=True)
    args = ap.parse_args(argv)
    out_dir = args.extractions.parent

    tasks = {}
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            tasks[t["task_id"]] = t

    results, missing = [], 0
    seen = set()
    with open(args.extractions, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = tasks.get(r["task_id"])
            if t is None:
                continue
            seen.add(r["task_id"])
            sc = score_extraction(list(t["questions"].values()),
                                  r.get("answers") or {})
            results.append({"task_id": r["task_id"],
                            "example_id": t["example_id"],
                            "draft": t["draft"], **sc})
    missing = len(tasks) - len(seen)

    with open(out_dir / "narrative_roundtrip.csv", "w",
              encoding="utf-8", newline="") as fh:
        fields = ["task_id", "example_id", "draft", "n_questions", "n_exact",
                  "n_not_stated", "n_wrong", "pass", "wrong_questions"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            r = dict(r)
            r["wrong_questions"] = ";".join(map(str, r["wrong_questions"]))
            w.writerow(r)

    fails = [r for r in results if not r["pass"]]
    with open(out_dir / "narrative_regen_list.jsonl", "w",
              encoding="utf-8", newline="\n") as fh:
        for r in fails:
            fh.write(json.dumps({"task_id": r["task_id"],
                                 "wrong_questions": r["wrong_questions"]},
                                ensure_ascii=False) + "\n")

    n_pass = sum(r["pass"] for r in results)
    total_q = sum(r["n_questions"] for r in results)
    total_exact = sum(r["n_exact"] for r in results)
    print(f"round-trip: {n_pass}/{len(results)} drafts pass the hard gate "
          f"({missing} tasks unscored); per-question recovery "
          f"{total_exact}/{total_q} "
          f"({100 * total_exact / max(total_q, 1):.1f}%)")
    print(f"{len(fails)} drafts on the regeneration list -> "
          f"{out_dir / 'narrative_regen_list.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
