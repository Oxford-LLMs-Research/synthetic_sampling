"""B3 stage 1: build the narrative generation tasks (RUN_CATALOGUE B3).

Substrate: the ladder-readout pairs at k=24, INFORMATIVE ordering (the
operating point), excluding the five duplicate-option hygiene targets. From
the 25 clean targets a seeded stratified sample of 30 pairs each gives 750
profiles; every profile gets TWO independent narrative drafts (separate
generation calls), so draft-vs-draft scoring agreement measures the
wording-variance ceiling that the narrative-vs-qa contrast is read against.

Outputs (outputs/narrative/inputs/):
  narrative_tasks.jsonl      one row per (pair, draft): task_id, profile q:a
  narrative_manifest.json    seed, counts, exclusions, prompt hashes
  narrative_generation_prompt.txt   the generation instruction (subagents)
  narrative_validation_prompt.txt   the blind round-trip instruction

The generation prompt enforces: third person, every fact preserved and
recoverable, nothing added, flowing prose (no lists, no q:a scaffolding),
120-450 words. The validator (a DIFFERENT model, claude-haiku per the
paraphrase playbook) later receives narrative + questions + option lists and
must pick, for every question, which option the narrative implies; exact
option match on all features is the hard coverage gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTER = REPO.parent
DEFAULT_IN = OUTER / "outputs_recovered" / "ladder_readout_set.jsonl"
OUT_DIR = REPO / "outputs" / "narrative" / "inputs"

SEED = 42
PAIRS_PER_TARGET = 30
DRAFTS = 2

GENERATION_PROMPT = """\
You will receive a survey respondent's answers as a list of question:answer
pairs. Write a flowing third-person narrative portrait of this respondent.

Hard constraints:
1. PRESERVE EVERY FACT. Each of the respondent's answers must be recoverable
   from your narrative by a careful reader who has the original questions and
   answer options in front of them. Keep the meaning of each answer at the
   precision of the original option (if they said "somewhat satisfied", the
   narrative must not read as fully satisfied).
2. ADD NOTHING. No inferred demographics, no invented context, no
   explanations of why the respondent might hold a view.
3. FORM: continuous prose, third person ("The respondent..." or "This
   person..."). No lists, no bullet points, no question-and-answer
   scaffolding, no headings. 120 to 450 words.
4. Do not mention the survey, the questionnaire, or the interview.

Return ONLY the narrative text.
"""

VALIDATION_PROMPT = """\
You will receive a narrative portrait of a person, followed by a list of
survey questions, each with its full set of answer options.

For each question, decide which single option the narrative implies this
person chose. If the narrative genuinely does not contain the information,
answer NOT_STATED. Do not guess from vibes; use only what the narrative
states.

Return ONLY a JSON object mapping each question's number (as given) to the
exact text of the chosen option, or "NOT_STATED".
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ladder-set", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--pairs-per-target", type=int, default=PAIRS_PER_TARGET)
    args = ap.parse_args(argv)

    by_target: dict[tuple, list[dict]] = defaultdict(list)
    excluded_dup_targets = set()
    with open(args.ladder_set, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["example_id"].endswith("_ladder_informative_k024"):
                continue
            opts = r["option_sets"]["original"]
            key = (r["survey"], r["target_code"])
            if len(set(opts)) != len(opts):
                excluded_dup_targets.add(key)
                continue
            by_target[key].append(r)

    rng = random.Random(SEED)
    tasks, sampled = [], 0
    for key in sorted(by_target):
        rows = sorted(by_target[key], key=lambda r: r["example_id"])
        take = rng.sample(rows, min(args.pairs_per_target, len(rows)))
        for r in sorted(take, key=lambda r: r["example_id"]):
            sampled += 1
            for d in range(1, DRAFTS + 1):
                tasks.append({
                    "task_id": f"{r['example_id']}#d{d}",
                    "example_id": r["example_id"],
                    "draft": d,
                    "survey": r["survey"],
                    "target_code": r["target_code"],
                    "n_features": len(r["questions"]),
                    "questions": r["questions"],
                })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "narrative_tasks.jsonl", "w",
              encoding="utf-8", newline="\n") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    (args.out_dir / "narrative_generation_prompt.txt").write_text(
        GENERATION_PROMPT, encoding="utf-8", newline="\n")
    (args.out_dir / "narrative_validation_prompt.txt").write_text(
        VALIDATION_PROMPT, encoding="utf-8", newline="\n")

    manifest = {
        "seed": SEED,
        "substrate": "ladder_readout k024 informative",
        "pairs_per_target": args.pairs_per_target,
        "n_targets": len(by_target),
        "n_pairs": sampled,
        "n_tasks": len(tasks),
        "drafts_per_pair": DRAFTS,
        "excluded_dup_targets": sorted(map(list, excluded_dup_targets)),
        "generation_prompt_sha256": hashlib.sha256(
            GENERATION_PROMPT.encode()).hexdigest(),
        "validation_prompt_sha256": hashlib.sha256(
            VALIDATION_PROMPT.encode()).hexdigest(),
    }
    (args.out_dir / "narrative_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(f"{len(by_target)} clean targets ({len(excluded_dup_targets)} "
          f"dup-option targets excluded), {sampled} pairs, "
          f"{len(tasks)} generation tasks -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
