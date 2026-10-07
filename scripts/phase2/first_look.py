"""First look at Phase 2 grid outputs: one row per serving.

    python scripts/phase2/first_look.py ../outputs_recovered/phase2_grid_r0-50 \
        --out ../analysis/phase2/first_look_wave1.csv

Reads every `<serving>/<serving>_shard*of*.jsonl` under the folder. Reports,
for the `label_num` arm on the `original` option set:

- acc: share of instances whose predicted option is the respondent's answer
- norm_q: normalized accuracy (acc - 1/M) / (1 - 1/M), averaged within
  question, then across questions
- modal_acc / modal_norm_q: the same two for always predicting the
  question's most common answer in this file
- emits_modal: share of instances where the prediction is that modal answer
- replicate: agreement of the within-serving replicate (about 10% of rows)
- unread: share of instances where at least one option received no score

A descriptive check that the run is sane, not the paper's analysis.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os

try:
    import orjson
    loads = orjson.loads
except ImportError:  # pragma: no cover
    loads = json.loads

ARM = "original|label_num"
REP = "original_replicate|label_num"
PREMISES = ("original|echo_qonly", "original|echo_ctxfree")


def norm_acc(acc: float, m: int) -> float:
    return (acc - 1.0 / m) / (1.0 - 1.0 / m)


def serving_rows(folder: str):
    for path in sorted(glob.glob(os.path.join(folder, "*_shard*of*.jsonl"))):
        with open(path, "rb") as fh:
            for line in fh:
                yield loads(line)


def summarise(folder: str) -> dict:
    n = errors = unread = rep_n = rep_agree = 0
    # per question: [n, correct, truth counts, prediction counts, M]
    q = collections.defaultdict(
        lambda: [0, 0, collections.Counter(), collections.Counter(), 0])
    by_survey = collections.defaultdict(lambda: [0, 0])
    prem = {k: [0, 0] for k in PREMISES}
    for r in serving_rows(folder):
        res = r["results"]
        rec = res.get(ARM)
        n += 1
        if rec is None or "error" in rec:
            errors += 1
            continue
        truth = r["ground_truth_index"]
        pred = rec["predicted_index"]
        scores = rec["scores"]
        if any(math.isinf(v) for v in scores.values()):
            unread += 1
        key = (r["survey"], r["target_code"])
        cell = q[key]
        cell[0] += 1
        cell[1] += pred == truth
        cell[2][truth] += 1
        cell[3][pred] += 1
        cell[4] = len(scores)
        s = by_survey[r["survey"]]
        s[0] += 1
        s[1] += pred == truth
        rep = res.get(REP)
        if rep is not None and "error" not in rep:
            rep_n += 1
            rep_agree += rep["predicted_index"] == pred
        for k in PREMISES:
            p = res.get(k)
            if p is not None and "error" not in p:
                prem[k][0] += 1
                prem[k][1] += p["predicted_index"] == truth

    scored = sum(c[0] for c in q.values())
    correct = sum(c[1] for c in q.values())
    modal_correct = emits_modal = 0
    norm, modal_norm = [], []
    for c in q.values():
        mode, mode_n = c[2].most_common(1)[0]
        modal_correct += mode_n
        emits_modal += c[3][mode]
        if c[4] >= 2:
            norm.append(norm_acc(c[1] / c[0], c[4]))
            modal_norm.append(norm_acc(mode_n / c[0], c[4]))
    out = {
        "serving": os.path.basename(folder.rstrip("/\\")),
        "rows": n,
        "errors": errors,
        "questions": len(q),
        "acc": correct / scored,
        "norm_q": sum(norm) / len(norm),
        "modal_acc": modal_correct / scored,
        "modal_norm_q": sum(modal_norm) / len(modal_norm),
        "emits_modal": emits_modal / scored,
        "replicate_n": rep_n,
        "replicate": rep_agree / rep_n if rep_n else float("nan"),
        "unread": unread / scored,
    }
    for k in PREMISES:
        out[f"acc_{k.split('|')[1]}"] = (
            prem[k][1] / prem[k][0] if prem[k][0] else float("nan"))
    for survey, (sn, sc) in sorted(by_survey.items()):
        out[f"acc_{survey}"] = sc / sn
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for folder in sorted(glob.glob(os.path.join(args.root, "*", ""))):
        if not glob.glob(os.path.join(folder, "*_shard*of*.jsonl")):
            continue
        row = summarise(folder)
        rows.append(row)
        print(f"{row['serving']:36s} rows {row['rows']:7d}  acc {row['acc']:.4f}"
              f"  norm_q {row['norm_q']:.4f}  replicate {row['replicate']:.4f}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = list(rows[0])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
