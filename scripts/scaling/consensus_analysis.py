#!/usr/bin/env python
r"""What does the model know, as opposed to what did a procedure produce?

Every accuracy figure is model plus prompt plus readout. Changing only the
readout moved normalized accuracy from 0.116 to 0.257 on identical instances,
so no single number separates the model's knowledge from the apparatus that
elicited it.

This takes a different definition: knowledge is what survives across fair
elicitations. Three genuinely different procedures are available on the same
4,800 instances, differing in what the model sees and in what is read out:

    echo_plain    options hidden, rank the option text by mean token logprob
    echo_listed   options shown, same ranking rule
    label_num     options shown, read the log-probability of the option number

Where all three agree, the model has a stable belief and we can ask whether it
is right. Where they disagree there is no belief to report, only an artefact of
the procedure. That is a stricter and more interpretable claim than the best
single readout, which measures how much a favourable apparatus can extract.

The control that decides whether consensus means anything: a readout set could
agree trivially by all naming the question's modal answer, which requires no
knowledge of the respondent at all. So consensus is split by whether the agreed
answer IS the modal one, and on the subset where the model departs from the mode
its accuracy is compared against what sticking with the mode would have scored
on those same respondents. Departing from the mode and being right is the only
thing here that cannot be done without person-level information.

    python .../consensus_analysis.py
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
OUT = REPO.parent / "analysis" / "readout"

# label_num_natural is excluded: it shares label_num's prompt and differs only in
# the Latin-square rotation, so counting both would inflate agreement without
# adding an elicitation. generate is excluded because it agrees with itself only
# 57.7% of the time, so it cannot testify to anyone's stability.
READOUTS = ["echo_plain", "echo_listed", "label_num"]


def qmean(v: np.ndarray, q: np.ndarray) -> float:
    return float(pd.Series(v).groupby(q).mean().mean())


def boot_ci(v: np.ndarray, q: np.ndarray, n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(q, return_inverse=True)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    m = np.array([v[inv == i].mean() for i in range(len(keys))])
    d = m[rng.integers(0, len(keys), size=(n, len(keys)))].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=SCALE / "readout_results" / "readout_results_qwen_qwen3-32b.jsonl")
    ap.add_argument("--input", type=Path, default=SCALE / "readout_set.jsonl")
    args = ap.parse_args()

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    rows = []
    for line in open(args.results, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        preds = {}
        for a in READOUTS:
            d = r["results"].get(f"original|{a}")
            if d and d.get("predicted") is not None:
                preds[a] = d["predicted"]
        if len(preds) != len(READOUTS):
            continue
        rows.append({"eid": r["example_id"], "q": f"{m['survey']}|{m['target_code']}",
                     "truth": m["ground_truth"],
                     "n_options": len(m["option_sets"]["original"]),
                     **{f"p_{a}": preds[a] for a in READOUTS}})
    d = pd.DataFrame(rows)
    print(f"{len(d):,} instances with all {len(READOUTS)} readouts, "
          f"{d.q.nunique()} questions\n")

    # The question's modal answer, from the respondents in this sample.
    mode = d.groupby("q").truth.agg(lambda s: s.value_counts().idxmax())
    d["mode"] = d.q.map(mode)
    cols = [f"p_{a}" for a in READOUTS]
    d["consensus"] = d[cols].nunique(axis=1) == 1
    d["pred"] = d[cols[0]].where(d.consensus)
    d["correct"] = (d.pred == d.truth).where(d.consensus)
    for a in READOUTS:
        d[f"ok_{a}"] = (d[f"p_{a}"] == d.truth).astype(float)
    d["mode_ok"] = (d["mode"] == d.truth).astype(float)

    M = d.groupby("q").truth.nunique()
    d["M"] = d.q.map(M)
    norm = lambda a, M_: (a - 1 / M_) / (1 - 1 / M_)

    print("=== how often do the three procedures agree? ===")
    rate = qmean(d.consensus.to_numpy().astype(float), d.q.to_numpy())
    lo, hi = boot_ci(d.consensus.to_numpy().astype(float), d.q.to_numpy())
    print(f"  all three agree on {rate:.1%} of instances [{lo:.1%}, {hi:.1%}]")
    pair = {}
    for i, a in enumerate(READOUTS):
        for b in READOUTS[i + 1:]:
            pair[f"{a} vs {b}"] = (d[f"p_{a}"] == d[f"p_{b}"]).mean()
    for k, v in pair.items():
        print(f"    {k:<32}{v:.1%}")

    print("\n=== accuracy where they agree, and where they do not ===")
    print(f"{'subset':<26}{'share':>8}{'raw acc':>10}{'norm acc':>10}")
    for name, sub in [("consensus", d[d.consensus]), ("no consensus", d[~d.consensus])]:
        if sub.empty:
            continue
        raw = qmean((sub.pred == sub.truth).to_numpy().astype(float)
                    if name == "consensus"
                    else sub[[f"ok_{a}" for a in READOUTS]].mean(axis=1).to_numpy(),
                    sub.q.to_numpy())
        na = qmean(norm((sub.pred == sub.truth).to_numpy().astype(float)
                        if name == "consensus"
                        else sub[[f"ok_{a}" for a in READOUTS]].mean(axis=1).to_numpy(),
                        sub.M.to_numpy()), sub.q.to_numpy())
        print(f"{name:<26}{len(sub)/len(d):>8.1%}{raw:>10.3f}{na:>10.3f}")
    print("  (for no-consensus rows the three readouts are averaged, since there "
          "is no single prediction)")

    print("\n=== is consensus just everyone naming the modal answer? ===")
    con = d[d.consensus]
    is_mode = (con.pred == con["mode"])
    print(f"  consensus predictions that ARE the question's modal answer: "
          f"{is_mode.mean():.1%}")
    for name, sub in [("consensus, agreed on the mode", con[is_mode]),
                      ("consensus, DEPARTS from the mode", con[~is_mode])]:
        if sub.empty:
            continue
        acc = (sub.pred == sub.truth).to_numpy().astype(float)
        mo = sub.mode_ok.to_numpy()
        a_lo, a_hi = boot_ci(acc - mo, sub.q.to_numpy())
        print(f"\n  {name}   n={len(sub):,} ({len(sub)/len(d):.1%} of all)")
        print(f"    model accuracy            {acc.mean():.3f}")
        print(f"    predicting the mode here  {mo.mean():.3f}")
        print(f"    difference                {acc.mean() - mo.mean():+.3f} "
              f"[{a_lo:+.3f}, {a_hi:+.3f}]")

    print("\nDeparting from the modal answer and being right is the only outcome")
    print("here that cannot be produced without information about the person.")

    # Tag the output by the results file. A fixed name meant running a second
    # model silently overwrote the first model's per-instance output.
    tag = args.results.stem.replace("readout_results", "").strip("_") or "default"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"consensus_per_instance_{tag}.csv"
    d.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
