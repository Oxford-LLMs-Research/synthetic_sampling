#!/usr/bin/env python
r"""Is the wording instability in the scoring rule or in the model?

Validated meaning-preserving paraphrases of the answer options, scored against
the originals under both readouts in one serving. Three reference points frame
every agreement rate:

    replicate ceiling   the unchanged options scored twice; the measurement's
                        own noise (historically ~100% within a serving)
    chance agreement    expected index-agreement if the two sets' predictions
                        were drawn independently from their own per-question
                        marginals (the January figure was ~22.9%)
    the old number      the withdrawn synonym arm flipped ~63% of echo_plain
                        predictions; this run replaces it with a valid one

The falsifiable prediction: if echo scoring's instability is the fluency term,
paraphrase agreement under label_num approaches the replicate ceiling while
echo_plain stays low. If both are low, the model's belief itself is unstable
under rewording, and no readout repairs that.

Comparisons are on predicted_index (option position), which the paraphrase
sets preserve by construction.

    python analyze_paraphrase.py --results <paraphrase_results_<tag>.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_readout import boot_ci, qmean, tag_of          # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCALE = REPO / "outputs" / "scaling_experiment"
PARA = SCALE / "paraphrase"
OUT = REPO.parent / "analysis" / "readout"

ARMS = ("echo_plain", "label_num")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=SCALE / "paraphrase_set.jsonl")
    args = ap.parse_args()
    tag = tag_of(args.results).replace("paraphrase_results_", "")

    meta = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r

    recs = []
    for line in open(args.results, encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        row = {"eid": r["example_id"],
               "q": f"{m['survey']}|{m['target_code']}",
               "set_id": m.get("paraphrase_set_id"),
               "gt_idx": m["ground_truth_index"],
               "M": len(set(m["option_sets"]["original"]))}
        ok = True
        for arm in ARMS:
            for setname, col in (("original", "o"), ("paraphrase", "p"),
                                 ("original_replicate", "r")):
                d = r["results"].get(f"{setname}|{arm}")
                if setname != "original_replicate" and (
                        not d or "error" in d or d.get("predicted_index") is None):
                    ok = False
                    break
                row[f"{col}_{arm}"] = (d or {}).get("predicted_index")
            if not ok:
                break
        if ok:
            recs.append(row)
    df = pd.DataFrame(recs)
    print(f"{len(df):,} instances with both sets under both arms, "
          f"{df.q.nunique()} questions, model file {args.results.name}\n")

    # Distinct-truth M per question for normalized accuracy, paper convention.
    # gt answers: use ground_truth_index tallies per question.
    M_by_q = df.groupby("q").gt_idx.nunique()
    df["Mq"] = df.q.map(M_by_q)

    summary, per_q_rows = [], []
    for arm in ARMS:
        o = df[f"o_{arm}"].to_numpy()
        p = df[f"p_{arm}"].to_numpy()
        qq = df.q.to_numpy()
        agree = (o == p).astype(float)
        m = qmean(agree, qq)
        lo, hi = boot_ci(agree, qq)

        # Chance: expected agreement were the two sets' predictions drawn
        # independently from their own per-question marginals.
        chance_qs = []
        for q, g in df.groupby("q"):
            po = g[f"o_{arm}"].value_counts(normalize=True)
            pp = g[f"p_{arm}"].value_counts(normalize=True)
            chance_qs.append(sum(po.get(k, 0) * pp.get(k, 0) for k in
                                 set(po.index) | set(pp.index)))
        chance = float(np.mean(chance_qs))

        rep = df[df[f"r_{arm}"].notna()]
        ceiling = float((rep[f"o_{arm}"] == rep[f"r_{arm}"]).mean()) \
            if len(rep) else float("nan")

        acc = df[df.Mq >= 2]
        na = {}
        for col in ("o", "p"):
            correct = (acc[f"{col}_{arm}"] == acc.gt_idx).astype(float)
            na[col] = qmean(((correct - 1 / acc.Mq) / (1 - 1 / acc.Mq)).to_numpy(),
                            acc.q.to_numpy())
        d_acc = ((acc[f"p_{arm}"] == acc.gt_idx).astype(float)
                 - (acc[f"o_{arm}"] == acc.gt_idx).astype(float)).to_numpy()
        d_lo, d_hi = boot_ci(d_acc, acc.q.to_numpy())

        print(f"=== {arm} ===")
        print(f"  original vs paraphrase agreement {m:.1%} [{lo:.1%}, {hi:.1%}]")
        print(f"  replicate ceiling {ceiling:.1%}   chance {chance:.1%}   "
              f"(the withdrawn synonym arm sat at ~36%)")
        print(f"  norm acc original {na['o']:.4f} -> paraphrase {na['p']:.4f} "
              f"(paired raw diff {qmean(d_acc, acc.q.to_numpy()):+.4f} "
              f"[{d_lo:+.4f}, {d_hi:+.4f}])\n")
        summary.append({"arm": arm, "agreement": m, "lo": lo, "hi": hi,
                        "replicate_ceiling": ceiling, "chance": chance,
                        "norm_acc_original": na["o"],
                        "norm_acc_paraphrase": na["p"],
                        "d_acc": qmean(d_acc, acc.q.to_numpy()),
                        "d_lo": d_lo, "d_hi": d_hi, "n": len(df)})
        for q, g in df.groupby("q"):
            per_q_rows.append({"arm": arm, "q": q, "n": len(g),
                               "agreement": (g[f"o_{arm}"] == g[f"p_{arm}"]).mean()})

    # Length-delta covariate: does the size of the wording change predict the
    # flip rate? Under the fluency story it should, for echo, and not for the
    # label readout.
    per_q = pd.DataFrame(per_q_rows)
    val = pd.read_csv(PARA / "paraphrase_validation.csv", encoding="utf-8")
    val["abs_wdelta"] = (val.words_paraphrase - val.words_original).abs()
    delta_by_set = val.groupby("set_id").abs_wdelta.mean()
    set_by_q = df.groupby("q").set_id.agg(lambda s: s.mode().iloc[0])
    per_q["mean_wdelta"] = per_q.q.map(set_by_q).map(delta_by_set)
    print("=== length-delta covariate (per question) ===")
    for arm in ARMS:
        g = per_q[(per_q.arm == arm)].dropna(subset=["mean_wdelta"])
        r = g[["agreement", "mean_wdelta"]].corr(method="spearman").iloc[0, 1]
        print(f"  {arm:<12} Spearman(agreement, mean |word delta|) {r:+.3f} "
              f"over {len(g)} questions")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / f"paraphrase_agreement_{tag}.csv",
                                 index=False)
    per_q.to_csv(OUT / f"paraphrase_per_question_{tag}.csv", index=False)
    print(f"\nwrote paraphrase_agreement_{tag}.csv and "
          f"paraphrase_per_question_{tag}.csv to {OUT}")


if __name__ == "__main__":
    main()
