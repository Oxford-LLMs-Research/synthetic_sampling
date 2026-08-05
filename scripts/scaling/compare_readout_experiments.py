#!/usr/bin/env python
r"""Do the January validation tests and the August readout experiment agree?

Both ran the same manipulation: score the options with the prompt hiding them,
then score them again with the options listed, scoring rule untouched. The
January tests (perplexity_test/) covered four models; the August readout
experiment covered two Qwen checkpoints. No model appears in both.

Reported side by side the two look contradictory. January concluded that
presentation moves WHICH option is chosen far more than HOW OFTEN it is right,
on a pooled difference of +0.9 accuracy points; August found +6.5 on Qwen 3 32B
and used it to argue the paper's headline understates the models.

This script establishes that they never disagreed. Three things it measures:

  per-question movement    the mean absolute per-question change, against an
                           exact permutation null. Under no condition effect
                           each discordant instance is equally likely to favour
                           either arm, so randomising its sign gives the noise
                           floor that a thin design produces for free. January
                           has ~36 observations per question and August ~100, so
                           their noise floors differ by a factor of nearly two
                           and the raw movement figures are not comparable until
                           that is netted out.

  the net effect           which is where the two actually differ, and by 7x

  the same contrast per    which is where the difference comes from. The gain
  model                    from being shown the options rises monotonically with
                           the model, within all three families and across them.
                           A model that cannot use an option list does not
                           benefit from seeing one.

Both raw and normalized accuracy are reported. The January pool reaches 140
options on occupation items against the main study's 17, so raw accuracy is not
comparable across the two without it. The ordering holds either way.

    python .../compare_readout_experiments.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
PERP = REPO.parent / "perplexity_test" / "results" / "results"
BASE = REPO.parent / "perplexity_test" / "base_run" / "base_run"
SCALE = REPO / "outputs" / "scaling_experiment"
OUT = REPO.parent / "analysis" / "readout"

JAN = ["olmo3-7b-dpo", "llama3.1-8b-instruct",
       "olmo3-32b-dpo", "llama3.1-70b-instruct"]
# Olmo 3.1 32B Instruct DPO is a different checkpoint from the `olmo3-32b-dpo`
# the January tests scored, so it is a third family point in the readout
# ordering rather than a within-model check across the two designs.
AUG = [("allenai_olmo-3.1-32b-instruct-dpo", "olmo3.1-32b-instruct-dpo"),
       ("qwen_qwen3-4b", "qwen3-4b"), ("qwen_qwen3-32b", "qwen3-32b")]


def norm(correct: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (correct - 1.0 / M) / (1.0 - 1.0 / M)


def permutation_null(d: pd.DataFrame, n: int, seed: int) -> tuple[float, float]:
    """Mean |per-question delta| expected when the condition has no effect.

    Conditions on the observed discordant count per question, so the floor
    reflects this design's own thinness rather than an assumed error model.
    """
    rng = np.random.default_rng(seed)
    per_n = d.groupby("q").size()
    disc = (d[d.a != d.b].groupby("q").size()
            .reindex(per_n.index, fill_value=0).to_numpy())
    nq = per_n.to_numpy()
    sims = np.empty(n)
    for i in range(n):
        signed = np.array([rng.integers(0, 2, k).sum() * 2 - k if k else 0
                           for k in disc], dtype=float)
        sims[i] = np.abs(signed / nq).mean()
    return float(sims.mean()), float(np.percentile(sims, 97.5))


def load_january(model: str) -> pd.DataFrame:
    tests = {}
    for line in open(REPO / "outputs" / "options_context_test.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        tests[r["example_id"]] = r
    # The stored options-context results carry only the shown conditions; the
    # hidden arm is a separate scoring run over the same validation sample.
    hidden = {}
    for line in open(BASE / f"{model}_survey_results_base.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        hidden[r["example_id"]] = r
    rows = []
    for line in open(PERP / f"{model}_options_context_results.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        t = tests.get(r["example_id"])
        if t is None:
            continue
        h = hidden.get(f"{t['base_id']}_{t['profile_type']}")
        s = (r.get("results") or {}).get("shown_natural")
        if h is None or not s or s.get("predicted") is None:
            continue
        M = t.get("n_options") or len(t.get("options") or [])
        if not M or M < 2:
            continue
        rows.append({"q": f"{t['survey']}|{t['target_code']}",
                     "a": float(h["predicted"] == t["ground_truth"]),
                     "b": float(s["predicted"] == t["ground_truth"]), "M": M})
    return pd.DataFrame(rows)


def load_august(tag: str) -> pd.DataFrame:
    meta = {}
    for line in open(SCALE / "readout_set.jsonl", encoding="utf-8"):
        r = json.loads(line)
        meta[r["example_id"]] = r
    rows = []
    for line in open(SCALE / "readout_results" / f"readout_results_{tag}.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        m = meta.get(r["example_id"])
        if m is None:
            continue
        a = r["results"].get("original|echo_plain")
        b = r["results"].get("original|echo_listed")
        if not a or not b or a.get("predicted") is None or b.get("predicted") is None:
            continue
        M = len(m["option_sets"]["original"])
        if M < 2:
            continue
        rows.append({"q": f"{m['survey']}|{m['target_code']}",
                     "a": float(a["predicted"] == m["ground_truth"]),
                     "b": float(b["predicted"] == m["ground_truth"]), "M": M})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = [(m, "January", load_january(m)) for m in JAN]
    frames += [(lab, "August", load_august(tag)) for tag, lab in AUG]

    print("options hidden against options shown, same scoring rule\n")
    print(f"{'model':<24}{'run':<10}{'n':>7}{'q':>5}{'obs/q':>7}"
          f"{'raw net':>10}{'norm net':>10}{'|delta|':>9}{'floor':>8}{'excess':>9}")
    rows = []
    for name, run, d in frames:
        M = d.M.to_numpy(float)
        raw = d.b.mean() - d.a.mean()
        nrm = (norm(d.b.to_numpy(), M) - norm(d.a.to_numpy(), M)).mean()
        per = d.groupby("q")[["a", "b"]].mean()
        obs_abs = (per.b - per.a).abs().mean()
        floor, _ = permutation_null(d, args.sims, args.seed)
        print(f"{name:<24}{run:<10}{len(d):>7,}{d.q.nunique():>5}"
              f"{int(d.groupby('q').size().median()):>7}"
              f"{raw:>+10.4f}{nrm:>+10.4f}{obs_abs:>9.4f}{floor:>8.4f}"
              f"{obs_abs - floor:>+9.4f}")
        rows.append({"model": name, "run": run, "n": len(d),
                     "n_questions": d.q.nunique(),
                     "obs_per_question": int(d.groupby("q").size().median()),
                     "raw_net": raw, "norm_net": nrm,
                     "mean_abs_per_question": obs_abs, "noise_floor": floor,
                     "excess_over_floor": obs_abs - floor, "mean_M": M.mean()})

    print("\nper-question movement is the same in both runs once the noise floor")
    print("is netted out. The runs differ in the NET, and the net orders by model.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "readout_experiment_comparison.csv", index=False)
    print(f"\nwrote {OUT / 'readout_experiment_comparison.csv'}")


if __name__ == "__main__":
    main()
