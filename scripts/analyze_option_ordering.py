"""Does showing answer options, and in which order, change the conclusions?

Validation Test 2 previously reported only how many predictions *shift* when
options are shown. That leaves the important question open: a shift is only a
problem if it changes accuracy. This script answers it.

Four conditions per instance:
  hidden      options never shown; perplexity scored independently (our method)
  natural     options shown in the original survey order
  reversed    options shown in reversed order
  averaged    reviewer-suggested control: average the model's answer
              distribution over the two orderings, then take the argmax

Two averaging rules are computed, since "average the distributions" is
ambiguous. `avg_prob` softmaxes each ordering's per-option mean token logprob
into a distribution and averages the two; `avg_logppl` averages the two
log-perplexities per option and takes the argmin. They agree almost always;
we report the agreement rate so the choice can be shown not to matter.

Data: perplexity_test/base_run (hidden, carries ground_truth) joined to
perplexity_test/results/results (both shown orderings) by stripping the `oc_`
prefix from the options-context example_id.

Outputs to analysis/option_ordering/:
  accuracy_by_condition.csv   per model x condition accuracy
  shift_rates.csv             pairwise prediction-shift rates
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
BASE = ROOT / "perplexity_test/base_run/base_run"
SHOWN = ROOT / "perplexity_test/results/results"
OUT = ROOT / "analysis/option_ordering"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "llama3.1-8b-instruct": "Llama 3.1 8B inst.",
    "llama3.1-70b-instruct": "Llama 3.1 70B inst.",
    "olmo3-7b-dpo": "OLMo 3 7B inst.",
    "olmo3-32b-dpo": "OLMo 3 32B inst.",
}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_hidden(model: str) -> dict:
    """example_id -> (ground_truth, predicted, options)."""
    out = {}
    with open(BASE / f"{model}_survey_results_base.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["example_id"]] = (d["ground_truth"], d["predicted"], d["options"])
    return out


def load_shown(model: str) -> dict:
    """example_id (base form) -> {condition: {option: perplexity}}."""
    out = {}
    with open(SHOWN / f"{model}_options_context_results.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            key = d["example_id"]
            if key.startswith("oc_"):
                key = key[3:]
            res = d["results"]
            if "shown_natural" not in res:
                continue
            out[key] = {c: res[c]["option_perplexities"] for c in
                        ("shown_natural", "shown_reversed") if c in res}
    return out


def process(model: str) -> tuple[dict, dict, dict]:
    """Returns (scale-question results, full-sample results, shift rates).

    The reversed ordering was only generated for scale questions, so the
    four-condition comparison is restricted to those. Hidden vs natural is
    additionally reported on the full sample, which includes categorical items.
    """
    hidden, shown = load_hidden(model), load_shown(model)
    keys = [k for k in shown if k in hidden]

    rec = {c: [] for c in ("hidden", "natural", "reversed", "avg_prob", "avg_logppl")}
    preds = {c: [] for c in rec}
    truth = []
    full = {"hidden": [], "natural": []}
    full_preds = {"hidden": [], "natural": []}

    for k in keys:
        gt, hid_pred, _ = hidden[k]
        nat = shown[k]["shown_natural"]
        full_opts = list(nat)
        if len(full_opts) >= 2:
            s_full = np.array([-np.log(nat[o]) for o in full_opts])
            nat_pred_full = full_opts[int(np.argmax(s_full))]
            full["hidden"].append(hid_pred == gt)
            full["natural"].append(nat_pred_full == gt)
            full_preds["hidden"].append(hid_pred)
            full_preds["natural"].append(nat_pred_full)

        if "shown_reversed" not in shown[k]:
            continue
        rev = shown[k]["shown_reversed"]
        options = [o for o in nat if o in rev]
        if len(options) < 2:
            continue

        # mean per-token logprob = -log(perplexity)
        s_nat = np.array([-np.log(nat[o]) for o in options])
        s_rev = np.array([-np.log(rev[o]) for o in options])

        p_avg = (softmax(s_nat) + softmax(s_rev)) / 2
        logppl_avg = (np.log([nat[o] for o in options])
                      + np.log([rev[o] for o in options])) / 2

        chosen = {
            "hidden": hid_pred,
            "natural": options[int(np.argmax(s_nat))],
            "reversed": options[int(np.argmax(s_rev))],
            "avg_prob": options[int(np.argmax(p_avg))],
            "avg_logppl": options[int(np.argmin(logppl_avg))],
        }
        truth.append(gt)
        for c, p in chosen.items():
            preds[c].append(p)
            rec[c].append(p == gt)

    acc = {c: float(np.mean(v)) for c, v in rec.items()}
    acc["n"] = len(truth)

    full_acc = {f"full_{c}": float(np.mean(v)) for c, v in full.items()}
    full_acc["full_n"] = len(full["hidden"])

    shifts = {
        "hidden_vs_natural": np.mean([a != b for a, b in zip(preds["hidden"], preds["natural"])]),
        "natural_vs_reversed": np.mean([a != b for a, b in zip(preds["natural"], preds["reversed"])]),
        "hidden_vs_averaged": np.mean([a != b for a, b in zip(preds["hidden"], preds["avg_prob"])]),
        "avgrule_disagreement": np.mean([a != b for a, b in zip(preds["avg_prob"], preds["avg_logppl"])]),
        "full_hidden_vs_natural": np.mean(
            [a != b for a, b in zip(full_preds["hidden"], full_preds["natural"])]),
    }
    return acc, full_acc, shifts


def main() -> None:
    acc_rows, shift_rows = [], []
    for model, disp in MODELS.items():
        acc, full_acc, shifts = process(model)
        acc_rows.append({"model": disp, **acc, **full_acc})
        shift_rows.append({"model": disp, **{k: float(v) for k, v in shifts.items()}})
        print(f"{disp:<22} n={acc['n']:<6} "
              f"hidden {acc['hidden']:.3f}  natural {acc['natural']:.3f}  "
              f"reversed {acc['reversed']:.3f}  averaged {acc['avg_prob']:.3f}")

    acc_df = pd.DataFrame(acc_rows)
    shift_df = pd.DataFrame(shift_rows)
    acc_df.to_csv(OUT / "accuracy_by_condition.csv", index=False)
    shift_df.to_csv(OUT / "shift_rates.csv", index=False)

    print("\naccuracy by condition (scale questions, all four conditions):")
    print(acc_df[["model", "n", "hidden", "natural", "reversed",
                  "avg_prob", "avg_logppl"]].round(3).to_string(index=False))
    print("\nfull sample (scale + categorical), hidden vs natural only:")
    print(acc_df[["model", "full_n", "full_hidden", "full_natural"]].round(3).to_string(index=False))
    print("\nprediction shift rates:")
    print(shift_df.round(3).to_string(index=False))
    print("\nmean across the 4 models (scale subset):")
    for c in ("hidden", "natural", "reversed", "avg_prob"):
        print(f"  {c:<11} {acc_df[c].mean():.3f}")
    print("mean across the 4 models (full sample):")
    for c in ("full_hidden", "full_natural"):
        print(f"  {c:<12} {acc_df[c].mean():.3f}")


if __name__ == "__main__":
    main()
