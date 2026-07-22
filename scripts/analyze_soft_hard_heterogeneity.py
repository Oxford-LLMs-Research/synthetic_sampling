"""Does the flattening result survive using probabilities instead of argmax?

The paper measures heterogeneity from the *tally of predicted answers*: each
synthetic respondent contributes one answer, and we compare the resulting
answer shares with the human ones. That is the "hard" reading, and it is what a
practitioner querying a synthetic panel actually obtains.

A reader can reasonably object that the argmax discards the model's uncertainty:
maybe the model knows the distribution and we destroyed it by taking the mode.
This script tests that. For every respondent we convert the per-option mean
token logprobs into a distribution with a softmax, average those distributions
over the question's respondents, and recompute both metrics on the result.

Per question we report, for hard and soft alike:
  entropy ratio  H(predicted) / H(empirical)   < 1 means flattening
  JS distance    scipy jensenshannon, base 2   0 means identical

Both definitions match the ones used for the main figures, so hard numbers here
reproduce the published values and the soft numbers are directly comparable.

Reads the raw per-instance JSONL (which retains option_logprobs) rather than
results_data.csv, restricted to rich profiles.

Outputs to analysis/soft_hard/:
  soft_hard_by_model.csv   per model, mean over questions
  soft_hard_by_question.csv
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
RESULTS = ROOT / "results"
OUT = ROOT / "analysis" / "soft_hard"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_SUFFIX = "_s6m4"

MODEL_DISPLAY = {
    "qwen3-32b": "Qwen 3 32B", "deepseek-v3p1-terminus": "DeepSeek-V3",
    "llama3.1-8b-instruct": "Llama 3.1 8B inst.", "gpt_oss": "GPT-OSS 120B",
    "llama3.1-70b-instruct": "Llama 3.1 70B inst.", "olmo3-7b-dpo": "OLMo 3 7B inst.",
    "gemma-3-27b-instruct": "Gemma 3 27B", "qwen3-4b": "Qwen 3 4B",
    "olmo3-32b-dpo": "OLMo 3 32B inst.", "olmo3-7b-base": "OLMo 3 7B base",
    "olmo3-32b-base": "OLMo 3 32B base", "llama3.1-70b-base": "Llama 3.1 70B base",
    "llama3.1-8b-base": "Llama 3.1 8B base",
}


def softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def process_model(model_dir: Path) -> pd.DataFrame:
    emp: dict[tuple, Counter] = defaultdict(Counter)
    hard: dict[tuple, Counter] = defaultdict(Counter)
    soft: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    nresp: dict[tuple, int] = defaultdict(int)

    for f in sorted(model_dir.glob("*_results.jsonl")):
        survey = f.stem.split("_survey_")[1].rsplit("_results", 1)[0]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                eid = d["example_id"]
                if not eid.endswith(PROFILE_SUFFIX):
                    continue
                lp = d.get("option_logprobs") or {}
                if len(lp) < 2:
                    continue
                # target code sits between respondent id and the profile suffix;
                # grouping only needs a stable key, so use the full option set
                # signature plus the trailing code recovered from example_id.
                stem = eid[: -len(PROFILE_SUFFIX)]
                key = (survey, stem.rsplit("_", 1)[-1], tuple(sorted(lp)))
                emp[key][d["ground_truth"]] += 1
                hard[key][d["predicted"]] += 1
                opts = list(lp)
                probs = softmax(np.array([lp[o] for o in opts], dtype=float))
                for o, p in zip(opts, probs):
                    soft[key][o] += float(p)
                nresp[key] += 1

    rows = []
    for key, emp_c in emp.items():
        opts = sorted(key[2])
        if len(opts) < 2:
            continue
        e = np.array([emp_c.get(o, 0) for o in opts], dtype=float)
        h = np.array([hard[key].get(o, 0) for o in opts], dtype=float)
        s = np.array([soft[key].get(o, 0.0) for o in opts], dtype=float)
        if e.sum() == 0 or h.sum() == 0 or s.sum() == 0:
            continue
        e, h, s = e / e.sum(), h / h.sum(), s / s.sum()
        h_emp = entropy(e[e > 0])
        if h_emp <= 0:
            continue
        rows.append({
            "survey": key[0], "target_code": key[1], "n": nresp[key],
            "er_hard": entropy(h[h > 0]) / h_emp,
            "er_soft": entropy(s[s > 0]) / h_emp,
            "jsd_hard": float(jensenshannon(h, e, base=2)),
            "jsd_soft": float(jensenshannon(s, e, base=2)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    per_model, per_question = [], []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name not in MODEL_DISPLAY:
            continue
        df = process_model(d)
        df["model"] = MODEL_DISPLAY[d.name]
        per_question.append(df)
        per_model.append({
            "model": MODEL_DISPLAY[d.name], "n_questions": len(df),
            "er_hard": df["er_hard"].mean(), "er_soft": df["er_soft"].mean(),
            "jsd_hard": df["jsd_hard"].mean(), "jsd_soft": df["jsd_soft"].mean(),
            "pct_flat_hard": (df["er_hard"] < 1).mean() * 100,
            "pct_flat_soft": (df["er_soft"] < 1).mean() * 100,
        })
        r = per_model[-1]
        print(f"{r['model']:<22} q={r['n_questions']:<4} "
              f"ER hard {r['er_hard']:.3f} soft {r['er_soft']:.3f} | "
              f"JSD hard {r['jsd_hard']:.3f} soft {r['jsd_soft']:.3f}")

    out = pd.DataFrame(per_model).sort_values("er_hard")
    out.to_csv(OUT / "soft_hard_by_model.csv", index=False)
    pd.concat(per_question).to_csv(OUT / "soft_hard_by_question.csv", index=False)

    print("\n=== per model (rich profiles) ===")
    print(out.round(3).to_string(index=False))
    print("\nmeans across 13 models:")
    for c in ("er_hard", "er_soft", "jsd_hard", "jsd_soft",
              "pct_flat_hard", "pct_flat_soft"):
        print(f"  {c:<14} {out[c].mean():.3f}")


if __name__ == "__main__":
    main()
