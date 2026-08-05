"""Why do the models fall below chance on some topics?

The paper attributes below-chance topics to models falling back on a default
answer instead of reading the profile. This tests that directly: for every
fine-grained topic it measures how often the models emit a content-free default
("Don't know" and its relatives) against how often humans do, and relates that
rate to the topic's normalized accuracy.

Outputs analysis/topic_difficulty/default_answers.csv.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_ids import repair

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
JSONL_DIR = (ROOT / "synthetic_sampling" / "outputs"
             / "main_data_smaller_20_jan_26" / "main_data")
OUT = ROOT / "analysis" / "topic_difficulty"
PROFILE = "s6m4"

MODELS = ["deepseek", "gemma3-27b", "gpt-oss",
          "llama3.1_70b_base", "llama3.1_70b_instruct",
          "llama3.1_8b_base", "llama3.1_8b_instruct",
          "olmo3_32b_base", "olmo3_32b_dpo",
          "olmo3_7b_base", "olmo3_7b_dpo",
          "qwen3-32b", "qwen3-4b"]

# Answer strings that carry no information about the respondent's position.
DEFAULTS = {"don't know", "dont know", "do not know", "refusal", "refused",
            "no answer", "not applicable", "missing", "don't know / refusal",
            "can't choose", "no opinion", "haven't heard enough to say",
            "haven't heard of this", "not asked", "declined to answer"}


def is_default(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in DEFAULTS


def topic_tags() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(str(JSONL_DIR / "*_instances.jsonl"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                k = (d["survey"], d["target_code"])
                if k not in out:
                    out[k] = d.get("target_topic_tag")
    return out


def main() -> None:
    tags = topic_tags()
    frames = []
    for m in MODELS:
        df = pd.read_csv(ROOT / "analysis" / m / "results_data.csv",
                         usecols=["example_id", "survey", "respondent_id",
                                  "target_code", "profile_type",
                                  "ground_truth", "predicted", "correct"],
                         dtype=str)
        df = repair(df[df["profile_type"] == PROFILE], verbose=False)
        df["model"] = m
        frames.append(df)
    a = pd.concat(frames, ignore_index=True)
    a["tag"] = [tags.get(k) for k in zip(a["survey"], a["target_code"])]
    a = a[a["tag"].notna() & (a["tag"] != "national_ethnic_identity")]
    a["correct"] = a["correct"].astype(str).str.lower().eq("true")
    a["pred_default"] = a["predicted"].map(is_default)
    a["true_default"] = a["ground_truth"].map(is_default)

    rows = []
    for t, g in a.groupby("tag"):
        top_pred, n_top = Counter(g["predicted"]).most_common(1)[0]
        rows.append({
            "tag": t, "n": len(g),
            "model_default_rate": g["pred_default"].mean(),
            "human_default_rate": g["true_default"].mean(),
            "top_prediction": top_pred,
            "top_prediction_share": n_top / len(g),
            "raw_acc": g["correct"].mean(),
        })
    d = pd.DataFrame(rows)

    topics = pd.read_csv(OUT / "by_topic.csv")
    d = d.merge(topics[["tag", "llm", "xgb", "n_questions"]], on="tag", how="left")
    d = d.sort_values("llm").reset_index(drop=True)

    print(d[["tag", "n_questions", "llm", "model_default_rate",
             "human_default_rate", "top_prediction_share", "top_prediction"]]
          .round(3).to_string(index=False))

    ok = d["llm"].notna()
    r = np.corrcoef(d.loc[ok, "model_default_rate"], d.loc[ok, "llm"])[0, 1]
    rh = np.corrcoef(d.loc[ok, "human_default_rate"], d.loc[ok, "llm"])[0, 1]
    print(f"\ncorr(model default rate, topic normalized accuracy) r = {r:.3f}")
    print(f"corr(human default rate, topic normalized accuracy) r = {rh:.3f}")
    below = d[d["llm"] < 0]
    above = d[d["llm"] >= 0]
    print(f"\nbelow-chance topics ({len(below)}): model default "
          f"{below['model_default_rate'].mean():.1%} vs human "
          f"{below['human_default_rate'].mean():.1%}")
    print(f"other topics ({len(above)}): model default "
          f"{above['model_default_rate'].mean():.1%} vs human "
          f"{above['human_default_rate'].mean():.1%}")

    print("\nper model, on the below-chance topics:")
    sub = a[a["tag"].isin(below["tag"])]
    for m in MODELS:
        s = sub[sub["model"] == m]
        if not len(s):
            continue
        tp, n_tp = Counter(s["predicted"]).most_common(1)[0]
        print(f"  {m:<24} default {s['pred_default'].mean():>6.1%}  "
              f"top prediction {n_tp/len(s):>6.1%}  {str(tp)[:34]}")

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "default_answers.csv", index=False)
    print(f"\nwrote {OUT / 'default_answers.csv'}")


if __name__ == "__main__":
    main()
