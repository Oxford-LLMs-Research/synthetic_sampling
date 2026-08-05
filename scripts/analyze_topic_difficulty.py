"""Is topic difficulty a property of the questions or of the models?

The paper reports which topics the language models predict worst. That ranking
on its own cannot say why: a topic could be hard because a randomly drawn
profile carries no signal about it, or because the models fail to use signal
that is there. XGBoost was fit to the same randomly sampled features, so
comparing the two by topic separates the cases.

Outputs analysis/topic_difficulty/by_topic.csv with, per fine-grained topic,
the number of questions, the 13-model mean normalized accuracy, XGBoost's, and
the gap.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
JSONL_DIR = (ROOT / "synthetic_sampling" / "outputs"
             / "main_data_smaller_20_jan_26" / "main_data")
PER_CELL = ROOT / "analysis" / "llm_vs_xgb" / "per_cell_comparison.csv"
OUT = ROOT / "analysis" / "topic_difficulty"

PROFILE = "s6m4"
# Two ESS questions, both with n < 15 and all responses coded as missing.
EXCLUDE = {"national_ethnic_identity"}


def topic_tags() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(str(JSONL_DIR / "*_instances.jsonl"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                d = json.loads(line)
                key = (d["survey"], d["target_code"])
                if key not in out:
                    out[key] = d.get("target_topic_tag")
    return out


def main() -> None:
    tags = topic_tags()
    pc = pd.read_csv(PER_CELL)
    pc = pc[pc["profile_type"] == PROFILE].copy()
    pc["tag"] = [tags.get(k) for k in zip(pc["survey"], pc["target_code"])]
    pc = pc[pc["tag"].notna() & ~pc["tag"].isin(EXCLUDE)]

    g = (pc.groupby("tag")
         .agg(n_questions=("target_code", "nunique"),
              llm=("norm_acc", "mean"),
              xgb=("xgb_norm_acc_matched", "mean"))
         .reset_index())
    g["gap"] = g["xgb"] - g["llm"]
    g = g.sort_values("llm").reset_index(drop=True)

    print(g.round(3).to_string(index=False))
    for thr in (1, 5):
        sub = g[g["n_questions"] >= thr]
        r_acc = sub[["llm", "xgb"]].corr().iloc[0, 1]
        r_gap = sub[["llm", "gap"]].corr().iloc[0, 1]
        print(f"\ntopics with >= {thr} question(s): {len(sub)}")
        print(f"  LLM below chance: {(sub['llm'] < 0).sum()}   "
              f"XGBoost below chance: {(sub['xgb'] < 0).sum()}   "
              f"XGBoost minimum: {sub['xgb'].min():.3f} ({sub.loc[sub['xgb'].idxmin(), 'tag']})")
        print(f"  corr(LLM, XGBoost) r = {r_acc:.3f}; "
              f"corr(LLM accuracy, gap) r = {r_gap:.3f}")

    # Per model as well as pooled: the mean alone cannot say whether a topic is
    # below chance for every model or only for some.
    pm = (pc.groupby(["tag", "model"])["norm_acc"].mean()
          .reset_index(name="llm"))
    spread = (pm.groupby("tag")["llm"]
              .agg(llm_min="min", llm_max="max",
                   n_models_below_chance=lambda s: int((s < 0).sum()))
              .reset_index())
    g = g.merge(spread, on="tag", how="left")

    below = g[g["llm"] < 0]
    print(f"\non the {len(below)} topics below chance on the 13-model mean, "
          f"models below chance individually: "
          f"{below['n_models_below_chance'].min()}--"
          f"{below['n_models_below_chance'].max()} of 13")

    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "by_topic.csv", index=False)
    pm.to_csv(OUT / "by_topic_model.csv", index=False)
    print(f"\nwrote {OUT / 'by_topic.csv'} and by_topic_model.csv")


if __name__ == "__main__":
    main()
