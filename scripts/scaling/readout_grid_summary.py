#!/usr/bin/env python
r"""One table for the whole readout grid: model x arm, and the gain ordering.

Assembles the per-tag CSVs that analyze_readout.py writes into a single
long-format table, and prints the echo-to-label gain against family and size,
which is the quantity the Tier 3 decision in REBUILD_DECISION.md turns on: the
3 Aug "scales with the model" framing died when the third family landed (a 4B
gains more than a 32B of another family), and the grid decides what the
defensible cross-family statement is.

Run after run_readout_analysis_all.py. Output: analysis/readout/grid_summary.csv

    python .../readout_grid_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO.parent / "analysis" / "readout"

# tag -> (display, family, size in B params, post-trained). Sizes are total
# parameters; MoE active counts noted in display only.
META = {
    "qwen_qwen3-4b": ("Qwen 3 4B", "Qwen 3", 4, True),
    "qwen_qwen3-32b": ("Qwen 3 32B", "Qwen 3", 32, True),
    "qwen_qwen3-235b-a22b-fp8": ("Qwen 3 235B (A22B, fp8)", "Qwen 3", 235, True),
    "google_gemma-3-27b-it": ("Gemma 3 27B", "Gemma 3", 27, True),
    "openai_gpt-oss-120b": ("GPT-OSS 120B (A5B)", "GPT-OSS", 120, True),
    "meta-llama_llama-3.1-8b": ("Llama 3.1 8B base", "Llama 3.1", 8, False),
    "meta-llama_llama-3.1-8b-instruct": ("Llama 3.1 8B inst.", "Llama 3.1", 8, True),
    "meta-llama_llama-3.1-70b": ("Llama 3.1 70B base", "Llama 3.1", 70, False),
    "meta-llama_llama-3.1-70b-instruct": ("Llama 3.1 70B inst.", "Llama 3.1", 70, True),
    "allenai_olmo-3-1025-7b": ("OLMo 3 7B base", "OLMo 3", 7, False),
    "allenai_olmo-3-7b-dpo": ("OLMo 3 7B DPO", "OLMo 3", 7, True),
    "allenai_olmo-3-1025-32b": ("OLMo 3 32B base", "OLMo 3", 32, False),
    "allenai_olmo-3.1-32b-instruct-dpo": ("OLMo 3.1 32B inst. DPO", "OLMo 3", 32, True),
    "deepseek-ai_deepseek-v3.1-terminus": ("DeepSeek-V3.1 (A37B)", "DeepSeek", 685, True),
}


def main() -> None:
    rows = []
    for f in sorted(OUT.glob("readout_summary_*.csv")):
        tag = f.stem[len("readout_summary_"):]
        if tag not in META:
            print(f"note: tag '{tag}' not in META; carried with raw tag")
        display, family, size, post = META.get(tag, (tag, "?", None, None))
        s = pd.read_csv(f)
        rep_f = OUT / f"readout_replicate_{tag}.csv"
        rep = (pd.read_csv(rep_f).set_index("arm")["agreement"]
               if rep_f.exists() else pd.Series(dtype=float))
        for _, r in s.iterrows():
            rows.append({
                "tag": tag, "model": display, "family": family,
                "size_b": size, "post_trained": post, "arm": r["arm"],
                "norm_acc": r["norm_acc"], "auc": r.get("auc"),
                "prior_only": r.get("prior_only"),
                "corrected": r.get("corrected"),
                "replicate_agreement": rep.get(r["arm"]),
            })
    if not rows:
        sys.exit(f"no readout_summary_*.csv under {OUT}; run the suite first")
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "grid_summary.csv", index=False)
    print(f"{d.tag.nunique()} models, {d.arm.nunique()} arms -> grid_summary.csv\n")

    wide = d.pivot_table(index=["family", "size_b", "post_trained", "model"],
                         columns="arm", values="norm_acc").reset_index()
    have = [c for c in ("echo_plain", "echo_listed", "label_num") if c in wide]
    for gain_col, a, b in (("gain_shown", "echo_listed", "echo_plain"),
                           ("gain_label", "label_num", "echo_plain")):
        if a in wide and b in wide:
            wide[gain_col] = wide[a] - wide[b]
    wide = wide.sort_values(["family", "size_b", "post_trained"])

    print("normalized accuracy by arm, and the echo-to-label gain:")
    cols = (["model"] + have
            + [c for c in ("gain_shown", "gain_label") if c in wide])
    print(wide[cols].round(4).to_string(index=False))

    if "gain_label" in wide:
        print("\nwithin-family ordering of the label gain (the claim under "
              "test: gain tracks size\nWITHIN a family; across families, "
              "family and post-training dominate):")
        for fam, g in wide.dropna(subset=["gain_label"]).groupby("family"):
            g = g.sort_values("size_b")
            if len(g) < 2:
                continue
            mono = g.gain_label.is_monotonic_increasing
            path = "  ->  ".join(f"{m}: {v:+.3f}"
                                 for m, v in zip(g.model, g.gain_label))
            print(f"  {fam:<10} {'monotone' if mono else 'NOT monotone'}: {path}")


if __name__ == "__main__":
    main()
