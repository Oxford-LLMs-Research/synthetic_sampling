"""Pin every C3 number quoted anywhere against the analysis CSVs.

Run after `analyze_c3.py` (amended 13 Aug protocol: native deployment,
never a logprob read on an injected trace). Nothing about C3 goes into
PAPER_STATE or the paper until this exits 0.

Verdicts these rows carry, against the pre-registration as re-read under
the amendment:
- Endpoint: the thinking-trained sibling UNDERPERFORMS its Instruct
  sibling — stated K=1 vs the chat ceiling -0.0585 (CI -0.101..-0.018,
  excluding zero); vs raw label_num -0.0507 (CI excluding zero). The
  +0.04 falsifier ("reasoning training breaks the ceiling") is nowhere
  approached. Readout-asymmetry caveat: C2 measured stated-vs-chat at
  -0.044 on fixed weights (different model), so the defensible verdict
  is "no gain, plausibly a real cost beyond the readout asymmetry".
- Elicitation flawless across all 10 draws: 100% blocks closed, 0
  truncations, 0 parse failures, 1.9% mean loop rate — not an artifact.
- Majority vote over K=10 does not rescue it (0.5054 raw, norm 0.2673 —
  no better than K=1), consistent with self-consistency 0.82.
- Aggregates: SAMPLING the thinking model gives worse marginal recovery
  than the Instruct sibling's first-token distribution (mean TV 0.2765
  vs 0.2430).
- Both siblings sit far below the prompt-parity supervised ceiling
  (0.4654, XGB-CEILING-FULL): reasoning-as-training does not close the
  conversion gap either.

    python scripts/native_thinking/verify_c3_numbers.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT.parent / "analysis" / "thinking"
TAG = "qwen_qwen3-30b-a3b-thinking-2507"

bad: list[str] = []
n_checked = 0


def check(name: str, want: float, got: float, tol: float = 5e-5) -> None:
    global n_checked
    n_checked += 1
    try:
        ok = abs(float(got) - want) <= tol
    except (TypeError, ValueError):
        ok = False
    if not ok:
        bad.append(f"{name}: want {want}, got {got}")


def main() -> int:
    lev = pd.read_csv(ANALYSIS / f"c3_levels_{TAG}.csv").set_index("cell")
    for cell, acc, norm in (
            ("instruct_chat_label_num", 0.5668, 0.3633),
            ("instruct_label_num", 0.5572, 0.3529),
            ("instruct_echo_plain", 0.5422, 0.3137),
            ("thinking_stated_k1", 0.5054, 0.2750),
            ("thinking_majority_k10", 0.5054, 0.2673)):
        check(f"{cell} n", 734, lev.loc[cell, "n"], 0.5)
        check(f"{cell} acc", acc, lev.loc[cell, "acc_raw"])
        check(f"{cell} norm", norm, lev.loc[cell, "norm_acc"])
    for cell, conf, e in (("instruct_chat_label_num", 0.9179, 0.3511),
                          ("instruct_label_num", 0.8866, 0.3294),
                          ("instruct_echo_plain", 0.7700, 0.2289)):
        check(f"{cell} conf", conf, lev.loc[cell, "mean_conf"])
        check(f"{cell} ece", e, lev.loc[cell, "ece"])
    check("stated_k1 miss", 0.0, lev.loc["thinking_stated_k1", "miss_rate"])

    con = pd.read_csv(ANALYSIS / f"c3_contrasts_{TAG}.csv") \
        .set_index("contrast")
    for name, d, lo, hi, flip in (
            ("thinking_stated-instruct_chat_label_num",
             -0.0585, -0.1011, -0.0183, 0.2725),
            ("thinking_stated-instruct_label_num",
             -0.0507, -0.0880, -0.0173, 0.2507)):
        check(f"{name} n", 734, con.loc[name, "n_pairs"], 0.5)
        check(f"{name} delta", d, con.loc[name, "delta_acc"])
        check(f"{name} ci_lo", lo, con.loc[name, "ci_lo"])
        check(f"{name} ci_hi", hi, con.loc[name, "ci_hi"])
        check(f"{name} flip", flip, con.loc[name, "flip_rate"])
    for arm in ("chat_label_num", "label_num", "echo_plain"):
        row = con.loc[f"instruct_{arm}_replicate"]
        check(f"{arm} replicate n", 179, row["n_pairs"], 0.5)
        check(f"{arm} replicate agree", 1.0, row["agree_rate"])

    cen = pd.read_csv(ANALYSIS / f"c3_census_{TAG}.csv")
    check("census draws", 10, len(cen), 0.5)
    check("census n_error", 0, cen["n_error"].sum(), 0.5)
    check("census blocks", 1.0, cen["block_rate"].mean())
    check("census closed", 1.0, cen["closed_rate"].mean())
    check("census truncated", 0, cen["truncated"].sum(), 0.5)
    check("census loops", 0.0189, cen["loop_rate"].mean())
    check("census parse_fail", 0.0, cen["parse_fail_rate"].mean())
    check("census draw1 acc", 0.5054,
          cen.set_index("draw").loc[1, "acc"])

    agg = pd.read_csv(ANALYSIS / f"c3_aggregate_{TAG}.csv")
    check("agg targets", 25, len(agg), 0.5)
    check("agg tv thinking", 0.2765, agg["tv_thinking_sampled"].mean())
    check("agg tv instruct", 0.2430, agg["tv_instruct_chat"].mean())
    check("agg self-consistency", 0.8215, agg["self_consistency"].mean())

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n_checked} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
