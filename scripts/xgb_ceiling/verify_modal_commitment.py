"""Pin every quoted modal-commitment number against the CSVs the run wrote.

Also re-pins the four headline dissenter splits against their original
sources (xgb_full_dissenter_split / xgb_llm_dissenter_split), so the new
tables are proven consistent with the already-landed record.

    python scripts/xgb_ceiling/verify_modal_commitment.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AN = REPO.parent / "analysis" / "xgb_ceiling"

bad: list[str] = []
n_checked = 0


def check(name: str, want: float, got: float, tol: float = 5e-5) -> None:
    global n_checked
    n_checked += 1
    if abs(float(got) - want) > tol:
        bad.append(f"{name}: want {want}, got {got}")


def main() -> int:
    s = pd.read_csv(AN / "modal_commitment_summary.csv").set_index("predictor")

    llm = s.loc["llm(c2_toff_label_num)"]
    check("llm n", 734, llm["n"], 0.5)
    check("llm acc_modal", 0.6487, llm["acc_modal"], 5e-4)
    check("llm acc_dissenter", 0.3736, llm["acc_dissenter"], 5e-4)
    check("llm pooled_dev_rate", 0.4550, llm["pooled_dev_rate"], 5e-4)
    check("llm pooled_dev_payoff", 0.3683, llm["pooled_dev_payoff"], 5e-4)
    check("llm dev_modal_resp_share", 0.3982,
          llm["pooled_dev_modal_resp_share"], 5e-4)
    check("llm dev_modal_rank2", 0.7669, llm["dev_modal_rank2_share"], 5e-4)
    check("llm dev_modal_p_modal", 0.1326, llm["dev_modal_p_modal_mean"], 5e-4)
    check("llm dev_modal_margin", 0.6302, llm["dev_modal_margin_mean"], 5e-4)
    check("llm mean_max_p", 0.8045, llm["mean_max_p"], 5e-4)
    check("llm mean_p_modal", 0.5081, llm["mean_p_modal"], 5e-4)

    xp = s.loc["xgb_within_prompt24"]
    check("xp acc_modal", 0.7449, xp["acc_modal"], 5e-4)
    check("xp acc_dissenter", 0.3929, xp["acc_dissenter"], 5e-4)
    check("xp pooled_dev_rate", 0.4046, xp["pooled_dev_rate"], 5e-4)
    check("xp pooled_dev_payoff", 0.4815, xp["pooled_dev_payoff"], 5e-4)
    check("xp dev_modal_resp_share", 0.3098,
          xp["pooled_dev_modal_resp_share"], 5e-4)
    check("xp dev_modal_rank2", 0.8370, xp["dev_modal_rank2_share"], 5e-4)
    check("xp dev_modal_p_modal", 0.2317, xp["dev_modal_p_modal_mean"], 5e-4)
    check("xp dev_modal_margin", 0.4316, xp["dev_modal_margin_mean"], 5e-4)
    check("xp mean_max_p", 0.7708, xp["mean_max_p"], 5e-4)

    xg = s.loc["xgb_grouped"]
    check("xg n", 704, xg["n"], 0.5)
    check("xg acc_modal", 0.7809, xg["acc_modal"], 5e-4)
    check("xg pooled_dev_payoff", 0.5000, xg["pooled_dev_payoff"], 5e-4)
    check("xg mean_max_p", 0.6889, xg["mean_max_p"], 5e-4)

    xw = s.loc["xgb_within"]
    check("xw acc_modal", 0.7829, xw["acc_modal"], 5e-4)
    check("xw acc_dissenter", 0.4489, xw["acc_dissenter"], 5e-4)
    check("xw pooled_dev_payoff", 0.5374, xw["pooled_dev_payoff"], 5e-4)

    # Consistency with the already-landed splits (same convention).
    full = pd.read_csv(AN / "xgb_full_dissenter_split.csv")
    for regime in ("grouped", "within", "within_prompt24"):
        r = full[full["regime"] == regime]
        if "regime" in full.columns and len(r):
            check(f"landed {regime} acc_modal",
                  float(r["acc_modal"].mean()
                        if "acc_modal" in r else r.iloc[0]["modal"]),
                  s.loc[f"xgb_{regime}", "acc_modal"], 5e-4)

    ov = pd.read_csv(
        AN / "modal_commitment_deviation_overlap.csv").set_index("regime")
    op = ov.loc["within_prompt24"]
    check("ov n_modal_shared", 416, op["n_modal_shared"], 0.5)
    check("ov llm_dev_rate", 0.3197, op["llm_dev_rate"], 5e-4)
    check("ov xgb_dev_rate", 0.2212, op["xgb_dev_rate"], 5e-4)
    check("ov jaccard", 0.2931, op["jaccard"], 5e-4)
    check("ov xgb_acc_where_llm_deviates", 0.6165,
          op["xgb_acc_where_llm_deviates"], 5e-4)
    check("ov llm_acc_where_xgb_deviates", 0.4457,
          op["llm_acc_where_xgb_deviates"], 5e-4)

    tgt = pd.read_csv(AN / "modal_commitment_by_target.csv")
    check("by_target rows", 4 * 25 - 1, len(tgt), 0.5)

    if bad:
        print("FAIL")
        for b in bad:
            print(" ", b)
        return 1
    print(f"all {n_checked} modal-commitment checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
