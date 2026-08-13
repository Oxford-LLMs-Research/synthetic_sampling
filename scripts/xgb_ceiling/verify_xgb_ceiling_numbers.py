"""Pin every quoted XGB-ceiling number against the CSVs the run wrote.

The reproduced quantities (readout per-target table, ladder by-rung table)
are already gated inside run_xgb_ceiling.py against T0.1's pinned outputs;
this script pins the NEW quantities — the pooled readout row, the 734-pair
k=24 anchor, and the dissenter split — so nothing is quoted from a
terminal scrollback.

    python scripts/xgb_ceiling/verify_xgb_ceiling_numbers.py
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
    s = pd.read_csv(AN / "xgb_readout_summary.csv").iloc[0]
    check("readout n", 4545, s["n_scored"], 0.5)
    check("readout acc", 0.4746, s["acc"])
    check("readout norm_acc", 0.2733, s["norm_acc"])
    check("readout tv", 0.0837, s["tv_marginal"])

    a = pd.read_csv(AN / "xgb_anchor_k24.csv")
    p = a[a["target"] == "POOLED"].iloc[0]
    check("anchor n", 660, p["n"], 0.5)
    check("anchor n_targets", 22, p["n_targets"], 0.5)
    check("anchor acc", 0.5000, p["acc"])
    check("anchor norm_acc", 0.2764, p["norm_acc"])
    check("anchor tv", 0.1591, p["tv_marginal"])

    d = pd.read_csv(AN / "xgb_dissenter_split.csv")
    for sub, modal, diss, ratio, n_t in (
            ("readout", 0.6597, 0.1681, 0.555, 46),
            ("ladder_k24_informative", 0.7201, 0.1648, 0.534, 27)):
        g = d[d["substrate"] == sub]
        check(f"{sub} n_targets", n_t, len(g), 0.5)
        check(f"{sub} modal acc", modal, g["acc_modal"].mean())
        check(f"{sub} dissenter acc", diss, g["acc_dissenter"].mean())
        check(f"{sub} entropy ratio", ratio, g["entropy_ratio"].mean(),
              tol=5e-4)

    m = pd.read_csv(AN / "xgb_anchor_matched_model.csv").iloc[0]
    check("matched n", 660, m["n"], 0.5)
    check("matched model norm_acc", 0.3560, m["norm_acc"])
    check("matched xgb norm_acc", 0.2764, m["xgb_norm_acc"])
    check("matched gap", 0.0796, m["gap_model_minus_xgb"])

    # Selection-era large-sample diagnostics quoted in the 13 Aug
    # deliberation (PAPER_STATE): within-country all-features XGB.
    c = pd.read_csv(REPO.parent / "analysis" / "feature_importance_xgb"
                    / "cell_diagnostics.csv")
    ok = c[c["status"] == "ok"]
    check("diag cells", 132, len(ok), 0.5)
    check("diag mean n_used", 2200, ok["n_used"].mean(), 1.0)
    check("diag mean majority", 0.5256, ok["majority_baseline"].mean())
    check("diag mean model_acc", 0.6238, ok["model_acc"].mean())
    check("diag mean raw lift", 0.0982,
          (ok["model_acc"] - ok["majority_baseline"]).mean())
    check("diag mean norm_lift", 0.1791, ok["norm_lift"].mean())
    check("diag frac lift<=0", 0.212, (ok["norm_lift"] <= 0).mean(), 5e-4)

    # XGB-CEILING-FULL: the properly-fit bracket (13 Aug, second landing).
    f = pd.read_csv(AN / "xgb_full_anchor_summary.csv").set_index("regime")
    for regime, n, acc, nacc in (
            ("grouped", 704, 0.6463, 0.4911),
            ("within", 734, 0.6771, 0.5369),
            ("within_prompt24", 734, 0.6362, 0.4654)):
        check(f"full {regime} n", n, f.loc[regime, "n"], 0.5)
        check(f"full {regime} acc", acc, f.loc[regime, "acc"])
        check(f"full {regime} norm_acc", nacc, f.loc[regime, "norm_acc"])

    fd = pd.read_csv(AN / "xgb_full_dissenter_split.csv")
    for regime, modal, diss in (
            ("grouped", 0.7809, 0.4038),
            ("within", 0.7829, 0.4489),
            ("within_prompt24", 0.7449, 0.3929)):
        g = fd[fd["regime"] == regime]
        check(f"full {regime} modal", modal, g["acc_modal"].mean())
        check(f"full {regime} dissenter", diss, g["acc_dissenter"].mean())

    ld = pd.read_csv(AN / "xgb_llm_dissenter_split.csv")
    check("llm split targets", 25, len(ld), 0.5)
    check("llm modal", 0.6487, ld["acc_modal"].mean())
    check("llm dissenter", 0.3736, ld["acc_dissenter"].mean())

    ladder_dump = AN / "xgb_ladder_dump.jsonl"
    n_lines = sum(1 for _ in ladder_dump.open(encoding="utf-8"))
    check("ladder dump rows", 13652, n_lines, 0.5)
    n_lines = sum(1 for _ in (AN / "xgb_readout_dump.jsonl").open(
        encoding="utf-8"))
    check("readout dump rows", 4545, n_lines, 0.5)

    for m in bad:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures over {n_checked} checked values")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
