"""Figure: the dissenter penalty, the mode-following signature.

Panel (a) One row per model. Filled dot = normalized accuracy on respondents
whose answer matches their question x country modal answer; open dot =
accuracy on dissenters. The connecting bar is the penalty. Sorted by modal
accuracy, so the widening of the bar down the ranking is visible: the better a
model is at this task, the larger its penalty.

Panel (b) The mechanism. As the modal answer's share of a cell rises, accuracy
on modal respondents climbs steeply while accuracy on dissenters is flat. A
predictor that tracked individuals would improve on both; a predictor that
tracks the mode improves only on the people who agree with it.

Source: analysis/equity_audit/dissenter_penalty_by_{model,share}.csv
Output: analysis/figures/emnlp_revision/figure_dissenter_penalty.pdf, copied
into the AAAI paper tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
EQ = ROOT / "analysis" / "equity_audit"
OUT_DIR = ROOT / "analysis" / "figures" / "emnlp_revision"
AAAI_FIGS = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "qwen3-32b": "Qwen 3 32B", "deepseek": "DeepSeek-V3",
    "olmo3_7b_dpo": "OLMo 3 7B inst.", "qwen3-4b": "Qwen 3 4B",
    "gemma3-27b": "Gemma 3 27B", "olmo3_7b_base": "OLMo 3 7B base",
    "llama3.1_8b_instruct": "Llama 3.1 8B inst.", "olmo3_32b_dpo": "OLMo 3 32B inst.",
    "gpt-oss": "GPT-OSS 120B", "llama3.1_70b_instruct": "Llama 3.1 70B inst.",
    "olmo3_32b_base": "OLMo 3 32B base", "llama3.1_8b_base": "Llama 3.1 8B base",
    "llama3.1_70b_base": "Llama 3.1 70B base",
}

FAMILY = {
    "deepseek": "DeepSeek", "gpt-oss": "GPT-OSS", "gemma3-27b": "Gemma",
    "llama3.1_70b_base": "Llama", "llama3.1_70b_instruct": "Llama",
    "llama3.1_8b_base": "Llama", "llama3.1_8b_instruct": "Llama",
    "olmo3_32b_base": "OLMo", "olmo3_32b_dpo": "OLMo",
    "olmo3_7b_base": "OLMo", "olmo3_7b_dpo": "OLMo",
    "qwen3-32b": "Qwen", "qwen3-4b": "Qwen",
}

FAMILY_COLORS = {
    "DeepSeek": "#6A994E", "GPT-OSS": "#C73E1D", "Gemma": "#BC4749",
    "Llama": "#2E86AB", "OLMo": "#A23B72", "Qwen": "#F18F01",
}

MODAL_C = "#2E5090"
DISS_C = "#B0413E"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(EQ / "dissenter_penalty_by_model.csv")
    m["label"] = m["model"].map(DISPLAY)
    m["color"] = m["model"].map(FAMILY).map(FAMILY_COLORS)
    m = m.sort_values("norm_modal").reset_index(drop=True)

    s = pd.read_csv(EQ / "dissenter_penalty_by_share.csv", header=[0, 1], index_col=0)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return m, s


def main() -> None:
    m, s = load()

    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "axes.linewidth": 0.6, "lines.linewidth": 0.8,
        "figure.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- panel (a): dumbbell -------------------------------------------
    y = range(len(m))
    for i, r in m.iterrows():
        ax1.plot([r["norm_dissenter"], r["norm_modal"]], [i, i],
                 color=r["color"], alpha=0.55, linewidth=2.0, zorder=2,
                 solid_capstyle="round")
        ax1.scatter(r["norm_dissenter"], i, s=26, facecolor="white",
                    edgecolor=r["color"], linewidth=0.9, zorder=3)
        ax1.scatter(r["norm_modal"], i, s=34, color=r["color"],
                    edgecolor="none", zorder=4)

    ax1.axvline(0, color="gray", linestyle=":", linewidth=0.7, zorder=1)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(m["label"], fontsize=7.5)
    ax1.set_xlabel("Normalized accuracy", fontsize=9)
    ax1.set_title("(a) Majority view versus dissent", fontsize=9, loc="left")
    ax1.grid(axis="x", linestyle="--", alpha=0.22, linewidth=0.4)
    ax1.set_axisbelow(True)
    ax1.set_ylim(-0.8, len(m) - 0.2)

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                      markerfacecolor="#555555", markeredgecolor="none",
                      label="Agrees with local majority"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                      markerfacecolor="white", markeredgecolor="#555555",
                      markeredgewidth=0.9, label="Dissents (45% of people)"),
    ]
    ax1.legend(handles=handles, fontsize=7.2, loc="lower right",
               frameon=False, handletextpad=0.4, borderpad=0.2)

    # ---- panel (b): mechanism ------------------------------------------
    order = ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"]
    s = s.reindex(order)
    x = range(len(order))
    ax2.plot(x, s["norm_True"], marker="o", markersize=4.5, color=MODAL_C,
             linewidth=1.4, label="Agrees with majority")
    ax2.plot(x, s["norm_False"], marker="o", markersize=4.5, color=DISS_C,
             linewidth=1.4, markerfacecolor="white", markeredgewidth=0.9,
             label="Dissents")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(order, fontsize=8)
    ax2.set_xlabel("Modal answer's share of the country", fontsize=9)
    ax2.set_ylabel("Normalized accuracy", fontsize=9)
    ax2.set_title("(b) Where the gain comes from", fontsize=9, loc="left")
    ax2.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.5, loc="upper left", frameon=False,
               handletextpad=0.5, borderpad=0.2)

    fig.tight_layout(pad=0.6, w_pad=1.6)

    pdf = OUT_DIR / "figure_dissenter_penalty.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf}")

    if AAAI_FIGS.exists():
        shutil.copy(pdf, AAAI_FIGS / pdf.name)
        print(f"Copied to {AAAI_FIGS}")

    print("\npanel (b) values:")
    print(s[["norm_True", "norm_False", "n_True", "n_False"]].round(3).to_string())


if __name__ == "__main__":
    main()
