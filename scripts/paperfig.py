"""Shared figure geometry and style for the AAAI submission.

Every figure in the paper was previously drawn at slide dimensions (10x8,
16x5.5) and then shrunk by \\includegraphics to fit a 7-inch two-column page.
The scale factors ran from 0.35 to 0.70, so 9pt labels rendered between 3.2pt
and 6.3pt against 10pt body text, and a figure* float that carried a
half-width image left the other half of the page blank.

This module fixes the physical size at draw time instead. Figures are drawn at
exactly the width they will occupy, saved without a tight bounding box so the
PDF media box equals the requested figsize, and included at width=\\columnwidth
or width=\\textwidth for a scale factor of exactly 1.0. Font sizes set here are
therefore the sizes that appear on the printed page.

Geometry is read from aaai2027.sty:
    \\textwidth   7.0in
    \\columnsep   0.375in
    => columnwidth = (7.0 - 0.375) / 2 = 3.3125in
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- physical geometry, inches -------------------------------------------
COL = 3.3125   # \columnwidth  -> use with width=\columnwidth
FULL = 7.0     # \textwidth    -> use with width=\textwidth (figure*)

ROOT = Path(r"C:\Users\murrn\cursor\synthetic_sampling")
OUT_DIR = ROOT / "analysis" / "figures" / "emnlp_revision"
AAAI_FIGS = Path(r"C:\Users\murrn\cursor\synthetic_sampling_aaai\emnlp\figures")

# --- palette --------------------------------------------------------------
FAMILY_COLORS = {
    "DeepSeek": "#6A994E",
    "GPT-OSS": "#C73E1D",
    "Gemma": "#5C7A8A",   # cool slate; was red-brown and collided with GPT-OSS
    "Llama": "#2E86AB",
    "OLMo": "#A23B72",
    "Qwen": "#F18F01",
}

# Semantic colors, used consistently across figures: the group a model tracks
# versus the group it abandons.
MODAL_C = "#2E5090"    # respondents who hold their country's majority view
DISS_C = "#C0392B"     # dissenters
BASELINE_C = "#111111"
XGB_C = "#5A6B7C"
GRID_A = 0.20

# --- legend convention ----------------------------------------------------
# One rule for every figure in both documents, in priority order:
#
#   1. No key, if the axis labels already carry the distinction (Figure 1: every
#      open marker sits on a row whose own label ends in "base").
#   2. Otherwise a key inside the axes, in a white box with a thin border --
#      inside, because a key on its own line under a figure spends a whole line
#      of page height on two markers; boxed, because an unboxed key placed among
#      the data reads as more data. Widening an axis past the data to open a
#      space for it is fair game.
#   3. Only where no interior space exists, a single row below the axes. This is
#      the appendix figures whose keys run to six or eight entries.
#
# Placement (loc) is per figure -- it has to go where that figure is empty --
# but everything else comes from here so the boxes cannot drift apart.
LEGEND_BOX = dict(
    frameon=True, framealpha=1.0, facecolor="white", edgecolor="#C9CDD3",
    fancybox=False, borderpad=0.4, labelspacing=0.3, handletextpad=0.4,
    fontsize=9,
)


def boxed_legend(ax, handles, loc, **kw):
    """The paper's only legend style. Extra kwargs override LEGEND_BOX."""
    opts = dict(LEGEND_BOX)
    opts.update(kw)
    leg = ax.legend(handles=handles, loc=loc, **opts)
    leg.get_frame().set_linewidth(0.5)
    return leg


MODEL_META = {
    "llama3.1_8b_base":      ("Llama 3.1 8B base",   "Llama",    False),
    "llama3.1_8b_instruct":  ("Llama 3.1 8B inst.",  "Llama",    True),
    "llama3.1_70b_base":     ("Llama 3.1 70B base",  "Llama",    False),
    "llama3.1_70b_instruct": ("Llama 3.1 70B inst.", "Llama",    True),
    "olmo3_7b_base":         ("OLMo 3 7B base",      "OLMo",     False),
    "olmo3_7b_dpo":          ("OLMo 3 7B inst.",     "OLMo",     True),
    "olmo3_32b_base":        ("OLMo 3 32B base",     "OLMo",     False),
    "olmo3_32b_dpo":         ("OLMo 3 32B inst.",    "OLMo",     True),
    "qwen3-4b":              ("Qwen 3 4B",           "Qwen",     True),
    "qwen3-32b":             ("Qwen 3 32B",          "Qwen",     True),
    "gpt-oss":               ("GPT-OSS 120B",        "GPT-OSS",  True),
    # The evaluated checkpoint is deepseek-v3p1-terminus, i.e. V3.1, not V3.
    "deepseek":              ("DeepSeek-V3.1",       "DeepSeek", True),
    "gemma3-27b":            ("Gemma 3 27B",         "Gemma",    True),
}


def use_style() -> None:
    """Font sizes here are final printed sizes: figures are never rescaled.

    AAAI-2027 requires that "labels and other text with the actual
    illustration must be at least nine-point type". Because every figure is
    drawn at its final width and included at scale 1.0, the sizes set here are
    what prints, so 9 is a hard floor rather than a preference. Nothing in any
    figure script may pass a smaller fontsize.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def save(fig, name: str, expect_width: float) -> Path:
    """Save at exact figsize and verify the PDF media box matches.

    No bbox_inches='tight': cropping changes the width, which would reintroduce
    an unpredictable scale factor at \\includegraphics time.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)

    from pypdf import PdfReader
    box = PdfReader(str(pdf)).pages[0].mediabox
    w, h = float(box.width) / 72, float(box.height) / 72
    if abs(w - expect_width) > 0.02:
        raise AssertionError(
            f"{name}: media box {w:.3f}in != expected {expect_width:.3f}in; "
            "figure would be rescaled by \\includegraphics")

    if AAAI_FIGS.exists():
        shutil.copy2(pdf, AAAI_FIGS / pdf.name)
    print(f"  {name}.pdf  {w:.3f} x {h:.3f} in  (scale 1.0)")
    return pdf
