from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

TARGETS: List[str] = ["Q47", "Q57", "Q199", "Q235", "Q164"]
TARGET_TITLES: Dict[str, str] = {
    "Q47": "Q47",
    "Q57": "Q57",
    "Q199": "Q199",
    "Q235": "Q235",
    "Q164": "Q164",
}
COUNTRY_ORDER: List[str] = ["Germany", "Nigeria", "Japan", "Brazil", "Egypt"]


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_PREFIX = "phase0a"

METADATA_PATH = SCRIPT_DIR.parents[1] / "src/synthetic_sampling/profiles/metadata/wvs/pulled_metadata_wvs.json"
OUTPUT_FILE = "phase0a_summary_compact.md"





def format_num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def load_metadata_labels(metadata_path: Path) -> Dict[str, str]:
    with metadata_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    labels: Dict[str, str] = {}
    for section_values in raw.values():
        if not isinstance(section_values, dict):
            continue
        for variable, variable_meta in section_values.items():
            if not isinstance(variable_meta, dict):
                continue
            question = str(variable_meta.get("question", "")).strip()
            description = str(variable_meta.get("description", "")).strip()
            labels[variable] = description or question or variable

    return labels


def feature_label(feature_code: str, labels: Dict[str, str]) -> str:
    return f"{feature_code} - {labels.get(feature_code, feature_code)}"


def build_markdown(
    importance_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    labels: Dict[str, str],
) -> str:
    sections: List[str] = []

    for target in TARGETS:
        cells_t = cells_df[cells_df["target_variable"] == target].copy()
        rho_t = rho_df[rho_df["target_variable"] == target]
        imp_t = importance_df[importance_df["target_variable"] == target].copy()

        acc_mean = float(cells_t["cv_accuracy_mean"].mean()) if len(cells_t) else np.nan
        baseline_mean = float(cells_t["majority_baseline"].mean()) if len(cells_t) else np.nan
        delta = acc_mean - baseline_mean if pd.notna(acc_mean) and pd.notna(baseline_mean) else np.nan
        mean_rho = float(rho_t["mean_pairwise_rho"].iloc[0]) if len(rho_t) else np.nan

        pivot = imp_t.pivot_table(
            index="feature_variable",
            columns="country_name",
            values="importance_mean",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=[c for c in COUNTRY_ORDER if c in pivot.columns])

        universal_candidates = []
        if not pivot.empty:
            for feat in pivot.index:
                top_quartile_hits = 0
                for country in pivot.columns:
                    series = pivot[country].dropna().sort_values(ascending=False)
                    if len(series) == 0:
                        continue
                    top_k = max(1, int(np.ceil(len(series) * 0.25)))
                    if feat in series.head(top_k).index:
                        top_quartile_hits += 1
                if top_quartile_hits >= 4:
                    universal_candidates.append((feat, top_quartile_hits))

        universal_features = [
            feature_label(feat, labels)
            for feat, _ in sorted(universal_candidates, key=lambda x: (-x[1], x[0]))[:6]
        ]

        specific_candidates = []
        if not pivot.empty:
            for feat in pivot.index:
                row = pivot.loc[feat].dropna()
                if len(row) < 2:
                    continue
                top_country = row.idxmax()
                top_val = float(row.max())
                gap = float(row.max() - row.median())
                specific_candidates.append((gap, feat, top_country, top_val))

        specific_candidates = sorted(specific_candidates, reverse=True)[:6]
        specific_features = [
            (
                f"{feature_label(feat, labels)} "
                f"({country}; importance={top_val:.3f}, gap={gap:.3f})"
            )
            for gap, feat, country, top_val in specific_candidates
        ]

        if specific_candidates:
            gap, feat, country, top_val = specific_candidates[0]
            noteworthy = (
                f"{feature_label(feat, labels)} peaks most in {country} "
                f"(importance={top_val:.3f}, gap vs median={gap:.3f})."
            )
        else:
            noteworthy = "NA"

        paragraph_1 = (
            f"CV accuracy (mean across countries) is **{format_num(acc_mean)}**, "
            f"compared with a majority-class baseline of **{format_num(baseline_mean)}** "
            f"(delta **{format_num(delta)}**). Mean pairwise Spearman rho across countries "
            f"is **{format_num(mean_rho)}**."
        )

        paragraph_2 = (
            "Universally important features: "
            + (
                "; ".join(universal_features)
                if universal_features
                else "none identified under current threshold"
            )
            + ". Country-specific features (largest country concentration): "
            + ("; ".join(specific_features) if specific_features else "none")
            + f" Noteworthy pattern: {noteworthy}"
        )

        sections.append(
            f"## {TARGET_TITLES.get(target, target)}\n\n"
            f"{paragraph_1}\n\n"
            f"{paragraph_2}\n"
        )

    global_acc = float(cells_df["cv_accuracy_mean"].mean())
    global_baseline = float(cells_df["majority_baseline"].mean())
    global_delta = float((cells_df["cv_accuracy_mean"] - cells_df["majority_baseline"]).mean())
    global_rho = float(rho_df["mean_pairwise_rho"].mean())

    total_cells = int(len(cells_df))
    total_skipped = int(cells_df["skipped"].sum()) if "skipped" in cells_df.columns else 0
    total_low_n = int(cells_df["low_n_flag"].sum()) if "low_n_flag" in cells_df.columns else 0
    total_sparse = int(cells_df["sparse_class_flag"].sum()) if "sparse_class_flag" in cells_df.columns else 0

    overall_paragraph = (
        f"Across all targets, average CV accuracy is **{format_num(global_acc)}** "
        f"vs average majority baseline **{format_num(global_baseline)}** "
        f"(average delta **{format_num(global_delta)}**), and average mean pairwise "
        f"Spearman rho is **{format_num(global_rho)}**. Completed cells: **{total_cells}**, "
        f"skipped cells: **{total_skipped}**, low-n flags: **{total_low_n}**, "
        f"sparse-class flags: **{total_sparse}**."
    )

    chunks: List[str] = []
    chunks.append("# Phase 0a Summary (Compact)\n")
    chunks.append(
        "This file compiles key output information from the generated Phase 0a CSV/PNG artifacts.\n"
    )
    chunks.extend(sections)
    chunks.append("## Overall Verdict Inputs\n")
    chunks.append(overall_paragraph + "\n")

    return "\n".join(chunks)


# Main execution
base_dir = Path(BASE_DIR)
metadata_path = Path(METADATA_PATH)

importance_path = base_dir / f"{OUTPUT_PREFIX}_importance_table.csv"
cell_path = base_dir / f"{OUTPUT_PREFIX}_cell_diagnostics.csv"
rho_path = base_dir / f"{OUTPUT_PREFIX}_mean_pairwise_rho.csv"

importance_df = pd.read_csv(importance_path)
cells_df = pd.read_csv(cell_path)
rho_df = pd.read_csv(rho_path)

labels = load_metadata_labels(metadata_path)
markdown = build_markdown(importance_df, cells_df, rho_df, labels)

output_path = base_dir / OUTPUT_FILE
output_path.write_text(markdown, encoding="utf-8")

print(f"Saved summary markdown: {output_path}")

