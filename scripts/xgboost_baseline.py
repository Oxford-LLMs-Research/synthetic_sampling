"""
XGBoost baseline for EMNLP revision.

Trains one XGBoost classifier per (survey, target_question, profile_type) using
exactly the same feature subsets shown to LLMs in the main evaluation. XGBoost's
native NaN handling means missing features (those not in a respondent's sampled
profile) are treated as a learnable missing direction rather than imputed.

Outputs: analysis/xgboost_baseline/results.csv

Usage:
    python xgboost_baseline.py                        # all surveys
    python xgboost_baseline.py --surveys wvs afrobarometer
    python xgboost_baseline.py --dry-run              # wvs only, first 10 targets
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # synthetic_sampling/
JSONL_DIR = ROOT / "synthetic_sampling/outputs/main_data_smaller_20_jan_26/main_data"
META_DIR = ROOT / "synthetic_sampling/src/synthetic_sampling/profiles/metadata/pulled_metadata"
OUT_DIR = ROOT / "analysis/xgboost_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURVEY_CONFIG = {
    "wvs":             ("wvs_instances.jsonl",            "pulled_metadata_wvs.json"),
    "afrobarometer":   ("afrobarometer_instances.jsonl",  "pulled_metadata_afrobarometer.json"),
    "arabbarometer":   ("arabbarometer_instances.jsonl",  "pulled_metadata_arabbarometer.json"),
    "asianbarometer":  ("asianbarometer_instances.jsonl", "pulled_metadata_asianbarometer.json"),
    "ess_wave_10":     ("ess_wave_10_instances.jsonl",    "pulled_metadata_ess10.json"),
    "ess_wave_11":     ("ess_wave_11_instances.jsonl",    "pulled_metadata_ess11.json"),
    "latinobarometer": ("latinobarometer_instances.jsonl","pulled_metadata_latinobarometer.json"),
}

N_SPLITS = 5
MIN_SAMPLES = 20      # skip (target, profile_type) cells with fewer respondents
MIN_CLASSES = 2       # skip single-class targets


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(survey: str):
    """Return (text_to_varcode, varcode_to_valmap) for a survey.

    text_to_varcode:   {question_text: var_code}
    varcode_to_valmap: {var_code: {answer_text: numeric_code}}  (positive codes only)
    """
    _, meta_file = SURVEY_CONFIG[survey]
    path = META_DIR / meta_file
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        meta = json.load(f)

    text_to_varcode = {}
    varcode_to_valmap = {}

    for section in meta.values():
        for var_code, var_info in section.items():
            qt = var_info.get("question", "")
            text_to_varcode[qt] = var_code

            label_to_code = {}
            values = var_info.get("values", {})
            if isinstance(values, dict):
                for code_str, label in values.items():
                    try:
                        code_int = int(code_str)
                        if code_int > 0:
                            label_to_code[label] = code_int
                    except ValueError:
                        pass
            varcode_to_valmap[var_code] = label_to_code

    return text_to_varcode, varcode_to_valmap


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(survey: str):
    """Load all instances for a survey into a list of dicts."""
    jsonl_file, _ = SURVEY_CONFIG[survey]
    path = JSONL_DIR / jsonl_file
    instances = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            instances.append(json.loads(line))
    return instances


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(instances, text_to_varcode, varcode_to_valmap):
    """Build X (feature matrix) and y (target labels) for a set of instances.

    All instances are assumed to share the same (target_code, profile_type).
    Returns:
        X: pd.DataFrame, rows=respondents, cols=union of var_codes in profiles
        y_labels: list of str (raw answer text for each respondent)
        n_options: int (number of valid answer options for this question)
    """
    rows = []
    y_labels = []
    n_options = None

    for inst in instances:
        if n_options is None:
            # Distinct labels, not option slots: see compute_normalized_accuracy.py.
            n_options = len(set(inst["options"]))

        row = {}
        for question_text, answer_text in inst["questions"].items():
            var_code = text_to_varcode.get(question_text)
            if var_code is None:
                continue
            val_map = varcode_to_valmap.get(var_code, {})
            numeric = val_map.get(answer_text, np.nan)
            row[var_code] = numeric

        rows.append(row)
        y_labels.append(inst["answer"])

    X = pd.DataFrame(rows)   # NaN for any var_code not in a given respondent's row
    return X, y_labels, n_options


# ---------------------------------------------------------------------------
# XGBoost cross-validation
# ---------------------------------------------------------------------------

def make_clf():
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
    )


def run_cv(X: pd.DataFrame, y_int: np.ndarray, n_splits: int = 5):
    """5-fold stratified CV; returns mean accuracy and macro F1."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, f1_scores = [], []

    for train_idx, test_idx in skf.split(X, y_int):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # Skip folds where train set lacks some classes (rare edge case)
        if len(np.unique(y_train)) < MIN_CLASSES:
            continue

        clf = make_clf()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc_scores.append((preds == y_test).mean())
        f1_scores.append(f1_score(y_test, preds, average="macro", zero_division=0))

    if not acc_scores:
        return np.nan, np.nan
    return float(np.mean(acc_scores)), float(np.mean(f1_scores))


def normalized_accuracy(acc: float, n_options: int) -> float:
    """(acc - 1/M) / (1 - 1/M) — maps random chance to 0."""
    if n_options <= 1:
        return np.nan
    chance = 1.0 / n_options
    if chance >= 1.0:
        return np.nan
    return (acc - chance) / (1.0 - chance)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_survey(survey: str, dry_run: bool = False) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Survey: {survey}")

    text_to_varcode, varcode_to_valmap = load_metadata(survey)
    print(f"  Metadata loaded: {len(text_to_varcode)} variables")

    instances = load_jsonl(survey)
    print(f"  Instances loaded: {len(instances):,}")

    # Group by (target_code, profile_type)
    groups = defaultdict(list)
    for inst in instances:
        groups[(inst["target_code"], inst["profile_type"])].append(inst)

    target_codes = sorted({tc for tc, _ in groups})
    if dry_run:
        target_codes = target_codes[:10]
        print(f"  [dry-run] Limiting to {len(target_codes)} targets")

    profile_types = ["s3m2", "s4m3", "s6m4"]
    results = []

    for target_code in target_codes:
        for profile_type in profile_types:
            key = (target_code, profile_type)
            cell_instances = groups.get(key, [])

            if len(cell_instances) < MIN_SAMPLES:
                continue

            X, y_labels, n_options = build_feature_matrix(
                cell_instances, text_to_varcode, varcode_to_valmap
            )

            if X.empty or X.shape[1] == 0:
                continue

            # Drop columns that are entirely NaN (feature never observed with a valid code)
            X = X.dropna(axis=1, how="all")
            if X.shape[1] == 0:
                continue

            # Encode target
            le = LabelEncoder()
            y_int = le.fit_transform(y_labels)

            # Drop classes with fewer samples than n_splits — guarantees each class
            # appears in at least one training fold and avoids XGBoost class-index errors
            class_counts_arr = np.bincount(y_int)
            rare_classes = np.where(class_counts_arr < N_SPLITS)[0]
            if len(rare_classes) > 0:
                keep_mask = ~np.isin(y_int, rare_classes)
                if keep_mask.sum() < MIN_SAMPLES:
                    continue
                X = X.iloc[keep_mask].reset_index(drop=True)
                y_labels = [y for y, k in zip(y_labels, keep_mask) if k]
                le = LabelEncoder()
                y_int = le.fit_transform(y_labels)

            n_classes = len(le.classes_)
            if n_classes < MIN_CLASSES:
                continue

            # Majority class accuracy
            class_counts = np.bincount(y_int)
            majority_acc = class_counts.max() / len(y_int)

            xgb_acc, xgb_f1 = run_cv(X, y_int, N_SPLITS)

            results.append({
                "survey": survey,
                "target_code": target_code,
                "profile_type": profile_type,
                "n_respondents": len(cell_instances),
                "n_features_union": X.shape[1],
                "n_classes": n_classes,
                "n_options": n_options,
                "majority_acc": round(majority_acc, 4),
                "xgb_acc": round(xgb_acc, 4) if not np.isnan(xgb_acc) else np.nan,
                "xgb_macro_f1": round(xgb_f1, 4) if not np.isnan(xgb_f1) else np.nan,
                "majority_norm_acc": round(normalized_accuracy(majority_acc, n_options), 4),
                "xgb_norm_acc": round(normalized_accuracy(xgb_acc, n_options), 4) if not np.isnan(xgb_acc) else np.nan,
            })

        sys.stdout.write(f"\r  Processed {target_code} ({len(results)} cells so far)")
        sys.stdout.flush()

    print(f"\n  Done: {len(results)} (target, profile_type) cells")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surveys", nargs="+", choices=list(SURVEY_CONFIG.keys()),
                        default=list(SURVEY_CONFIG.keys()))
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output filename (relative to analysis/xgboost_baseline/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only first 10 targets per survey (for testing)")
    args = parser.parse_args()

    all_results = []
    for survey in args.surveys:
        rows = process_survey(survey, dry_run=args.dry_run)
        all_results.extend(rows)

    df = pd.DataFrame(all_results)
    out_path = OUT_DIR / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Quick aggregate summary per profile_type
    print("\nAggregate mean accuracy by profile_type:")
    print(df.groupby("profile_type")[["majority_acc", "xgb_acc", "majority_norm_acc", "xgb_norm_acc"]].mean().round(3))


if __name__ == "__main__":
    main()
