"""Verify every reported number against its source CSV (never by eye)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol


def verify_numbers(
    reported: Mapping[str, float],
    source_csv: Union[str, Path],
    *,
    key_col: str = "metric",
    value_col: str = "value",
    tol: float = 1e-6,
) -> List[str]:
    """Compare a dict of reported numbers to rows in a source CSV.

    Returns a list of mismatch messages (empty = all verified).
    """
    df = pd.read_csv(source_csv)
    if key_col not in df.columns or value_col not in df.columns:
        return [f"source CSV missing columns {key_col!r}/{value_col!r}"]
    src = {str(r[key_col]): float(r[value_col]) for _, r in df.iterrows()}
    mismatches: List[str] = []
    for name, val in reported.items():
        if name not in src:
            mismatches.append(f"missing in source: {name}={val}")
            continue
        if not _close(float(val), src[name], tol):
            mismatches.append(
                f"mismatch {name}: reported={val} source={src[name]}")
    return mismatches


def verify_table_against_csv(
    table_rows: Iterable[Mapping[str, Any]],
    source_csv: Union[str, Path],
    *,
    join_keys: Tuple[str, ...],
    value_cols: Tuple[str, ...],
    tol: float = 1e-6,
) -> List[str]:
    """Row-wise join of an in-memory table against a source CSV."""
    src = pd.read_csv(source_csv)
    mismatches: List[str] = []
    for row in table_rows:
        mask = pd.Series(True, index=src.index)
        for k in join_keys:
            mask &= src[k].astype(str) == str(row[k])
        hits = src[mask]
        if hits.empty:
            mismatches.append(f"no source row for { {k: row[k] for k in join_keys} }")
            continue
        if len(hits) > 1:
            mismatches.append(f"ambiguous source rows for { {k: row[k] for k in join_keys} }")
            continue
        src_row = hits.iloc[0]
        for col in value_cols:
            a, b = float(row[col]), float(src_row[col])
            if not _close(a, b, tol):
                mismatches.append(
                    f"mismatch {join_keys}={tuple(row[k] for k in join_keys)} "
                    f"{col}: reported={a} source={b}")
    return mismatches
