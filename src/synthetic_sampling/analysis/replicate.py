"""Replicate and serving-control summaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Cross-serving floor measured on the paper control: same model, same
# instances, different serving stack. Agreement rates are read against this,
# not against 1.0. Within-serving replicate ceiling is near 100%.
CROSS_SERVING_FLOOR = 0.644


def _predicted(rec: Dict[str, Any]) -> Optional[str]:
    if "error" in rec:
        return None
    return rec.get("predicted")


def pair_agreement(
    rows: Iterable[dict],
    arm_a: str,
    arm_b: str,
    set_a: str = "original",
    set_b: str = "original",
) -> Tuple[float, int]:
    """Fraction of instances where two arm keys agree on the predicted option."""
    key_a = f"{set_a}|{arm_a}"
    key_b = f"{set_b}|{arm_b}"
    agree = total = 0
    for r in rows:
        results = r.get("results") or {}
        if key_a not in results or key_b not in results:
            continue
        pa, pb = _predicted(results[key_a]), _predicted(results[key_b])
        if pa is None or pb is None:
            continue
        total += 1
        if pa == pb:
            agree += 1
    return (agree / total if total else float("nan"), total)


def replicate_agreement(
    rows: Iterable[dict],
    arm: str = "label_num",
) -> Tuple[float, int]:
    """Agreement between original and original_replicate for one arm."""
    return pair_agreement(
        rows, arm, arm, set_a="original", set_b="original_replicate")


def summarize_controls(
    rows: List[dict],
    arm: str = "label_num",
) -> dict:
    """Replicate ceiling vs cross-serving floor for reporting."""
    rate, n = replicate_agreement(rows, arm=arm)
    return {
        "arm": arm,
        "replicate_agreement": rate,
        "n_replicate_pairs": n,
        "cross_serving_floor": CROSS_SERVING_FLOOR,
        "note": (
            "Read agreement against 64.4% (cross-serving), not 100%. "
            "Replicate is the within-serving ceiling."
        ),
    }
