"""Option-set and metadata hygiene hooks.

Phase 0 fills the bodies of these functions after the census review. Until then
they are identity / empty so the remade loader path is wired for the fixes
without inventing unverified exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class HygieneFinding:
    """One option-set or metadata defect found by a scan."""

    survey_id: str
    var_code: str
    kind: str  # typo_duplicate | residual_missingness | mojibake | bare_numeric | other
    detail: str
    options: Tuple[str, ...] = ()


@dataclass
class HygieneReport:
    findings: List[HygieneFinding] = field(default_factory=list)

    @property
    def n_findings(self) -> int:
        return len(self.findings)


# Known defects from the instrument work (5 Aug). Phase 0 decides whether each
# becomes an exclusion, a rewrite, or a documented keep.
KNOWN_DEFECTS: Tuple[HygieneFinding, ...] = (
    HygieneFinding(
        survey_id="wvs",
        var_code="Q149",
        kind="typo_duplicate",
        detail='Option set contains typo duplicate "None at all" / "None et all"',
    ),
    HygieneFinding(
        survey_id="wvs",
        var_code="set001",
        kind="bare_numeric",
        detail="Option set carries residual missingness code 94.0",
    ),
)


def scan_option_sets(
    metadata: Dict[str, Any],
    survey_id: str = "",
) -> HygieneReport:
    """Scripted scan for option-set defects. Phase 0 extends the heuristics."""
    findings: List[HygieneFinding] = []
    root = metadata.get("variables") or metadata
    if not isinstance(root, dict):
        return HygieneReport(findings=findings)

    # Metadata is usually section -> var_code -> info; also accept flat var maps.
    entries: List[Tuple[str, dict]] = []
    for key, val in root.items():
        if not isinstance(val, dict):
            continue
        if any(
            isinstance(v, dict)
            and ("values" in v or "value_labels" in v or "question" in v)
            for v in val.values()
        ):
            for var_code, meta in val.items():
                if isinstance(meta, dict):
                    entries.append((str(var_code), meta))
        else:
            entries.append((str(key), val))

    for var_code, meta in entries:
        values = meta.get("values") or meta.get("value_labels") or {}
        if isinstance(values, dict):
            labels = [str(v) for v in values.values()]
        elif isinstance(values, list):
            labels = [str(v) for v in values]
        else:
            continue

        for lab in labels:
            stripped = lab.strip()
            if stripped.replace(".", "", 1).isdigit() and "." in stripped:
                findings.append(HygieneFinding(
                    survey_id=survey_id,
                    var_code=var_code,
                    kind="bare_numeric",
                    detail=f"Bare-numeric option label: {lab!r}",
                    options=tuple(labels),
                ))
                break

        norms = [
            (lab, "".join(c for c in lab.lower() if c.isalnum()))
            for lab in labels
        ]
        for i, (lab_a, na) in enumerate(norms):
            for lab_b, nb in norms[i + 1:]:
                if not na or not nb or na == nb:
                    continue
                if abs(len(na) - len(nb)) > 2 or len(na) < 6:
                    continue
                diffs = sum(1 for x, y in zip(na, nb) if x != y)
                diffs += abs(len(na) - len(nb))
                if diffs <= 2:
                    findings.append(HygieneFinding(
                        survey_id=survey_id,
                        var_code=var_code,
                        kind="typo_duplicate",
                        detail=f"Near-duplicate options: {lab_a!r} / {lab_b!r}",
                        options=tuple(labels),
                    ))
                    break

    return HygieneReport(findings=findings)


def filter_missingness_codes(
    options: Sequence[str],
    codes_to_drop: Optional[Sequence[str]] = None,
) -> List[str]:
    """Drop residual missingness labels from an option list.

    Default drop set is empty until Phase 0 census locks the codes. Callers
    that already know a code (e.g. '94.0') may pass it explicitly.
    """
    drop = set(codes_to_drop or ())
    return [o for o in options if o not in drop]


def attach_interview_date_field(
    respondent_row: Dict[str, Any],
    *,
    date_col: Optional[str],
    date_format: Optional[str] = None,
) -> Optional[str]:
    """Carry interview date from the microdata row when the column is configured.

    Returns the raw string (or None). Parsing/normalisation is Phase 0 work;
    this only ensures the field is not dropped at load time.
    """
    if not date_col:
        return None
    val = respondent_row.get(date_col)
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return None
    return str(val)


def apply_harmonisation(
    metadata: Dict[str, Any],
    survey_id: str = "",
) -> Dict[str, Any]:
    """Return metadata after hygiene rewrites. Identity until Phase 0 lands."""
    _ = survey_id
    return metadata
