"""Option-set and metadata hygiene, implementing the Phase 0 decisions.

The decision round of 11 Aug 2026 (PAPER_STATE, same date) turned the census
findings into six recorded decisions; this module is their implementation.
The pulled metadata files stay exactly what was pulled — every fix here is a
derived view applied at load time (`loaders._load_metadata` calls
`apply_harmonisation`), so the harmonised state is regenerable and never
hand-edited.

The decisions, in code:
1. Duplicate labels: keep the deliberate scale binning; option sets are
   built from UNIQUE labels (`dedupe_option_labels`).
2. `netustm` is archived in minutes; both waves' question wording is
   rewritten to say minutes (QUESTION_REWRITES).
3. A code with no label never enters a profile (`code_enters_profile`);
   verified hand labels are added for the big sentinels (LABEL_ADDITIONS)
   and routed via PROFILE_DROP_CODES where they mean "no answer".
4. Proper-noun list variables are exempt from typo review (census-side);
   the one true typo, WVS Q48 "None et all", is rewritten (LABEL_REWRITES).
5. Unmatched metadata tails: Afrobarometer's six are a case mismatch and
   are renamed to the file's casing (VAR_RENAMES); Arab's structural
   expansions and Latino's degenerate multi-part blocks are dropped
   (VAR_DROPS), each recoverable as noted in PAPER_STATE.
6. KNOWN_DEFECTS carries Q48 (the 5 Aug Q149 reference was stale).
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


# Known defects from the instrument work (5 Aug), corrected 11 Aug: the typo
# pair lives on Q48, not Q149 (census rediscovery; Q48 is also one of B3's
# five duplicate-label exclusions).
KNOWN_DEFECTS: Tuple[HygieneFinding, ...] = (
    HygieneFinding(
        survey_id="wvs",
        var_code="Q48",
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


# ---------------------------------------------------------------------------
# Decision tables (all verified against raw data / codebooks, 11 Aug 2026).

# Decision 2 + 4: question / label rewrites, per survey -> var.
QUESTION_REWRITES: Dict[str, Dict[str, str]] = {
    "ess_wave_10": {
        # Archived in minutes (max 1440); pulled wording said "how much time".
        "netustm": ("On a typical day, how many minutes do you spend using "
                    "the internet on a computer, tablet, smartphone, or "
                    "other device?"),
    },
    "ess_wave_11": {
        # Archived in minutes; pulled wording invented "hours".
        "netustm": ("On a typical day, about how many minutes do you spend "
                    "using the internet on any device (computer, tablet, or "
                    "smartphone) for work or personal use?"),
    },
}

# Decision 4: exact label-text rewrites, per survey -> var -> {old: new}.
LABEL_REWRITES: Dict[str, Dict[str, Dict[str, str]]] = {
    "wvs": {"Q48": {"None et all": "None at all"}},
}

# Decision 3: verified hand labels added to values maps, survey -> var -> map.
# WVS -4 = "Not asked" (WVS's own convention on its other variables); the 19
# vars below carry -4 in the microdata (84,314 values) without a label.
_WVS_NOT_ASKED_VARS = (
    "Q119", "Q219", "Q220", "Q221", "Q222", "Q224", "Q225", "Q226", "Q227",
    "Q228", "Q229", "Q230", "Q231", "Q232", "Q233", "Q234", "Q234A", "Q237",
    "Q276",
)
LABEL_ADDITIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "wvs": {v: {"-4": "Not asked"} for v in _WVS_NOT_ASKED_VARS},
    "ess_wave_11": {
        # ESS-11 codebook: 444444 "Not classifiable", 555555 "No second
        # ancestry" (substantive: it enters profiles).
        "anctrya2": {"444444": "Not classifiable",
                     "555555": "No second ancestry"},
    },
    "arabbarometer": {
        # Monthly household income: the pulled 98/99 labels never occur in
        # the raw amounts; 99999 (17%) is the real refuse/no-answer code.
        "Q1015": {"99999": "Refused to answer"},
    },
}

# Decision 3: hand-labelled codes that mean "no substantive answer" — the
# respondent's q:a line is dropped, same as an unlabeled code.
PROFILE_DROP_CODES: Dict[str, Dict[str, frozenset]] = {
    "wvs": {v: frozenset({"-4"}) for v in _WVS_NOT_ASKED_VARS},
    "arabbarometer": {"Q1015": frozenset({"99999"})},
}

# Decision 3: census-locked sentinel codes for CONTINUOUS variables (no
# values map); these never pass into a profile as a numeric answer.
CONTINUOUS_SENTINELS = frozenset({
    "97", "98", "99", "997", "998", "999", "9997", "9998", "9999",
    "99997", "99998", "99999", "999997", "999998", "999999",
    "444444", "555555", "666666", "888888", "6666", "7777", "8888",
    "-1", "-2", "-3", "-4", "-5", "-7", "-8", "-9",
})

# Decision 5: variable renames to the raw file's naming, survey -> {old: new}.
VAR_RENAMES: Dict[str, Dict[str, str]] = {
    "afrobarometer": {
        "Q45pt1": "Q45PT1", "Q45pt2": "Q45PT2", "Q45pt3": "Q45PT3",
        "Q45pt1OTHER": "Q45PT1OTHER", "Q45pt2OTHER": "Q45PT2OTHER",
        "Q45pt3OTHER": "Q45PT3OTHER",
    },
    # S17 holds YYYYMMDD birth dates; year is the first four digits.
    "latinobarometer": {"S17.C": "S17"},
}

# Decision 5: variables dropped from the usable set (no matching data column
# or degenerate pulled metadata; recoverability recorded in PAPER_STATE).
VAR_DROPS: Dict[str, frozenset] = {
    "arabbarometer": frozenset({
        "Q1012C_MOR", "Q1034", "QKUW34", "QGAZA1", "QGAZA5A", "QGAZA5A2",
        "QGAZA5B", "QGAZA5C", "Q201A_41_Gaza", "QKUW40", "QMOR7",
        "Q104A_1", "Q104A_2", "Q104B", "Q873", "Q881A", "Q881B",
        "Q884A", "Q884B", "Q622C_IRQ", "Q622E_IRQ", "Q629", "Q130",
        "Q412A", "Q432",
    }),
    "latinobarometer": frozenset(
        {"REEDUC.1"}
        | {f"P38CSN.{i}" for i in range(1, 8)}
        | {f"P57ST.{i}" for i in range(1, 9)}
        | {f"S14M.{i}" for i in range(1, 9)}
        | {"S14M.96"}
    ),
}


# ---------------------------------------------------------------------------
# Hooks used by option-set / profile construction.

def dedupe_option_labels(labels: Sequence[str]) -> List[str]:
    """Decision 1: option sets present unique labels, first occurrence order.

    The scale binning that maps several codes to one label is deliberate;
    under `label_num` the label text is the whole estimand, so duplicate
    option text is incoherent for the model and is collapsed here.
    """
    return list(dict.fromkeys(labels))


def code_enters_profile(
    survey_id: str,
    var_code: str,
    code: Any,
    values_map: Optional[Dict[str, str]],
) -> bool:
    """Decision 3: no bare or no-answer code ever reaches a prompt.

    Labeled variable: a code absent from the values map, or routed by
    PROFILE_DROP_CODES, drops that respondent's q:a line. Continuous
    variable (no values map): numeric answers pass except the census-locked
    sentinel list.
    """
    c = _norm_code(code)
    drops = PROFILE_DROP_CODES.get(survey_id, {}).get(var_code)
    if drops and c in drops:
        return False
    if values_map:
        return c in values_map
    return c not in CONTINUOUS_SENTINELS


def _norm_code(val: Any) -> str:
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


# ---------------------------------------------------------------------------

def scan_option_sets(
    metadata: Dict[str, Any],
    survey_id: str = "",
) -> HygieneReport:
    """Scripted scan for option-set defects (the census carries the full
    detector set in scripts/phase0/hygiene_census.py; this is the in-package
    subset used by tests and spot checks)."""
    findings: List[HygieneFinding] = []
    root = metadata.get("variables") or metadata
    if not isinstance(root, dict):
        return HygieneReport(findings=findings)

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
    """Drop residual missingness labels from an option list."""
    drop = set(codes_to_drop or ())
    return [o for o in options if o not in drop]


def attach_interview_date_field(
    respondent_row: Dict[str, Any],
    *,
    date_col: Optional[str],
    date_format: Optional[str] = None,
) -> Optional[str]:
    """Carry interview date from the microdata row when the column is
    configured. Returns the raw string (or None)."""
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
    """Return the harmonised (derived) view of a pulled metadata dict.

    Applies, for the given survey: variable drops and renames (decision 5),
    question rewrites (decision 2), label rewrites (decision 4), and label
    additions (decision 3). The input dict is not mutated.
    """
    drops = VAR_DROPS.get(survey_id, frozenset())
    renames = VAR_RENAMES.get(survey_id, {})
    q_rw = QUESTION_REWRITES.get(survey_id, {})
    l_rw = LABEL_REWRITES.get(survey_id, {})
    l_add = LABEL_ADDITIONS.get(survey_id, {})

    out: Dict[str, Any] = {}
    for section, block in metadata.items():
        if not isinstance(block, dict):
            out[section] = block
            continue
        new_block: Dict[str, Any] = {}
        for var, meta in block.items():
            if not isinstance(meta, dict):
                new_block[var] = meta
                continue
            if var in drops:
                continue
            new_var = renames.get(var, var)
            new_meta = dict(meta)
            if new_var in q_rw or var in q_rw:
                new_meta["question"] = q_rw.get(var, q_rw.get(new_var))
            values = new_meta.get("values")
            if isinstance(values, dict):
                new_values = dict(values)
                rw = l_rw.get(var) or l_rw.get(new_var)
                if rw:
                    new_values = {
                        k: rw.get(str(v), v) for k, v in new_values.items()
                    }
                add = l_add.get(var) or l_add.get(new_var)
                if add:
                    for code, label in add.items():
                        new_values.setdefault(code, label)
                new_meta["values"] = new_values
            new_block[new_var] = new_meta
        out[section] = new_block
    return out
