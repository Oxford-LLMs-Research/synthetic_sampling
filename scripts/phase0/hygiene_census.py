"""Phase 0 hygiene census: detect and report, never repair.

Scans every survey's pulled metadata (question text + value labels — the
strings that reach prompts) and its microdata (the codes that select them)
for the defect classes accumulated in PAPER_STATE, plus generic detectors.
The census is the evidence base for the Phase 0 decisions; nothing here
rewrites anything, and `surveys.harmonise` stays identity until the
decisions are recorded.

    python scripts/phase0/hygiene_census.py

Outputs (WORK/analysis/hygiene/):
    census_findings.csv         one row per (survey, var, kind) finding
    census_unlabeled_codes.csv  microdata codes absent from the values map
    census_date_candidates.csv  date-ish columns per survey (for the
                                interview_date_col stubs)
    census_summary.txt          per-survey counts + known-item checklist

Exit is nonzero if any KNOWN item from the instrument work is NOT
rediscovered — the census must at minimum see everything we already know.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data"
META = ROOT / "src" / "synthetic_sampling" / "surveys" / "metadata"
OUTDIR = ROOT.parent / "analysis" / "hygiene"

SURVEYS = {
    # survey_id -> (metadata file, microdata path, loader kind)
    "wvs": ("pulled_metadata_wvs.json", DATA / "WVS" / "WVS_2017_22.csv", "csv"),
    "afrobarometer": ("pulled_metadata_afrobarometer.json",
                      DATA / "Afrobarometer" / "afrobarometer_r9_converted.csv", "csv"),
    "arabbarometer": ("pulled_metadata_arabbarometer.json",
                      DATA / "Arabbarometer" / "ArabBarometer_WaveVIII_English_v3.csv", "csv"),
    "asianbarometer": ("pulled_metadata_asianbarometer.json",
                       DATA / "Asianbarometer" / "asiabarom_combined_datasets.csv", "csv"),
    "latinobarometer": ("pulled_metadata_latinobarometer.json",
                        DATA / "Latinobarometro" / "Latinobarometro_2023_Eng_Spss_v1_0.sav", "sav"),
    "ess_wave_10": ("pulled_metadata_ess10.json",
                    DATA / "ESS" / "wave_10" / "ESS10_with_consolidations.csv", "csv"),
    "ess_wave_11": ("pulled_metadata_ess11.json",
                    DATA / "ESS" / "wave_11" / "ESS11_with_consolidations.csv", "csv"),
}

# ---------------------------------------------------------------- detectors

MOJIBAKE_PATTERNS = [
    ("replacement_char", re.compile("�")),
    ("cp1252_quote", re.compile("â€™|â€œ|â€\x9d|â€˜")),
    ("cp1252_dash", re.compile("â€“|â€”")),
    ("cp1252_generic", re.compile("â€|Ã[-¿]|Â[ -¿]")),
    ("stray_control", re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")),
]

DANGLING_WORDS = {
    "of", "to", "the", "and", "or", "in", "for", "with", "a", "an", "by",
    "at", "on", "as", "is", "are", "be", "than", "that", "who", "which",
    "your", "my", "their", "our",
}

MISSING_LABELS = {
    "don't know", "dont know", "no answer", "missing", "not asked",
    "refused", "refuse", "not applicable", "can't choose", "cant choose",
    "decline to answer", "declined", "no response", "do not understand",
    "not available", "no opinion", "unsure",
}

SENTINEL_CODES = {
    "97", "98", "99", "997", "998", "999", "9997", "9998", "9999",
    "99997", "99998", "99999", "999997", "999998", "999999",
    "555555", "888888", "-1", "-2", "-4", "-5", "-7", "-8", "-9",
    "77", "88", "777", "888", "7777", "8888",
}


def mojibake_hits(text: str) -> list[str]:
    return [name for name, pat in MOJIBAKE_PATTERNS if pat.search(text)]


def looks_truncated(label: str) -> str | None:
    s = label.strip()
    if len(s) < 4:
        return None
    if s.endswith(("...", "…")):
        return None  # deliberate ellipsis, not truncation
    if s[-1] in ",;:-/(":
        return f"ends with {s[-1]!r}"
    for opener, closer in (("(", ")"), ("[", "]")):
        if s.count(opener) > s.count(closer):
            return f"unbalanced {opener!r}"
    last = re.split(r"[^A-Za-z']+", s)[-1] if re.split(r"[^A-Za-z']+", s) else ""
    if last.lower() in DANGLING_WORDS:
        return f"ends on dangling word {last!r}"
    return None


def norm_label(label: str) -> str:
    return "".join(c for c in label.lower() if c.isalnum())


def near_duplicate_pairs(labels: list[str]) -> list[tuple[str, str]]:
    """Typo-scale near duplicates ('None at all' / 'None et all').

    Pairs whose differing characters include a digit are dropped: those are
    systematic distinct options (Statement 1 / Statement 2, waves, scales),
    not typos.
    """
    pairs = []
    norms = [(lab, norm_label(lab)) for lab in dict.fromkeys(labels)]
    for i, (lab_a, na) in enumerate(norms):
        for lab_b, nb in norms[i + 1:]:
            if not na or not nb or na == nb:
                continue
            if abs(len(na) - len(nb)) > 2 or min(len(na), len(nb)) < 6:
                continue
            diff_chars = [
                (x, y) for x, y in zip(na, nb) if x != y
            ] + [(c, "") for c in (na[len(nb):] or nb[len(na):])]
            if len(diff_chars) > 2:
                continue
            if any(x.isdigit() or y.isdigit() for x, y in diff_chars):
                continue
            pairs.append((lab_a, lab_b))
    return pairs


def norm_code(val) -> str:
    """Microdata value -> the code string convention of the metadata maps."""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    s = str(val).strip()
    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]
    return s


# ---------------------------------------------------------------- metadata walk

def iter_vars(metadata: dict):
    """Yield (var_code, question, values_dict) over section -> var maps."""
    for section, block in metadata.items():
        if not isinstance(block, dict):
            continue
        for var_code, meta in block.items():
            if not isinstance(meta, dict):
                continue
            values = meta.get("values") or meta.get("value_labels") or {}
            if not isinstance(values, dict):
                values = {}
            yield str(var_code), str(meta.get("question") or ""), values


def scan_metadata(survey_id: str, metadata: dict, findings: list[dict]) -> dict:
    """Text-level scans; returns var -> values map for the microdata pass."""
    values_by_var: dict[str, dict] = {}
    for var, question, values in iter_vars(metadata):
        values_by_var[var] = values
        labels = [str(v) for v in values.values()]

        for text, where in [(question, "question")] + [(l, "label") for l in labels]:
            hits = mojibake_hits(text)
            if hits:
                findings.append(dict(
                    survey=survey_id, var_code=var, kind="mojibake",
                    detail=f"{where}: {','.join(hits)}", example=text[:120],
                    count=1))
        for lab in labels:
            why = looks_truncated(lab)
            if why:
                findings.append(dict(
                    survey=survey_id, var_code=var, kind="truncated_label",
                    detail=why, example=lab[:120], count=1))
            if lab.strip() and re.fullmatch(r"-?\d+(\.\d+)?", lab.strip()):
                findings.append(dict(
                    survey=survey_id, var_code=var, kind="bare_numeric_label",
                    detail="option label is a bare number", example=lab, count=1))

        substantive = [
            str(v) for k, v in values.items()
            if str(v).strip().lower() not in MISSING_LABELS
            and not str(k).lstrip("-").isdigit() is False
        ]
        subst_nonmissing = [
            str(v) for k, v in values.items()
            if str(v).strip().lower() not in MISSING_LABELS
        ]
        dup_groups = {lab: n for lab, n in Counter(subst_nonmissing).items() if n > 1}
        if dup_groups:
            findings.append(dict(
                survey=survey_id, var_code=var, kind="exact_duplicate_labels",
                detail=f"{len(dup_groups)} label(s) shared by multiple codes "
                       "(binned scale or collapse artifact); option sets built "
                       "per-code will carry duplicates",
                example="; ".join(f"{l!r} x{n}" for l, n in list(dup_groups.items())[:3]),
                count=sum(dup_groups.values())))
        for lab_a, lab_b in near_duplicate_pairs(subst_nonmissing):
            findings.append(dict(
                survey=survey_id, var_code=var, kind="typo_duplicate",
                detail="near-duplicate option labels",
                example=f"{lab_a!r} / {lab_b!r}", count=1))
        _ = substantive
    return values_by_var


# ---------------------------------------------------------------- microdata

DATE_NAME_PAT = re.compile(
    r"date|inw|fecha|year|month|yr\b|intyear|dateintr|diaentre|mesentre", re.I)

HOURS_PER_DAY_PAT = re.compile(
    r"(hours?|much time).{0,80}(a|per|typical|each) day"
    r"|(a|per|typical|each) day.{0,80}hours?|daily.{0,40}hours?"
    r"|hours?.{0,40}daily", re.I | re.S)


def load_microdata(path: Path, kind: str, usecols: list[str]) -> pd.DataFrame:
    if kind == "sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(
            str(path), usecols=usecols or None, apply_value_formats=False)
        return df
    return pd.read_csv(path, usecols=usecols or None, low_memory=False)


def read_columns(path: Path, kind: str) -> list[str]:
    if kind == "sav":
        import pyreadstat
        _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
        return list(meta.column_names)
    return list(pd.read_csv(path, nrows=0).columns)


def scan_microdata(survey_id: str, path: Path, kind: str,
                   values_by_var: dict, questions: dict,
                   findings: list[dict], unlabeled: list[dict],
                   date_rows: list[dict]) -> str:
    columns = read_columns(path, kind)
    colset = set(columns)
    matched = [v for v in values_by_var if v in colset]

    for col in columns:
        if DATE_NAME_PAT.search(col):
            date_rows.append(dict(survey=survey_id, column=col))

    df = load_microdata(path, kind, matched)

    for var in matched:
        values = values_by_var[var]
        series = df[var].dropna()
        if series.empty:
            continue
        counts = Counter(norm_code(v) for v in series)
        n = sum(counts.values())
        labeled = set(values.keys())
        if values:
            for code, cnt in counts.most_common():
                if code in labeled:
                    continue
                kind_ = ("sentinel_candidate" if code in SENTINEL_CODES
                         or re.fullmatch(r"([1-9])\1{2,}", code.lstrip("-"))
                         else "unlabeled_code")
                unlabeled.append(dict(
                    survey=survey_id, var_code=var, code=code, count=cnt,
                    share=round(cnt / n, 4), kind=kind_))
        else:
            # No values map: numeric answer carried raw into profiles.
            for code, cnt in counts.most_common():
                if code in SENTINEL_CODES or re.fullmatch(r"([1-9])\1{4,}", code.lstrip("-")):
                    unlabeled.append(dict(
                        survey=survey_id, var_code=var, code=code, count=cnt,
                        share=round(cnt / n, 4), kind="sentinel_candidate"))

        # Implausible magnitudes for per-day hour questions. Labeled codes
        # and repdigit sentinels are excluded; everything else above 24 is a
        # substantive answer that cannot be true of a day.
        q = questions.get(var, "")
        if HOURS_PER_DAY_PAT.search(q):
            substantive = [
                v for v in series
                if norm_code(v) not in values
                and norm_code(v) not in SENTINEL_CODES
                and not re.fullmatch(r"([1-9])\1{1,}", norm_code(v).lstrip("-"))
            ]
            numeric = pd.to_numeric(pd.Series(substantive), errors="coerce").dropna()
            bad = numeric[numeric > 24]
            if len(bad):
                findings.append(dict(
                    survey=survey_id, var_code=var, kind="implausible_numeric",
                    detail=f"hours-per-day values over 24 (max {bad.max():g})",
                    example=str(sorted(set(bad.astype(int)))[:8]),
                    count=int(len(bad))))
    return f"{survey_id}: {len(matched)}/{len(values_by_var)} metadata vars matched to columns"


# ---------------------------------------------------------------- encoding

ENC_BAD = re.compile("[�\x80-\x9f]")


def scan_encoding(survey_id: str, meta_path: Path, data_path: Path, kind: str,
                  enc_rows: list[dict], nonascii: list[dict]) -> None:
    """Stream both sources for encoding damage; inventory metadata non-ASCII.

    Prior 'mojibake' reports ('Can<?>t choose', an ESS em-dash) turn out to
    be cp1252 console renderings of legitimate Unicode; this scan is the
    evidence: damage count per source, plus the codepoint inventory that
    shows what the console could not print.
    """
    raw = meta_path.read_text(encoding="utf-8", errors="replace")
    n_bad = len(ENC_BAD.findall(raw))
    enc_rows.append(dict(survey=survey_id, source=meta_path.name,
                         bad_char_count=n_bad))
    counts: Counter = Counter(c for c in raw if ord(c) > 126)
    for ch, cnt in counts.most_common():
        i = raw.find(ch)
        nonascii.append(dict(
            survey=survey_id, codepoint=f"U+{ord(ch):04X}", char=ch,
            count=cnt, example=raw[max(0, i - 40):i + 12].replace("\n", " ")))

    if kind == "csv":
        n_bad_lines = 0
        with open(data_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if ENC_BAD.search(line):
                    n_bad_lines += 1
        enc_rows.append(dict(survey=survey_id, source=data_path.name,
                             bad_char_count=n_bad_lines))
    else:
        import pyreadstat
        _, meta = pyreadstat.read_sav(str(data_path), metadataonly=True)
        texts = list(meta.column_labels or [])
        for mapping in (meta.variable_value_labels or {}).values():
            texts.extend(str(v) for v in mapping.values())
        n_bad = sum(len(ENC_BAD.findall(str(t))) for t in texts if t)
        enc_rows.append(dict(survey=survey_id, source=data_path.name,
                             bad_char_count=n_bad))


# ---------------------------------------------------------------- known items

def check_known_items(findings: list[dict], unlabeled: list[dict],
                      enc_rows: list[dict], nonascii: list[dict]) -> list[str]:
    """The census must rediscover everything the instrument work already knows.

    Two items RESOLVE rather than reproduce: the 'mojibake' apostrophe and
    ESS em-dash are legitimate Unicode that a cp1252 console cannot print
    (harmonise.KNOWN_DEFECTS also pins the typo pair to Q149; the census
    finds it on Q48 — the Q149 reference is stale). The checklist therefore
    requires the encoding scan to have covered every source with zero
    damage, and the em-dash to be located in the ESS inventory.
    """
    f_idx = {(r["survey"], r["var_code"], r["kind"]) for r in findings}
    fails = []

    def need(cond: bool, name: str):
        if not cond:
            fails.append(name)

    need(("wvs", "Q48", "typo_duplicate") in f_idx,
         "wvs Q48 'None at all'/'None et all' typo duplicate "
         "(KNOWN_DEFECTS says Q149; that reference is stale)")
    for sv, var in [("ess_wave_10", "euftf"), ("ess_wave_10", "stfeco"),
                    ("wvs", "Q178"), ("wvs", "Q243"), ("wvs", "Q48")]:
        need((sv, var, "exact_duplicate_labels") in f_idx,
             f"{sv} {var} duplicate-label target (B3 exclusion set)")
    need(len({r["survey"] for r in enc_rows}) >= 7,
         "encoding scan covered all 7 sources")
    need(all(r["bad_char_count"] == 0 for r in enc_rows)
         or any(r["kind"] == "mojibake" for r in findings),
         "encoding damage either absent everywhere or reported as findings")
    need(any(r["survey"].startswith("ess") and r["codepoint"] == "U+2014"
             for r in nonascii),
         "the observed ESS em-dash located in the non-ASCII inventory")
    need(any(r["code"] == "555555" for r in unlabeled),
         "ancestry sentinel 555555 in microdata")
    need(any(r["code"] in ("99999", "999999") for r in unlabeled),
         "income sentinel 99999 in microdata")
    need(any(r["kind"] == "implausible_numeric" and "hours" in r["detail"]
             for r in findings),
         "hours-per-day over 24 (the '120 hours/day' item)")
    return fails


# ---------------------------------------------------------------- main

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []
    unlabeled: list[dict] = []
    date_rows: list[dict] = []
    enc_rows: list[dict] = []
    nonascii: list[dict] = []
    lines: list[str] = []

    for survey_id, (meta_file, data_path, kind) in SURVEYS.items():
        metadata = json.load(open(META / meta_file, encoding="utf-8"))
        values_by_var = scan_metadata(survey_id, metadata, findings)
        questions = {v: q for v, q, _ in iter_vars(metadata)}
        line = scan_microdata(survey_id, data_path, kind, values_by_var,
                              questions, findings, unlabeled, date_rows)
        scan_encoding(survey_id, META / meta_file, data_path, kind,
                      enc_rows, nonascii)
        print(line)
        lines.append(line)

    fdf = pd.DataFrame(findings).sort_values(["survey", "kind", "var_code"])
    udf = pd.DataFrame(unlabeled).sort_values(
        ["survey", "var_code", "count"], ascending=[True, True, False])
    ddf = pd.DataFrame(date_rows)
    edf = pd.DataFrame(enc_rows)
    ndf = pd.DataFrame(nonascii)
    fdf.to_csv(OUTDIR / "census_findings.csv", index=False)
    udf.to_csv(OUTDIR / "census_unlabeled_codes.csv", index=False)
    ddf.to_csv(OUTDIR / "census_date_candidates.csv", index=False)
    edf.to_csv(OUTDIR / "census_encoding.csv", index=False)
    ndf.to_csv(OUTDIR / "census_nonascii_inventory.csv", index=False,
               encoding="utf-8")

    fails = check_known_items(findings, unlabeled, enc_rows, nonascii)

    with open(OUTDIR / "census_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("Phase 0 hygiene census - summary\n")
        fh.write("=" * 60 + "\n\n")
        for line in lines:
            fh.write(line + "\n")
        fh.write("\nFindings by (survey, kind):\n")
        for (sv, kd), n in sorted(Counter(
                (r["survey"], r["kind"]) for r in findings).items()):
            fh.write(f"  {sv:18s} {kd:24s} {n}\n")
        fh.write("\nUnlabeled/sentinel codes by survey:\n")
        for sv, n in sorted(Counter(r["survey"] for r in unlabeled).items()):
            fh.write(f"  {sv:18s} {n} distinct (var, code) rows\n")
        fh.write("\nEncoding integrity (bad chars per source):\n")
        for r in enc_rows:
            fh.write(f"  {r['survey']:18s} {r['source']:44s} {r['bad_char_count']}\n")
        fh.write("\nNon-ASCII inventory: "
                 f"{len(ndf)} (survey, codepoint) rows — legitimate Unicode; "
                 "prior 'mojibake' reports were cp1252 console renderings.\n")
        fh.write("\nKnown-item checklist:\n")
        if fails:
            for f in fails:
                fh.write(f"  MISSED {f}\n")
        else:
            fh.write("  all known items rediscovered\n")

    print(f"\nwrote {OUTDIR / 'census_findings.csv'} ({len(fdf)} rows)")
    print(f"wrote {OUTDIR / 'census_unlabeled_codes.csv'} ({len(udf)} rows)")
    print(f"wrote {OUTDIR / 'census_date_candidates.csv'} ({len(ddf)} rows)")
    print(f"wrote {OUTDIR / 'census_encoding.csv'} ({len(edf)} rows)")
    print(f"wrote {OUTDIR / 'census_nonascii_inventory.csv'} ({len(ndf)} rows)")
    if fails:
        for f in fails:
            print("MISSED", f)
        print("CENSUS INCOMPLETE: known items not rediscovered")
        return 1
    print("known-item checklist: all rediscovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
