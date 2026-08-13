"""Pin the A3 default-option set before it is staged. Exit 0 or it does
not ship.

Checks:
  1. 784 pairs x 2 cells = 1,568 rows, 16 targets, cells adjacent per pair.
  2. Per pair: dkabsent options == dkpresent options minus exactly the
     frozen DK strings, order preserved; at least one DK option removed.
  3. Ground truth: dkpresent index resolves to ground_truth; dkabsent index
     resolves for substantive truths and is null exactly when truth_is_dk.
  4. DK-regex sweep: a BROAD non-substantive pattern over every option in
     the set matches only the frozen strings (nothing missed, nothing
     extra); no duplicate-option target present.
  5. Sidecar a3_dk_meta.json: per-target n and human_dk_share match a
     recount from the rows.
  6. build_prompt renders both cells: DK strings present in the dkpresent
     option block, absent from the dkabsent block.

Usage:
  python scripts/default_options/verify_a3_set.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from synthetic_sampling.scoring.prompts import build_prompt  # noqa: E402

IN_DIR = REPO / "outputs" / "default_options" / "inputs"
SET = IN_DIR / "a3_dk_set.jsonl"
META = IN_DIR / "a3_dk_meta.json"

N_PAIRS = 784
N_TARGETS = 16
DK_OPTIONS = frozenset({"Don't know", "Do not know", "Refusal"})
BROAD_DK = re.compile(
    r"^(don'?t know|do not know|refusal|refused?\b|no answer|not applicable"
    r"|can'?t (remember|choose|say)|cannot (remember|choose|say)"
    r"|haven'?t heard|not stated|no opinion|undecided|missing|it depends)",
    re.IGNORECASE)


def main() -> int:
    bad: list[str] = []

    rows = [json.loads(l) for l in SET.open(encoding="utf-8")]
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        eid, cell = r["example_id"].rsplit("_", 1)
        by_pair[eid][cell] = r

    if len(rows) != N_PAIRS * 2:
        bad.append(f"{len(rows)} rows, want {N_PAIRS * 2}")
    if len(by_pair) != N_PAIRS:
        bad.append(f"{len(by_pair)} pairs, want {N_PAIRS}")
    targets = {(r["survey"], r["target_code"]) for r in rows}
    if len(targets) != N_TARGETS:
        bad.append(f"{len(targets)} targets, want {N_TARGETS}")

    for i in range(0, len(rows), 2):
        if rows[i]["base_id"] != rows[i + 1]["base_id"]:
            bad.append(f"rows {i},{i + 1} not one pair")
            break

    recount: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for eid, cells in by_pair.items():
        if set(cells) != {"dkpresent", "dkabsent"}:
            bad.append(f"{eid}: cells {sorted(cells)}")
            continue
        pres, absent = cells["dkpresent"], cells["dkabsent"]
        p_opts = pres["option_sets"]["original"]
        a_opts = absent["option_sets"]["original"]

        if len(set(p_opts)) != len(p_opts):
            bad.append(f"{eid}: duplicate-option target leaked in")
        want_absent = [o for o in p_opts if o not in DK_OPTIONS]
        if a_opts != want_absent:
            bad.append(f"{eid}: dkabsent is not present-minus-DK in order")
        if len(a_opts) == len(p_opts):
            bad.append(f"{eid}: no DK option was actually removed")

        truth = pres["ground_truth"]
        if p_opts[pres["ground_truth_index"]] != truth:
            bad.append(f"{eid}: dkpresent index does not resolve")
        if pres["truth_is_dk"] != (truth in DK_OPTIONS):
            bad.append(f"{eid}: truth_is_dk flag wrong")
        if pres["truth_is_dk"]:
            if absent["ground_truth_index"] is not None:
                bad.append(f"{eid}: DK truth but dkabsent index not null")
        else:
            if a_opts[absent["ground_truth_index"]] != truth:
                bad.append(f"{eid}: dkabsent index does not resolve")

        rc = recount[(pres["survey"], pres["target_code"])]
        rc[0] += 1
        rc[1] += int(pres["truth_is_dk"])

        for o in p_opts:
            if BROAD_DK.search(o.strip()) and o not in DK_OPTIONS:
                bad.append(f"{eid}: option {o!r} looks DK-ish but is not "
                           f"in the frozen set")

    meta = json.loads(META.read_text(encoding="utf-8"))
    if sorted(DK_OPTIONS) != meta["dk_option_strings"]:
        bad.append("meta dk_option_strings != frozen set")
    for key, v in meta["targets"].items():
        s, t = key.split("|", 1)
        n, ndk = recount[(s, t)]
        if v["n"] != n or v["n_dk_truth"] != ndk:
            bad.append(f"meta {key}: n/n_dk_truth {v['n']}/{v['n_dk_truth']} "
                       f"vs recount {n}/{ndk}")
        if abs(v["human_dk_share"] - ndk / n) > 1e-6:
            bad.append(f"meta {key}: human_dk_share off")

    def option_block(prompt: str) -> list[str]:
        """The numbered options as rendered, not the rest of the prompt
        (a profile answer can legitimately BE "Don't know")."""
        seg = prompt.partition("\n\nOptions:\n")[2].partition(
            "\n\nInstructions:")[0]
        return [line.partition(". ")[2] for line in seg.splitlines()]

    for eid in sorted(by_pair)[::131]:
        pres, absent = (by_pair[eid]["dkpresent"], by_pair[eid]["dkabsent"])
        p = option_block(build_prompt(
            pres, pres["option_sets"]["original"], "label_num"))
        a = option_block(build_prompt(
            absent, absent["option_sets"]["original"], "label_num"))
        for dk in pres["dk_options"]:
            if dk not in p:
                bad.append(f"{eid}: DK option missing from dkpresent block")
            if dk in a:
                bad.append(f"{eid}: DK option leaked into dkabsent block")

    for m in bad[:20]:
        print("FAIL ", m)
    print(f"\n{len(bad)} failures; {len(rows)} rows, {len(by_pair)} pairs, "
          f"{len(targets)} targets")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
