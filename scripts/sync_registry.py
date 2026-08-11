"""Mirror EXPERIMENT_REGISTRY.md to the out-of-repo backup, and verify it.

The registry is held in two places on purpose (decided 9 Aug 2026): tracked in
git so revisions have a history, and mirrored to `outputs_recovered/` so a
non-regenerable document survives losing the checkout. Two copies only work if
exactly one is editable, so:

    CANONICAL   CODE/EXPERIMENT_REGISTRY.md          <- edit this one
    MIRROR      WORK/outputs_recovered/experiment_registry/   <- never edit

    python scripts/sync_registry.py            # copy canonical -> mirror, rehash
    python scripts/sync_registry.py --check    # exit 1 if they differ

Run the sync after every registry edit, in the same commit. Run --check when
you want to know whether the mirror is stale without touching it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "EXPERIMENT_REGISTRY.md"
MIRROR_DIR = ROOT.parent / "outputs_recovered" / "experiment_registry"
MIRROR = MIRROR_DIR / "EXPERIMENT_REGISTRY.md"
SUMS = MIRROR_DIR / "SHA256SUMS.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rehash() -> int:
    """Rewrite SHA256SUMS.json over every file in the mirror folder."""
    sums = {
        p.relative_to(MIRROR_DIR).as_posix(): sha256(p)
        for p in sorted(MIRROR_DIR.rglob("*"))
        if p.is_file() and p.name != SUMS.name
    }
    SUMS.write_text(
        json.dumps({"files": len(sums), "sha256": sums}, indent=1),
        encoding="utf-8", newline="\n")
    return len(sums)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not write. Exit 1 if the mirror is stale.")
    args = ap.parse_args(argv)

    if not CANONICAL.exists():
        print(f"FAIL canonical missing: {CANONICAL}")
        return 1
    if not MIRROR_DIR.exists():
        if args.check:
            print(f"FAIL mirror folder missing: {MIRROR_DIR}")
            return 1
        MIRROR_DIR.mkdir(parents=True)

    want = sha256(CANONICAL)
    have = sha256(MIRROR) if MIRROR.exists() else None

    if args.check:
        if have is None:
            print(f"STALE mirror does not exist: {MIRROR}")
            return 1
        if have != want:
            print(f"STALE mirror differs from canonical\n"
                  f"  canonical {want}\n  mirror    {have}\n"
                  f"  fix: python scripts/sync_registry.py")
            return 1
        recorded = json.loads(SUMS.read_text(encoding="utf-8"))["sha256"] if SUMS.exists() else {}
        if recorded.get(MIRROR.name) != want:
            print("STALE SHA256SUMS.json does not match the mirror it describes\n"
                  "  fix: python scripts/sync_registry.py")
            return 1
        print(f"OK mirror in sync ({want[:12]}...)")
        return 0

    if have == want:
        n = rehash()
        print(f"OK already in sync; rehashed {n} files")
        return 0

    shutil.copyfile(CANONICAL, MIRROR)
    n = rehash()
    print(f"SYNCED {CANONICAL.name} -> {MIRROR}")
    print(f"       sha256 {want}")
    print(f"       rehashed {n} files into {SUMS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
