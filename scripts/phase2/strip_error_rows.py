"""Move error rows out of a score file so a resumed job rescores them.

    python scripts/phase2/strip_error_rows.py SHARD.jsonl

The runner resumes by example_id, so a row whose arms failed (server gone)
counts as scored. This rewrites SHARD.jsonl without those rows and keeps
them beside it as SHARD.jsonl.error_rows. Stdlib only.
"""
import json
import os
import sys


def main(path: str) -> int:
    side, tmp = path + ".error_rows", path + ".tmp"
    if os.path.exists(side):
        print(f"{side} exists; refusing to overwrite")
        return 1
    kept = bad = 0
    with open(path, encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8", newline="") as fok, \
            open(side, "w", encoding="utf-8", newline="") as fbad:
        for line in fin:
            results = json.loads(line).get("results", {})
            if any("error" in r for r in results.values()):
                fbad.write(line)
                bad += 1
            else:
                fok.write(line)
                kept += 1
    os.replace(tmp, path)
    print(f"kept {kept}  moved {bad} error rows to {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
