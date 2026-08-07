"""Bundled survey metadata JSON (one file per survey)."""

from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parent


def metadata_path(filename: str) -> Path:
    return METADATA_DIR / filename
