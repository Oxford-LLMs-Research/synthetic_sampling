"""Checks: smoke gate, coverage, number verification."""

from .coverage import coverage_report
from .smoke import check_smoke, check_smoke_file

__all__ = [
    "coverage_report",
    "check_smoke",
    "check_smoke_file",
    "verify_numbers",
    "verify_table_against_csv",
]


def __getattr__(name):
    # number_verify needs pandas, which the cluster venv may not carry;
    # keep it lazy so the in-job smoke/coverage gates run without it.
    if name in ("verify_numbers", "verify_table_against_csv"):
        from . import number_verify
        return getattr(number_verify, name)
    raise AttributeError(name)
