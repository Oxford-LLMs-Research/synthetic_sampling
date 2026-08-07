#!/usr/bin/env python
"""Thin wrapper: prefer `ss-score` after `pip install -e .`."""
from synthetic_sampling.cli_score import main

if __name__ == "__main__":
    raise SystemExit(main())
