#!/usr/bin/env python
"""Thin wrapper: prefer `ss-analyze` after `pip install -e .`."""
from synthetic_sampling.cli_analyze import main

if __name__ == "__main__":
    raise SystemExit(main())
