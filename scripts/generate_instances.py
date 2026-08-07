#!/usr/bin/env python
"""Thin wrapper: prefer `ss-generate` after `pip install -e .`."""
from synthetic_sampling.cli_generate import main

if __name__ == "__main__":
    raise SystemExit(main())
