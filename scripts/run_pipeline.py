#!/usr/bin/env python3
"""Convenience wrapper: python scripts/run_pipeline.py [--config PATH]."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5_forecasting.cli import main

if __name__ == "__main__":
    # Default to run-all when no subcommand is provided
    if len(sys.argv) == 1 or (len(sys.argv) >= 2 and sys.argv[1].startswith("--")):
        sys.argv.insert(1, "run-all")
    main()
