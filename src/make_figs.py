#!/usr/bin/env python3
"""Grader-friendly entry point for schema-checked assignment figures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_figs import main  # noqa: E402


if __name__ == "__main__":
    main()
