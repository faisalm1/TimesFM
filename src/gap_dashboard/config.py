"""Runtime paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path

# Project root: .../TimesFM
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "cache"

DEFAULT_GAP_THRESHOLD_PCT = 10.0
DEFAULT_FORWARD_DAYS = 5
DEFAULT_CONTEXT_DAYS = 1024
DEFAULT_HISTORY_YEARS = 5


def cache_dir() -> Path:
    p = Path(os.environ.get("GAP_DASHBOARD_CACHE", str(DEFAULT_CACHE_DIR)))
    p.mkdir(parents=True, exist_ok=True)
    return p
