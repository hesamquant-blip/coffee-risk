"""Project paths. Everything under data/ is gitignored (D-006)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
MANUAL = RAW / "manual"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

MANIFEST = RAW / "MANIFEST.csv"
FETCH_LOG = RAW / "FETCH_LOG.csv"
PANEL = PROCESSED / "panel_daily.parquet"
