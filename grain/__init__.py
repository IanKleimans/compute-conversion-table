"""Grain conversion table: measured AI work per GPU-hour, tied to rent and token prices."""

from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
