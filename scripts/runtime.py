"""Ensure src/ is on sys.path when running scripts from scripts/."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_src_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
