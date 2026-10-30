#!/usr/bin/env python3
"""Explicit synthetic-only executable; no NN model calls or hidden fixtures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from synsteg_audit.framed_interval_public_v1_cli import main
if __name__ == "__main__":
    raise SystemExit(main())
