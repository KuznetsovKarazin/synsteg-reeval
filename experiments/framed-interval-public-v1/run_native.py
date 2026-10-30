#!/usr/bin/env python3
"""Standalone public-v1 native sender/receiver; no campaign/fixture imports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from synsteg_audit.framed_interval_public_v1_orchestration import endpoint_main
if __name__=="__main__":raise SystemExit(endpoint_main())
