#!/usr/bin/env python3
"""Standalone ending-v2 sender/receiver; public inputs only, no experiment imports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from synsteg_audit.framed_interval_public_ending_v2_orchestration import endpoint_main
if __name__=="__main__":raise SystemExit(endpoint_main())
