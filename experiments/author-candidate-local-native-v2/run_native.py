#!/usr/bin/env python3
"""Explicit native-v2 coordinator. Default activation file is disabled."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.author_candidate_local_native_v2 import run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute required; source/activation Git readback and internal NN lease enforced")
    result = run_campaign(args.root, args.output, args.activation, args.intent, args.readback)
    print(json.dumps({k: result[k] for k in ("status", "attempts_executed",
        "format_accepted_chunks_observed", "complete_campaign_format_accepted_chunks")}))
    return 0 if result["status"] == "complete_diagnostic_pending_Git_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
