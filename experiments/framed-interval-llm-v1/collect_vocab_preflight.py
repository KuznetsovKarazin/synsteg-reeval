#!/usr/bin/env python3
"""Run exactly one separately frozen vocabulary-only native preflight."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_vocab_collect import run_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--activation-sha256", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_collection(ROOT, args.activation, args.activation_sha256, args.output, args.intent)
    print(json.dumps({"state": result["state"], "planned_fixtures": 15, "planned_prompts": 2,
                      "forward_calls": 0}, sort_keys=True))
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "collection_rejected", "type": type(exc).__name__, "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
