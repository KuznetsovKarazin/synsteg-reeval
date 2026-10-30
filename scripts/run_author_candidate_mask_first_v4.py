"""Explicit future launch entry point. No activation/intent is provided here."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="launch only after all frozen gates pass")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from synsteg_audit.author_candidate_mask_first_v4_native import run_campaign, verify_gate
    if args.execute:
        result = run_campaign(args.root, args.activation, args.intent, args.readback)
    else:
        requests, profile, receipt = verify_gate(args.root, args.activation, args.intent, args.readback)
        result = {"gate_valid": True, "execution_requested": False, "requests": len(requests),
                  "source_commit": receipt["source_commit"], "model_calls": 0, "tokenizer_calls": 0}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
