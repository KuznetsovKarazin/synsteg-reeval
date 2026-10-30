#!/usr/bin/env python3
"""Build metadata-only source closure; never activates or calls native code."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.author_candidate_local_native_v3 import (
    ID, PROTOCOL, DEFAULT_ACTIVATION, CLOSURE, REQUIRED_CLOSURE,
    protocol_document, validate_inputs, sha_file, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--initial-metadata", action="store_true")
    parser.add_argument("--closure-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.initial_metadata:
        write_json(root / PROTOCOL, protocol_document())
        write_json(root / DEFAULT_ACTIVATION, {"schema": ID + "-activation", "enabled": False,
            "reason": "No source freeze, independent integrated-native review, enabled intent or actual Git readback yet.",
            "must_create_new_enabled_activation_file": True, "native_calls": 0})
    if args.closure_output:
        validate_inputs(root)
        files = {p: sha_file(root / p) for p in sorted(REQUIRED_CLOSURE)}
        write_json(args.closure_output, {"schema": ID + "-source-closure", "files_sha256": files,
            "source_files": len(files), "native_generation_enabled": False,
            "model_tokenizer_calls": 0, "next": "Independent review, immutable Git source/readback, then separate activation/intent if coordinator chooses."})
    if not args.initial_metadata and not args.closure_output:
        parser.error("select --initial-metadata or --closure-output; neither enables generation")
    print(json.dumps({"native_calls": 0, "activation_enabled": False,
                      "closure_written": bool(args.closure_output)}))


if __name__ == "__main__":
    main()
