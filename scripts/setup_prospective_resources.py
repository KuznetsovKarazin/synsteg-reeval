#!/usr/bin/env python3
"""Fetch/verify exact NLTK pilot resources and print a new-protocol fragment.

Never changes a protocol, freeze record or old result. On another machine,
adapt a new protocol to the printed paths before creating a fresh freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RESOURCE_FILES = {
    "corpora/wordnet.zip": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
    "taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json": "d152ffa69d45b8357276341a3d592c6c025911d52ec0d9068d4f382560dd48b6",
    "taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json": "4c713731cb06727736962bc8cd94ad919217a040e76691059af99bc0c2c9c246",
    "taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json": "789e8e35f6fac656d9b7eccea29ba4e8a137cbb5b525dd31f39e4291a9e80832",
}
PACKAGES = ("wordnet", "averaged_perceptron_tagger_eng")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_existing(root: Path, *, require_all: bool) -> list[str]:
    if (root / "corpora/wordnet").exists():
        raise ValueError("An extracted WordNet directory can shadow the pinned ZIP; use a fresh directory")
    missing = []
    for relative, expected in RESOURCE_FILES.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256(path) != expected:
            raise ValueError(f"Existing resource differs from pinned pilot data: {path}; "
                             "use a fresh directory, do not replace an active run's resources")
    if require_all and missing:
        raise ValueError("Missing pinned resources: " + ", ".join(missing))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Explicit NLTK data directory; use as NLTK_DATA for the new run")
    parser.add_argument("--verify-only", action="store_true",
                        help="No downloads or filesystem changes")
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    missing = verify_existing(root, require_all=args.verify_only)
    if missing and not args.verify_only:
        try:
            import nltk
        except ImportError as exc:
            raise SystemExit("Install nltk==3.9.2 in the neural environment before downloading") from exc
        root.mkdir(parents=True, exist_ok=True)
        for package in PACKAGES:
            needed = ("corpora/wordnet.zip" in missing if package == "wordnet"
                      else any(path.startswith("taggers/") for path in missing))
            if needed and not nltk.download(package, download_dir=str(root), quiet=True, raise_on_error=True):
                raise RuntimeError(f"NLTK download failed: {package}")
        verify_existing(root, require_all=True)
    files = [{"path": str(root / relative), "sha256": digest} for relative, digest in RESOURCE_FILES.items()]
    print(json.dumps({
        "status": "verified",
        "resource_root": str(root),
        "environment": {"NLTK_DATA": str(root)},
        "protocol_fragment": {"external_resource_files": files},
        "notes": [
            "Copy these paths into a NEW protocol before freeze; never modify an old frozen run.",
            "Set NLTK_DATA to resource_root when invoking the prospective runner.",
            "Use an isolated NLTK environment without another WordNet directory that could shadow the ZIP.",
            "Pinned Hugging Face model weights require separate caching; this script only handles lexical/POS data.",
            "Downloads use the official NLTK downloader followed by exact SHA256 checks; changed upstream data fail verification."
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
