#!/usr/bin/env python3
"""Reassemble the exact first40 primary archive and restore verified files only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/prospective-development-transport-first40-v1-artifacts/manifest.json"


def path_in_root(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
        raise ValueError("Unsafe artifact path")
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Unsafe artifact destination")
    return path


def restore(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_id") != "prospective-development-transport-first40-v1":
        raise ValueError("Wrong experiment artifact")
    chunks = []
    names = set()
    for item in manifest["parts"]:
        if item["path"] in names:
            raise ValueError("Duplicate archive part")
        names.add(item["path"])
        data = path_in_root(item["path"]).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Archive part identity differs")
        chunks.append(data)
    data = b"".join(chunks)
    archive = manifest["archive"]
    if len(data) != archive["bytes"] or hashlib.sha256(data).hexdigest() != archive["sha256"]:
        raise ValueError("Reassembled archive identity differs")
    path = path_in_root(archive["path"])
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to overwrite different existing archive")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    spec = importlib.util.spec_from_file_location("continuation_restore", ROOT / "scripts/restore_native_continuation_artifact_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.restore(manifest_path, destination)
    return {"verified_parts": len(chunks), "archive_sha256": archive["sha256"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.destination), sort_keys=True))
