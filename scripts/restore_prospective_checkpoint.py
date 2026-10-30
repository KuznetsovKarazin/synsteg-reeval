#!/usr/bin/env python3
"""Restore a verified prospective stage as data; never load models or rerun a test.

The manifest must come from a trusted, immutable repository checkpoint. Hashes
detect differences from that manifest; they do not authenticate an unknown source.
Atomic no-overwrite publication is supported on Linux and Windows.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

SPEC = importlib.util.spec_from_file_location(
    "prospective_transport", Path(__file__).with_name("package_run_checkpoint.py"))
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)
STAGES = {"frozen_design", "train_generation", "development_generation",
          "attack_training", "test_generation", "test_evaluation", "final_verification"}


def read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != "prospective-stage-checkpoint-v1":
        raise ValueError("Unsupported prospective stage manifest")
    name = transport.safe_name(manifest.get("run_name"))
    if "/" in name or manifest.get("stage") not in STAGES:
        raise ValueError("Invalid run name or stage")
    if (not transport.valid_digest(manifest.get("archive_sha256"))
            or not transport.valid_digest(manifest.get("freeze_id"))
            or type(manifest.get("archive_bytes")) is not int or manifest["archive_bytes"] <= 0):
        raise ValueError("Invalid archive or freeze identity")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid source commit")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or "freeze.json" not in files:
        raise ValueError("Missing file inventory or freeze.json")
    for relative, digest in files.items():
        transport.safe_name(relative)
        if not transport.valid_digest(digest):
            raise ValueError(f"Invalid member hash: {relative}")
        if any(str(parent) in files for parent in PurePosixPath(relative).parents):
            raise ValueError("File/directory path collision")


def publish_no_overwrite(staged, output):
    """Atomically rename without the empty-directory overwrite allowed by POSIX rename."""
    if os.name == "nt":
        os.rename(staged, output)  # Windows rename refuses any existing target.
        return
    if not sys.platform.startswith("linux"):
        raise NotImplementedError("Atomic no-overwrite restoration requires Linux or Windows")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as error:
        raise NotImplementedError("This Linux libc does not provide renameat2") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100, RENAME_NOREPLACE=1; source and target share a filesystem.
    if rename(-100, os.fsencode(staged), -100, os.fsencode(output), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def restore_stage(manifest_path, archive_path, output):
    manifest_path = transport.no_symlink_path(manifest_path)
    archive_path = transport.no_symlink_path(archive_path)
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if not archive_path.is_file():
        raise ValueError("Archive must be a regular file")
    output = transport.new_destination(output)
    files, seen = manifest["files_sha256"], set()
    prefix = manifest["run_name"] + "/"
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prospective-restore-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "archive.tar.gz"
        # Validate an isolated copy, so later changes to the input cannot change extraction.
        shutil.copyfile(archive_path, archive)
        if (archive.stat().st_size != manifest["archive_bytes"]
                or transport.sha_file(archive) != manifest["archive_sha256"]):
            raise ValueError("Archive hash or size mismatch")
        staged = temporary / "run"
        staged.mkdir(mode=0o700)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                name = transport.safe_name(member.name)
                if not member.isfile() or member.issparse() or not name.startswith(prefix):
                    raise ValueError(f"Unsafe or unexpected archive member: {name}")
                relative = transport.safe_name(name[len(prefix):])
                if relative in seen or relative not in files:
                    raise ValueError(f"Unexpected or duplicate member: {relative}")
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with tar.extractfile(member) as source, target.open("xb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                        size += len(block)
                if size != member.size or digest.hexdigest() != files[relative]:
                    raise ValueError(f"Member hash or size mismatch: {relative}")
                seen.add(relative)
        if seen != set(files):
            raise ValueError("Archive file inventory differs from manifest")
        freeze = read_json(staged / "freeze.json")
        if not isinstance(freeze, dict) or freeze.get("freeze_id") != manifest["freeze_id"]:
            raise ValueError("Embedded freeze identity differs from stage manifest")
        canonical = json.dumps({k: v for k, v in freeze.items() if k != "freeze_id"},
                               sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != manifest["freeze_id"]:
            raise ValueError("Embedded freeze digest is invalid")
        publish_no_overwrite(staged, output)
    return {"status": "verified_stage_restored", "stage": manifest["stage"],
            "run_name": manifest["run_name"], "files": len(files),
            "freeze_id": manifest["freeze_id"], "source_commit": manifest["source_commit"],
            "archive_sha256": manifest["archive_sha256"], "output": str(output),
            "note": "Restored data only; this does not rerun or certify completion of an experiment."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(restore_stage(args.manifest, args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
