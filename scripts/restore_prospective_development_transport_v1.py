"""Verify/restore the complete first30 public transport archive without any model.

Existing differing files, symlinks, unsafe paths and unlisted ZIP members refuse.
The archive and every decompressed byte are checked before any restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "results/prospective-development-transport-v1-artifacts"
PREFIX = "results/prospective-development-transport-v1/milestones/tinyllama-first30/"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    destination = args.destination.resolve()
    manifest = json.loads((ROOT / ARTIFACT / "manifest.json").read_text())
    archive_path = ROOT / manifest["archive_path"]
    if sha(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise SystemExit("archive SHA256 mismatch")
    members = manifest["members"]
    if not 1 <= len(members) <= 20000 or sum(row["bytes"] for row in members.values()) > 1073741824:
        raise SystemExit("archive resource bound exceeded")
    existing = 0
    missing = []
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(members) or {row.filename for row in entries} != set(members):
            raise SystemExit("ZIP member inventory mismatch")
        for entry in entries:
            name = entry.filename
            relative = PurePosixPath(name)
            if (not name.startswith(PREFIX) or relative.is_absolute() or ".." in relative.parts
                    or str(relative) != name or entry.is_dir()
                    or ((entry.external_attr >> 16) & 0o170000) != 0o100000):
                raise SystemExit("unsafe archive member: " + name)
            expected = members[name]
            if entry.file_size != expected["bytes"]:
                raise SystemExit("member size mismatch: " + name)
            data = archive.read(entry)
            if len(data) != expected["bytes"] or sha(data) != expected["sha256"]:
                raise SystemExit("member SHA256 mismatch: " + name)
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise SystemExit("destination escapes root: " + name)
            for ancestor in (target, *target.parents):
                if ancestor == destination:
                    break
                if ancestor.is_symlink():
                    raise SystemExit("symlink destination refused: " + name)
            if target.exists():
                if not target.is_file() or sha(target.read_bytes()) != expected["sha256"]:
                    raise SystemExit("existing different file; overwrite refused: " + name)
                existing += 1
            else:
                missing.append(name)
        if not args.verify_only:
            for name in missing:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                handle, temporary = tempfile.mkstemp(prefix=".public-transport-restore-", dir=target.parent)
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write(archive.read(name))
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Hard-link publication is atomic and refuses an existing destination.
                    os.link(temporary, target)
                finally:
                    os.unlink(temporary)
    print(json.dumps({"status": "verified" if args.verify_only else "restored_or_verified",
                      "archive_sha256": manifest["archive_sha256"], "members_verified": len(members),
                      "existing_identical": existing, "missing": len(missing),
                      "restored": 0 if args.verify_only else len(missing),
                      "model_calls": 0, "paid_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
