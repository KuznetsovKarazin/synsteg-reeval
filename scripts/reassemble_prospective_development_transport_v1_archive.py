"""Reassemble the exact first30 ZIP from small Git-readable binary parts."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/prospective-development-transport-v1-artifacts/manifest.json"
PREFIX = "results/prospective-development-transport-v1-artifacts/parts/"


def reassemble(root, output=None):
    manifest = json.loads((root / MANIFEST).read_text())
    parts = manifest["git_parts"]
    if not 1 <= len(parts) <= 2048 or not 0 < manifest["archive_bytes"] <= 1073741824:
        raise ValueError("invalid archive limits")
    if sum(p["bytes"] for p in parts) != manifest["archive_bytes"]:
        raise ValueError("part sizes do not sum to archive size")
    if len({p["path"] for p in parts}) != len(parts):
        raise ValueError("duplicate part")
    verified = []
    for part in parts:
        rel = PurePosixPath(part["path"])
        if not str(rel).startswith(PREFIX) or rel.is_absolute() or ".." in rel.parts or str(rel) != part["path"]:
            raise ValueError("unsafe part path")
        p = root / part["path"]
        if any(a.is_symlink() for a in (p, *p.parents) if a != root) or not p.is_file():
            raise ValueError("part must be a regular file without symlink ancestors")
        raw = p.read_bytes()
        if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError("part identity mismatch: " + part["path"])
        verified.append(raw)
    raw = b"".join(verified)
    if len(raw) != manifest["archive_bytes"] or hashlib.sha256(raw).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("reassembled archive identity mismatch")
    rel = PurePosixPath(manifest["archive_path"])
    if rel.is_absolute() or ".." in rel.parts or str(rel) != manifest["archive_path"]:
        raise ValueError("unsafe archive path")
    target = Path(output) if output is not None else root / str(rel)
    if any(a.is_symlink() for a in (target, *target.parents)):
        raise ValueError("symlink output refused")
    if target.exists():
        if not target.is_file() or target.read_bytes() != raw:
            raise ValueError("different existing archive; overwrite refused")
        state = "verified_existing"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".first30-reassemble-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(tmp, target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(tmp)
        state = "reassembled"
    return {"status": state, "parts_verified": len(parts), "archive_bytes": len(raw),
            "archive_sha256": manifest["archive_sha256"], "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(reassemble(ROOT, args.output)))
