#!/usr/bin/env python3
"""Verify and extract the immutable submitted archive; never overwrite changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_archive(archive: Path) -> None:
    """Reassemble verified Git-stored chunks without publishing a partial ZIP."""
    if archive.exists():
        if sha256(archive) != EXPECTED_SHA256:
            raise ValueError("Existing source archive hash mismatch; refusing to overwrite")
        return
    parts_dir = archive.with_name(archive.name + ".parts")
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != 1
            or manifest.get("archive_name") != archive.name
            or manifest.get("archive_sha256") != EXPECTED_SHA256):
        raise ValueError("Chunk manifest does not identify the submitted archive")
    parts = manifest.get("parts")
    chunk_bytes, archive_bytes = manifest.get("chunk_bytes"), manifest.get("archive_bytes")
    if (not isinstance(parts, list) or not parts
            or type(chunk_bytes) is not int or chunk_bytes <= 0
            or type(archive_bytes) is not int or archive_bytes <= 0):
        raise ValueError("Invalid chunk manifest sizes or parts")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=".archive-", delete=False) as out:
            temporary = Path(out.name)
            for index, item in enumerate(parts):
                expected_name = f"part-{index:03d}.bin"
                if not isinstance(item, dict) or item.get("path") != expected_name:
                    raise ValueError(f"Invalid chunk order or path at index {index}")
                path = parts_dir / expected_name
                if path.is_symlink():
                    raise ValueError(f"Symbolic link in archive chunks: {expected_name}")
                data = path.read_bytes()
                expected_size = chunk_bytes if index < len(parts) - 1 else archive_bytes - total
                if (type(item.get("bytes")) is not int
                        or len(data) != item["bytes"] or len(data) != expected_size
                        or not 0 < len(data) <= chunk_bytes
                        or hashlib.sha256(data).hexdigest() != item.get("sha256")):
                    raise ValueError(f"Chunk hash or size mismatch: {expected_name}")
                out.write(data)
                digest.update(data)
                total += len(data)
            out.flush()
            os.fsync(out.fileno())
        if total != archive_bytes or digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError("Reassembled source archive hash or size mismatch")
        # Same-directory hard link atomically publishes the fully verified file,
        # and fails instead of overwriting if another process created the target.
        os.link(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_source(archive: Path, destination: Path) -> dict:
    if not archive.exists():
        assemble_archive(archive)
    actual = sha256(archive)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"Source archive hash mismatch: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(archive) as z:
        members = []
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if not path.parts or path.parts[0] != "synsteg_reproduction":
                raise ValueError(f"Unexpected archive root: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Symbolic link in archive: {item.filename}")
            if item.is_dir():
                continue
            relative = Path(*path.parts[1:])
            if str(relative) in manifest:
                raise ValueError(f"Duplicate archive member: {item.filename}")
            blob = z.read(item)
            manifest[str(relative)] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
            members.append((relative, blob))
        if destination.exists():
            for relative, item in manifest.items():
                p = destination / relative
                if not p.is_file() or sha256(p) != item["sha256"]:
                    raise ValueError(f"Existing extraction differs: {p}; use a fresh destination")
            extras = [str(p.relative_to(destination)) for p in destination.rglob("*")
                      if p.is_file() and str(p.relative_to(destination)) not in manifest
                      and "__pycache__" not in p.parts]
            if extras:
                raise ValueError(f"Unexpected files in existing extraction: {extras[:5]}")
        else:
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".extract-") as temporary:
                temp = Path(temporary) / "source"
                temp.mkdir()
                for relative, blob in members:
                    target = temp / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(blob)
                shutil.move(str(temp), str(destination))
    result = {"archive_sha256": actual, "file_count": len(manifest), "files": manifest}
    manifest_path = destination.parent / (destination.name + ".manifest.json")
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/source/synsteg_reproduction.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/source/synsteg_reproduction")
    args = parser.parse_args()
    result = import_source(args.archive.resolve(), args.destination.resolve())
    print(json.dumps({"archive_sha256": result["archive_sha256"], "files": result["file_count"],
                      "destination": str(args.destination.resolve())}, indent=2))


if __name__ == "__main__":
    main()
