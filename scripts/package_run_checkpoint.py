#!/usr/bin/env python3
"""Create/restore verified, small Git-storable snapshots of quiescent run files."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile

CHUNK_BYTES = 45000
SCHEMA = "synsteg-run-checkpoint-v1"
MAX_MANIFEST_BYTES = 45000


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if str(p) != value:
        raise ValueError(f"Noncanonical relative path: {value!r}")
    return value


def no_symlink_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symbolic link in path: {part}")
    return path


def new_destination(path):
    path = no_symlink_path(path)
    if os.path.lexists(path):
        raise ValueError(f"Destination exists; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scan(source, exclude_checkpoints=False):
    files, excluded = {}, []
    for p in sorted(source.rglob("*")):
        rel = p.relative_to(source).as_posix()
        safe_name(rel)
        mode = p.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symbolic link in source: {rel}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Nonregular source file: {rel}")
        if p.name.endswith(".lock") or p.name == ".writer.lock":
            raise ValueError(f"Lock file present; stop writers and resolve lock first: {rel}")
        if (p.name.endswith((".tmp", ".temp", "~")) or p.name.startswith(".tmp-")
                or (exclude_checkpoints and "checkpoints" in p.relative_to(source).parts)):
            excluded.append(rel)
            continue
        files[rel] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    if not files:
        raise ValueError("Source has no included regular files")
    return files, excluded


def publish_directory(staged, output):
    # Reserve the destination without overwriting even an empty directory.
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        for child in staged.iterdir():
            shutil.move(str(child), str(output / child.name))
    except BaseException:
        shutil.rmtree(output)
        raise


def write_inventory(staged, files):
    raw = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    parts = []
    for index, offset in enumerate(range(0, len(raw), CHUNK_BYTES)):
        block = raw[offset:offset + CHUNK_BYTES]
        name = f"inventory-part-{index:05d}.bin"
        (staged / name).write_bytes(block)
        parts.append({"path": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "parts": parts}


def expand_inventory(snapshot, manifest):
    if not isinstance(manifest, dict) or "file_inventory" not in manifest:
        return manifest
    if "files" in manifest:
        raise ValueError("Ambiguous inline and external file inventories")
    entry = manifest["file_inventory"]
    if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] <= 0
            or not valid_digest(entry.get("sha256")) or not isinstance(entry.get("parts"), list)):
        raise ValueError("Invalid external inventory")
    blocks, total = [], 0
    for index, part in enumerate(entry["parts"]):
        name = f"inventory-part-{index:05d}.bin"
        size = min(CHUNK_BYTES, entry["bytes"] - total)
        if (not isinstance(part, dict) or part.get("path") != name or type(part.get("bytes")) is not int
                or part["bytes"] != size or not 0 < size <= CHUNK_BYTES or not valid_digest(part.get("sha256"))):
            raise ValueError("Invalid external inventory chunk")
        path = no_symlink_path(snapshot / name)
        if not path.is_file():
            raise ValueError("Missing external inventory chunk")
        block = path.read_bytes()
        if len(block) != size or hashlib.sha256(block).hexdigest() != part["sha256"]:
            raise ValueError("External inventory chunk hash/size mismatch")
        blocks.append(block)
        total += size
    raw = b"".join(blocks)
    if total != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("External inventory total hash/size mismatch")
    files = json.loads(raw)
    if (not isinstance(files, dict) or type(manifest.get("file_count")) is not int
            or len(files) != manifest["file_count"]):
        raise ValueError("External inventory file count mismatch")
    return {**manifest, "files": files}


def create_snapshot(source, output, *, quiescent=False, exclude_checkpoints=False):
    if not quiescent:
        raise ValueError("Creation requires --quiescent: stop writers or pause the completed stage")
    source = no_symlink_path(source)
    if not source.is_dir():
        raise ValueError("Source must be an existing directory")
    output = no_symlink_path(output)
    if source == output or source in output.parents:
        raise ValueError("Snapshot output must be outside its source directory")
    output = new_destination(output)
    files, excluded = scan(source, exclude_checkpoints)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".snapshot-") as temp:
        temp = Path(temp)
        archive = temp / "archive.tar.gz"
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=9) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for name, entry in files.items():
                        blob = (source / name).read_bytes()
                        if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                            raise ValueError(f"Source changed while packaging: {name}")
                        item = tarfile.TarInfo(name)
                        item.size = len(blob)
                        item.mode = 0o644
                        item.mtime = item.uid = item.gid = 0
                        item.uname = item.gname = ""
                        tar.addfile(item, io.BytesIO(blob))
        if scan(source, exclude_checkpoints) != (files, excluded):
            raise ValueError("Source changed during snapshot; pause writers and retry")
        staged = temp / "parts"
        staged.mkdir()
        parts = []
        with archive.open("rb") as stream:
            for index, block in enumerate(iter(lambda: stream.read(CHUNK_BYTES), b"")):
                name = f"part-{index:05d}.bin"
                (staged / name).write_bytes(block)
                parts.append({"path": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        manifest = {"schema": SCHEMA, "archive_format": "tar+gzip", "chunk_bytes": CHUNK_BYTES,
                    "archive_bytes": archive.stat().st_size, "archive_sha256": sha_file(archive),
                    "files": files, "parts": parts, "excluded_paths": excluded,
                    "exclude_checkpoints": bool(exclude_checkpoints),
                    "consistency": "caller_asserted_quiescent_and_two_file_hash_scans_matched"}
        encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if len(encoded) > MAX_MANIFEST_BYTES:
            manifest["file_inventory"] = write_inventory(staged, manifest.pop("files"))
            manifest["file_count"] = len(files)
            encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("Snapshot has too many transport entries; split source into smaller snapshots")
        (staged / "manifest.json").write_bytes(encoded)
        publish_directory(staged, output)
    return {**manifest, "files": files}


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(m):
    if not isinstance(m, dict):
        raise ValueError("Checkpoint manifest must be an object")
    if m.get("schema") != SCHEMA or m.get("archive_format") != "tar+gzip" or m.get("chunk_bytes") != CHUNK_BYTES:
        raise ValueError("Unsupported checkpoint manifest")
    if type(m.get("archive_bytes")) is not int or m["archive_bytes"] <= 0 or not valid_digest(m.get("archive_sha256")):
        raise ValueError("Invalid archive identity")
    if not isinstance(m.get("parts"), list) or not m["parts"] or not isinstance(m.get("files"), dict) or not m["files"]:
        raise ValueError("Empty or invalid checkpoint members")
    for name, entry in m["files"].items():
        safe_name(name)
        if not isinstance(entry, dict) or type(entry.get("bytes")) is not int or entry["bytes"] < 0 or not valid_digest(entry.get("sha256")):
            raise ValueError(f"Invalid file identity: {name}")
        for parent in PurePosixPath(name).parents:
            if str(parent) in m["files"]:
                raise ValueError("File/directory path collision")
    total = 0
    for i, part in enumerate(m["parts"]):
        expected = min(CHUNK_BYTES, m["archive_bytes"] - total)
        if (not isinstance(part, dict) or part.get("path") != f"part-{i:05d}.bin"
                or type(part.get("bytes")) is not int or part["bytes"] != expected
                or not 0 < expected <= CHUNK_BYTES or not valid_digest(part.get("sha256"))):
            raise ValueError(f"Invalid chunk identity/order: {i}")
        total += expected
    if total != m["archive_bytes"]:
        raise ValueError("Incomplete chunk sequence")


def verify_tar(archive, files):
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = safe_name(member.name)
            if not member.isfile() or member.issparse() or name in seen or name not in files:
                raise ValueError(f"Invalid/duplicate archive member: {name}")
            seen.add(name)
            expected = files[name]
            if member.size != expected["bytes"]:
                raise ValueError(f"Archive member size mismatch: {name}")
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable member: {name}")
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {name}")
    if seen != set(files):
        raise ValueError("Archive file inventory differs from manifest")


def restore_snapshot(snapshot, output):
    snapshot = no_symlink_path(snapshot)
    output = new_destination(output)
    manifest_path = no_symlink_path(snapshot / "manifest.json")
    m = expand_inventory(snapshot, json.loads(manifest_path.read_text(encoding="utf-8")))
    validate_manifest(m)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".restore-") as temp:
        temp = Path(temp)
        archive = temp / "archive.tar.gz"
        with archive.open("wb") as stream:
            for part in m["parts"]:
                p = no_symlink_path(snapshot / part["path"])
                if not p.is_file() or p.stat().st_size != part["bytes"] or sha_file(p) != part["sha256"]:
                    raise ValueError(f"Chunk hash or size mismatch: {part['path']}")
                with p.open("rb") as data:
                    shutil.copyfileobj(data, stream)
        if archive.stat().st_size != m["archive_bytes"] or sha_file(archive) != m["archive_sha256"]:
            raise ValueError("Combined archive hash or size mismatch")
        verify_tar(archive, m["files"])
        # All identities have been checked before any member is extracted.
        staged = temp / "restored"
        staged.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                path = staged / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as data, path.open("xb") as target:
                    shutil.copyfileobj(data, target)
        publish_directory(staged, output)
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    c = subs.add_parser("create")
    c.add_argument("--source", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--quiescent", action="store_true")
    c.add_argument("--exclude-checkpoints", action="store_true", help="Omit per-text cache files; keep final scores")
    r = subs.add_parser("restore")
    r.add_argument("--snapshot", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    m = (create_snapshot(a.source, a.output, quiescent=a.quiescent, exclude_checkpoints=a.exclude_checkpoints)
         if a.action == "create" else restore_snapshot(a.snapshot, a.output))
    print(json.dumps({"action": a.action, "files": len(m["files"]), "parts": len(m["parts"]),
                      "archive_bytes": m["archive_bytes"], "archive_sha256": m["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
