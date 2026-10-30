#!/usr/bin/env python3
"""Preserve all completed offline study files; no scientific recomputation.

Produces compatible native-continuation manifest + ordered 196608-byte parts.
Reassembles the parts and restores every exact byte with frozen generic helpers
in a new temporary tree. The logical ZIP need not be a direct Git blob.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile

ID = "author-candidate-mask-order-v1"
RESULT = Path("results") / ID
OUT = Path("results") / (ID + "-artifacts")
HELPERS = {
    "scripts/restore_native_continuation_artifact_v1.py": "612260031d56dec54a6bee71319ef78acd7a9b43705003b8ef1b80c4551d05e3",
    "scripts/reassemble_baseline_primary_chunks_v1.py": "dcea5c193b04620749c0c161202d27a1bc8efe0aa835b1029a91c2d6af17fe55",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_file(path):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "regular nonsymlink file required")
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) and len(raw) == after.st_size, "changed file")
    return raw


def write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def json_new(path, value):
    write_new(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def inventory(root, relative):
    need(not (root / relative).is_symlink(), "symlink result tree")
    out = {}
    for path in sorted((root / relative).rglob("*")):
        need(not path.is_symlink(), "symlink inside result tree")
        if path.is_dir():
            continue
        out[path.relative_to(root).as_posix()] = raw_file(path)
    return out


def package(root):
    root = Path(root).resolve()
    out = root / OUT
    need(not out.exists(), "existing package refused")
    helpers = {name: raw_file(root / name) for name in HELPERS}
    need(all(sha(raw) == HELPERS[name] for name, raw in helpers.items()), "frozen helper mismatch")
    data = inventory(root, RESULT)
    required = {(RESULT / name).as_posix() for name in ("summary.json", "probe-comparisons.jsonl")}
    need(required <= set(data) and (RESULT / "failure.json").as_posix() not in data,
         "complete analysis required; preserve failures separately")
    summary = json.loads(data[(RESULT / "summary.json").as_posix()])
    need(summary["experiment_id"] == ID and summary["factual_probes"] == 361
         and summary["recorded_rows"] == 18050 and summary["scope"]["native_calls"] == 0,
         "completed scientific denominator differs")
    need(summary["probe_rows_sha256"] == sha(data[(RESULT / "probe-comparisons.jsonl").as_posix()]),
         "probe JSONL hash differs")
    out.mkdir(parents=True, exist_ok=False)
    try:
        archive_path = out / "full.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_LZMA) as archive:
            for name, raw in data.items():
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 26, 0, 0, 0))
                info.compress_type = zipfile.ZIP_LZMA
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw, compress_type=zipfile.ZIP_LZMA)
        archive_raw = raw_file(archive_path)
        archive_identity = {"path": (OUT / "full.zip").as_posix(),
                            "bytes": len(archive_raw), "sha256": sha(archive_raw)}
        manifest = {"schema": "native-continuation-artifact-bundle-v1", "artifact_id": ID,
                    "archive": archive_identity,
                    "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in data.items()}}
        native_path = OUT / "manifest.json"
        json_new(root / native_path, manifest)
        chunks = []
        for index, offset in enumerate(range(0, len(archive_raw), 196608)):
            raw = archive_raw[offset:offset + 196608]
            path = OUT / f"full.zip.part{index:03d}"
            write_new(root / path, raw)
            chunks.append({"path": path.as_posix(), "bytes": len(raw), "sha256": sha(raw), "offset": offset})
        native_raw = raw_file(root / native_path)
        transport = {"schema": "baseline-primary-chunk-transport-v1", "archive": archive_identity,
                     "chunk_bytes_max": 196608, "chunk_count": len(chunks), "chunks": chunks,
                     "native_manifest": {"path": native_path.as_posix(), "bytes": len(native_raw), "sha256": sha(native_raw)}}
        transport_path = OUT / "transport-manifest.json"
        json_new(root / transport_path, transport)
        with tempfile.TemporaryDirectory(prefix="synsteg-mask-order-fresh-") as temporary:
            fresh = Path(temporary)
            for name, raw in helpers.items():
                write_new(fresh / name, raw)
            for path in [native_path, transport_path] + [Path(row["path"]) for row in chunks]:
                write_new(fresh / path, raw_file(root / path))
            commands = [
                [sys.executable, "-B", str(fresh / "scripts/reassemble_baseline_primary_chunks_v1.py"),
                 "--root", str(fresh), "--manifest", str(fresh / transport_path), "--output-root", str(fresh)],
                [sys.executable, "-B", str(fresh / "scripts/restore_native_continuation_artifact_v1.py"),
                 "--manifest", str(fresh / native_path), "--destination", str(fresh / "restored")],
            ]
            processes = []
            for order, command in enumerate(commands):
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                value = {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout,
                         "stderr": proc.stderr, "native_calls": 0}
                processes.append(value)
                json_new(out / f"restore-process-{order}.json", value)
                need(proc.returncode == 0, "frozen reassembly/restore failed")
            restored = inventory(fresh / "restored", RESULT)
            need(restored == data, "fresh restore exact inventory or bytes differ")
            need(raw_file(fresh / OUT / "full.zip") == archive_raw, "fresh logical ZIP differs")
        need(inventory(root, RESULT) == data, "primary input changed during packaging")
        status = {"schema": ID + "-local-preservation", "status": "verified_locally_pending_git",
                  "archive": archive_identity, "primary_file_count": len(data), "chunk_count": len(chunks),
                  "exact_primary_inventory_and_bytes": True, "fresh_restore_from_parts": True,
                  "frozen_helpers": HELPERS, "native_calls": 0, "analysis_reexecuted": False,
                  "git_readback_verified": False,
                  "scope": "Direct small files duplicate archived bytes, not additional scientific observations."}
        json_new(out / "status.json", status)
        direct = []
        for name, raw in data.items():
            if len(raw) > 500000:
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                continue  # Nontext optional primary is preserved in the archive.
            direct.append(name)
        package_text = [(OUT / name).as_posix() for name in ["manifest.json", "transport-manifest.json",
                         "restore-process-0.json", "restore-process-1.json", "status.json"]]
        plan = {"schema": ID + "-primary-save-plan", "include_this_plan": True,
                "text_paths": direct + package_text,
                "binary_paths": [row["path"] for row in chunks],
                "archive_only_primary_paths": [name for name in data if name not in direct],
                "logical_archive_is_not_direct_git_blob": True,
                "restore_helpers_already_in_repository": HELPERS,
                "all_primary_files": manifest["files"]}
        json_new(out / "save-plan.json", plan)
        return status
    except BaseException as exc:
        json_new(out / "preservation-failure.json", {"exception": type(exc).__name__,
                 "message": str(exc), "native_calls": 0, "analysis_reexecuted": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(package(args.root), sort_keys=True))
