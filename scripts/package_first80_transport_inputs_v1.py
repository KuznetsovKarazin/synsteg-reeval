#!/usr/bin/env python3
"""Preserve/restore exact first80 metadata binding; no codec/model calls."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ID = "prospective-development-transport-first80-v1"
BINDING = "results/" + ID + "/input-binding"
ART = "results/" + ID + "/input-binding-artifacts"
NAMES = ("records.jsonl.gz", "input_bound_grid.jsonl.gz", "manifest.json")
HELPER = "scripts/package_first80_transport_inputs_v1.py"
LIMIT = 196608
MAX = 32 * 1024 * 1024


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, "duplicate key")
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite value")))


def path(root, name):
    rel = PurePosixPath(name)
    need(type(name) is str and str(rel) == name and not rel.is_absolute()
         and ".." not in rel.parts and "\\" not in name, "unsafe path")
    target = root / name
    for p in (target, *target.parents):
        if p == root:
            break
        need(not p.is_symlink(), "symlink path")
    need(target.resolve().is_relative_to(root), "outside root")
    return target


def new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def source(root):
    directory = path(root, BINDING)
    need({p.name for p in directory.iterdir()} == set(NAMES), "binding exact inventory")
    data = {BINDING + "/" + name: path(root, BINDING + "/" + name).read_bytes() for name in NAMES}
    need(sum(map(len, data.values())) <= MAX, "binding byte limit")
    m = strict(data[BINDING + "/manifest.json"])
    need(m["schema"] == "prospective-first80-input-binding-v1" and m["experiment_id"] == ID,
         "binding identity")
    need(m["model_record_pairs"] == 300 and m["cells"] == 15300 and m["completed_covers"] == 80
         and m["reused_public_conditions"] == 1440 and m["new_public_conditions"] == 480,
         "binding complete denominator")
    need(all(m[k] == 0 for k in ("encode_calls", "decode_calls", "model_calls"))
         and m["new_outcomes_known"] is False, "preexecution metadata only")
    need(set(m["files_sha256"]) == set(NAMES[:2]) and
         all(m["files_sha256"][name] == sha(data[BINDING + "/" + name]) for name in NAMES[:2]),
         "binding file hashes")
    return data


def package(root):
    root = Path(root).resolve()
    need(not path(root, ART).exists(), "package exists; no overwrite")
    data = source(root)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_LZMA) as z:
        for name, raw in sorted(data.items()):
            entry = zipfile.ZipInfo(name, (2026, 10, 26, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            z.writestr(entry, raw, compress_type=zipfile.ZIP_LZMA)
    archive = memory.getvalue()
    need(len(archive) <= MAX, "archive byte limit")
    path(root, ART).mkdir(parents=True, exist_ok=False)
    chunks = []
    for offset in range(0, len(archive), LIMIT):
        raw = archive[offset:offset + LIMIT]
        name = ART + f"/parts/input.zip.part{offset // LIMIT:03d}"
        new(path(root, name), raw)
        chunks.append({"path": name, "offset": offset, "bytes": len(raw), "sha256": sha(raw)})
    m = {"schema": "prospective-first80-binding-chunked-v1", "experiment_id": ID,
         "archive": {"format": "ZIP_LZMA", "bytes": len(archive), "sha256": sha(archive)},
         "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(data.items())},
         "chunk_count": len(chunks), "chunk_bytes_max": LIMIT, "chunks": chunks,
         "helper": {"path": HELPER, "sha256": sha(path(root, HELPER).read_bytes())},
         "encoder_calls": 0, "decoder_calls": 0, "model_calls": 0}
    new(path(root, ART + "/manifest.json"), canonical(m) + b"\n")
    need(source(root) == data, "binding changed during packaging")
    return m


def restore(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    need(not destination.exists(), "restore destination exists; no overwrite")
    m = strict(path(root, ART + "/manifest.json").read_bytes())
    need(set(m) == {"schema", "experiment_id", "archive", "files", "chunk_count", "chunk_bytes_max", "chunks", "helper", "encoder_calls", "decoder_calls", "model_calls"}
         and m["schema"] == "prospective-first80-binding-chunked-v1" and m["experiment_id"] == ID,
         "closed artifact schema")
    need(m["helper"] == {"path": HELPER, "sha256": sha(path(root, HELPER).read_bytes())}, "helper identity")
    need(set(m["files"]) == {BINDING + "/" + name for name in NAMES}, "exact binding members")
    need(type(m["chunk_count"]) is int and 0 < m["chunk_count"] == len(m["chunks"]) <= (MAX + LIMIT - 1) // LIMIT
         and type(m["chunk_bytes_max"]) is int and m["chunk_bytes_max"] == LIMIT, "chunk count/type")
    need(all(type(m[k]) is int and m[k] == 0 for k in ("encoder_calls", "decoder_calls", "model_calls")), "no execution scope")
    need({p.name for p in path(root, ART + "/parts").iterdir()} == {f"input.zip.part{i:03d}" for i in range(m["chunk_count"])}, "exact parts inventory")
    combined = bytearray()
    for i, row in enumerate(m["chunks"]):
        need(set(row) == {"path", "offset", "bytes", "sha256"} and row["path"] == ART + f"/parts/input.zip.part{i:03d}"
             and type(row["offset"]) is int and row["offset"] == len(combined)
             and type(row["bytes"]) is int and 0 < row["bytes"] <= LIMIT, "chunk schema/order")
        raw = path(root, row["path"]).read_bytes()
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "chunk identity")
        combined.extend(raw)
        need(len(combined) <= MAX, "assembled byte limit")
    a = m["archive"]
    need(set(a) == {"format", "bytes", "sha256"} and a["format"] == "ZIP_LZMA"
         and type(a["bytes"]) is int and len(combined) == a["bytes"] and sha(combined) == a["sha256"], "archive identity")
    files = {}
    with zipfile.ZipFile(io.BytesIO(combined)) as z:
        need(z.namelist() == sorted(m["files"]), "exact sorted archive inventory")
        for entry in z.infolist():
            name = entry.filename
            info = m["files"][name]
            need(set(info) == {"bytes", "sha256"} and type(info["bytes"]) is int and 0 <= info["bytes"] <= MAX
                 and entry.file_size == info["bytes"] and stat.S_ISREG(entry.external_attr >> 16)
                 and entry.compress_type == zipfile.ZIP_LZMA, "member type/size")
            raw = z.read(entry)
            need(len(raw) == info["bytes"] and sha(raw) == info["sha256"], "member identity")
            files[name] = raw
    for name, raw in files.items():
        new(path(destination, name), raw)
    need(source(destination) == files, "fresh binding identity")
    return {"schema": "prospective-first80-binding-fresh-restore-v1", "status": "passed",
            "files": len(files), "chunks": len(m["chunks"]), "archive_sha256": a["sha256"],
            "all_primary_bytes_exact": True, "encoder_calls": 0, "decoder_calls": 0, "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "restore"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    need(not args.receipt.exists(), "receipt exists")
    if args.mode == "restore":
        need(args.destination is not None, "destination required")
        result = restore(args.root, args.destination)
    else:
        result = package(args.root)
    new(args.receipt, canonical(result) + b"\n")
    print(json.dumps({"status": result.get("status", "packaged"), "encoder_calls": 0, "decoder_calls": 0, "model_calls": 0}))
