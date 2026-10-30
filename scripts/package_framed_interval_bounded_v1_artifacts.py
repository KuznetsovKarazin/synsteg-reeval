#!/usr/bin/env python3
"""Package/restore completed bounded-v1 evidence. Never imports or calls a model.

All source bytes, including partial/UNKNOWN/.writing artifacts, are preserved.
ZIP_LZMA bundles exceeding the conservative direct-file limit are represented
only by <=512KiB parts. This script does NOT audit scientific validity or repeat
generation/decoding; every manifest count refers to artifact preservation.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "framed-interval-bounded-v1"
DIRECT_LIMIT = 1_000_000
PART_LIMIT = 512 * 1024
DEFAULT_CAMPAIGN = "results/framed-interval-bounded-v1/native-pilot-v1"
DEFAULT_WRAPPER = "results/framed-interval-bounded-v1-process/native-pilot-v1"
DEFAULT_OUTPUT = "results/framed-interval-bounded-v1-artifacts"
EXPECTED_RUNS = tuple(f"{prompt}-{config}-{arm}" for prompt in ("station", "library")
    for config in ("c00", "c01", "c02", "c03") for arm in ("stego", "ordinary"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("canonical relative path required")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or str(p) != value:
        raise ValueError("unsafe relative path")
    return p


def safe_path(root, relative):
    rel = relative_path(relative)
    root = Path(root).resolve()
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlinked artifact path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("artifact escapes root")
    return path


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def identity(path, *, relative=None):
    data = Path(path).read_bytes()
    result = {"bytes": len(data), "sha256": sha(data)}
    if relative is not None:
        result = {"path": relative, **result}
    return result


def inventory(root, directory):
    path = safe_path(root, directory)
    if not path.is_dir():
        raise ValueError("input directory missing: "+directory)
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("symlink inside artifact input")
        if item.is_file():
            name = item.relative_to(root).as_posix()
            files[name] = identity(item)
        elif not item.is_dir():
            raise ValueError("nonregular artifact input")
    return files


def check_identity(raw, row, label):
    if type(row) is not dict or type(row.get("bytes")) is not int or row["bytes"] < 0:
        raise ValueError("invalid identity: "+label)
    if len(raw) != row["bytes"] or sha(raw) != row.get("sha256"):
        raise ValueError("identity differs: "+label)


def archive_bytes(root, manifest):
    archive = manifest["archive"]
    storage = archive["storage"]
    if not storage or [p["index"] for p in storage] != list(range(len(storage))):
        raise ValueError("archive storage ordering differs")
    if archive["storage_kind"] not in {"single", "parts"}:
        raise ValueError("unknown archive storage kind")
    if archive["storage_kind"] == "single" and len(storage) != 1:
        raise ValueError("single archive has multiple objects")
    chunks = []
    for part in storage:
        raw = safe_path(root, part["path"]).read_bytes()
        check_identity(raw, part, part["path"])
        if archive["storage_kind"] == "parts" and len(raw) > PART_LIMIT:
            raise ValueError("archive part exceeds512KiB")
        chunks.append(raw)
    raw = b"".join(chunks)
    check_identity(raw, archive, manifest["artifact_id"])
    return raw


def verified_members(root, manifest):
    if manifest.get("schema") != VERSION+"-artifact-bundle-v1":
        raise ValueError("unexpected bundle schema")
    raw = archive_bytes(root, manifest)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("exact archive member set differs")
        output = {}
        for name in names:
            relative_path(name)
            info = z.getinfo(name)
            if info.is_dir():
                raise ValueError("file-only bundle expected")
            data = z.read(name)
            check_identity(data, manifest["files"][name], name)
            output[name] = data
    return output


def restore_bundle(root, manifest, destination):
    dest = Path(destination).resolve()
    files = verified_members(root, manifest)
    pending = []
    for name, data in files.items():
        target = safe_path(dest, name)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError("existing different restore target: "+name)
        else:
            pending.append((target, data))
    for path, data in pending:
        write_once(path, data)
    return {"verified_members": len(files), "written_members": len(pending),
        "verified_bytes": sum(len(data) for data in files.values()), "model_calls": 0}


def make_bundle(root, output, artifact_id, files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, meta in sorted(files.items()):
            data = safe_path(root, name).read_bytes()
            check_identity(data, meta, name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, data, compress_type=zipfile.ZIP_LZMA)
    raw = buffer.getvalue()
    part_kind = "single" if len(raw) <= DIRECT_LIMIT else "parts"
    if part_kind == "single":
        segments = [(f"{artifact_id}.zip", raw)]
    else:
        segments = [(f"{artifact_id}.zip.part{offset//PART_LIMIT:04d}", raw[offset:offset+PART_LIMIT])
            for offset in range(0, len(raw), PART_LIMIT)]
    storage = []
    for index, (name, data) in enumerate(segments):
        relative = output + "/archives/" + name
        write_once(safe_path(root, relative), data)
        storage.append({"index": index, "path": relative, "bytes": len(data), "sha256": sha(data)})
    manifest = {"schema": VERSION+"-artifact-bundle-v1", "artifact_id": artifact_id,
        "compression": "ZIP_LZMA", "archive": {"logical_name": artifact_id+".zip",
            "bytes": len(raw), "sha256": sha(raw), "storage_kind": part_kind, "storage": storage},
        "files": files, "model_calls": 0}
    path = output + "/manifests/" + artifact_id + ".json"
    write_once(safe_path(root, path), json_bytes(manifest))
    return path, manifest


def terminal_inputs(root, campaign, wrapper):
    if not campaign.startswith("results/"+VERSION+"/"):
        raise ValueError("dedicated bounded campaign required")
    if not wrapper.startswith("results/"+VERSION+"-process/"):
        raise ValueError("dedicated process wrapper required")
    campaign_path, wrapper_path = safe_path(root, campaign), safe_path(root, wrapper)
    if campaign_path == wrapper_path or campaign_path in wrapper_path.parents or wrapper_path in campaign_path.parents:
        raise ValueError("campaign and wrapper overlap")
    manifest = json.loads((campaign_path/"campaign-manifest.json").read_bytes())
    if (manifest.get("schema") != VERSION+"-sixteen-case-manifest-v1"
            or tuple(r["run_id"] for r in manifest["public_runs"]) != EXPECTED_RUNS
            or manifest.get("planned_cases") != 16
            or manifest.get("planned_fresh_stego_receivers") != 8):
        raise ValueError("complete fixed16+8 campaign manifest required")
    process = json.loads((wrapper_path/"process.json").read_bytes())
    if (type(process.get("exit_code")) is not int
            or type(process.get("finished_at_unix")) not in (int, float)
            or not math.isfinite(process["finished_at_unix"])):
        raise ValueError("terminal process receipt required before packaging")
    for name in ("launch.json", "stdout.log", "stderr.log"):
        if not (wrapper_path/name).is_file():
            raise ValueError("missing process wrapper: "+name)
    # Aborts/final fallback are retained alongside the older status snapshot.
    status_name = next((name for name in ("campaign-abort.json", "campaign-final-status.json",
        "campaign-status.json") if (campaign_path/name).exists()), None)
    if status_name is None:
        raise ValueError("campaign status evidence missing")
    ledger = json.loads((campaign_path/status_name).read_bytes())
    return manifest, process, ledger, status_name


def package(root, campaign, wrapper, output):
    root = Path(root).resolve()
    if not output.startswith("results/"+VERSION+"-artifacts"):
        raise ValueError("dedicated new artifact output required")
    manifest, process, ledger, status_name = terminal_inputs(root, campaign, wrapper)
    campaign_files, wrapper_files = inventory(root, campaign), inventory(root, wrapper)
    if set(campaign_files) & set(wrapper_files):
        raise ValueError("duplicate primary input")
    sources = {**campaign_files, **wrapper_files}
    destination = safe_path(root, output)
    if destination.exists():
        raise FileExistsError("artifact output exists; no overwrite")
    destination.mkdir(parents=True)
    try:
        bundles, covered, slots, text_copies = [], set(), [], []
        for run in manifest["public_runs"]:
            for role in ("sender", "receiver") if run["arm"] == "stego" else ("sender",):
                prefix = campaign+"/"+run["run_id"]+"/"+role
                selected = {p: info for p, info in campaign_files.items() if p.startswith(prefix+"/")}
                artifact_id = run["run_id"]+"-"+role
                path, bundle = make_bundle(root, output, artifact_id, selected)
                bundles.append({"manifest": path, "artifact_id": artifact_id})
                covered.update(selected)
                slots.append({"run_id": run["run_id"], "role": role, "path": prefix,
                    "directory_exists": safe_path(root, prefix).is_dir(),
                    "member_count": len(selected), "manifest": path})
                if role == "sender":
                    wire_path = prefix+"/committed-wire.bin"
                    row = {"run_id": run["run_id"], "source": wire_path, "available": wire_path in selected}
                    if row["available"]:
                        raw = safe_path(root, wire_path).read_bytes()
                        row.update(bytes=len(raw), sha256=sha(raw))
                        try:
                            text = raw.decode("utf-8", "strict")
                        except UnicodeDecodeError as exc:
                            row.update(utf8_valid=False, reason=str(exc))
                        else:
                            text_path = output+"/texts/"+run["run_id"]+".txt"
                            write_once(safe_path(root, text_path), raw)
                            row.update(utf8_valid=True, path=text_path, whitespace_words=len(text.split()),
                                exact_source_bytes=True)
                    text_copies.append(row)
        remainder = {p: info for p, info in sources.items() if p not in covered}
        path, bundle = make_bundle(root, output, "campaign-and-process", remainder)
        bundles.append({"manifest": path, "artifact_id": "campaign-and-process"})
        covered.update(remainder)
        if covered != set(sources) or len(slots) != 24:
            raise AssertionError("complete primary partition differs")
        validation = {"schema": VERSION+"-artifact-fresh-restore-v1", "status": "passed",
            "bundle_count": len(bundles), "endpoints": 24, "bundles": [], "model_calls": 0}
        with tempfile.TemporaryDirectory(prefix="synsteg-bounded-fresh-restore-") as tmp:
            restored = Path(tmp)
            for bundle_row in bundles:
                m = json.loads(safe_path(root, bundle_row["manifest"]).read_bytes())
                result = restore_bundle(root, m, restored)
                validation["bundles"].append({**bundle_row, **result})
            recovered_files = {p.relative_to(restored).as_posix(): identity(p)
                for p in restored.rglob("*") if p.is_file()}
            if recovered_files != sources:
                raise ValueError("fresh full restore differs from complete input inventory")
        # A process or another writer changing evidence during packaging is an error.
        if inventory(root, campaign) != campaign_files or inventory(root, wrapper) != wrapper_files:
            raise ValueError("primary input changed during packaging")
        validation["verified_primary_files"] = len(sources)
        validation["verified_primary_bytes"] = sum(row["bytes"] for row in sources.values())
        write_once(destination/"fresh-restore.json", json_bytes(validation))
        summary = {"schema": VERSION+"-packaged-outcome-summary-v1",
            "scope": "Copies declared terminal metadata; not an independent scientific audit",
            "campaign": campaign, "process_wrapper": wrapper,
            "selected_terminal_status": campaign+"/"+status_name,
            "campaign_state": ledger.get("state"), "process_exit_code": process["exit_code"],
            "planned_cases": 16, "planned_fresh_stego_receivers": 8,
            "declared_authenticated_payload_deliveries": ledger.get("authenticated_payload_deliveries"),
            "declared_authenticated_delivery_denominator": ledger.get("authenticated_delivery_denominator"),
            "all_failure_unknown_and_unattempted_outcomes_preserved": True,
            "text_copies": text_copies, "model_calls": 0}
        write_once(destination/"summary.json", json_bytes(summary))
        artifacts = inventory(root, output)
        index_path = output+"/index.json"
        upload_files = sorted([*artifacts, index_path])
        index = {"schema": VERSION+"-complete-primary-artifacts-v1", "status": "packaged_and_fresh_restore_verified",
            "campaign": campaign, "process_wrapper": wrapper, "primary_files": sources,
            "primary_file_count": len(sources), "endpoint_slots": slots, "bundles": bundles,
            "artifacts_excluding_this_index": artifacts, "upload_files": upload_files,
            "index_self_hash_external": True, "direct_file_max_bytes": DIRECT_LIMIT,
            "split_part_max_bytes": PART_LIMIT, "model_calls": 0,
            "freeze_and_actual_Git_readback_still_required": True}
        write_once(destination/"index.json", json_bytes(index))
        return {"status": index["status"], "index": index_path,
            "index_sha256": sha(json_bytes(index)), "primary_files": len(sources),
            "endpoint_slots": len(slots), "bundles": len(bundles), "upload_file_count": len(upload_files),
            "model_calls": 0}
    except Exception as exc:
        failure = destination/"packaging-failure.json"
        if not failure.exists():
            write_once(failure, json_bytes({"schema": VERSION+"-packaging-failure-v1",
                "error_type": type(exc).__name__, "error": str(exc), "source_modified_by_packager": False,
                "source_snapshot_stability": "unknown_after_failure",
                "note": "Partial output retained; no input source or model was modified/called."}))
        raise


def restore(root, index_path, destination):
    root = Path(root).resolve()
    index = json.loads(safe_path(root, index_path).read_bytes())
    if index.get("schema") != VERSION+"-complete-primary-artifacts-v1":
        raise ValueError("complete artifact index required")
    for path, row in index["artifacts_excluding_this_index"].items():
        check_identity(safe_path(root, path).read_bytes(), row, path)
    seen, results = {}, []
    for row in index["bundles"]:
        m = json.loads(safe_path(root, row["manifest"]).read_bytes())
        if set(seen) & set(m["files"]):
            raise ValueError("duplicate primary member in bundles")
        seen.update(m["files"])
        results.append({"artifact_id": row["artifact_id"], **restore_bundle(root, m, destination)})
    if seen != index["primary_files"]:
        raise ValueError("complete bundle/index membership differs")
    return {"status": "verified_and_restored", "verified_primary_files": len(seen),
        "written_primary_files": sum(r["written_members"] for r in results),
        "bundle_count": len(results), "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("package")
    p.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    p.add_argument("--process-wrapper", default=DEFAULT_WRAPPER)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    r = subs.add_parser("restore")
    r.add_argument("--index", default=DEFAULT_OUTPUT+"/index.json")
    r.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        result = package(ROOT, args.campaign, args.process_wrapper, args.output)
    else:
        result = restore(ROOT, args.index, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
