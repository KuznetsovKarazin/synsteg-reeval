#!/usr/bin/env python3
"""Archive one successful fixed batch41–50 and verify chunk reassembly plus fresh restore.

This utility never imports a generation runner, a model, or a tokenizer.
It is only valid after the single intended pair finishes successfully.
A failed/unknown run must be preserved separately, without inference retry.
Existing outputs are refused; the unchanged independent auditor is run once.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = None
BATCH_NUMBER = BEFORE = AFTER = RUN_COUNT = BATCH_COUNT = PRIMARY_COUNT = None
BATCH_ID = BATCH = OUT = AUDIT = AUDIT_FAILURE = None
RUN = Path("runs/large-campaign-development-v2/models/tinyllama11")
AUDITOR = Path("results/large-campaign-continuation-v3-audit/audit_bounded_batch.py")
AUDITOR_SHA = "5c53ace045bdee7e67ee03b3d880ef93b9b2fb63f41485cfd960a4099e94b035"
RESTORER = Path("scripts/restore_native_continuation_artifact_v1.py")
RESTORER_SHA = "612260031d56dec54a6bee71319ef78acd7a9b43705003b8ef1b80c4551d05e3"
REASSEMBLER = Path("scripts/reassemble_baseline_primary_chunks_v1.py")
REASSEMBLER_SHA = "dcea5c193b04620749c0c161202d27a1bc8efe0aa835b1029a91c2d6af17fe55"
LOGICAL_ARCHIVE_MAX_BYTES = 16 * 1024 * 1024
PART_BYTES = 196608

def configure(root, batch_number):
    global ROOT,BATCH_NUMBER,BEFORE,AFTER,RUN_COUNT,BATCH_COUNT,PRIMARY_COUNT,BATCH_ID,BATCH,OUT,AUDIT,AUDIT_FAILURE
    require(type(batch_number) is int and 41 <= batch_number <= 50, "Only fixed batches41–50")
    ROOT=Path(root).resolve();BATCH_NUMBER=batch_number;BEFORE=2*batch_number-2;AFTER=BEFORE+2
    RUN_COUNT=AFTER+6;BATCH_COUNT=2*AFTER+21;PRIMARY_COUNT=3*AFTER+27
    BATCH_ID=f"batch{batch_number:02d}"
    BATCH=Path("results/large-campaign-development-v2/batches")/BATCH_ID
    OUT=Path(f"results/continuation-20261026-{BATCH_ID}/artifacts")
    AUDIT=AUDITOR.parent/f"{BATCH_ID}-postrun-audit.json"
    AUDIT_FAILURE=AUDITOR.parent/f"{BATCH_ID}-verification-failure.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def files_below(relative):
    found = {}
    for path in sorted((ROOT / relative).rglob("*")):
        require(not path.is_symlink(), "Refuse symlink: " + str(path))
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            found[name] = path.read_bytes()
    return found

def main():
    require(not (ROOT / OUT).exists(), "Archive output exists; do not replace it")
    require(not (ROOT / AUDIT).exists() and not (ROOT / AUDIT_FAILURE).exists(),
            "Batch audit already exists; do not repeat it")
    require(sha((ROOT / AUDITOR).read_bytes()) == AUDITOR_SHA, "Auditor changed")
    require(sha((ROOT / RESTORER).read_bytes()) == RESTORER_SHA, "Restorer changed")
    require(sha((ROOT / REASSEMBLER).read_bytes()) == REASSEMBLER_SHA, "Reassembler changed")
    process = read_json(ROOT / BATCH / "process.json")
    require(process["status"] == "batch_completed_pending_remote_checkpoint"
            and process["exit_code"] == 0 and process["timed_out"] is False
            and process["new_attempts"] == 2 and process["completed_total"] == AFTER
            and process["prior_evidence_issues"] == [],
            "Require successful pair; preserve failures separately and never retry inference")
    checkpoint = read_json(ROOT / BATCH / "after-checkpoint.json")
    require(checkpoint["completed_generations"] == checkpoint["attempted"] == AFTER
            and checkpoint["planned_model_record_pairs"] == 300
            and checkpoint["status"] == "verified", "Unexpected complete denominator")
    run_data = files_below(RUN)
    batch_data = files_below(BATCH)
    prior_receipt = read_json(ROOT / BATCH / "prior-git-receipt.json")
    require(prior_receipt["expected_completed"] == BEFORE
            and len(prior_receipt["run_files_sha256"]) == BEFORE+6, "Prior snapshot denominator changed")
    derived_after_run = len(prior_receipt["run_files_sha256"]) + 2
    derived_batch = len(prior_receipt["run_files_sha256"]) + derived_after_run + 11
    require((derived_after_run, derived_batch, derived_batch + derived_after_run) == (RUN_COUNT, BATCH_COUNT, PRIMARY_COUNT),
            "Unexpected derived artifact counts")
    require(len(run_data) == derived_after_run and len(batch_data) == derived_batch,
            "Unexpected primary member count")
    require({name: sha(data) for name, data in run_data.items()}
            == checkpoint["run_files_sha256"], "Cumulative run differs from immutable after")
    recorded_batch = read_json(ROOT / BATCH / "artifact-manifest.json")["files_sha256"]
    require(set(recorded_batch) == set(batch_data) - {(BATCH / "artifact-manifest.json").as_posix()},
            "Batch artifact manifest does not cover every other member")
    require(all(sha(batch_data[name]) == expected for name, expected in recorded_batch.items()),
            "Batch member differs from its immutable artifact manifest")
    after_hashes = read_json(ROOT / BATCH / "after/files.json")
    require(after_hashes == checkpoint["run_files_sha256"], "After snapshot identity mismatch")
    for name, data in run_data.items():
        require(batch_data[(BATCH / "after" / name).as_posix()] == data,
                "After snapshot bytes differ: " + name)
    # The auditor writes its own immutable success/failure receipt. Its result
    # is retained even if archiving subsequently fails.
    command = [sys.executable, str(ROOT / AUDITOR), "--root", str(ROOT), "--batch-id", BATCH_ID]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (ROOT / OUT).mkdir(parents=True, exist_ok=False)
    with (ROOT / OUT / "audit.stdout.log").open("x") as stream:
        stream.write(completed.stdout)
    with (ROOT / OUT / "audit.stderr.log").open("x") as stream:
        stream.write(completed.stderr)
    write_json_new(ROOT / OUT / "audit-process.json", {
        "command": command, "exit_code": completed.returncode,
        "auditor_sha256": AUDITOR_SHA, "new_model_calls": 0,
        "new_tokenizer_calls": 0})
    require(completed.returncode == 0 and (ROOT / AUDIT).exists(), "Independent audit failed")
    audit = read_json(ROOT / AUDIT)
    require(audit["status"] == "passed" and audit["completed"] == AFTER
            and audit["prior_completed_entries_preserved"] == BEFORE
            and audit["current_cumulative_run_matches_immutable_after_snapshot"] is True,
            "Independent audit acceptance conditions failed")
    combined = dict(sorted({**batch_data, **run_data}.items()))
    require(len(combined) == PRIMARY_COUNT, "Unexpected primary denominator")
    archive_path = ROOT / OUT / "full.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, strict_timestamps=True) as archive:
        for name, data in combined.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with archive_path.open("rb") as stream:
        os.fsync(stream.fileno())
    archive_bytes = archive_path.read_bytes()
    require(0 < len(archive_bytes) <= LOGICAL_ARCHIVE_MAX_BYTES,
            "Logical archive exceeds16MiB operational bound; preserve exact failure, no audit/inference retry")
    manifest = {
        "schema": "native-continuation-artifact-bundle-v1",
        "artifact_id": "baseline-"+BATCH_ID,
        "archive": {"path": (OUT / "full.zip").as_posix(),
                    "bytes": len(archive_bytes), "sha256": sha(archive_bytes)},
        "files": {name: {"bytes": len(data), "sha256": sha(data)}
                  for name, data in combined.items()}}
    write_json_new(ROOT / OUT / "manifest.json", manifest)
    # Ensure input bytes stayed unchanged while the auditor and ZIP were running.
    require(all((ROOT / name).read_bytes() == data for name, data in combined.items()),
            "Primary input changed during packaging")
    run_list = [{"path": name, "bytes": len(data), "chars": len(data.decode("utf-8")),
                 "sha256": sha(data)} for name, data in run_data.items()]
    write_json_new(ROOT / OUT / "run-files-for-git.json", run_list)
    transport = chunk_archive(archive_bytes, manifest)
    chunks = transport["chunks"]
    restore_result = verify_chunk_restore(manifest, combined)
    write_json_new(ROOT / OUT / "fresh-restore.json", restore_result)
    status = {
        "schema": "baseline81-100-primary-preservation-local-v1", "batch_id": BATCH_ID, "status": "verified_locally_pending_git",
        "primary_members": PRIMARY_COUNT, "immutable_batch_members": BATCH_COUNT, "cumulative_run_files": RUN_COUNT,
        "archive": manifest["archive"], "fresh_restore": restore_result,
        "completed": AFTER, "full_denominator": 300, "prior_completed_prefix_preserved": BEFORE,
        "audit_path": AUDIT.as_posix(), "audit_sha256": sha((ROOT / AUDIT).read_bytes()),
        "new_completed_rows": audit["new_completed_rows"],
        "cumulative_generated_output_tokens": audit["cumulative_generated_output_tokens"],
        "new_model_calls_by_packager": 0, "remote_git_verified": False,
        "next": f"Root saves all parts and all{RUN_COUNT}run files, actual Git readback, then immutable receipt+closure+handoff."}
    write_json_new(ROOT / OUT / "status.json", status)
    text_paths = sorted(set(run_data) | {AUDIT.as_posix()} |
        {p.relative_to(ROOT).as_posix() for p in (ROOT / OUT).iterdir()
         if p.is_file() and p.name != "full.zip" and not p.name.startswith("full.zip.part")})
    text_files = [{"path": name, "bytes": (ROOT/name).stat().st_size,
                   "sha256": sha((ROOT/name).read_bytes())} for name in text_paths]
    # Reading every listed text file as UTF-8 is a storage check only.
    for name in text_paths:
        (ROOT/name).read_bytes().decode("utf-8")
    save_plan = {"schema": "baseline81-100-primary-save-plan-v1", "batch_id": BATCH_ID,
        "text_files": text_files, "binary_parts": chunks,
        "include_this_plan": True, "logical_archive_direct_git_blob": False,
        "primary_members": PRIMARY_COUNT, "current_run_files": RUN_COUNT,
        "required_after_save": f"Read back every listed Git path at immutable primary commit; reconstruct all archive members from actual fetched bytes before issuing after-{BATCH_ID} receipt.",
        "remote_git_verified": False, "model_calls": 0}
    write_json_new(ROOT / OUT / "primary-save-plan.json", save_plan)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def chunk_archive(archive_bytes, manifest):
    require(0 < len(archive_bytes) <= LOGICAL_ARCHIVE_MAX_BYTES, "Logical archive bound")
    # Small raw Git parts retain the exact logical archive and avoid upload limits.
    chunks = []
    for index, offset in enumerate(range(0, len(archive_bytes), PART_BYTES)):
        data = archive_bytes[offset:offset+PART_BYTES]
        path = OUT / ("full.zip.part" + str(index).zfill(3))
        with (ROOT / path).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        chunks.append({"path": path.as_posix(), "bytes": len(data),
                       "sha256": sha(data), "offset": offset})
    manifest_path = OUT / "manifest.json"
    native_raw = (ROOT / manifest_path).read_bytes()
    transport = {"schema": "baseline-primary-chunk-transport-v1",
        "archive": manifest["archive"], "chunk_bytes_max": PART_BYTES,
        "chunk_count": len(chunks), "chunks": chunks,
        "native_manifest": {"path": manifest_path.as_posix(),
                            "bytes": len(native_raw), "sha256": sha(native_raw)}}
    write_json_new(ROOT / OUT / "transport-manifest.json", transport)
    return transport

def verify_chunk_restore(manifest, combined):
    """Reassemble from parts in a fresh tree and invoke the unchanged restorer once."""
    require(sha((ROOT / REASSEMBLER).read_bytes())==REASSEMBLER_SHA,"Reassembler changed")
    require(sha((ROOT / RESTORER).read_bytes())==RESTORER_SHA,"Restorer changed")
    spec=importlib.util.spec_from_file_location("frozen_baseline_reassembler",ROOT/REASSEMBLER)
    reassembler=importlib.util.module_from_spec(spec);spec.loader.exec_module(reassembler)
    with tempfile.TemporaryDirectory(prefix=f"synsteg-{BATCH_ID}-parts-restore-") as temporary:
        fresh=Path(temporary);archive_root=fresh/'archive-root';archive_root.mkdir()
        result=reassembler.reassemble(ROOT,ROOT/OUT/'transport-manifest.json',archive_root)
        require(result['status']=='passed' and result['archive_written'] is True,"Fresh part reassembly failed")
        raw=(archive_root/manifest['archive']['path']).read_bytes()
        require(raw==(ROOT/OUT/'full.zip').read_bytes(),"Reassembled archive differs from original")
        copied_manifest=archive_root/OUT/'manifest.json';copied_manifest.write_bytes((ROOT/OUT/'manifest.json').read_bytes())
        helper=archive_root/RESTORER;helper.parent.mkdir(parents=True,exist_ok=True);helper.write_bytes((ROOT/RESTORER).read_bytes())
        rspec=importlib.util.spec_from_file_location("frozen_native_restorer_from_parts",helper)
        restorer=importlib.util.module_from_spec(rspec);rspec.loader.exec_module(restorer)
        destination=fresh/'restored';restore_result=restorer.restore(copied_manifest,destination)
        require(restore_result=={'verified_members':len(combined),'written_members':len(combined),'model_calls':0},"Fresh restore result differs")
        require(all((destination/name).read_bytes()==data for name,data in combined.items()),"Fresh restore bytes differ")
    return {**restore_result,'reassembled_from_parts':True,'archive_bytes':manifest['archive']['bytes'],'archive_sha256':manifest['archive']['sha256']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch-number', type=int, choices=range(41,51), required=True)
    args=parser.parse_args()
    configure(args.root,args.batch_number)
    main()
