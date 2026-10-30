"""Artifact preservation tests use fake terminal files only; no model/tokenizer."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bounded_packager", ROOT/"scripts/package_framed_interval_bounded_v1_artifacts.py")
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def put(root, path, data):
    target = root/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if type(data) is bytes else json.dumps(data).encode())
    return target


def fake_inputs(tmp_path):
    campaign, wrapper = pack.DEFAULT_CAMPAIGN, pack.DEFAULT_WRAPPER
    runs = [{"run_id": rid, "arm": rid.rsplit("-", 1)[1]} for rid in pack.EXPECTED_RUNS]
    put(tmp_path, campaign+"/campaign-manifest.json", {
        "schema": pack.VERSION+"-sixteen-case-manifest-v1", "public_runs": runs,
        "planned_cases": 16, "planned_fresh_stego_receivers": 8})
    put(tmp_path, campaign+"/campaign-status.json", {"state": "finished_all_planned_outcomes",
        "authenticated_payload_deliveries": 0, "authenticated_delivery_denominator": 8})
    for run in runs:
        for role in ("sender", "receiver") if run["arm"] == "stego" else ("sender",):
            prefix = campaign+"/"+run["run_id"]+"/"+role
            put(tmp_path, prefix+"/worker-result.json", {"status": "synthetic_refusal"})
            if role == "sender":
                put(tmp_path, prefix+"/committed-wire.bin", b"Known synthetic prefix.")
    put(tmp_path, campaign+"/station-c00-stego/sender/partial.json.writing", b'{"incomplete":')
    raw = b"".join(hashlib.sha256(str(n).encode()).digest() for n in range(400))
    put(tmp_path, campaign+"/station-c00-stego/sender/token-journal.jsonl", raw)
    put(tmp_path, wrapper+"/launch.json", {"automatic_retries": 0})
    put(tmp_path, wrapper+"/process.json", {"exit_code": 0, "finished_at_unix": 1.0})
    put(tmp_path, wrapper+"/stdout.log", b"")
    put(tmp_path, wrapper+"/stderr.log", b"")
    return campaign, wrapper


def test_complete_roundtrip_split_parts_exact_text_and_no_overwrite(tmp_path, monkeypatch):
    campaign, wrapper = fake_inputs(tmp_path)
    monkeypatch.setattr(pack, "DIRECT_LIMIT", 800)
    monkeypatch.setattr(pack, "PART_LIMIT", 512)
    before = {**pack.inventory(tmp_path, campaign), **pack.inventory(tmp_path, wrapper)}
    result = pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    assert result["bundles"] == 25 and result["endpoint_slots"] == 24
    index = json.loads((tmp_path/result["index"]).read_bytes())
    assert index["primary_files"] == before
    assert result["upload_file_count"] == len(index["upload_files"])
    assert all((tmp_path/path).is_file() for path in index["upload_files"])
    assert any(path.endswith(".part0001") for path in index["upload_files"])
    assert all((tmp_path/path).stat().st_size <= 512
        for path in index["upload_files"] if ".zip.part" in path)
    assert any(path.endswith(".writing") for path in index["primary_files"])
    destination = tmp_path/"restored"
    restored = pack.restore(tmp_path, result["index"], destination)
    assert restored["verified_primary_files"] == len(before) == restored["written_primary_files"]
    assert {p.relative_to(destination).as_posix(): pack.identity(p)
        for p in destination.rglob("*") if p.is_file()} == before
    assert pack.restore(tmp_path, result["index"], destination)["written_primary_files"] == 0
    text = tmp_path/pack.DEFAULT_OUTPUT/"texts/station-c00-stego.txt"
    assert text.read_bytes() == b"Known synthetic prefix."
    existing = destination/campaign/"station-c00-stego/sender/committed-wire.bin"
    existing.write_bytes(b"DIFFERENT")
    with pytest.raises(ValueError, match="existing different"):
        pack.restore(tmp_path, result["index"], destination)
    assert existing.read_bytes() == b"DIFFERENT"
    with pytest.raises(FileExistsError):
        pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    assert {**pack.inventory(tmp_path, campaign), **pack.inventory(tmp_path, wrapper)} == before


def test_corrupted_storage_rejected_before_restore_writes(tmp_path):
    campaign, wrapper = fake_inputs(tmp_path)
    result = pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    index = json.loads((tmp_path/result["index"]).read_bytes())
    archive = next(path for path in index["upload_files"] if path.endswith(".zip"))
    with (tmp_path/archive).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="identity differs"):
        pack.restore(tmp_path, result["index"], tmp_path/"restored")
    assert not (tmp_path/"restored").exists()


def test_terminal_receipt_required_before_any_artifact_write(tmp_path):
    campaign, wrapper = fake_inputs(tmp_path)
    put(tmp_path, wrapper+"/process.json", {"exit_code": None})
    with pytest.raises(ValueError, match="terminal process"):
        pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    assert not (tmp_path/pack.DEFAULT_OUTPUT).exists()


def test_unknown_zero_invalid_utf8_and_missing_slots_are_retained(tmp_path):
    campaign, wrapper = fake_inputs(tmp_path)
    put(tmp_path, campaign+"/station-c00-stego/sender/committed-wire.bin", b"")
    put(tmp_path, campaign+"/station-c00-ordinary/sender/committed-wire.bin", b"\xff")
    receiver = tmp_path/campaign/"station-c00-stego/receiver"
    (receiver/"worker-result.json").unlink(); receiver.rmdir()
    put(tmp_path, campaign+"/campaign-abort.json", {"state": "aborted_campaign_error",
        "authenticated_payload_deliveries": 0, "authenticated_delivery_denominator": 8,
        "unknown": True})
    result = pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    index = json.loads((tmp_path/result["index"]).read_bytes())
    slot = next(s for s in index["endpoint_slots"]
        if s["run_id"] == "station-c00-stego" and s["role"] == "receiver")
    assert slot["member_count"] == 0 and not slot["directory_exists"]
    summary = json.loads((tmp_path/pack.DEFAULT_OUTPUT/"summary.json").read_bytes())
    assert summary["campaign_state"] == "aborted_campaign_error"
    rows = {r["run_id"]: r for r in summary["text_copies"]}
    assert rows["station-c00-stego"]["bytes"] == 0
    assert rows["station-c00-ordinary"]["utf8_valid"] is False
    assert "path" not in rows["station-c00-ordinary"]
    assert any(p.endswith("campaign-status.json") for p in index["primary_files"])
    assert any(p.endswith("campaign-abort.json") for p in index["primary_files"])


def test_source_symlink_refused(tmp_path):
    campaign, wrapper = fake_inputs(tmp_path)
    (tmp_path/campaign/"private-link").symlink_to(tmp_path/wrapper/"process.json")
    with pytest.raises(ValueError, match="symlink"):
        pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    assert not (tmp_path/pack.DEFAULT_OUTPUT).exists()


def test_source_mutation_during_packaging_is_detected_and_partial_evidence_kept(tmp_path, monkeypatch):
    campaign, wrapper = fake_inputs(tmp_path)
    original = pack.make_bundle
    changed = False
    def mutate(root, output, artifact_id, files):
        nonlocal changed
        result = original(root, output, artifact_id, files)
        if not changed:
            changed = True
            put(tmp_path, campaign+"/station-c00-stego/sender/worker-result.json", b"changed")
        return result
    monkeypatch.setattr(pack, "make_bundle", mutate)
    with pytest.raises(ValueError, match="changed during packaging"):
        pack.package(tmp_path, campaign, wrapper, pack.DEFAULT_OUTPUT)
    output = tmp_path/pack.DEFAULT_OUTPUT
    assert (output/"packaging-failure.json").is_file()
    failure = json.loads((output/"packaging-failure.json").read_bytes())
    assert failure["source_modified_by_packager"] is False
    assert failure["source_snapshot_stability"] == "unknown_after_failure"
    assert "source_files_changed" not in failure
    assert not (output/"index.json").exists()
    assert list((output/"archives").glob("*.zip"))
