import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import restore_recovery_cloud_state as restore
import run_budget_generation as runner


RUNTIME = {"schema": "synsteg-local-runtime-recovery-v1", "python": "fixture-python",
           "python_implementation": "CPython", "platform": "fixture-platform", "machine": "fixture-cpu",
           "processor": "fixture-cpu", "packages": {name: "fixture-version" for name in
               ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")}}


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def refresh_manifest(repo, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    source = repo / manifest["snapshot_root"]
    manifest["files"] = [{"path": p.relative_to(source).as_posix(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
        for p in sorted(source.rglob("*")) if p.is_file()]
    write_json(manifest_path, manifest)


@pytest.fixture
def snapshot(tmp_path):
    repo = tmp_path / "portable-repository"
    repo.mkdir()
    for name in ("scripts/run_budget_generation.py", "src/synsteg_audit/budget_ledger.py"):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, destination)
    prompts = tmp_path / "fixture-prompts.jsonl"
    prompts.write_text("".join(json.dumps({"record_id": "record-" + str(i), "source_group_id": "source-" + str(i),
        "split": "development", "topic": "topic", "genre": "prose", "target_words": 250,
        "prompt": "Synthetic test prompt " + str(i)}) + "\n" for i in range(2)))
    generated = tmp_path / "fixture-run"
    runner.run(prompts=prompts, output=generated, provider="local", model="Qwen/Qwen2.5-0.5B-Instruct",
        revision=runner.PINNED_LOCAL["Qwen/Qwen2.5-0.5B-Instruct"], execute=True, max_new_records=1,
        runtime_reader=lambda: RUNTIME, local_generator=lambda *args: {
            "cover_text": "Synthetic fixture only", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 2}})
    source = repo / "results/snapshot"
    shutil.copytree(generated, source, ignore=shutil.ignore_patterns("*.lock"))
    manifest = repo / "snapshot-manifest.json"
    write_json(manifest, {"schema": "synsteg-cloud-snapshot-v1", "snapshot_root": "results/snapshot",
                         "suggested_target": "runs/restored", "type": "standalone", "files": []})
    refresh_manifest(repo, manifest)
    return repo, manifest, source


def operate(snapshot, **kwargs):
    repo, manifest, _ = snapshot
    return restore.operate(manifest, repo=repo, runtime_reader=lambda: RUNTIME, **kwargs)


def test_read_only_verify_and_exact_idempotent_restore(snapshot):
    repo, _, source = snapshot
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = operate(snapshot)
    assert report["status"] == "verified_no_files_written"
    assert report["resume_eligible_in_current_runtime"]
    assert not report["new_requests_authorized_by_verification"]
    assert not (repo / "runs").exists()
    assert operate(snapshot, restore=True)["status"] == "restored_verified_state"
    assert operate(snapshot, restore=True)["status"] == "already_restored_identical"
    assert {p.relative_to(repo / "runs/restored"): p.read_bytes()
            for p in (repo / "runs/restored").rglob("*") if p.is_file()} == original
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == original


def test_missing_ledger_cannot_repeat_completed_attempt(snapshot):
    repo, manifest, source = snapshot
    (source / "local_ledger/ledger.json").unlink()
    refresh_manifest(repo, manifest)
    with pytest.raises(restore.RestoreError, match="missing_ledger_would_allow_duplicate_run"):
        operate(snapshot, restore=True)
    assert not (repo / "runs").exists()


def test_missing_ledger_entry_cannot_repeat_completed_attempt(snapshot):
    repo, manifest, source = snapshot
    path = source / "local_ledger/ledger.json"
    ledger = json.loads(path.read_text())
    ledger["attempts"] = {}
    write_json(path, ledger)
    refresh_manifest(repo, manifest)
    with pytest.raises(restore.RestoreError, match="missing_ledger_entry_would_allow_duplicate_run"):
        operate(snapshot)


def test_summary_prevents_partial_restore_with_both_attempts_and_ledger_missing(snapshot):
    repo, manifest, source = snapshot
    shutil.rmtree(source / "attempts")
    (source / "local_ledger/ledger.json").unlink()
    refresh_manifest(repo, manifest)
    with pytest.raises(restore.RestoreError, match="missing_ledger_would_allow_duplicate_run"):
        operate(snapshot)


def test_existing_changed_target_never_overwritten(snapshot):
    repo, _, _ = snapshot
    operate(snapshot, restore=True)
    path = repo / "runs/restored/summary.json"
    path.write_bytes(b"newer-live-state")
    with pytest.raises(restore.RestoreError, match="mismatch"):
        operate(snapshot, restore=True)
    assert path.read_bytes() == b"newer-live-state"


@pytest.mark.parametrize("path", ["/absolute", "../escape", "C:/escape", "a\\b", "a/../b"])
def test_absolute_and_traversal_fields_rejected(snapshot, path):
    _, manifest, _ = snapshot
    obj = json.loads(manifest.read_text())
    obj["suggested_target"] = path
    write_json(manifest, obj)
    with pytest.raises(restore.RestoreError, match="unsafe_relative_path"):
        operate(snapshot, restore=True)


def test_casefold_collision_rejected(snapshot):
    _, manifest, _ = snapshot
    obj = json.loads(manifest.read_text())
    duplicate = dict(obj["files"][0])
    duplicate["path"] = duplicate["path"].upper()
    obj["files"].append(duplicate)
    write_json(manifest, obj)
    with pytest.raises(restore.RestoreError, match="collision"):
        operate(snapshot)


def test_symlink_and_lock_members_refused(snapshot):
    repo, manifest, source = snapshot
    (source / "unsafe-link").symlink_to(repo / "snapshot-manifest.json")
    with pytest.raises(restore.RestoreError, match="symlink"):
        operate(snapshot)
    (source / "unsafe-link").unlink()
    (source / "campaign.lock").write_bytes(b"0")
    with pytest.raises(restore.RestoreError, match="lock_file"):
        operate(snapshot)
    refresh_manifest(repo, manifest)
    with pytest.raises(restore.RestoreError, match="lock_file"):
        operate(snapshot)


def test_runtime_mismatch_preserved_and_blocks_resume_eligibility(snapshot):
    repo, manifest, _ = snapshot
    report = restore.operate(manifest, repo=repo, restore=True, runtime_reader=lambda: None)
    assert report["status"] == "restored_verified_state"
    assert not report["resume_eligible_in_current_runtime"]
    assert json.loads((repo / "runs/restored/runtime.json").read_text()) == RUNTIME


def test_unknown_attempt_can_be_preserved_but_not_resumed(snapshot):
    repo, manifest, source = snapshot
    attempt_path = next((source / "attempts").glob("*.json"))
    attempt = json.loads(attempt_path.read_text())
    attempt.update(status="reserved_outcome_unknown", result=None)
    write_json(attempt_path, attempt)
    ledger_path = source / "local_ledger/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    entry = ledger["attempts"][attempt["identity"]]
    entry["status"] = "reserved_outcome_unknown"
    entry.pop("result", None)
    write_json(ledger_path, ledger)
    refresh_manifest(repo, manifest)
    result = operate(snapshot, restore=True)
    assert not result["resume_eligible_in_current_runtime"]
    assert result["unresolved_or_failed_attempts"] == [attempt["identity"]]


def test_inconsistent_completed_result_rejected(snapshot):
    repo, manifest, source = snapshot
    path = next((source / "attempts").glob("*.json"))
    attempt = json.loads(path.read_text())
    attempt["result"]["cover_text"] = "Corrupted output"
    write_json(path, attempt)
    refresh_manifest(repo, manifest)
    with pytest.raises(restore.RestoreError, match="result_conflict"):
        operate(snapshot)


def test_changed_frozen_runner_rejected(snapshot):
    repo, _, _ = snapshot
    (repo / "scripts/run_budget_generation.py").write_bytes(b"changed implementation")
    with pytest.raises(restore.RestoreError, match="frozen_implementation"):
        operate(snapshot)
