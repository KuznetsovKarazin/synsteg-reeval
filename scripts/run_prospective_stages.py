#!/usr/bin/env python3
"""Run an already-frozen pilot, gating every stage on an external checkpoint receipt.

This controller never prepares/freezes a protocol or uploads files itself. It
does not import model libraries or print sample text. Start only after the
source has been committed and the operator has authorized the frozen run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time


STAGES = (
    ("frozen_design", ("verify",)),
    ("train_generation", ("generate", "--split", "train")),
    ("development_generation", ("generate", "--split", "development")),
    ("attack_training", ("train",)),
    ("test_generation", ("generate", "--split", "test")),
    ("test_evaluation", ("evaluate",)),
    ("final_verification", ("verify",)),
)
TERMINATION_SIGNAL = 0


def raise_if_terminated():
    if TERMINATION_SIGNAL:
        raise SystemExit(128 + TERMINATION_SIGNAL)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def emit(control: Path, stage: str, status: str, **details):
    value = {"stage": stage, "status": status, "time_unix": time.time(), **details}
    write_json(control / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


@contextmanager
def controller_lock(control: Path):
    lock = control / "controller.lock"
    control.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Existing controller lock: establish owned-session termination before manual recovery") from error
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "started_at_unix": time.time()}, stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def recover_known_exited_child_lock(run: Path, child_pid: int, stage: str):
    """Called only after wait() proved this controller's child exited."""
    lock = run / ".writer.lock"
    if not lock.exists():
        return
    if lock.read_text().strip() != str(child_pid):
        raise RuntimeError("Unexpected writer-lock owner; manual inspection required")
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "known_exited_child_stale_lock_recovery",
            "time_unix": time.time(), "stale_pid": child_pid, "stage": stage,
            "evidence": "Owned subprocess.wait() returned before removal of its same-PID lock."}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    lock.unlink()


def runner_module(repo: Path):
    spec = importlib.util.spec_from_file_location("prospective_controller_runner", repo / "scripts/prospective_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_completed_evaluation(run: Path, repo: Path):
    report = read_json(run / "evaluation.json")
    frozen = read_json(run / "freeze.json")
    opening_path = run / "test_opened.json"
    if report.get("freeze_id") != frozen["freeze_id"] or not opening_path.is_file():
        raise RuntimeError("Existing evaluation has invalid identity; do not repeat it")
    module = runner_module(repo)
    lock = module.verify_attack_lock(run, read_json(run / "protocol.json"), frozen)
    opening = read_json(opening_path)
    generation = read_json(run / "test_generation.json")
    if (report.get("lock_id") != lock["lock_id"] or opening.get("lock_id") != lock["lock_id"]
            or opening.get("freeze_id") != frozen["freeze_id"]
            or not generation.get("complete")
            or report.get("test_records_sha256") != generation.get("records_sha256")
            or opening.get("test_records_sha256") != generation.get("records_sha256")):
        raise RuntimeError("Existing evaluation references changed test or attack identities")
    # Test text is read only after confirming an existing opening/evaluation.
    if module.digest(module.read_records(run, "test")) != generation["records_sha256"]:
        raise RuntimeError("Evaluated test records changed")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    completions = [event for event in events if event.get("event") == "test_evaluation_completed"]
    openings = [event for event in events if event.get("event") == "test_opened_once"]
    if len(completions) != 1 or len(openings) != 1 or completions[0].get("evaluation_sha256") != sha256(run / "evaluation.json"):
        raise RuntimeError("Existing evaluation lacks one matching completion event; preserve it for inspection")


def stage_complete(run: Path, control: Path, stage: str, repo: Path | None = None) -> bool:
    repo = repo or Path(__file__).resolve().parents[1]
    if stage.endswith("_generation"):
        split = stage.removesuffix("_generation")
        path = run / f"{split}_generation.json"
        if not path.exists():
            return False
        data = read_json(path)
        expected = len(read_json(run / "manifests" / f"{split}.json"))
        if not (data.get("complete") and data.get("records") == expected
                and data.get("expected") == expected and data.get("records_sha256")):
            raise RuntimeError(f"Invalid completion manifest for {split}")
        if len(list((run / "records" / split).glob("*.json"))) != expected:
            raise RuntimeError(f"Record count changed for completed split {split}")
        return True
    if stage == "attack_training":
        if not (run / "attack_lock.json").is_file():
            return False
        runner_module(repo).verify_attack_lock(run, read_json(run / "protocol.json"), read_json(run / "freeze.json"))
        return True
    if stage == "test_evaluation":
        if not (run / "evaluation.json").is_file():
            return False
        validate_completed_evaluation(run, repo)
        return True
    return (control / f"{stage}.completed.json").is_file()


def execute_stage(repo: Path, run: Path, control: Path, stage: str, arguments: tuple[str, ...], *, force: bool = False):
    raise_if_terminated()
    if not force and stage_complete(run, control, stage, repo):
        emit(control, stage, "already_complete_not_repeated")
        return
    if (run / ".writer.lock").exists():
        raise RuntimeError("Existing writer lock; no automatic liveness inference from another exec's /proc")
    log = control / f"{stage}.log"
    command = [sys.executable, str(repo / "scripts/prospective_experiment.py"), *arguments, "--run", str(run)]
    with log.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        write_json(control / "owned_child.json", {"stage": stage, "pid": child.pid,
            "controller_pid": os.getpid(), "status": "running", "time_unix": time.time()})
        emit(control, stage, "running", child_pid=child.pid, log=str(log))
        heartbeat_at = time.monotonic()
        try:
            while child.poll() is None:
                raise_if_terminated()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - heartbeat_at >= 30:
                    split = stage.removesuffix("_generation") if stage.endswith("_generation") else None
                    count = len(list((run / "records" / split).glob("*.json"))) if split else None
                    emit(control, stage, "running", child_pid=child.pid, completed_records=count)
                    heartbeat_at = time.monotonic()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            recover_known_exited_child_lock(run, child.pid, stage)
            write_json(control / "owned_child.json", {"stage": stage, "pid": child.pid,
                "status": "exited_after_controller_signal", "returncode": child.returncode, "time_unix": time.time()})
            raise
    recover_known_exited_child_lock(run, child.pid, stage)
    write_json(control / "owned_child.json", {"stage": stage, "pid": child.pid,
        "status": "exited", "returncode": child.returncode, "time_unix": time.time()})
    if child.returncode != 0:
        emit(control, stage, "failed", exit_code=child.returncode, log=str(log))
        raise RuntimeError(f"Stage {stage} exited {child.returncode}; attempts remain preserved")
    write_json(control / f"{stage}.completed.json", {"exit_code": 0, "child_pid": child.pid,
                                                    "time_unix": time.time()})
    emit(control, stage, "completed", child_pid=child.pid, log=str(log))


def checkpoint(run: Path, control: Path, directory: Path, stage: str, freeze_id: str,
               source_commit: str) -> tuple[Path, dict]:
    stem = f"{run.name}-{stage}-{freeze_id[:12]}"
    manifest_path = directory / f"{stem}.json"
    archive_path = directory / f"{stem}.tar.gz"
    if (run / ".writer.lock").exists():
        raise RuntimeError("Checkpoint requires no active or unexplained writer lock")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get("freeze_id") != freeze_id or manifest.get("stage") != stage
                or manifest.get("source_commit") != source_commit
                or not archive_path.is_file() or sha256(archive_path) != manifest.get("archive_sha256")):
            raise RuntimeError("An existing checkpoint identity or archive hash does not match")
        # Later stages may add files and append ledger events, but cannot
        # replace a completed stage's records, attempts, protocol, or models.
        for relative, expected_hash in manifest["files_sha256"].items():
            if relative == "events.jsonl":
                with tarfile.open(archive_path, "r:gz") as archive:
                    earlier_events = archive.extractfile(f"{run.name}/events.jsonl").read()
                if not (run / relative).read_bytes().startswith(earlier_events):
                    raise RuntimeError("Event ledger no longer preserves the archived prefix")
            elif not (run / relative).is_file() or sha256(run / relative) != expected_hash:
                raise RuntimeError(f"Previously checkpointed file changed: {relative}")
        return manifest_path, manifest
    if archive_path.exists():
        raise RuntimeError("Orphaned archive has no manifest; preserve and inspect before recovery")
    paths = sorted(p for p in run.rglob("*") if p.is_file())
    if any(p.is_symlink() or p.name == ".writer.lock" for p in paths):
        raise RuntimeError("Checkpoint requires no active writer or symbolic file links")
    if not paths:
        raise RuntimeError("Cannot archive an empty run")
    file_hashes = {str(p.relative_to(run)): sha256(p) for p in paths}
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(Path(run.name) / path.relative_to(run)), recursive=False)
    # The stage child has exited; detect an unexpected writer during copying.
    after = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file()}
    if after != file_hashes:
        raise RuntimeError("Run changed while checkpointing; preserve temporary archive for inspection")
    temporary.replace(archive_path)
    manifest = {"schema": "prospective-stage-checkpoint-v1", "run_name": run.name,
        "stage": stage, "freeze_id": freeze_id, "source_commit": source_commit,
        "created_at_unix": time.time(), "archive_path": str(archive_path),
        "archive_sha256": sha256(archive_path), "archive_bytes": archive_path.stat().st_size,
        "files_sha256": file_hashes,
        "receipt_path": str(directory / f"{stem}.remote.json")}
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def validate_receipt(receipt: dict, manifest: dict):
    required = ("archive_sha256", "verified_remote_sha256", "freeze_id", "stage")
    expected = {"archive_sha256": manifest["archive_sha256"],
                "verified_remote_sha256": manifest["archive_sha256"],
                "freeze_id": manifest["freeze_id"], "stage": manifest["stage"]}
    if any(receipt.get(key) != expected[key] for key in required):
        raise RuntimeError("Remote receipt does not identify the verified checkpoint bytes")
    if not isinstance(receipt.get("remote_uri"), str) or not receipt["remote_uri"].strip():
        raise RuntimeError("Receipt needs an immutable remote URI or repository/commit/path identity")
    if not isinstance(receipt.get("verification_evidence"), str) or not receipt["verification_evidence"].strip():
        raise RuntimeError("Receipt needs an explanation of remote-byte verification")
    if not isinstance(receipt.get("verified_at_unix"), (int, float)):
        raise RuntimeError("Receipt needs a remote verification timestamp")


def await_receipt(control: Path, manifest_path: Path, manifest: dict, timeout: float):
    raise_if_terminated()
    receipt_path = Path(manifest["receipt_path"])
    start = heartbeat = time.monotonic()
    emit(control, manifest["stage"], "checkpoint_ready_waiting_for_remote_receipt",
         checkpoint_manifest=str(manifest_path), receipt_path=str(receipt_path),
         archive_sha256=manifest["archive_sha256"])
    while not receipt_path.exists():
        raise_if_terminated()
        if time.monotonic() - start >= timeout:
            raise TimeoutError("No verified remote checkpoint receipt; later stages were not started")
        time.sleep(min(5, max(0.01, timeout - (time.monotonic() - start))))
        if time.monotonic() - heartbeat >= 30:
            emit(control, manifest["stage"], "waiting_for_remote_receipt", receipt_path=str(receipt_path))
            heartbeat = time.monotonic()
    validate_receipt(read_json(receipt_path), manifest)
    emit(control, manifest["stage"], "remote_checkpoint_acknowledged", receipt_path=str(receipt_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--source-commit", required=True, help="Already-persisted remote source commit, not a future commit")
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("/tmp/synsteg_checkpoints"))
    parser.add_argument("--receipt-timeout-seconds", type=float, default=3600)
    parser.add_argument("--continuity-evidence", required=True,
                        help="Operator evidence that this is a never-started run or that every prior attempt marker is preserved")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source-commit must be an existing 40-character commit SHA")
    if args.receipt_timeout_seconds <= 0:
        parser.error("receipt timeout must be positive")
    if not args.continuity_evidence.strip():
        parser.error("Attempt-ledger continuity evidence must not be empty")
    repo, run = args.repo.resolve(), args.run.resolve()
    frozen = read_json(run / "freeze.json")
    if frozen.get("freeze_id") != args.freeze_id:
        parser.error("freeze ID differs from the already-frozen run")
    directory = args.checkpoint_directory.resolve()
    control = directory / f"{run.name}-controller"
    if directory.is_relative_to(run):
        parser.error("checkpoints must be outside the run directory")
    def terminate_controller(signum, frame):
        # Defer raising to a child-aware boundary; a signal during Popen must
        # not escape before its returned handle can be cleaned up.
        global TERMINATION_SIGNAL
        TERMINATION_SIGNAL = signum
    signal.signal(signal.SIGTERM, terminate_controller)
    signal.signal(signal.SIGINT, terminate_controller)
    with controller_lock(control):
        with (control / "continuity_attestations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"freeze_id": args.freeze_id, "source_commit": args.source_commit,
                "time_unix": time.time(), "evidence": args.continuity_evidence}, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Even a fully completed resumed run must still match its frozen code,
        # resources, and environment. This lightweight check loads no models.
        execute_stage(repo, run, control, "startup_verification", ("verify",), force=True)
        for stage, arguments in STAGES:
            execute_stage(repo, run, control, stage, arguments)
            manifest_path, manifest = checkpoint(run, control, directory, stage, args.freeze_id, args.source_commit)
            await_receipt(control, manifest_path, manifest, args.receipt_timeout_seconds)
        emit(control, "all", "completed_with_remote_receipts")


if __name__ == "__main__":
    main()
