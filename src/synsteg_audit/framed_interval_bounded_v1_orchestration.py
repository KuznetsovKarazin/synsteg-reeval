"""One-attempt16-case bounded campaign with eight isolated fresh receivers.

No model calls occur on import. Frozen old storage/process primitives remain
unchanged; new bounded crypto, stopping, requests and acceptance are explicit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .framed_interval_bounded_v1 import (
    THRESHOLD_BITS, TOKEN_CAP, generate, make_envelope, receive,
)
from .framed_interval_bounded_v1_activation import (
    INTENT, check_fixtures, load_activation,
)
from .framed_interval_bounded_v1_journal import (
    DurableEndpoint, atomic_json, recover_journal, result_record, sha,
)
from .framed_interval_bounded_v1_qwen import BoundedNativeService
from .framed_interval_llm_v1_activation import checked_bytes, digest, safe_path
from .framed_interval_llm_v1_orchestration import mkdir_durable, reconcile
from .integer_interval_channel_v1 import Config
from .seeded_pair_llm import canonical

SEND_STATUSES = {"support_refusal", "capacity_limit", "wire_limit_exceeded",
    "backend_error", "contract_error", "ordinary_threshold_generated",
    "ordinary_cap_generated", "insufficient_capacity", "threshold_pending_receiver"}
RECEIVE_STATUSES = {"invalid_wire", "contract_error", "capacity_limit",
    "backend_error", "incomplete_wire", "insufficient_capacity",
    "authentication_failed", "invalid_frame", "authenticated_delivery"}


def validate_request(request, activation, campaign_root):
    if type(request) is not dict: raise ValueError("strict worker request object required")
    common = {"schema", "role", "run_id", "activation_sha256", "campaign_manifest_sha256", "key_hex"}
    role = request.get("role")
    expected = common | ({"payload_hex"} if role == "generate" else {"wire", "envelope"})
    if type(request) is not dict or set(request) != expected or request["schema"] != "framed-interval-bounded-v1-worker-request-v1":
        raise ValueError("strict worker request fields required")
    if role not in {"generate", "receive"} or request["activation_sha256"] != activation.sha256:
        raise ValueError("worker role/activation differs")
    manifest_raw = (Path(campaign_root) / "campaign-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if (type(manifest) is not dict or set(manifest) != {
            "schema", "activation_sha256", "public_runs", "planned_cases",
            "planned_fresh_stego_receivers", "execution_kind",
            "attempts_per_endpoint", "retry_permitted"}
            or manifest.get("schema") != "framed-interval-bounded-v1-sixteen-case-manifest-v1"
            or manifest.get("attempts_per_endpoint") != 1 or manifest.get("retry_permitted") is not False
            or sha(manifest_raw) != request["campaign_manifest_sha256"] or manifest["activation_sha256"] != activation.sha256
            or manifest["public_runs"] != activation.public_protocol["runs"]
            or manifest["planned_cases"] != 16 or manifest["planned_fresh_stego_receivers"] != 8):
        raise ValueError("complete sixteen-case/eight-receiver manifest differs")
    matches = [r for r in activation.public_protocol["runs"] if r["run_id"] == request["run_id"]]
    if len(matches) != 1: raise ValueError("run outside fixed sixteen-case list")
    run = matches[0]
    key = request["key_hex"]
    if type(key) is not str or re.fullmatch("[0-9a-f]{64}", key) is None:
        raise ValueError("exact separately supplied32-byte key required")
    if role == "receive" and run["arm"] != "stego":
        raise ValueError("only eight stego receivers are planned")
    if role == "generate":
        fixtures = check_fixtures(activation)
        private = next(row for row in fixtures["runs"] if row["run_id"] == run["run_id"])
        if request["payload_hex"] != private["payload_hex"] or key != private["key_hex"]:
            raise ValueError("fixed sender payload/key differs")
    if role == "receive":
        # Validate this allowlist BEFORE opening either received artifact. A
        # descriptor must not redirect a receiver to sender fixtures or traces.
        sender = Path(campaign_root) / run["run_id"] / "sender"
        for name, filename in (("wire", "committed-wire.bin"), ("envelope", "envelope.json")):
            entry = request[name]
            expected_path = (sender / filename).relative_to(activation.root).as_posix()
            if (type(entry) is not dict or set(entry) != {"path", "sha256"}
                    or entry["path"] != expected_path
                    or type(entry["sha256"]) is not str
                    or re.fullmatch("[0-9a-f]{64}", entry["sha256"]) is None):
                raise ValueError("receiver may read only this run's wire and public envelope")
    # Receiver branch never receives/reads expected plaintext or the sender trace.
    return run


def execute_endpoint(request, activation, campaign_root, directory, *, service_factory=BoundedNativeService):
    directory = Path(directory)
    run = validate_request(request, activation, campaign_root)
    atomic_json(directory / "execution-started.json", {
        "schema": "framed-interval-bounded-v1-endpoint-start-v1", "run_id": run["run_id"],
        "role": request["role"], "activation_sha256": activation.sha256,
        "attempt": 1, "retry_permitted": False})
    deadline = time.monotonic()+activation.public_protocol["per_endpoint_timeout_seconds"]
    service = service_factory(activation, directory, deadline)
    durable = None
    output = {"schema": "framed-interval-bounded-v1-worker-result-v1", "run_id": run["run_id"],
        "role": request["role"], "execution_kind": service.kind, "activation_sha256": activation.sha256,
        "status": "worker_exception", "retry_permitted": False, "core": None}
    try:
        endpoint = service.prepare()
        protocol = activation.public_protocol
        env = make_envelope(endpoint, bytes.fromhex(run["nonce_hex"]), arm=run["arm"], prompt_id=run["prompt_id"],
            protocol_digest=activation.sha256, prompt_ids=tuple(activation.preflight["prompt_ids"][run["prompt_id"]]),
            config=Config(**protocol["interval_config"]), max_wire_bytes=protocol["max_wire_bytes"],
            ordinary_seed=bytes.fromhex(run["ordinary_seed_hex"]))
        if request["role"] == "receive":
            # Only these two explicit sender artifacts are readable by the decoder.
            wire = checked_bytes(activation.root, request["wire"])
            received_env = json.loads(checked_bytes(activation.root, request["envelope"]))
            if received_env != env: raise ValueError("wire envelope differs from fixed public resources/run")
        atomic_json(directory / "envelope.json", env)
        durable = DurableEndpoint(endpoint, env, bytes.fromhex(request["key_hex"]), directory)
        if request["role"] == "generate":
            payload = None if request["payload_hex"] is None else bytes.fromhex(request["payload_hex"])
            result = generate(bytes.fromhex(request["key_hex"]), env, durable, payload=payload)
        else:
            result = receive(wire, bytes.fromhex(request["key_hex"]), env, durable)
        core = result_record(result)
        traces = core.pop("traces")
        saved = recover_journal(directory / "token-journal.jsonl")
        committed = saved["committed_traces"]
        core["trace_sha256"] = digest(canonical(traces))
        core["committed_trace_matches_returned_prefix"] = canonical(traces[:len(committed)]) == canonical(committed)
        core["noncommitted_trace_suffix"] = traces[len(committed):]
        output.update(status="returned", core=core)
    except Exception as exc:
        output["error"] = {"type": type(exc).__name__, "message": str(exc)[:1024]}
    finally:
        close_errors = []
        for component in (durable, service):
            if component is not None:
                try: component.close()
                except Exception as exc:
                    close_errors.append({"type": type(exc).__name__, "message": str(exc)[:1024]})
        if close_errors:
            output["close_errors"] = close_errors
            output["status"] = "worker_exception"
    atomic_json(directory / "worker-result.json", output)
    try: output["reconciliation"] = reconcile(directory)
    except Exception as exc:
        atomic_json(directory / "reconciliation-error.json", {"type": type(exc).__name__, "message": str(exc)[:1024]})
    return output


def spawn_worker(root, activation_path, activation_sha, campaign_relative, endpoint_relative, request_relative, timeout):
    started = time.monotonic()
    root = Path(root)
    endpoint = safe_path(root, endpoint_relative)
    command = [sys.executable, str(root / "experiments/framed-interval-bounded-v1/run_bounded.py"), "worker",
        "--activation", activation_path, "--activation-sha256", activation_sha, "--intent", INTENT,
        "--campaign", campaign_relative, "--endpoint", endpoint_relative, "--request", request_relative]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    with (endpoint / "worker.stdout.log").open("xb") as stdout, (endpoint / "worker.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            outcome = {"status": "exited", "exit_code": code, "pid": process.pid}
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
            outcome = {"status": "timeout", "exit_code_before_cleanup": process.returncode, "pid": process.pid}
        finally:
            # Also kills surviving grandchildren when Python exits first, or a
            # bridge constructor/wait fails. This group was created by this Popen.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=3)
        outcome["exit_code"] = process.returncode
        outcome["wall_seconds"] = time.monotonic()-started
        return outcome


def _collect(directory, process, role, expected_kind, run_id, activation_sha):
    directory = Path(directory)
    outcome = {"process": process, "worker_status": "missing_terminal_record", "authenticated_delivery": False}
    path = directory / "worker-result.json"
    if path.exists():
        worker = json.loads(path.read_bytes())
        if (worker.get("role") != role or worker.get("execution_kind") != expected_kind
                or worker.get("run_id") != run_id or worker.get("activation_sha256") != activation_sha
                or worker.get("status") not in {"returned", "worker_exception", "prelaunch_error"}):
            raise ValueError("worker role/execution kind differs")
        outcome["worker_status"] = worker["status"]
        outcome["worker"] = worker
        core = worker.get("core")
        if core is not None:
            allowed = SEND_STATUSES if role == "generate" else RECEIVE_STATUSES
            if core.get("status") not in allowed: raise ValueError("unknown core outcome")
            success = core["status"] == "authenticated_delivery"
            if core.get("authenticated_delivery") is not success:
                raise ValueError("payload-success guard inconsistent")
            if role == "receive" and ((core.get("payload_hex") is not None) != success):
                raise ValueError("plaintext must exist iff authenticated delivery")
            if success and (type(core.get("processed_tokens")) is not int
                    or not 1 <= core["processed_tokens"] <= TOKEN_CAP
                    or core.get("wire_tokens") != core["processed_tokens"]
                    or core.get("first_threshold_token") != core["processed_tokens"]
                    or type(core.get("resolved_bits")) is not int
                    or core["resolved_bits"] < THRESHOLD_BITS
                    or type(core.get("payload_hex")) is not str):
                raise ValueError("authenticated delivery requires complete bounded stopping replay")
            outcome["authenticated_delivery"] = (success and worker["status"] == "returned"
                and process.get("status") == "exited" and process.get("exit_code") == 0)
    try: outcome["reconciliation"] = reconcile(directory)
    except Exception as exc:
        outcome["artifact_error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
        outcome["authenticated_delivery"] = False
    if outcome["authenticated_delivery"]:
        evidence = outcome["reconciliation"]
        request = json.loads((directory / "request.json").read_bytes())
        core = outcome["worker"]["core"]
        if (not evidence.get("wire_available") or evidence.get("known_committed_tokens") != core["processed_tokens"]
                or evidence.get("unknown_native_step") or evidence.get("torn_last_record")
                or not core.get("committed_trace_matches_returned_prefix")
                or core.get("noncommitted_trace_suffix") != []
                or evidence.get("committed_trace_sha256") != core.get("trace_sha256")
                or request.get("wire", {}).get("sha256") != evidence.get("wire_sha256")):
            outcome["authenticated_delivery"] = False
            outcome["delivery_evidence_error"] = "complete clean bounded receiver journal/trace required"
    if outcome["authenticated_delivery"]:
        saved = recover_journal(directory / "token-journal.jsonl")
        resolved, crossing = 0, None
        for index, trace in enumerate(saved["committed_traces"], 1):
            bits = trace.get("emitted_bits")
            if type(bits) is not list or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
                raise ValueError("invalid bounded receiver emitted-bit trace")
            resolved += len(bits)
            if resolved >= THRESHOLD_BITS and crossing is None: crossing = index
        if (crossing != core["processed_tokens"] or resolved != core["resolved_bits"]
                or crossing != core["first_threshold_token"]):
            outcome["authenticated_delivery"] = False
            outcome["delivery_evidence_error"] = "journal first crossing/resolved bits differ"
    return outcome


def run_campaign(root, activation_path, activation_sha, output_relative, intent, *,
                 activation_loader=load_activation, worker_runner=spawn_worker, expected_kind="native"):
    campaign_started = time.monotonic()
    activation = activation_loader(root, activation_path, activation_sha, intent)
    root = activation.root
    if not output_relative.startswith("results/framed-interval-bounded-v1/"):
        raise ValueError("new dedicated experiment output path required")
    output = safe_path(root, output_relative)
    mkdir_durable(output, exclusive=True)  # No automatic resume or retry of an existing attempt.
    manifest = {"schema": "framed-interval-bounded-v1-sixteen-case-manifest-v1", "activation_sha256": activation.sha256,
        "public_runs": activation.public_protocol["runs"], "planned_cases": 16, "planned_fresh_stego_receivers": 8,
        "execution_kind": expected_kind, "attempts_per_endpoint": 1, "retry_permitted": False}
    atomic_json(output / "campaign-manifest.json", manifest)
    manifest_sha = sha((output / "campaign-manifest.json").read_bytes())
    ledger = {"schema": "framed-interval-bounded-v1-campaign-status-v1", "activation_sha256": activation.sha256,
        "execution_kind": expected_kind, "planned_cases": 16, "planned_fresh_stego_receivers": 8,
        "retry_permitted": False, "state": "running", "cases": [
            {"run_id": run["run_id"], "arm": run["arm"], "sender": {"worker_status": "not_started"},
             "receiver": {"worker_status": "not_started"} if run["arm"] == "stego" else None}
            for run in activation.public_protocol["runs"]]}
    atomic_json(output / "campaign-status.json", ledger)
    try:
        # Parent alone reads private expected-payload fields. Receiver workers use public protocol.
        proposal = check_fixtures(activation)
        private_runs = {run["run_id"]: run for run in proposal["runs"]}
        for index, run in enumerate(activation.public_protocol["runs"]):
            case = ledger["cases"][index]
            case_dir = output / run["run_id"]
            mkdir_durable(case_dir)
            sender = case_dir / "sender"; mkdir_durable(sender)
            request = {"schema": "framed-interval-bounded-v1-worker-request-v1", "role": "generate",
                "run_id": run["run_id"], "activation_sha256": activation.sha256,
                "campaign_manifest_sha256": manifest_sha, "key_hex": private_runs[run["run_id"]]["key_hex"],
                "payload_hex": private_runs[run["run_id"]]["payload_hex"]}
            atomic_json(sender / "request.json", request)
            for role, directory in [("generate", sender), ("receive", case_dir / "receiver")]:
                slot = "sender" if role == "generate" else "receiver"
                if role == "receive":
                    if run["arm"] != "stego": continue
                    mkdir_durable(directory)
                    envelope_path, wire_path = sender / "envelope.json", sender / "committed-wire.bin"
                    if (not envelope_path.exists() or not wire_path.exists()
                            or case["sender"].get("artifact_error")
                            or not case["sender"].get("reconciliation", {}).get("wire_available")):
                        case[slot] = {"worker_status": "dependency_blocked", "reason": "no_verified_wire_or_envelope",
                                      "authenticated_delivery": False, "retry_permitted": False}
                        atomic_json(directory / "dependency-blocked.json", case[slot])
                        atomic_json(output / "campaign-status.json", ledger, replace=True)
                        continue
                    request = {k: v for k, v in request.items() if k != "payload_hex"}
                    request.update(role="receive", wire={"path": wire_path.relative_to(root).as_posix(), "sha256": sha(wire_path.read_bytes())},
                        envelope={"path": envelope_path.relative_to(root).as_posix(), "sha256": sha(envelope_path.read_bytes())})
                    atomic_json(directory / "request.json", request)
                case[slot] = {"worker_status": "running", "retry_permitted": False}
                atomic_json(output / "campaign-status.json", ledger, replace=True)
                endpoint_started = time.monotonic()
                try:
                    process = worker_runner(root, activation_path, activation.sha256, output_relative,
                        directory.relative_to(root).as_posix(), (directory / "request.json").relative_to(root).as_posix(),
                        activation.public_protocol["per_endpoint_timeout_seconds"])
                except Exception as exc:
                    process = {"status": "spawn_or_wait_error", "error_type": type(exc).__name__, "error": str(exc)[:1024],
                               "wall_seconds": time.monotonic()-endpoint_started}
                try:
                    case[slot] = _collect(directory, process, role, expected_kind, run["run_id"], activation.sha256)
                    if role == "receive" and case[slot].get("authenticated_delivery"):
                        observed = case[slot]["worker"]["core"]["payload_hex"]
                        exact = observed == private_runs[run["run_id"]]["payload_hex"]
                        case[slot]["exact_expected_payload"] = exact
                        case[slot]["authenticated_delivery"] = exact
                except Exception as exc:
                    case[slot] = {"worker_status": "orchestration_error", "process": process,
                        "error_type": type(exc).__name__, "error": str(exc)[:1024], "artifact_error": True,
                        "authenticated_delivery": False, "retry_permitted": False}
                atomic_json(output / "campaign-status.json", ledger, replace=True)
    except Exception as exc:
        ledger["state"] = "aborted_campaign_error"
        ledger["campaign_error"] = {"type": type(exc).__name__, "message": str(exc)[:1024]}
        for case in ledger["cases"]:
            for name in ("sender", "receiver"):
                slot = case[name]
                if slot and slot.get("worker_status") in {"not_started", "running"}:
                    prior = slot["worker_status"]
                    case[name] = {"worker_status": "not_attempted_due_to_campaign_abort" if prior == "not_started" else "unknown_after_campaign_abort",
                        "reason": "campaign_storage_or_orchestration_error", "authenticated_delivery": False,
                        "retry_permitted": False}
        ledger["authenticated_payload_deliveries"] = sum(bool(case["receiver"] and case["receiver"].get("authenticated_delivery")) for case in ledger["cases"])
        ledger["authenticated_delivery_denominator"] = 8
        ledger["real_model_success_claimed"] = expected_kind == "native" and ledger["authenticated_payload_deliveries"] > 0
        ledger["wall_seconds"] = time.monotonic()-campaign_started
        # A failed replacement may leave campaign-status.json.writing intact.
        # Publish a separate final record without deleting that evidence.
        atomic_json(output / "campaign-abort.json", ledger)
        return ledger
    ledger["state"] = "finished_all_planned_outcomes"
    ledger["wall_seconds"] = time.monotonic()-campaign_started
    ledger["authenticated_payload_deliveries"] = sum(bool(case["receiver"] and case["receiver"].get("authenticated_delivery"))
                                                     for case in ledger["cases"])
    ledger["authenticated_delivery_denominator"] = 8
    ledger["real_model_success_claimed"] = expected_kind == "native" and ledger["authenticated_payload_deliveries"] > 0
    try: atomic_json(output / "campaign-status.json", ledger, replace=True)
    except Exception as exc:
        ledger["status_publication_error"] = {"type": type(exc).__name__, "message": str(exc)[:1024]}
        atomic_json(output / "campaign-final-status.json", ledger)
    return ledger
