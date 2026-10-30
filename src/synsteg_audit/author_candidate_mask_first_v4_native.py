"""Source-only v4 worker/coordinator; no native work at import or before gates.

The unchanged v3 guard supplies sealed prepare/commit evidence. The new v4
support order and separate auditor supply the distribution semantics. A parent
process imposes the wall deadline on an owned session; uncertain execution is
UNKNOWN and consumes its reservation. No activation/intent is shipped here.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import time

from . import candidate_journal_guard_v3 as guard
from .author_candidate_mask_first_v4 import filter_then_quantize
from .author_candidate_schema_v3 import classify_prefix, constrain_support, select_token

ID = "author-candidate-mask-first-v4"
BASE = "experiments/" + ID
PROPOSAL = BASE + "/protocol.proposed.json"
OUTPUT = "results/" + ID + "/native-diagnostic-v4"
REQUESTS = "experiments/author-candidate-format-v2/requests.json"
PROFILE = "experiments/framed-interval-public-v1/native-public-profile.json"
CAP, SECONDS = 512, 600
REQUEST_HASHES = (
    "15f413e60449231fe505a45091d838f36155d15307bcdf16b27d1215ffa05104",
    "d9f3597b716fb4fbb888a83ceccd881f94635537132ac5dc00c0748ddd14b4f4",
)
FROZEN = {
    PROPOSAL: "b12ff46ff99e6d8038afc4f924b395e5807b831ba110f095142bc9f74e652b1a",
    REQUESTS: "0cc5750f8251cb7b10bb29d00ec858d5b005d0df42fb7d2781152c807a4ae421",
    PROFILE: "f32097750dc898fce418a29146587a4229eae1dcb76e3cbcd374b692846adc8a",
    "src/synsteg_audit/author_candidate_schema_v3.py": "cb17dd1780f5ee737e6b53454c0c7a911eb7867a9c2d2779f82790201d0b695f",
    "src/synsteg_audit/author_candidate_mask_first_v4.py": "39547825b20d3259b4a2ed8192ae8201a0e2b4dd49fcbb0b4f3ad2008c13d813",
    "src/synsteg_audit/candidate_journal_guard_v3.py": "62799921d139c5454ba9612e0eebd01809e17d511ecdb7bcea463c3ce39760f6",
    "src/synsteg_audit/author_candidate_local_native_v1.py": "b64719970d45968c879f105488a35c0ee968bf8e66fe29b1f85d6889728a83ab",
    "src/synsteg_audit/framed_interval_public_v1_native.py": "5d4d238371e15fc4efc75832916f090005a387ca70287c1fc8abd3cd69e68078",
    "src/synsteg_audit/author_candidate_format_v2.py": "20f70a75be1a7ab12825a00714b72e96e11c2fc8ae24a09d85af78508fc62273",
}
canonical, digest = guard.canonical, guard.sha


def sha_file(path):
    return digest(guard._read_stable(path))


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite metadata")
    return json.loads(guard._read_stable(path), object_pairs_hook=pairs, parse_constant=bad)


def mkdir(path, *, exist_ok=False):
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError:
        if not exist_ok or not stat.S_ISDIR(path.lstat().st_mode):
            raise
    guard._sync_dir(path.parent)


def write_json(path, value):
    guard._exclusive(Path(path), canonical(value))


def validate_request(request):
    index = request.get("chunk_order")
    if type(index) is not int or index not in (0, 1):
        raise ValueError("fixed chunk0/1 required")
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if request.get("request_sha256") != REQUEST_HASHES[index] or digest(canonical(body)) != REQUEST_HASHES[index]:
        raise ValueError("fixed request changed")


def key_hex(request_sha256):
    if request_sha256 not in REQUEST_HASHES:
        raise ValueError("fixed request required")
    # Preserve v3 fixture BYTES despite a new experiment ID.
    return digest(canonical(["author-candidate-local-native-v3", "fixed-public-diagnostic-key", request_sha256]))


def sampling_evidence(raw, request, rows, ids):
    mask = filter_then_quantize(raw, request["words"], rows, classify_prefix)
    law = None
    if mask["state"] == "ready":
        positive = [{"token_id": r["token_id"], "piece": bytes.fromhex(r["piece_hex"]), "mass": r["weight"]}
                    for r in mask["rows"] if r["weight"] > 0]
        law = constrain_support(raw, request["words"], positive)
        if law["state"] != "ready" or law["retained_mass"] != 1 << 24 or law["rejected"]:
            raise ValueError("positive v4 support changed during compatible-law adaptation")
        selection = select_token(law, bytes.fromhex(key_hex(request["request_sha256"])),
                                 request["request_sha256"], 0, len(ids))
    elif mask["state"] == "empty_schema_support":
        selection = {"state": "empty_support", "reason": "empty_schema_support"}
    elif mask["state"] == "resource_refusal":
        selection = {"state": "resource_refusal", "reason": mask["reason"]}
    else:
        raise ValueError("unexpected v4 support state")
    return {"request_sha256": request["request_sha256"], "key_hex": key_hex(request["request_sha256"]),
            "prefix_ids": list(ids), "prefix_raw_sha256": digest(raw), "top50_rows": rows,
            "mask_first": mask, "conditional_law": law, "selection": selection}


def binding_for(profile, request, protocol_sha256):
    from .framed_interval_public_v1_native import public_digest
    return {"experiment_id": ID, "attempt_id": f"chunk{request['chunk_order']}-attempt0",
            "request_sha256": request["request_sha256"], "resource_sha256": public_digest(profile),
            "protocol_sha256": protocol_sha256, "token_cap": CAP}


def _worker(root, profile, request, directory, commitment_path, binding, deadline, *, bridge_factory=None):
    """Private child only. Parent must reserve and supervise it; no retry here."""
    from .author_candidate_local_native_v1 import chatml, production_bridge
    validate_request(request)
    if bridge_factory is None:
        bridge_factory = production_bridge
    directory = Path(directory)
    native_dir = directory / "native"
    mkdir(native_dir)
    write_json(directory / "launch.json", {"schema": ID + "-attempt-launch", "binding": binding,
        "key_hex": key_hex(request["request_sha256"]), "request": request,
        "prompt_text_sha256": digest(chatml(request).encode()), "deadline_monotonic": deadline,
        "token_cap": CAP, "wall_seconds": SECONDS})
    writer = guard.AttemptWriter(directory / "sealed", binding)
    bridge = None
    ids, raw, prompt_ids = [], b"", []
    stop = proof = error = cleanup_error = None
    phase = "factory_not_invoked"
    counters = {"factory_invoked": 0, "tokenize_invoked": 0, "prompt_invoked": 0,
                "distribution_invoked": 0, "render_invoked": 0, "step_invoked": 0}
    cleanup_attempted = False

    def before(operation):
        nonlocal phase
        if time.monotonic() >= deadline:
            raise TimeoutError("worker deadline before " + operation)
        phase = operation + "_invoked_outcome_pending"
        counters[operation + "_invoked"] += 1

    def acknowledged(operation):
        nonlocal phase
        phase = operation + "_acknowledged"
        if time.monotonic() >= deadline:
            raise TimeoutError("worker deadline after " + operation)

    try:
        before("factory"); bridge = bridge_factory(root, profile, native_dir, deadline); acknowledged("factory")
        evaluated = getattr(bridge, "evaluated", None)
        before("tokenize"); prompt_ids = bridge.tokenize(chatml(request), True, True); acknowledged("tokenize")
        if (getattr(bridge, "evaluated", None) != evaluated or type(prompt_ids) is not list or not prompt_ids
                or any(type(t) is not int or not 0 <= t < bridge.vocabulary_size for t in prompt_ids)
                or len(prompt_ids) + CAP > bridge.context_tokens):
            raise ValueError("prompt IDs/context/counter differ")
        write_json(directory / "tokenized-prompt.json", {"prompt_ids": prompt_ids, "add_special": True,
                   "parse_special": True, "request_sha256": request["request_sha256"]})
        prepared = writer.prepare_prompt(prompt_ids)
        before("prompt"); bridge.start(prompt_ids); acknowledged("prompt")
        writer.commit_prompt(prepared, getattr(bridge, "evaluated", None))
        for _ in range(CAP):
            evaluated = len(prompt_ids) + len(ids)
            if getattr(bridge, "evaluated", None) != evaluated:
                raise ValueError("counter before probe differs")
            prepared = writer.prepare_distribution()
            before("distribution"); logits = bridge.distribution(); acknowledged("distribution")
            if (type(logits) is not list or len(logits) != 50
                    or len({r[0] for r in logits}) != 50
                    or any(type(t) is not int or not 0 <= t < bridge.vocabulary_size
                           or t in profile["special_ids"] for t, _ in logits)
                    or getattr(bridge, "evaluated", None) != evaluated):
                raise ValueError("native masked top50/counter differs")
            rows = []
            for token, logit in logits:
                before("render"); rendered = bridge.render(ids + [token]); acknowledged("render")
                if (type(rendered) is not bytes or not rendered.startswith(raw) or len(rendered) <= len(raw)
                        or getattr(bridge, "evaluated", None) != evaluated):
                    raise ValueError("contextual rendering/counter differs")
                rows.append({"token_id": token, "logit_hex": logit, "piece_hex": rendered[len(raw):].hex()})
            evidence = sampling_evidence(raw, request, rows, ids)
            writer.commit_distribution(prepared, evaluated, sample_evidence=evidence)
            selection = evidence["selection"]
            if selection["state"] != "selected":
                stop = selection["state"]
                break
            token = selection["token_id"]
            intended = raw + bytes.fromhex(selection["piece_hex"])
            prepared = writer.prepare_step(token, intended, sample_evidence=evidence)
            before("step"); bridge.step(token); acknowledged("step")
            writer.commit_step(prepared, getattr(bridge, "evaluated", None))
            ids.append(token); raw = intended
            state = classify_prefix(raw, request["words"])["state"]
            if state == "complete":
                stop = "complete"
                break
            if state != "incomplete":
                raise ValueError("committed prefix violates schema")
        if stop is None:
            stop = "token_cap"
        phase = "cleanup_outcome_pending"; cleanup_attempted = True
        bridge.close()
        process = getattr(bridge, "process", None)
        if process is None or process.poll() is None:
            raise ValueError("owned native process was not reaped after cleanup")
        phase = "cleanup_acknowledged"
        if time.monotonic() >= deadline:
            raise TimeoutError("worker deadline before seal")
        proof = writer.seal(ids, raw, stop_label=stop)
        write_json(commitment_path, {"schema": ID + "-terminal-commitment", "binding": binding,
            "terminal_sha256": proof["terminal_sha256"],
            "source": "in-memory seal return persisted outside attempt bundle",
            "git_output_readback_completed": False})
        phase = "sealed_pending_independent_audit"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "detail": str(exc)[:1024]}
    finally:
        if bridge is not None and not cleanup_attempted:
            cleanup_attempted = True
            try:
                bridge.close()
            except BaseException as exc:
                cleanup_error = {"type": type(exc).__name__, "detail": str(exc)[:512]}
        writer.close()
    result = {"schema": ID + "-worker-result", "binding": binding,
              "status": "sealed_pending_audit" if proof and not error and not cleanup_error else "unknown",
              "phase": phase, "invoked": counters, "committed_output_tokens": len(ids),
              "terminal_sha256": None if proof is None else proof["terminal_sha256"],
              "raw_sha256": digest(raw), "raw_bytes": len(raw), "stop_reason": stop,
              "error": error, "cleanup_error": cleanup_error, "retry_allowed": False}
    write_json(directory / "worker-result.json", result)
    return result


def _group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _supervise(directory, worker, *, seconds=SECONDS):
    """One owned session, hard monotonic deadline, terminate BEFORE reaping.

    The child cannot enter factory/model work until the parent verifies its new
    session and sends GO. WNOWAIT retains the leader identity until owned-group
    shutdown. A post-reap existence check can only refuse; it never sends signals.
    Delayed kernel reaping/uninterruptible IO remains UNKNOWN, not a success.
    """
    if type(seconds) not in (int, float) or not 0 < seconds <= SECONDS:
        raise ValueError("bounded wall budget required")
    directory = Path(directory)
    started = time.monotonic(); deadline = started + seconds
    ready_read, ready_write = os.pipe()
    go_read, go_write = os.pipe()
    os.set_blocking(ready_read, False)
    try:
        pid = os.fork()
    except BaseException:
        for fd in (ready_read, ready_write, go_read, go_write):
            os.close(fd)
        raise
    if pid == 0:
        os.close(ready_read); os.close(go_write)
        try:
            os.setsid()
            write_json(directory / "supervisor-child.json", {"pid": os.getpid(), "session_id": os.getsid(0),
                       "deadline_monotonic": deadline})
            os.write(ready_write, b"R"); os.close(ready_write)
            if os.read(go_read, 1) != b"G":
                raise RuntimeError("supervisor did not authorize owned session")
            os.close(go_read)
            worker(deadline)
            os._exit(0)
        except BaseException as exc:
            try:
                write_json(directory / "worker-crash.json", {"type": type(exc).__name__, "detail": str(exc)[:1024]})
            finally:
                os._exit(70)
    os.close(ready_write); os.close(go_read)
    group_owned = expired = False
    observed_exit = status = None
    shutdown_error = None
    old_term = signal.getsignal(signal.SIGTERM)
    def terminate_supervisor(signum, frame):
        raise SystemExit("supervisor interrupted by SIGTERM")
    signal.signal(signal.SIGTERM, terminate_supervisor)
    try:
        while True:
            # Do not reap: the unreaped child pins PID/PGID against reuse.
            observed = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if observed is not None:
                observed_exit = observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
                expired = time.monotonic() >= deadline
                break
            if time.monotonic() >= deadline:
                expired = True
                break
            if not group_owned:
                try:
                    ready = os.read(ready_read, 1)
                except BlockingIOError:
                    ready = None
                if ready == b"R":
                    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
                        raise ValueError("child did not create its owned session")
                    group_owned = True
                    os.write(go_write, b"G")
                elif ready == b"":
                    raise ValueError("child exited before session handshake")
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
    except BaseException:
        expired = True
        raise
    finally:
        # No native work can start before GO. Once GO is sent, the unreaped
        # leader pins the verified group ID through this shutdown operation.
        try:
            if group_owned:
                os.killpg(pid, signal.SIGKILL)
            elif observed_exit is None:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            shutdown_error = {"type": type(exc).__name__, "detail": str(exc)[:512]}
        os.close(ready_read); os.close(go_write)
        reap_deadline = time.monotonic() + 2.0
        while time.monotonic() < reap_deadline:
            waited, observed = os.waitpid(pid, os.WNOHANG)
            if waited:
                status = observed
                break
            time.sleep(0.01)
        group_present = _group_exists(pid) if group_owned else False
        observation = {"schema": ID + "-supervisor-result", "pid": pid,
            "wall_seconds": seconds, "elapsed_seconds": time.monotonic() - started,
            "deadline_expired": expired, "observed_exit_before_reap": observed_exit,
            "wait_status": status, "child_exit_code": None if status is None else os.waitstatus_to_exitcode(status),
            "session_verified_before_worker": group_owned,
            "group_shutdown_before_reap": group_owned and shutdown_error is None,
            "owned_group_present_after_cleanup": group_present, "shutdown_error": shutdown_error,
            "status": ("completed_child" if not expired and observed_exit == 0 and status == 0
                       and group_owned and not group_present and shutdown_error is None else "unknown"),
            "retry_allowed": False}
        signal.signal(signal.SIGTERM, old_term)
        write_json(directory / "supervisor-result.json", observation)
    return observation


def audit_finished_attempt(directory, commitment, binding, profile, request):
    from .author_candidate_mask_first_v4_audit import audit_attempt
    directory = Path(directory)
    supervisor = read_json(directory / "supervisor-result.json")
    if supervisor.get("status") != "completed_child":
        raise ValueError("supervisor did not certify normal owned child completion")
    worker = read_json(directory / "worker-result.json")
    pin = read_json(commitment)
    if (worker.get("status") != "sealed_pending_audit" or worker.get("binding") != binding
            or worker.get("error") is not None or worker.get("cleanup_error") is not None
            or set(pin) != {"schema", "binding", "terminal_sha256", "source", "git_output_readback_completed"}
            or pin["schema"] != ID + "-terminal-commitment" or canonical(pin["binding"]) != canonical(binding)
            or pin["source"] != "in-memory seal return persisted outside attempt bundle"
            or pin["git_output_readback_completed"] is not False
            or pin["terminal_sha256"] != worker["terminal_sha256"]):
        raise ValueError("worker/commitment identity differs")
    audited = audit_attempt(directory / "sealed", binding, profile, request, pin["terminal_sha256"])
    for name in ("committed_output_tokens", "raw_sha256", "raw_bytes", "stop_reason"):
        if canonical(worker.get(name)) != canonical(audited[name]):
            raise ValueError("worker disagrees with sealed audit: " + name)
    expected = {"factory_invoked": 1, "tokenize_invoked": 1, "prompt_invoked": 1,
                "distribution_invoked": audited["distribution_probes"],
                "render_invoked": 50 * audited["distribution_probes"],
                "step_invoked": audited["committed_output_tokens"]}
    if canonical(worker["invoked"]) != canonical(expected):
        raise ValueError("physical call counters differ from sealed audit")
    return {**audited, "binding": binding, "terminal_sha256": pin["terminal_sha256"],
            "retry_allowed": False, "post_save_Git_evidence_verified": False}


def required_source_paths():
    # Freeze the unchanged bridge's transitive source closure plus NEW source.
    from .author_candidate_local_native_v1 import REQUIRED_CLOSURE
    return set(REQUIRED_CLOSURE) | set(FROZEN) | {
        PROPOSAL, "src/synsteg_audit/candidate_journal_guard_v3.py",
        "src/synsteg_audit/author_candidate_format_v2.py",
        "src/synsteg_audit/author_candidate_mask_first_v4_native.py",
        "src/synsteg_audit/author_candidate_mask_first_v4_audit.py",
        "tests/test_author_candidate_mask_first_v4_native.py",
        "tests/test_author_candidate_mask_first_v4_audit.py",
        "scripts/run_author_candidate_mask_first_v4.py",
    }


def audit_campaign_ledger(output, profile, requests, protocol_sha256, trusted_terminal_pins):
    """Read-only crash reconciliation; missing evidence means UNKNOWN, no resume.

    Pins must come from actual Git restoration or another independent trusted
    source. Never promote a same-directory self-hash into trusted input. This
    reports the fixed denominator even if the coordinator died before result.json.
    """
    output = Path(output)
    if len(requests) != 2 or [r.get("chunk_order") for r in requests] != [0, 1]:
        raise ValueError("fixed two-request order required")
    for request in requests:
        validate_request(request)
    allowed = {"chunk0-attempt0", "chunk1-attempt0"}
    if (type(trusted_terminal_pins) is not dict or set(trusted_terminal_pins) - allowed
            or any(type(v) is not str or re.fullmatch("[0-9a-f]{64}", v) is None
                   for v in trusted_terminal_pins.values())):
        raise ValueError("external terminal pin map differs")
    if not stat.S_ISDIR(output.lstat().st_mode):
        raise ValueError("existing nonsymlink campaign directory required")
    reservation_dir = output / "reservations"
    if os.path.lexists(reservation_dir):
        if not stat.S_ISDIR(reservation_dir.lstat().st_mode):
            raise ValueError("reservation directory must not be a symlink")
        actual_reservations = {p.name for p in reservation_dir.iterdir()}
    else:
        # A crash immediately after campaign mkdir precedes every reservation
        # and backend. Preserve the denominator even in this empty-start state.
        actual_reservations = set()
    if actual_reservations - {name + ".json" for name in allowed}:
        raise ValueError("unexpected reservation outside fixed two attempts")
    states, scheduling_violations = [], []
    terminal_before = False
    for request in requests:
        binding = binding_for(profile, request, protocol_sha256)
        name = binding["attempt_id"]
        directory = output / name
        reserved = name + ".json" in actual_reservations
        state = {"chunk_order": request["chunk_order"], "request_sha256": request["request_sha256"],
                 "reservation_present": reserved, "status": "not_attempted"}
        if not reserved:
            if (os.path.lexists(directory) or os.path.lexists(output / "commitments" / (name + ".json"))
                    or name in trusted_terminal_pins):
                state.update(status="unknown", reason="attempt evidence without reservation")
                terminal_before = True
            states.append(state)
            continue
        if terminal_before:
            scheduling_violations.append(name + " reserved after prior unknown/resource or unattempted predecessor")
        if request["chunk_order"] == 1 and states[0]["status"] == "not_attempted":
            scheduling_violations.append(name + " reserved before fixed predecessor")
        try:
            reservation = read_json(output / "reservations" / (name + ".json"))
            if (reservation.get("schema") != ID + "-reservation" or reservation.get("no_retry") is not True
                    or reservation.get("outcome_known") is not False
                    or canonical(reservation.get("binding")) != canonical(binding)):
                raise ValueError("reservation identity differs")
            if name not in trusted_terminal_pins:
                raise ValueError("no externally pinned terminal; reservation outcome unknown")
            commitment = output / "commitments" / (name + ".json")
            if read_json(commitment).get("terminal_sha256") != trusted_terminal_pins[name]:
                raise ValueError("local commitment differs from external terminal pin")
            result = audit_finished_attempt(directory, commitment, binding, profile, request)
            state.update(status=result["outcome"], audit=result)
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            state.update(status="unknown", reason=type(exc).__name__ + ": " + str(exc)[:1024])
        terminal_before = terminal_before or state["status"] in {"unknown", "resource_refusal"}
        states.append(state)
    return {"schema": ID + "-offline-ledger-audit", "chunks": states, "requested_chunks": 2,
            "reservations": sum(s["reservation_present"] for s in states),
            "format_accepted_chunks_observed": sum(s["status"] == "format_accepted" for s in states),
            "scheduling_violations": scheduling_violations,
            "campaign_complete": not scheduling_violations and all(s["status"] in
                {"format_accepted", "empty_schema_support", "token_cap_refusal"} for s in states),
            "usable_candidate_rows": None, "capacity_bits": None, "retry_allowed": False,
            "native_calls": 0, "tokenizer_calls": 0, "read_only": True}


def verify_gate(root, activation_path, intent_path, readback_path):
    """Require separate future Git-pinned activation; proposal remains disabled."""
    from .framed_interval_public_v1_native import validate_profile
    root = Path(root).resolve()
    if Path(__file__).resolve() != root / "src/synsteg_audit/author_candidate_mask_first_v4_native.py":
        raise ValueError("fresh worker import must come from the pinned checkout")
    for path, expected in FROZEN.items():
        if sha_file(root / path) != expected:
            raise ValueError("frozen input changed: " + path)
    proposal = read_json(root / PROPOSAL)
    expected_fields = {"experiment_id": ID, "execution_enabled": False, "token_cap_per_attempt": CAP,
        "wall_seconds_per_attempt": SECONDS, "maximum_physical_attempts": 2, "attempts_per_chunk": 1,
        "fixed_request_sha256": list(REQUEST_HASHES), "requests_sha256": FROZEN[REQUESTS]}
    if any(canonical(proposal.get(k)) != canonical(v) for k, v in expected_fields.items()):
        raise ValueError("saved proposal differs")
    requests = read_json(root / REQUESTS)["requests"]
    if len(requests) != 2:
        raise ValueError("full fixed denominator2 required")
    for order, request in enumerate(requests):
        validate_request(request)
        if request["chunk_order"] != order:
            raise ValueError("fixed order differs")
    profile = read_json(root / PROFILE); validate_profile(profile)
    activation_path, intent_path, readback_path = map(Path, (activation_path, intent_path, readback_path))
    closure_path = root / BASE / "source-closure-v1.json"
    closure = read_json(closure_path)
    if set(closure) != {"schema", "files_sha256"} or closure["schema"] != ID + "-source-closure-v1":
        raise ValueError("source closure schema differs")
    if set(closure["files_sha256"]) != required_source_paths():
        raise ValueError("full source closure differs")
    common = {"experiment_id": ID, "proposal_sha256": sha_file(root / PROPOSAL),
        "source_closure_sha256": sha_file(closure_path), "output": OUTPUT,
        "logical_requests": 2, "maximum_physical_attempts": 2,
        "automatic_retry": False, "paid_calls_allowed": False, "closed_test_allowed": False}
    intent = {"schema": ID + "-execution-intent-v1", **common, "outcomes_known": False}
    activation = {"schema": ID + "-activation-v1", **common, "enabled": True,
                  "execution_intent_sha256": sha_file(intent_path), "root_exclusive_NN_lease_required": True}
    if canonical(read_json(intent_path)) != canonical(intent) or canonical(read_json(activation_path)) != canonical(activation):
        raise ValueError("separate enabled activation/intent absent or differs")
    expected = {**closure["files_sha256"], str(closure_path.relative_to(root)): sha_file(closure_path)}
    for path in (activation_path, intent_path):
        relative = path.resolve().relative_to(root).as_posix()
        if not relative.startswith(BASE + "/") or relative in expected:
            raise ValueError("distinct v4 activation and intent paths required")
        expected[relative] = sha_file(path)
    receipt = read_json(readback_path)
    if (set(receipt) != {"schema", "source_commit", "activation_commit", "all_files_exact", "files_sha256"}
            or receipt["schema"] != ID + "-source-readback-v1" or receipt["all_files_exact"] is not True
            or any(re.fullmatch("[0-9a-f]{40}", receipt[k]) is None for k in ("source_commit", "activation_commit"))
            or receipt["files_sha256"] != expected):
        raise ValueError("actual Git source and activation readback required")
    for path, expected_sha in expected.items():
        if sha_file(root / path) != expected_sha:
            raise ValueError("pinned source changed: " + path)
    return requests, profile, receipt


def run_campaign(root, activation_path, intent_path, readback_path):
    root = Path(root).resolve()
    requests, profile, receipt = verify_gate(root, activation_path, intent_path, readback_path)
    output = root / OUTPUT
    lease = os.open(root / "results/project-nn.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lease).st_mode):
            raise ValueError("NN lease must be regular")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mkdir(output.parent, exist_ok=True); mkdir(output)
        mkdir(output / "reservations"); mkdir(output / "commitments")
        write_json(output / "campaign-start.json", {"schema": ID + "-campaign-start", "source_readback": receipt,
                   "requested_chunks": 2, "maximum_physical_attempts": 2, "outcomes_known": False})
        states = [{"chunk_order": i, "request_sha256": r["request_sha256"], "status": "not_attempted", "attempts": []}
                  for i, r in enumerate(requests)]
        write_json(output / "initial-denominator.json", {"chunks": states, "outside_diagnostic_chunks": 107})
        terminal = None
        for state, request in zip(states, requests):
            binding = binding_for(profile, request, sha_file(root / PROPOSAL))
            name = binding["attempt_id"]
            # Durable exclusive reservation precedes attempt directory/fork/factory.
            write_json(output / "reservations" / (name + ".json"), {"schema": ID + "-reservation",
                       "binding": binding, "source_commit": receipt["source_commit"],
                       "activation_commit": receipt["activation_commit"], "no_retry": True, "outcome_known": False})
            directory = output / name
            commitment = output / "commitments" / (name + ".json")
            observed = None
            try:
                mkdir(directory)
                observed = _supervise(directory, lambda deadline: _worker(root, profile, request, directory,
                                      commitment, binding, deadline))
                result = audit_finished_attempt(directory, commitment, binding, profile, request)
            except BaseException as exc:
                result = {"outcome": "unknown", "binding": binding, "retry_allowed": False,
                          "supervisor_observation": observed, "error": {"type": type(exc).__name__, "detail": str(exc)[:1024]}}
            state["attempts"].append(result); state["status"] = result["outcome"]
            write_json(output / ("after-" + name + ".json"), {"chunks": states})
            if state["status"] in {"unknown", "resource_refusal"}:
                terminal = state["status"]
                break
            if state["status"] not in {"format_accepted", "empty_schema_support", "token_cap_refusal"}:
                raise ValueError("unexpected outcome; stop without retry")
        result = {"schema": ID + "-campaign-result", "chunks": states, "requested_chunks": 2,
                  "attempts_reserved": sum(bool(s["attempts"]) for s in states),
                  "attempts_physical_scope": "reservations consume attempts; abrupt crash may leave physical execution unknown",
                  "format_accepted_chunks_observed": sum(s["status"] == "format_accepted" for s in states),
                  "status": terminal + "_terminal" if terminal else "complete_diagnostic_pending_Git_audit",
                  "full_cover_chunks": 109, "outside_diagnostic_chunks": 107,
                  "usable_candidate_rows": None, "capacity_bits": None, "payload_delivery": None,
                  "receiver_eligibility_invariance": None, "paid_calls": 0, "closed_test_opened": False,
                  "automatic_retries": 0, "post_save_Git_evidence_verified": False}
        write_json(output / "result.json", result)
        return result
    finally:
        os.close(lease)
