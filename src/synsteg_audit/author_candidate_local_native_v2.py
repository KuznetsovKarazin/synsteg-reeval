"""Disabled-by-default native candidate v2, integrating frozen format and guard.

No inference/tokenization at import. Composite new condition: prompt, shape,
prefix stop, seed domain and durable gate differ from v1. Not GPT-4o replication.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import time

from . import candidate_journal_guard_v2 as guard
from .author_candidate_format_v2 import classify_prefix, validate_response
from .author_candidate_local_native_v1 import production_bridge, chatml, REQUIRED_CLOSURE as V1_SOURCE_CLOSURE
from .framed_interval_public_v1_native import validate_profile, public_digest
from .seeded_pair_llm import quantize

ID = "author-candidate-local-native-v2"
BASE = "experiments/" + ID
PROTOCOL = BASE + "/protocol.json"
CLOSURE = BASE + "/source-closure.json"
DEFAULT_ACTIVATION = BASE + "/activation.disabled.json"
REQUESTS = "experiments/author-candidate-format-v2/requests.json"
FORMAT_PROTOCOL = "experiments/author-candidate-format-v2/protocol.json"
PROFILE = "experiments/framed-interval-public-v1/native-public-profile.json"
OUTPUT = "results/" + ID + "/native-diagnostic-v2"
CAP = 512
SECONDS = 600
TOTAL = 1 << 24
REQUEST_HASHES = (
    "15f413e60449231fe505a45091d838f36155d15307bcdf16b27d1215ffa05104",
    "d9f3597b716fb4fbb888a83ceccd881f94635537132ac5dc00c0748ddd14b4f4",
)
FROZEN = {
    "src/synsteg_audit/candidate_journal_guard_v2.py": "48b7536a28e623b2633616108dda3ea286cb7af562d6edd250ba8ece570a0477",
    "src/synsteg_audit/author_candidate_format_v2.py": "20f70a75be1a7ab12825a00714b72e96e11c2fc8ae24a09d85af78508fc62273",
    FORMAT_PROTOCOL: "d19c4ce18167601fcc05cd3a167c3752aa952cee9f2f64fbd46f8743d31495da",
    REQUESTS: "0cc5750f8251cb7b10bb29d00ec858d5b005d0df42fb7d2781152c807a4ae421",
    PROFILE: "f32097750dc898fce418a29146587a4229eae1dcb76e3cbcd374b692846adc8a",
}
REQUIRED_CLOSURE = frozenset(V1_SOURCE_CLOSURE) | frozenset(FROZEN) | frozenset({
    PROTOCOL, DEFAULT_ACTIVATION, BASE + "/run_native.py",
    "src/synsteg_audit/candidate_journal_guard_v2.py",
    "src/synsteg_audit/author_candidate_format_v2.py",
    "src/synsteg_audit/author_candidate_local_native_v2.py",
    "scripts/prepare_author_candidate_local_native_v2.py",
    "tests/test_author_candidate_local_native_v2.py",
    "docs/author_candidate_local_native_v2_design_ru.md",
})

canonical = guard.canonical
digest = guard.sha


def sha_file(path):
    # This helper hashes bounded metadata/source artifacts, never model weights.
    return digest(guard._read_stable(path))


def read_json(path):
    def pairs(items):
        value = {}
        for key, entry in items:
            if key in value:
                raise ValueError("duplicate metadata JSON key")
            value[key] = entry
        return value
    def constant(_):
        raise ValueError("nonfinite metadata JSON")
    return json.loads(guard._read_stable(path), object_pairs_hook=pairs, parse_constant=constant)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir_durable(path, *, exist_ok=False):
    """Publish each new directory entry before any backend can be created."""
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError:
        if not exist_ok or not stat.S_ISDIR(path.lstat().st_mode):
            raise
    fsync_directory(path.parent)


def write_new(path, raw):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("zero-progress write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def write_json(path, value):
    write_new(path, canonical(value))


def protocol_document():
    return {
        "schema": ID + "-protocol", "id": ID,
        "default_native_activation_enabled": False,
        "activation_policy": "Separate enabled activation and execution intent must be Git-pinned and read back before execution.",
        "composite_variant_not_prompt_only_causal_comparison": True,
        "not_author_GPT4o_replication": True,
        "fixed_request_sha256": list(REQUEST_HASHES), "requests_path": REQUESTS,
        "frozen_component_files_sha256": FROZEN,
        "diagnostic_chunks": 2, "full_cover_chunks": 109, "outside_diagnostic_chunks": 107,
        "five_token_context_unchanged": True,
        "model": "Qwen2.5-1.5B-Instruct Q4_K_M", "profile": PROFILE,
        "profile_scope": "resources/runtime/READY/resident identity only, not inherited temperature0.8",
        "temperature": 0.7, "top_k": 50, "integer_total": TOTAL,
        "token_cap": CAP, "attempt_seconds": SECONDS, "attempts_per_chunk_max": 3,
        "maximum_physical_attempts": 6, "cold_process_per_attempt": True,
        "EOS_policy": "existing EOS/special mask; not vanilla EOS stopping",
        "seed_domain": ID, "seed_search": False,
        "stop": "first committed complete, invalid_prefix or resource_refusal; otherwise512committedtokens",
        "whole_final_token_preserved": True, "response_repair_or_fence_removal": False,
        "format_abstention": "[] passes shape and ends chunk; useful candidates/capacity remain null",
        "known_format_refusal": "May use only next predeclared attempt, max3 per chunk.",
        "unknown_or_resource_refusal": "Stop campaign, preserve evidence, never retry/resume.",
        "external_terminal_pin": "Durable coordinator commitment outside sealed bundle, then exact Git readback before final scientific acceptance.",
        "attempt_registry": "One fixed output path; immutable per-attempt reservations before backend creation; existing campaign directory refused.",
        "output": OUTPUT, "native_calls_at_design": 0, "paid_calls_allowed": False,
        "closed_test_allowed": False, "full_candidate_resource_complete": False,
        "semantics_and_capacity_measured": False,
    }


def validate_request(request, chunk_order):
    if type(chunk_order) is not int or chunk_order not in (0, 1):
        raise ValueError("fixed chunk0/1 required")
    if type(request) is not dict or request.get("chunk_order") != chunk_order:
        raise ValueError("request identity")
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if request.get("request_sha256") != REQUEST_HASHES[chunk_order] or digest(canonical(body)) != REQUEST_HASHES[chunk_order]:
        raise ValueError("fixed request changed")


def validate_inputs(root):
    root = Path(root)
    for relative, expected in FROZEN.items():
        if sha_file(root / relative) != expected:
            raise ValueError("frozen component changed: " + relative)
    protocol = read_json(root / PROTOCOL)
    if canonical(protocol) != canonical(protocol_document()):
        raise ValueError("native v2 protocol changed")
    requests = read_json(root / REQUESTS)
    if len(requests.get("requests", [])) != 2 or requests.get("generation_enabled") is not False:
        raise ValueError("fixed offline request cohort differs")
    for order, request in enumerate(requests["requests"]):
        validate_request(request, order)
    profile = read_json(root / PROFILE)
    validate_profile(profile)
    return protocol, requests["requests"], profile


def seed_hex(request_sha256, attempt_order):
    if type(attempt_order) is not int or not 0 <= attempt_order < 3:
        raise ValueError("fixed attempt0..2 required")
    if request_sha256 not in REQUEST_HASHES:
        raise ValueError("fixed request hash required")
    return digest(canonical([ID, "fixed-public-diagnostic-seed", request_sha256, attempt_order]))


def choose(weights, seed, request_sha256, attempt_order, ids):
    if sum(x["weight"] for x in weights) != TOTAL:
        raise ValueError("categorical total differs")
    context = [ID, request_sha256, attempt_order, len(ids), digest(canonical(ids))]
    draw = int.from_bytes(hmac.new(bytes.fromhex(seed), canonical(context), hashlib.sha256).digest()[:3], "big")
    remaining = draw
    for row in weights:
        if remaining < row["weight"]:
            return row["token_id"], draw, digest(canonical(context))
        remaining -= row["weight"]
    raise ValueError("categorical law incomplete")


def binding_for(profile, request, attempt_order, protocol_sha256):
    return {"experiment_id": ID, "attempt_id": f"chunk{request['chunk_order']}-attempt{attempt_order}",
            "request_sha256": request["request_sha256"], "resource_sha256": public_digest(profile),
            "protocol_sha256": protocol_sha256, "token_cap": CAP}


def assess_verified_raw(raw, words):
    prefix = classify_prefix(raw)
    if prefix["state"] == "resource_refusal":
        return "parser_resource_refusal", prefix, None
    if prefix["state"] != "complete":
        return "format_rejected", prefix, None
    try:
        return "format_accepted", prefix, validate_response(raw, words)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        return "format_rejected", {**prefix, "shape_error": type(exc).__name__ + ": " + str(exc)[:512]}, None


def read_external_commitment(path, binding):
    value = guard.strict_json(guard._read_stable(path))
    expected_keys = {"schema", "binding", "terminal_sha256", "source", "git_output_readback_completed"}
    if (set(value) != expected_keys or value["schema"] != ID + "-terminal-commitment"
            or canonical(value["binding"]) != canonical(binding)
            or value["source"] != "in-memory seal return persisted outside attempt bundle"
            or value["git_output_readback_completed"] is not False
            or not re.fullmatch("[0-9a-f]{64}", value["terminal_sha256"])):
        raise ValueError("external commitment differs")
    return value


def run_attempt(root, profile, request, attempt_order, directory, commitment_path, *,
                protocol_sha256, bridge_factory=production_bridge):
    """Exactly one reserved physical attempt. Caller owns campaign/NN lease."""
    validate_request(request, request["chunk_order"])
    seed = seed_hex(request["request_sha256"], attempt_order)
    binding = binding_for(profile, request, attempt_order, protocol_sha256)
    directory = Path(directory); mkdir_durable(directory)
    native_dir = directory / "native"; mkdir_durable(native_dir)
    write_json(directory / "launch.json", {"schema": ID + "-attempt-launch", "binding": binding,
        "seed_hex": seed, "request": request, "prompt_text_sha256": digest(chatml(request).encode()),
        "token_cap": CAP, "attempt_seconds": SECONDS})
    writer = guard.AttemptWriter(directory / "sealed", binding)
    bridge = None; ids = []; raw = b""; prompt_ids = []
    prompt_invoked = False; prompt_committed = False; steps_invoked = 0
    distribution_calls = 0; pending_step = False; stop = None; cleanup_error = None
    error = None; proof = None; format_result = None; prefix = None; outcome = "unknown"
    cleanup_attempted = False
    started = time.monotonic()
    try:
        bridge = bridge_factory(root, profile, native_dir, started + SECONDS)
        before = getattr(bridge, "evaluated", None)
        prompt_ids = bridge.tokenize(chatml(request), True, True)
        if getattr(bridge, "evaluated", None) != before:
            raise ValueError("TOKENIZE mutated evaluated counter")
        if (type(prompt_ids) is not list or not prompt_ids
                or any(type(i) is not int or not 0 <= i < bridge.vocabulary_size for i in prompt_ids)
                or len(prompt_ids) + CAP > bridge.context_tokens):
            raise ValueError("prompt/context IDs invalid")
        write_json(directory / "tokenized-prompt.json", {"prompt_ids": prompt_ids,
            "request_sha256": request["request_sha256"], "add_special": True, "parse_special": True})
        prepared = writer.prepare_prompt(prompt_ids)
        prompt_invoked = True; bridge.start(prompt_ids)
        writer.commit_prompt(prepared, getattr(bridge, "evaluated", None))
        prompt_committed = True
        for index in range(CAP):
            evaluated = len(prompt_ids) + len(ids)
            if getattr(bridge, "evaluated", None) != evaluated:
                raise ValueError("native history before distribution differs")
            distribution_calls += 1; logits = bridge.distribution()
            if (getattr(bridge, "evaluated", None) != evaluated or len(logits) != 50
                    or len({token for token, _ in logits}) != 50
                    or any(type(token) is not int or not 0 <= token < bridge.vocabulary_size
                           or token in profile["special_ids"] for token, _ in logits)):
                raise ValueError("native masked top50 support/counter differs")
            weights = quantize(logits, temperature=0.7, total=TOTAL)
            token, draw, context_sha = choose(weights, seed, request["request_sha256"], attempt_order, ids)
            intended_ids = ids + [token]; intended_raw = bridge.render(intended_ids)
            if getattr(bridge, "evaluated", None) != evaluated or type(intended_raw) is not bytes:
                raise ValueError("TEXT mutated counter or returned non-bytes")
            prepared = writer.prepare_step(token, intended_raw, sample_evidence={
                "request_sha256": request["request_sha256"], "seed_hex": seed,
                "prefix_ids": ids, "weights": weights, "draw_uint24": draw,
                "draw_context_sha256": context_sha, "temperature": 0.7, "integer_total": TOTAL})
            pending_step = True; steps_invoked += 1; bridge.step(token)
            writer.commit_step(prepared, getattr(bridge, "evaluated", None))
            pending_step = False; ids = intended_ids; raw = intended_raw
            prefix = classify_prefix(raw)
            if prefix["state"] != "incomplete":
                stop = prefix["state"]; break
        if stop is None:
            stop = "token_cap"
        # Cleanup is part of known-outcome completeness; a failed cleanup never seals.
        cleanup_attempted = True
        try:
            bridge.close()
        except BaseException as exc:
            cleanup_error = {"type": type(exc).__name__, "detail": str(exc)[:512]}
            raise
        proof = writer.seal(ids, raw, stop_label=stop)
        # This commitment comes from the writer's return, not a reread self-hash
        # in the mutable bundle. It is separately persisted before acceptance.
        write_json(commitment_path, {"schema": ID + "-terminal-commitment", "binding": binding,
            "terminal_sha256": proof["terminal_sha256"],
            "source": "in-memory seal return persisted outside attempt bundle",
            "git_output_readback_completed": False})
        commitment = read_external_commitment(commitment_path, binding)
        checked = guard.verify_bundle(directory / "sealed", binding,
                                      expected_terminal_sha256=commitment["terminal_sha256"])
        outcome, prefix, format_result = assess_verified_raw(checked["response"], request["words"])
    except BaseException as exc:
        error = {"type": type(exc).__name__, "detail": str(exc)[:1024]}
        outcome = "unknown"
    finally:
        if bridge is not None and not cleanup_attempted:
            try:
                bridge.close()
            except BaseException as exc:
                cleanup_error = {"type": type(exc).__name__, "detail": str(exc)[:512]}
                outcome = "unknown"
        writer.close()
    write_new(directory / "observed-response.raw", raw)
    process = getattr(bridge, "process", None)
    result = {"schema": ID + "-attempt-result", "binding": binding, "chunk_order": request["chunk_order"],
        "attempt_order": attempt_order, "request_sha256": request["request_sha256"], "seed_hex": seed,
        "outcome": outcome, "stop_reason": stop, "prefix": prefix,
        "raw_sha256": digest(raw), "raw_bytes": len(raw), "generated_ids": ids,
        "committed_output_tokens": len(ids), "prompt_token_count": len(prompt_ids),
        "prompt_invoked": prompt_invoked, "prompt_committed": prompt_committed,
        "distribution_calls_invoked": distribution_calls, "step_calls_invoked": steps_invoked,
        "step_calls_committed": len(ids), "pending_step_unknown": pending_step,
        "format_result": format_result if outcome == "format_accepted" else None,
        "terminal_sha256": None if proof is None else proof["terminal_sha256"],
        "commitment_sha256": sha_file(commitment_path) if Path(commitment_path).is_file() else None,
        "format_acceptance_scope": "local persisted gate; final independent Git output readback still required",
        "usable_candidate_rows": None, "capacity_bits": None, "semantic_equivalence_verified": False,
        "elapsed_seconds": time.monotonic() - started, "native_pid": getattr(process, "pid", None),
        "native_returncode_after_cleanup": None if process is None else process.poll(),
        "error": error, "cleanup_error": cleanup_error, "provider_usage": None,
        "retry_allowed": False, "EOS_sampled": False}
    write_json(directory / "attempt-result.json", result)
    return result


def verify_attempt_outcome(profile, request, attempt_order, directory, commitment_path, protocol_sha256, observed):
    binding = binding_for(profile, request, attempt_order, protocol_sha256)
    expected_identity = {"schema": ID + "-attempt-result", "chunk_order": request["chunk_order"],
        "attempt_order": attempt_order, "request_sha256": request["request_sha256"],
        "seed_hex": seed_hex(request["request_sha256"], attempt_order), "binding": binding}
    if type(observed) is not dict or any(canonical(observed.get(k)) != canonical(v) for k, v in expected_identity.items()):
        raise ValueError("attempt returned wrong identity")
    if canonical(observed) != guard._read_stable(Path(directory) / "attempt-result.json"):
        raise ValueError("returned result differs from persisted result")
    if observed.get("outcome") == "unknown":
        return observed
    commitment = read_external_commitment(commitment_path, binding)
    if observed.get("commitment_sha256") != sha_file(commitment_path):
        raise ValueError("commitment changed")
    checked = guard.verify_bundle(Path(directory) / "sealed", binding,
                                  expected_terminal_sha256=commitment["terminal_sha256"])
    history = verify_sampling_and_stop(Path(directory) / "sealed", binding, profile, request,
                                       attempt_order, commitment["terminal_sha256"])
    outcome, prefix, format_result = assess_verified_raw(checked["response"], request["words"])
    expected = {"outcome": outcome, "prefix": prefix, "format_result": format_result,
        "raw_sha256": digest(checked["response"]), "raw_bytes": len(checked["response"]),
        "generated_ids": checked["output_ids"], "committed_output_tokens": checked["committed_tokens"],
        "step_calls_committed": checked["committed_tokens"], "terminal_sha256": checked["terminal_sha256"],
        "stop_reason": history["stop_reason"], "prompt_token_count": history["prompt_tokens"],
        "prompt_invoked": True, "prompt_committed": True,
        "pending_step_unknown": False, "error": None, "cleanup_error": None,
        "usable_candidate_rows": None, "capacity_bits": None, "semantic_equivalence_verified": False}
    if any(canonical(observed.get(k)) != canonical(v) for k, v in expected.items()):
        raise ValueError("claimed format/counters differ from independently reopened bundle")
    if (type(observed.get("step_calls_invoked")) is not int or observed["step_calls_invoked"] != checked["committed_tokens"]
            or type(observed.get("distribution_calls_invoked")) is not int or observed["distribution_calls_invoked"] != checked["committed_tokens"]):
        raise ValueError("native command counters differ")
    return observed


def verify_sampling_and_stop(bundle, binding, profile, request, attempt_order, terminal_sha256):
    """Same-implementation mechanical replay; independent native audit is additional."""
    terminal_raw = guard._read_stable(bundle / "terminal.json")
    if digest(terminal_raw) != terminal_sha256:
        raise ValueError("terminal changed during coordinator replay")
    manifest = guard.strict_json(terminal_raw)
    journal = guard._read_stable(bundle / "journal.jsonl")
    if digest(journal) != manifest["files"]["journal.jsonl"]["sha256"]:
        raise ValueError("journal changed during coordinator replay")
    state = guard.verify_trace(journal, binding)
    prepared = [guard.strict_json(line)["event"] for line in journal.splitlines()
                if guard.strict_json(line)["event"]["kind"] == "step_prepared"]
    seed = seed_hex(request["request_sha256"], attempt_order)
    ids = []
    for index, event in enumerate(prepared):
        evidence = event["sample_evidence"]
        weights = evidence.get("weights")
        if (type(weights) is not list or len(weights) != 50
                or any(type(w) is not dict or set(w) != {"token_id", "logit_hex", "weight"}
                       or type(w["token_id"]) is not int or w["token_id"] in profile["special_ids"]
                       or not 0 <= w["token_id"] < profile["ready"]["vocabulary_size"]
                       or type(w["weight"]) is not int or w["weight"] < 0 for w in weights)):
            raise ValueError("saved categorical evidence schema differs")
        computed = quantize([(w["token_id"], w["logit_hex"]) for w in weights], temperature=0.7, total=TOTAL)
        if canonical(computed) != canonical(weights):
            raise ValueError("saved quantization differs")
        token, draw, context = choose(weights, seed, request["request_sha256"], attempt_order, ids)
        expected = {"request_sha256": request["request_sha256"], "seed_hex": seed,
                    "prefix_ids": ids, "weights": computed, "draw_uint24": draw,
                    "draw_context_sha256": context, "temperature": 0.7, "integer_total": TOTAL}
        if canonical(evidence) != canonical(expected) or event["token_id"] != token:
            raise ValueError("saved keyed selection differs")
        prefix = classify_prefix(bytes.fromhex(event["intended_raw_hex"]))
        if index < len(prepared) - 1 and prefix["state"] != "incomplete":
            raise ValueError("continued after first terminal prefix")
        ids.append(token)
    last = classify_prefix(state["raw"])
    stop = "token_cap" if last["state"] == "incomplete" else last["state"]
    if (last["state"] == "incomplete" and len(ids) != CAP) or state["stop_label"] != stop:
        raise ValueError("stopping rule differs")
    return {"stop_reason": stop, "prompt_tokens": len(state["prompt_ids"]),
            "saved_distribution_rows": len(ids) * 50, "same_implementation_check": True}


def execution_intent(root):
    root = Path(root)
    return {"schema": ID + "-execution-intent", "run_id": "native-diagnostic-v2",
        "protocol_sha256": sha_file(root / PROTOCOL), "source_closure_sha256": sha_file(root / CLOSURE),
        "output": OUTPUT, "logical_requests": 2, "maximum_physical_attempts": 6,
        "outcomes_known": False, "automatic_retry": False, "paid_calls_allowed": False, "closed_test_allowed": False}


def enabled_activation(root, intent_path):
    root = Path(root)
    return {"schema": ID + "-activation", "enabled": True,
        "protocol_sha256": sha_file(root / PROTOCOL), "source_closure_sha256": sha_file(root / CLOSURE),
        "execution_intent_sha256": sha_file(intent_path), "output": OUTPUT,
        "logical_requests": 2, "maximum_physical_attempts": 6,
        "root_exclusive_nn_lease_required": True, "paid_calls_allowed": False, "closed_test_allowed": False}


def verify_gate(root, activation_path, intent_path, readback_path):
    root = Path(root).resolve()
    protocol, requests, profile = validate_inputs(root)
    activation_path, intent_path, readback_path = map(Path, (activation_path, intent_path, readback_path))
    activation = read_json(activation_path)
    if activation.get("enabled") is not True:
        raise ValueError("native-v2 activation disabled")
    intent = read_json(intent_path)
    if canonical(intent) != canonical(execution_intent(root)) or canonical(activation) != canonical(enabled_activation(root, intent_path)):
        raise ValueError("activation/intent contract differs")
    closure = read_json(root / CLOSURE)
    if closure.get("schema") != ID + "-source-closure" or set(closure.get("files_sha256", {})) != REQUIRED_CLOSURE:
        raise ValueError("source closure set differs")
    expected = {**closure["files_sha256"], CLOSURE: sha_file(root / CLOSURE)}
    for p in (activation_path, intent_path):
        relative = p.resolve().relative_to(root).as_posix()
        if not relative.startswith(BASE + "/") or relative in expected:
            raise ValueError("activation and intent require distinct new experiment paths")
        expected[relative] = sha_file(p)
    receipt = read_json(readback_path)
    if (receipt.get("schema") != ID + "-source-readback" or receipt.get("all_files_exact") is not True
            or not re.fullmatch("[0-9a-f]{40}", receipt.get("source_commit", ""))
            or not re.fullmatch("[0-9a-f]{40}", receipt.get("activation_commit", ""))
            or receipt.get("files_sha256") != expected):
        raise ValueError("actual Git source/activation readback required")
    for relative, expected_sha in expected.items():
        if sha_file(root / relative) != expected_sha:
            raise ValueError("frozen input changed: " + relative)
    return protocol, requests, profile, receipt


def run_campaign(root, output, activation_path, intent_path, readback_path, *, attempt_runner=run_attempt):
    root = Path(root).resolve(); output = Path(output).resolve()
    protocol, requests, profile, receipt = verify_gate(root, activation_path, intent_path, readback_path)
    if output != root / OUTPUT:
        raise ValueError("one fixed campaign output path required; no alternate-directory rerun")
    nn_lease = os.open(root / "results/project-nn.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(nn_lease).st_mode):
            raise ValueError("project NN lease must be regular")
        fcntl.flock(nn_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mkdir_durable(output.parent, exist_ok=True)
        mkdir_durable(output)
        mkdir_durable(output / "reservations"); mkdir_durable(output / "commitments")
        write_json(output / "intent.json", {"schema": ID + "-campaign-launch", "protocol": protocol,
            "source_readback": receipt, "outcomes_known": False, "requested_chunks": 2, "maximum_physical_attempts": 6})
        states = [{"chunk_order": i, "request_sha256": r["request_sha256"], "status": "not_attempted", "attempts": []}
                  for i, r in enumerate(requests)]
        write_json(output / "initial-denominator.json", {"chunks": states, "full_cover_chunks": 109, "outside_diagnostic_chunks": 107})
        terminal = None
        for chunk, request in zip(states, requests):
            if terminal:
                break
            for attempt_order in range(3):
                name = f"chunk{chunk['chunk_order']}-attempt{attempt_order}"
                directory = output / name; commitment = output / "commitments" / (name + ".json")
                binding = binding_for(profile, request, attempt_order, sha_file(root / PROTOCOL))
                write_json(output / "reservations" / (name + ".json"), {"schema": ID + "-attempt-reservation",
                    "binding": binding, "source_commit": receipt["source_commit"], "activation_commit": receipt["activation_commit"],
                    "no_retry": True, "outcome_known": False})
                observed = None
                try:
                    observed = attempt_runner(root, profile, request, attempt_order, directory, commitment,
                                              protocol_sha256=binding["protocol_sha256"])
                    result = verify_attempt_outcome(profile, request, attempt_order, directory, commitment,
                                                    binding["protocol_sha256"], observed)
                except BaseException as exc:
                    try:
                        canonical(observed); preserved_observed = observed
                    except (ValueError, TypeError, RecursionError):
                        preserved_observed = {"non_json_repr": repr(observed)[:4096]}
                    result = {"schema": ID + "-coordinator-unknown", "binding": binding, "outcome": "unknown",
                        "error": {"type": type(exc).__name__, "detail": str(exc)[:1024]},
                        "observed_returned_result": preserved_observed, "retry_allowed": False}
                chunk["attempts"].append(result); chunk["status"] = result["outcome"]
                write_json(output / ("after-" + name + ".json"), {"chunks": states})
                if result["outcome"] == "format_accepted":
                    break
                if result["outcome"] in {"unknown", "parser_resource_refusal"}:
                    terminal = result["outcome"]; break
                if result["outcome"] != "format_rejected":
                    raise ValueError("unexpected attempt status")
            if chunk["status"] == "format_rejected":
                chunk["status"] = "format_attempts_exhausted"
        accepted = sum(s["status"] == "format_accepted" for s in states)
        result = {"schema": ID + "-campaign-result", "chunks": states, "requested_chunks": 2,
            "format_accepted_chunks_observed": accepted,
            "complete_campaign_format_accepted_chunks": None if terminal == "unknown" else accepted,
            "attempts_executed": sum(len(s["attempts"]) for s in states), "max_attempts": 6,
            "status": terminal + "_terminal" if terminal else "complete_diagnostic_pending_Git_audit",
            "full_cover_chunks": 109, "outside_diagnostic_chunks": 107,
            "full_cover_candidate_resource_complete": False, "original_fixed_transport_outcomes": 0,
            "usable_candidate_rows": None, "capacity_bits": None, "semantic_equivalence_verified": False,
            "provider": "new composite EOS-masked Qwen condition, not author GPT4o or prompt-only causal comparison",
            "paid_calls": 0, "closed_test_opened": False, "automatic_retries": 0,
            "post_save_Git_evidence_verified": False}
        write_json(output / "result.json", result)
        return result
    finally:
        os.close(nn_lease)
