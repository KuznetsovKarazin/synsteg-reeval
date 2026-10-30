"""Evidence-only collector for public-ending-v2; never launches a model/tokenizer.

Journal hash chains attest saved process evidence, not an independent rerun of
the probability law. The native receiver performs keyed full-stream replay.
Only the experiment coordinator may supply an expected public test payload.
"""
import json
import re
from pathlib import Path

from .framed_interval_public_ending_v2 import VERSION, THRESHOLD_BITS, TOKEN_CAP, terminal_punctuation
from .framed_interval_public_ending_v2_orchestration import reconcile
from .framed_interval_llm_v1_journal import recover_journal, sha
from .framed_interval_public_ending_v2 import canonical_json

SUCCESS_SENDERS = {"ending_pending_receiver", "ordinary_ending_generated"}


def _load(path):
    return json.loads(Path(path).read_bytes())


def _ending_evidence(directory, core, saved, rec, role, wire, expected_payload):
    traces = saved["committed_traces"]
    count = core.get("processed_tokens" if role == "receive" else "emitted_tokens")
    if type(count) is not int or not 1 <= count <= TOKEN_CAP or len(traces) != count:
        raise ValueError("complete bounded committed token count differs")
    if role == "receive" and core.get("wire_tokens") != count:
        raise ValueError("receiver did not process all wire tokens")
    if (saved["unknown_native_step"] or saved["torn_last_record"] or not rec.get("wire_available")
            or rec.get("known_committed_tokens") != count or rec.get("unknown_native_step")
            or rec.get("torn_last_record") or core.get("committed_trace_matches_returned_prefix") is not True
            or core.get("noncommitted_trace_suffix") != []):
        raise ValueError("committed trace is incomplete or has unknown suffix")
    trace_hash = sha(canonical_json(traces))
    if (trace_hash != core.get("trace_sha256") or trace_hash != rec.get("committed_trace_sha256")
            or _load(directory / "traces.json") != traces):
        raise ValueError("full committed/returned/saved trace identity differs")
    # recover_journal already validated each prepare/commit identity and hash
    # chain. Reconstruct exact prefix bytes from committed suffixes, with no
    # tokenizer and no text repair, to independently locate the first ending.
    suffixes = []
    pending = None
    for line in (directory / "token-journal.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        if row["kind"] == "prepared":
            pending = bytes.fromhex(row["suffix_hex"])
        elif row["kind"] == "committed":
            suffixes.append(pending)
            pending = None
    if len(suffixes) != count:
        raise ValueError("committed suffix count differs")
    resolved, crossing, first_ending, raw = 0, None, None, b""
    threshold_wire_bytes = None
    for i, (trace, suffix) in enumerate(zip(traces, suffixes), 1):
        if trace.get("prefix_sha256") != sha(raw):
            raise ValueError("trace prefix hash differs from committed wire")
        bits = trace.get("emitted_bits")
        if type(bits) is not list or any(type(b) is not int or b not in (0, 1) for b in bits):
            raise ValueError("invalid emitted bit evidence")
        resolved += len(bits)
        raw += suffix
        raw.decode("utf-8")
        if crossing is None and resolved >= THRESHOLD_BITS:
            crossing = i
            threshold_wire_bytes = len(raw)
        if crossing is not None and first_ending is None and terminal_punctuation(raw):
            first_ending = i
    if (crossing is None or first_ending != count or core.get("first_threshold_token") != crossing
            or core.get("ending_token") != first_ending or core.get("tail_tokens") != count - crossing
            or not 0 <= count - crossing <= 96 or resolved != core.get("resolved_bits")
            or core.get("has_terminal_punctuation") is not True):
        raise ValueError("first threshold/first ending/tail bound differs")
    if (raw != saved["committed_wire"] or len(raw) > 65536 or rec.get("wire_sha256") != sha(raw)
            or rec.get("wire_bytes") != len(raw) or (directory / "committed-wire.bin").read_bytes() != raw):
        raise ValueError("exact committed wire identity or byte bound differs")
    if role == "generate":
        if core.get("wire_sha256") != sha(raw) or core.get("wire_bytes") != len(raw):
            raise ValueError("sender returned wire identity differs")
        expected_status = "ordinary_ending_generated" if core.get("arm") == "ordinary" else "ending_pending_receiver"
        if core.get("arm") not in {"stego", "ordinary"} or core.get("status") != expected_status:
            raise ValueError("sender arm/status differs")
    else:
        if type(wire) is not bytes or raw != wire:
            raise ValueError("receiver committed wire differs from exact supplied wire")
        payload_hex = core.get("payload_hex")
        if (type(payload_hex) is not str or re.fullmatch("(?:[0-9a-f]{2}){0,16}", payload_hex) is None
                or core.get("authenticated_delivery") is not True):
            raise ValueError("authenticated payload evidence differs")
        payload = bytes.fromhex(payload_hex)
        if (directory / "received-payload.bin").read_bytes() != payload:
            raise ValueError("received payload artifact differs")
        if expected_payload is not None and payload != expected_payload:
            raise ValueError("payload differs from coordinator public test fixture")
    return {"committed_tokens": count, "first_threshold_token": crossing, "ending_token": first_ending,
            "tail_tokens": count-crossing, "resolved_bits": resolved, "wire_bytes": len(raw),
            "threshold_wire_bytes": threshold_wire_bytes, "tail_wire_bytes": len(raw)-threshold_wire_bytes,
            "end_wire_bytes": len(raw), "tail_seconds": None,
            "tail_seconds_scope": "not_measured; frozen journal has no per-token timestamps; endpoint wall_seconds separate",
            "terminal_punctuation_only_not_semantic_completion": True}


def collect(directory, process, role, wire=None, *, expected_payload=None):
    directory = Path(directory)
    if role not in {"generate", "receive"}:
        raise ValueError("known endpoint role required")
    outcome = {"process": process, "worker_status": "missing_terminal_record", "authenticated_delivery": False}
    path = directory / "worker-result.json"
    if not path.exists():
        outcome["artifact_error"] = "missing terminal worker record"
        return outcome
    try:
        worker = _load(path)
        outcome.update(worker=worker, worker_status=worker.get("status"))
        if (worker.get("schema") != VERSION + "-native-worker-result" or worker.get("role") != role
                or worker.get("execution_kind") != "native"):
            raise ValueError("worker identity differs")
        core = worker.get("core") or {}
        rec = reconcile(directory)
        outcome["reconciliation"] = rec
        clean = (worker.get("status") == "returned" and process.get("status") == "exited"
                 and process.get("exit_code") == 0 and worker.get("model_process_attempted") is True
                 and not any("error" in key for key in worker))
        success = core.get("status") in SUCCESS_SENDERS if role == "generate" else core.get("status") == "authenticated_delivery"
        if success:
            if not clean:
                raise ValueError("successful status lacks clean terminal process evidence")
            outcome["ending_evidence"] = _ending_evidence(directory, core,
                recover_journal(directory / "token-journal.jsonl"), rec, role, wire, expected_payload)
            outcome["authenticated_delivery"] = role == "receive"
            if role == "receive" and expected_payload is not None:
                outcome["exact_payload"] = True
        elif core.get("authenticated_delivery") is True or core.get("payload_hex") is not None:
            raise ValueError("payload/authentication advertised without successful ending")
    except Exception as exc:
        outcome["delivery_evidence_error"] = type(exc).__name__ + ": " + str(exc)[:512]
        outcome["authenticated_delivery"] = False
    return outcome
