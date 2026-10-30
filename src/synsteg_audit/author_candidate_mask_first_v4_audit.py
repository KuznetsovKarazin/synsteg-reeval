"""Independent offline arithmetic/trace audit for the proposed mask-first v4.

No model, tokenizer, worker or v4 support imports. The frozen v3 parser and
journal verifier are shared dependencies, not independently reimplemented.
Saved float32 logits and contextual pieces are checked for internal consistency;
this audit cannot prove they came from a genuine native model or tokenizer.
Syntax completion, semantic validity and receiver reproducibility are distinct.
"""
from __future__ import annotations

import hashlib
import hmac
import math
from pathlib import Path
import re
import struct

from . import candidate_journal_guard_v3 as guard
from .author_candidate_schema_v3 import classify_prefix
from .author_candidate_format_v2 import validate_response

ID = "author-candidate-mask-first-v4"
TOTAL = 16777216
CAP = 512
MAX_BYTES = 262144
TEMPERATURE = 0.7
canonical = guard.canonical
digest = guard.sha


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _whitespace(raw, suffix):
    # Count quotes preceded by an even backslash run. A quote inside a JSON
    # string is escaped exactly when the immediately preceding run is odd.
    in_string = False
    backslashes = 0
    for char in raw:
        if char == 34 and (not in_string or backslashes % 2 == 0):
            in_string = not in_string
        backslashes = backslashes + 1 if char == 92 else 0
    return not in_string and bool(suffix) and not set(suffix).difference(b" \t\n\r")


def _rows(rows):
    _require(type(rows) is list and len(rows) == 50, "exactly all 50 rows required")
    result = []
    for row in rows:
        _require(type(row) is dict and set(row) == {"token_id", "logit_hex", "piece_hex"},
                 "top50 row schema differs")
        token, logit_hex, piece_hex = row["token_id"], row["logit_hex"], row["piece_hex"]
        _require(type(token) is int and 0 <= token < 2**31, "top50 token ID differs")
        _require(type(logit_hex) is str and re.fullmatch(r"[0-9a-f]{8}", logit_hex),
                 "float32 logit encoding differs")
        _require(type(piece_hex) is str and 0 < len(piece_hex) <= 2 * (MAX_BYTES + 65536)
                 and re.fullmatch(r"(?:[0-9a-f]{2})+", piece_hex), "contextual suffix encoding differs")
        logit = struct.unpack(">f", bytes.fromhex(logit_hex))[0]
        _require(math.isfinite(logit), "nonfinite logit")
        result.append((token, logit, bytes.fromhex(piece_hex)))
    ids = [r[0] for r in result]
    _require(len(set(ids)) == 50, "duplicate token ID")
    _require(result == sorted(result, key=lambda r: (-r[1], r[0])), "top50 order differs")
    return result


def _mask_first(raw, words, rows):
    _require(type(raw) is bytes, "exact raw bytes required")
    decoded = _rows(rows)
    initial = classify_prefix(raw, words)["state"]
    if initial == "resource_refusal":
        return {"schema": ID, "state": "resource_refusal", "reason": "prefix byte budget"}
    _require(initial == "incomplete", "probe requires an incomplete prefix")
    observations = []
    eligible_indexes = []
    for index, ((token, logit, piece), row) in enumerate(zip(decoded, rows)):
        state = classify_prefix(raw + piece, words)["state"]
        _require(state in {"incomplete", "complete", "invalid_prefix", "resource_refusal"},
                 "unrecognized grammar classification")
        observations.append(dict(row, prefix_state=state, weight=0,
                                 structural_whitespace=_whitespace(raw, piece)))
        if state in {"incomplete", "complete"}:
            eligible_indexes.append(index)
    if any(row["prefix_state"] == "resource_refusal" for row in observations):
        return {"schema": ID, "state": "resource_refusal", "reason": "candidate byte budget",
                "rows": observations}
    if not eligible_indexes:
        return {"schema": ID, "state": "empty_schema_support", "rows": observations,
                "integer_total": 0, "positive_tokens": 0}

    maximum = max(decoded[i][1] for i in eligible_indexes)
    exponentials = {i: math.exp((decoded[i][1] - maximum) / TEMPERATURE) for i in eligible_indexes}
    denominator = math.fsum(exponentials.values())
    fractional = {}
    for i in eligible_indexes:
        scaled = exponentials[i] / denominator * TOTAL
        whole = math.floor(scaled)
        observations[i]["weight"] = whole
        fractional[i] = scaled - whole
    residual = TOTAL - sum(r["weight"] for r in observations)
    _require(0 <= residual <= len(eligible_indexes), "largest remainder residual differs")
    priority = sorted(eligible_indexes, key=lambda i: (-fractional[i], decoded[i][0]))
    for i in priority[:residual]:
        observations[i]["weight"] += 1
    positive = [r for r in observations if r["weight"]]
    _require(sum(r["weight"] for r in positive) == TOTAL and positive, "integer mass not conserved")
    return {
        "schema": ID, "state": "ready", "rows": observations,
        "integer_total": TOTAL, "temperature": TEMPERATURE,
        "eligible_tokens": len(eligible_indexes), "positive_tokens": len(positive),
        "positive_support_structural_whitespace_only": all(r["structural_whitespace"] for r in positive),
        "structural_whitespace_mass": {
            "numerator": sum(r["weight"] for r in positive if r["structural_whitespace"]),
            "denominator": TOTAL},
        "exponential_underflow_ids": [decoded[i][0] for i in eligible_indexes if exponentials[i] == 0.0],
        "semantic_capacity_bits": None,
    }


def _law(raw, words, mask):
    positive = sorted((r for r in mask["rows"] if r["weight"]), key=lambda r: r["token_id"])
    base = [{"token_id": r["token_id"], "piece": r["piece_hex"], "mass": r["weight"]}
            for r in positive]
    accepted = [{"token_id": r["token_id"], "piece_hex": r["piece_hex"], "mass": r["weight"],
                 "conditional_numerator": r["weight"], "conditional_denominator": TOTAL}
                for r in positive]
    law = {"schema": "author-candidate-schema-v3-conditional-law", "state": "ready",
           "prefix_sha256": digest(raw), "words_sha256": digest(canonical(list(words))),
           "input_mass": TOTAL, "retained_mass": TOTAL,
           "retained_fraction": {"numerator": TOTAL, "denominator": TOTAL},
           "accepted": accepted, "rejected": [], "base_law_sha256": digest(canonical(base))}
    law["law_sha256"] = digest(canonical(law))
    return law


def _draw(law, key, request_hash, step):
    # Keep the full rejection contract even though TOTAL=2**24 divides 2**256.
    limit = (2**256 // law["retained_mass"]) * law["retained_mass"]
    for counter in range(128):
        context = {"domain": "author-candidate-schema-v3/draw-v1", "request_sha256": request_hash,
                   "attempt": 0, "step": step, "law_sha256": law["law_sha256"], "counter": counter}
        integer = int.from_bytes(hmac.digest(key, canonical(context), hashlib.sha256), "big")
        if integer >= limit:
            continue
        ticket = integer % law["retained_mass"]
        lower = 0
        for row in law["accepted"]:
            if lower <= ticket < lower + row["mass"]:
                return {"state": "selected", "token_id": row["token_id"], "piece_hex": row["piece_hex"],
                        "ticket": ticket, "counter": counter, "law_sha256": law["law_sha256"]}
            lower += row["mass"]
        raise ValueError("ticket outside reconstructed intervals")
    return {"state": "resource_refusal", "reason": "HMAC draw block limit", "law_sha256": law["law_sha256"]}


def expected_evidence(raw, words, rows, request_hash, ids):
    """Reconstruct exact sampling evidence without importing the implementation.

    Supports synthetic request hashes for source tests. The campaign coordinator
    must additionally enforce the preregistered two-request allowlist.
    """
    _require(_hash(request_hash), "request digest required")
    _require(type(ids) is list and len(ids) < CAP
             and all(type(t) is int and 0 <= t < 2**31 for t in ids), "prefix token IDs/cap differ")
    key = bytes.fromhex(digest(canonical([
        "author-candidate-local-native-v3", "fixed-public-diagnostic-key", request_hash])))
    mask = _mask_first(raw, words, rows)
    law = None
    if mask["state"] == "ready":
        law = _law(raw, words, mask)
        selection = _draw(law, key, request_hash, len(ids))
    elif mask["state"] == "empty_schema_support":
        selection = {"state": "empty_support", "reason": "empty_schema_support"}
    else:
        selection = {"state": "resource_refusal", "reason": mask["reason"]}
    return {"request_sha256": request_hash, "key_hex": key.hex(), "prefix_ids": list(ids),
            "prefix_raw_sha256": digest(raw), "top50_rows": rows,
            "mask_first": mask, "conditional_law": law, "selection": selection}


def _context(binding, profile, request):
    guard.validate_binding(binding)
    _require(binding["experiment_id"] == ID and binding["token_cap"] == CAP, "v4 binding ID/cap differs")
    _require(type(request) is dict and type(request.get("chunk_order")) is int
             and request["chunk_order"] in (0, 1), "fixed cohort chunk index required")
    _require(binding["attempt_id"] == f"chunk{request['chunk_order']}-attempt0", "attempt identity differs")
    request_hash = request.get("request_sha256")
    _require(_hash(request_hash) and request_hash == binding["request_sha256"]
             and digest(canonical({k: v for k, v in request.items() if k != "request_sha256"})) == request_hash,
             "request content/binding differs")
    _require(type(profile) is dict and digest(canonical(profile)) == binding["resource_sha256"],
             "public resource binding differs")
    ready, special = profile.get("ready"), profile.get("special_ids")
    _require(type(ready) is dict and type(ready.get("vocabulary_size")) is int
             and 1 <= ready["vocabulary_size"] <= 2**31, "public vocabulary differs")
    _require(type(special) is list
             and all(type(t) is int and 0 <= t < ready["vocabulary_size"] for t in special)
             and special == sorted(set(special)), "public special-ID list differs")
    _require("words" in request, "request words missing")
    classify_prefix(b"", request["words"])
    return ready["vocabulary_size"], set(special)


def audit_attempt(bundle, binding, profile, request, terminal_sha256):
    """Accept only an externally pinned complete sealed trace and exact replay.

    The caller pins request/profile/protocol and the external terminal digest.
    No wall-clock deadline or genuine inference is proved by saved journal data.
    Any exception means UNKNOWN, never an accepted candidate or known refusal.
    """
    vocabulary, special = _context(binding, profile, request)
    bundle = Path(bundle)
    sealed = guard.verify_bundle(bundle, binding, expected_terminal_sha256=terminal_sha256)
    terminal_raw = guard._read_stable(bundle / "terminal.json")
    _require(digest(terminal_raw) == terminal_sha256, "terminal changed during audit")
    manifest = guard.strict_json(terminal_raw)
    journal = guard._read_stable(bundle / "journal.jsonl")
    _require(digest(journal) == manifest["files"]["journal.jsonl"]["sha256"], "journal changed during audit")
    trace = guard.verify_trace(journal, binding)
    _require(all(t < vocabulary for t in trace["prompt_ids"]), "prompt token outside vocabulary")
    raw, ids, pending = b"", [], None
    probes = whitespace_steps = streak = maximum_streak = whitespace_only_probes = 0
    for encoded in journal.splitlines():
        event = guard.strict_json(encoded)["event"]
        kind = event["kind"]
        if kind == "distribution_committed":
            _require(classify_prefix(raw, request["words"])["state"] == "incomplete",
                     "distribution after first complete prefix")
            evidence = event["sample_evidence"]
            rows = evidence.get("top50_rows")
            decoded = _rows(rows)
            _require(all(t < vocabulary and t not in special for t, _, _ in decoded),
                     "special or out-of-vocabulary top50 token")
            expected = expected_evidence(raw, request["words"], rows, request["request_sha256"], ids)
            _require(canonical(expected) == canonical(evidence), "saved sampling evidence differs")
            pending = expected
            probes += 1
            if expected["mask_first"].get("positive_support_structural_whitespace_only", False):
                whitespace_only_probes += 1
        elif kind == "step_prepared":
            _require(pending is not None and pending["selection"]["state"] == "selected",
                     "STEP without independently selected probe")
            selected = pending["selection"]
            _require(canonical(event["sample_evidence"]) == canonical(pending)
                     and event["token_id"] == selected["token_id"]
                     and bytes.fromhex(event["intended_raw_hex"]) == raw + bytes.fromhex(selected["piece_hex"]),
                     "STEP token/raw/selection mismatch")
        elif kind == "step_committed":
            _require(pending is not None and pending["selection"]["state"] == "selected",
                     "committed STEP without selection")
            suffix = bytes.fromhex(pending["selection"]["piece_hex"])
            structural = _whitespace(raw, suffix)
            whitespace_steps += int(structural)
            streak = streak + 1 if structural else 0
            maximum_streak = max(maximum_streak, streak)
            raw += suffix
            ids.append(pending["selection"]["token_id"])
            pending = None
    last = classify_prefix(raw, request["words"])
    if pending is not None:
        stop = pending["selection"]["state"]
        _require(stop in {"empty_support", "resource_refusal"}, "selected probe left unconsumed")
    elif last["state"] == "complete":
        stop = "complete"
    else:
        _require(last["state"] == "incomplete" and len(ids) == CAP, "early or invalid token-cap stop")
        stop = "token_cap"
    _require(trace["stop_label"] == stop and trace["ids"] == ids and trace["raw"] == raw,
             "sealed stopping state differs")
    _require(sealed["response"] == raw and sealed["output_ids"] == ids,
             "bundle changed between guard and arithmetic replay")
    outcome = {"complete": "format_accepted", "empty_support": "empty_schema_support",
               "resource_refusal": "resource_refusal", "token_cap": "token_cap_refusal"}[stop]
    return {
        "outcome": outcome, "stop_reason": stop, "generated_ids": ids,
        "raw_sha256": digest(raw), "raw_bytes": len(raw), "committed_output_tokens": len(ids),
        "prompt_token_count": len(trace["prompt_ids"]), "distribution_probes": probes,
        "saved_distribution_rows": 50 * probes, "render_calls": 50 * probes,
        "terminal_sha256": terminal_sha256,
        "format_result": validate_response(raw, request["words"]) if stop == "complete" else None,
        "whitespace": {"structural_whitespace_steps": whitespace_steps,
                       "max_structural_whitespace_run": maximum_streak,
                       "terminal_structural_whitespace_run": streak,
                       "whitespace_only_positive_support_probes": whitespace_only_probes},
        "usable_candidate_rows": None, "capacity_bits": None,
        "semantic_equivalence_verified": False,
        "independent_arithmetic_and_draw_replay": True,
        "shared_frozen_parser_and_journal_guard": True,
        "native_logits_or_tokenizer_recomputed": False,
        "genuine_native_execution_verified": False,
    }


def verify_bundle(directory, binding, profile, request, *, expected_terminal_sha256):
    """Named acceptance entry point with an explicitly external terminal pin."""
    return audit_attempt(directory, binding, profile, request, expected_terminal_sha256)
