#!/usr/bin/env python3
"""Conservative deny-only gate. No model calls, nonce writes, or launch authority.

This separately identified helper does not alter or automatically intercept any
frozen runner. A future outer launcher must call it using the actual job inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

VERSION = "continuation-replay-guard-v1"
GUARD_SCHEMA = "synsteg-interruption-no-replay-guard-v1"
REQUEST_SCHEMA = "synsteg-continuation-replay-request-v1"
GUARD_SHA256 = "ed7c17d68df5cc524ec8e441f3febb2a96e6d8108eda6f22f2ac154fae74db0c"
MAX_INPUT_BYTES = 65536
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX24 = re.compile(r"[0-9a-f]{24}\Z", re.ASCII)
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)


class GuardError(ValueError):
    """Missing, malformed or unpinned evidence: launch must remain blocked."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def exact_fields(value: object, fields: set[str], label: str) -> dict:
    require(type(value) is dict and set(value) == fields, label + " fields differ")
    return value


def strict_json(raw: bytes, label: str) -> object:
    require(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES, label + " bytes missing or oversized")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, label + " duplicate JSON key")
            result[key] = value
        return result

    def bad_constant(_):
        raise GuardError(label + " non-finite JSON number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GuardError(label + " invalid UTF-8/JSON") from exc


def hex_value(value: object, regex: re.Pattern, label: str) -> None:
    require(type(value) is str and regex.fullmatch(value) is not None, label + " must be lowercase fixed-width hex")


def label_value(value: object, label: str) -> None:
    require(type(value) is str and LABEL.fullmatch(value) is not None, label + " must be a nonempty ASCII label")


def order_value(value: object, label: str) -> None:
    require(type(value) is int and 1 <= value <= 100, label + " must be an integer in development order1..100")


def validate_guard(guard: object) -> dict:
    guard = exact_fields(guard, {
        "schema", "source_commit", "deny_experiment_ids", "conservatively_burned_nonce_pairs",
        "blocked_baseline_intents", "historical_unknown_baseline_orders", "next_unreserved_baseline_order",
        "does_not_change_historical_ledger", "proof_of_all_nonce_use", "rule",
    }, "guard")
    require(guard["schema"] == GUARD_SCHEMA, "guard schema differs")
    hex_value(guard["source_commit"], HEX40, "guard source commit")
    denied = guard["deny_experiment_ids"]
    require(type(denied) is list and len(denied) > 0, "guard denied experiments missing")
    for value in denied:
        label_value(value, "denied experiment")
    require(len(set(denied)) == len(denied), "duplicate denied experiment")

    nonce_pairs = guard["conservatively_burned_nonce_pairs"]
    require(type(nonce_pairs) is list and len(nonce_pairs) == 8, "guard must retain all eight nonce pairs")
    nonce_ids, case_ids = set(), set()
    for row in nonce_pairs:
        row = exact_fields(row, {"case_id", "arm", "nonce_hex", "key_sha256", "status"}, "nonce row")
        label_value(row["case_id"], "case id")
        require(row["arm"] in ("stego", "ordinary"), "nonce arm differs")
        hex_value(row["nonce_hex"], HEX24, "nonce")
        hex_value(row["key_sha256"], HEX64, "key fingerprint")
        require(row["status"] == "conservatively_burned_actual_use_unknown", "nonce status differs")
        pair = (row["key_sha256"], row["nonce_hex"])
        require(pair not in nonce_ids and row["case_id"] not in case_ids, "duplicate nonce pair/case")
        nonce_ids.add(pair)
        case_ids.add(row["case_id"])

    blocked = guard["blocked_baseline_intents"]
    require(type(blocked) is list and len(blocked) > 0, "blocked intent rows missing")
    identities, orders = set(), set()
    for row in blocked:
        row = exact_fields(row, {"model_alias", "source_order", "identity", "observed_ledger_status", "scheduling_status", "intent_path"}, "blocked intent")
        label_value(row["model_alias"], "model alias")
        order_value(row["source_order"], "source order")
        hex_value(row["identity"], HEX64, "intent identity")
        require(row["observed_ledger_status"] == "not_attempted", "blocked intent historical ledger status differs")
        require(row["scheduling_status"] == "blocked_reserved_intent_evidence_gap", "blocked intent scheduling status differs")
        require(type(row["intent_path"]) is str and row["intent_path"].startswith("results/") and ".." not in Path(row["intent_path"]).parts, "blocked intent path differs")
        pair = (row["model_alias"], row["source_order"])
        require(row["identity"] not in identities and pair not in orders, "duplicate blocked identity/order")
        identities.add(row["identity"])
        orders.add(pair)

    unknown = guard["historical_unknown_baseline_orders"]
    require(type(unknown) is list and len(unknown) > 0, "historical unknown rows missing")
    previous = set()
    for row in unknown:
        row = exact_fields(row, {"model_alias", "source_order"}, "historical unknown")
        label_value(row["model_alias"], "unknown model alias")
        order_value(row["source_order"], "unknown source order")
        pair = (row["model_alias"], row["source_order"])
        require(pair not in previous and pair not in orders, "duplicate/overlapping unknown order")
        previous.add(pair)
    order_value(guard["next_unreserved_baseline_order"], "next unreserved order")
    require(guard["does_not_change_historical_ledger"] is True, "historical ledger protection absent")
    require(guard["proof_of_all_nonce_use"] is False, "conservative ledger must not claim observed nonce use")
    require(type(guard["rule"]) is str and len(guard["rule"]) > 0, "guard rule missing")
    return guard


def validate_request(request: object) -> dict:
    request = exact_fields(request, {
        "schema", "experiment_id", "model_alias", "source_order", "intent_identity", "key_sha256", "nonces",
    }, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema differs")
    label_value(request["experiment_id"], "experiment id")
    require((request["model_alias"] is None) == (request["source_order"] is None), "model alias/source order must be supplied together")
    if request["model_alias"] is not None:
        label_value(request["model_alias"], "model alias")
        order_value(request["source_order"], "source order")
    if request["intent_identity"] is not None:
        hex_value(request["intent_identity"], HEX64, "intent identity")
    nonces = request["nonces"]
    require(type(nonces) is list and len(nonces) <= 100, "nonces must be a bounded list")
    require((request["key_sha256"] is None) == (len(nonces) == 0), "key fingerprint and nonempty nonce list must be supplied together")
    if request["key_sha256"] is not None:
        hex_value(request["key_sha256"], HEX64, "key fingerprint")
    for nonce in nonces:
        hex_value(nonce, HEX24, "nonce")
    require(len(set(nonces)) == len(nonces), "duplicate request nonce")
    return request


def evaluate_bytes(guard_bytes: bytes, request_bytes: bytes) -> dict:
    require(type(guard_bytes) is bytes, "guard bytes required")
    require(hashlib.sha256(guard_bytes).hexdigest() == GUARD_SHA256, "guard missing/changed: exact pinned receipt required")
    guard = validate_guard(strict_json(guard_bytes, "guard"))
    request = validate_request(strict_json(request_bytes, "request"))
    reasons = []
    if request["experiment_id"] in guard["deny_experiment_ids"]:
        reasons.append({"kind": "denied_experiment_id", "experiment_id": request["experiment_id"]})
    for row in guard["conservatively_burned_nonce_pairs"]:
        if request["key_sha256"] == row["key_sha256"] and row["nonce_hex"] in request["nonces"]:
            reasons.append({"kind": "conservatively_burned_key_nonce", "case_id": row["case_id"]})
    for row in guard["blocked_baseline_intents"]:
        if request["intent_identity"] == row["identity"]:
            reasons.append({"kind": "blocked_explicit_intent_identity", "identity": row["identity"]})
        if (request["model_alias"], request["source_order"]) == (row["model_alias"], row["source_order"]):
            reasons.append({"kind": "blocked_reserved_order", "model_alias": row["model_alias"], "source_order": row["source_order"]})
    for row in guard["historical_unknown_baseline_orders"]:
        if (request["model_alias"], request["source_order"]) == (row["model_alias"], row["source_order"]):
            reasons.append({"kind": "historical_unknown_order", "model_alias": row["model_alias"], "source_order": row["source_order"]})
    return {
        "schema": VERSION + "-decision", "status": "denied" if reasons else "not_denied_by_this_receipt",
        "launch_authorized": False, "guard_sha256": GUARD_SHA256,
        "guard_source_commit": guard["source_commit"], "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "reasons": reasons, "original_runtime_ledger_recovered": False,
        "required_other_gates": ["actual_job_request_binding", "fresh_git_head", "resource_inventory", "new_intent", "predecessor_receipt", "canonical_nn_lease"],
        "frozen_runners_automatically_intercepted": False,
    }


def read_regular(path: Path, label: str) -> bytes:
    require(not path.is_symlink() and path.is_file(), label + " must be an existing regular file")
    with path.open("rb") as stream:
        value = stream.read(MAX_INPUT_BYTES + 1)
    require(0 < len(value) <= MAX_INPUT_BYTES, label + " bytes missing or oversized")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        decision = evaluate_bytes(read_regular(args.guard, "guard"), read_regular(args.request, "request"))
    except (GuardError, OSError) as exc:
        print(json.dumps({"schema": VERSION + "-decision", "status": "invalid_or_missing_evidence", "launch_authorized": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(decision, sort_keys=True))
    return 3 if decision["status"] == "denied" else 0


if __name__ == "__main__":
    raise SystemExit(main())
