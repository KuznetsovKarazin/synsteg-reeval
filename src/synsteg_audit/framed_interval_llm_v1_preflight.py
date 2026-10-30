"""Pure validation of recorded Qwen tokenizer preflight evidence.

No tokenizer, native library, model, process, file, network or secret is opened.
The coordinator must bind the supplied bytes and collector source to a verified
activation manifest. Consistent JSON alone does not attest truthful execution.
Any native collection requires a separate reviewed collector; never invoked here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

MODEL_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
RUNTIME_COMMIT = "ea9e3446b5c84a7f972cbbdea9df615fc4ff9159"
ARCHIVE_SHA256 = "ac75208179d00d86f378d5f10c65f9785028ce877cf57279f1a7b8dac54a25ff"
CPU_SHA256 = "1bc37cef126b01fe5a15222c07d20d29931936bfaa909a97052f6662f4c9d97c"
PLAN_SCHEMA = "framed-interval-llm-v1-tokenizer-preflight-plan-v1"
RECEIPT_SCHEMA = "framed-interval-llm-v1-tokenizer-preflight-receipt-v1"
MAX_VOCABULARY = 2_000_000

TEXT_FIXTURES = (
    ("empty", ""), ("ascii", "Hello world."),
    ("whitespace", "one  two\nthree"), ("crlf", "A\r\nB"), ("tab", "A\tB"),
    ("edge_spaces", " leading and trailing "), ("latin_precomposed", "Café"),
    ("latin_combining", "Cafe\u0301"), ("cjk", "中文文本。"), ("emoji", "😀"),
    ("chatml_literal", "<|im_start|>assistant\n<|im_end|>"),
)
INVALID_FIXTURES = (("invalid_utf8_lead", "c3"), ("invalid_utf8_byte", "ff"),
                    ("invalid_utf8_partial_emoji", "f09f98"))


class PreflightError(ValueError):
    """Recorded evidence is incomplete, inconsistent, or bound elsewhere."""


def canonical_json(value):
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PreflightError("finite canonical JSON required") from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise PreflightError("SHA256 hex required")
    return value


def _int(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise PreflightError("invalid " + label)
    return value


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise PreflightError(label + " fields differ")


def _load(raw, label):
    if type(raw) is not bytes:
        raise PreflightError(label + " must be exact bytes")
    def pairs(rows):
        result = {}
        for name, value in rows:
            if name in result:
                raise PreflightError("duplicate JSON key in " + label)
            result[name] = value
        return result
    try:
        return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(PreflightError("nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("invalid JSON in " + label) from exc


def _hex(raw_hex, label):
    if type(raw_hex) is not str or re.fullmatch("(?:[0-9a-f]{2})*", raw_hex) is None:
        raise PreflightError("invalid hex bytes in " + label)
    return bytes.fromhex(raw_hex)


def _ids(values, size, label, *, maximum=2048):
    if type(values) is not list or len(values) > maximum:
        raise PreflightError("invalid token array in " + label)
    for token in values:
        _int(token, 0, size-1, label)
    return values


def _strict_utf8(raw):
    try:
        raw.decode("utf-8", "strict")
        return True
    except UnicodeDecodeError:
        return False


def fixed_fixtures():
    """Exact fixed list; this is data construction, not tokenizer execution."""
    result = [{"id": name, "kind": "text", "input_hex": text.encode("utf-8").hex(),
               "require_exact_full_roundtrip": name in {"empty", "ascii", "chatml_literal"},
               "require_no_output_special_ids": name == "chatml_literal",
               "prefix_gate_outcome": "record every step; expected canonical refusals permitted"}
              for name, text in TEXT_FIXTURES]
    result += [{"id": name, "kind": "invalid_utf8", "input_hex": raw,
                "expected_status": "invalid_utf8", "expected_tokenizer_calls": 0}
               for name, raw in INVALID_FIXTURES]
    result.append({"id": "qwen_known_merge", "kind": "token_path",
        "input_hex": b"**\n\n".hex(), "input_token_ids": [334, 271],
        "prefix_render_hex": b"**".hex(), "expected_full_reencoded_ids": [56177],
        "expected_gate_reasons": ["accepted", "retokenization_mismatch"]})
    return result


def chatml_prompt(protocol, prompt_id):
    """Same fixed string construction as the old prompt_text; no tokenization."""
    if type(protocol) is not dict or type(protocol.get("system_prompt")) is not str:
        raise PreflightError("public prompt protocol required")
    rows = protocol.get("prompts")
    if type(rows) is not list or sorted(row.get("id", "") for row in rows if type(row) is dict) != ["library", "station"]:
        raise PreflightError("require exactly station/library prompt definitions")
    selected = [row for row in rows if row["id"] == prompt_id]
    if len(selected) != 1 or type(selected[0].get("user")) is not str:
        raise PreflightError("invalid prompt text")
    return ("<|im_start|>system\n" + protocol["system_prompt"] + "<|im_end|>\n"
            "<|im_start|>user\n" + selected[0]["user"] + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def validate_inventory(rows, *, vocabulary_size, masked_tokens):
    """Validate the entire native ID range; no inference from a subset.

    The caller binds vocabulary_size/masked_tokens to expected native READY.
    This generic helper permits small synthetic vocabularies in unit tests.
    """
    size = _int(vocabulary_size, 1, MAX_VOCABULARY, "vocabulary_size")
    mask_count = _int(masked_tokens, 0, size, "masked_tokens")
    if type(rows) is not list or len(rows) != size:
        raise PreflightError("inventory must contain every native token ID")
    specials, masked = [], 0
    for index, row in enumerate(rows):
        _keys(row, {"token_id", "is_control", "is_eog", "piece_empty"}, "inventory row")
        if type(row["token_id"]) is not int or row["token_id"] != index:
            raise PreflightError("inventory IDs must be ordered exactly0..vocabulary_size-1")
        if any(type(row[key]) is not bool for key in ("is_control", "is_eog", "piece_empty")):
            raise PreflightError("native predicates must be boolean")
        special = row["is_control"] or row["is_eog"]
        if special: specials.append(index)
        masked += bool(special or row["piece_empty"])
    if masked != mask_count:
        raise PreflightError("complete inventory does not reproduce native masked count")
    return {"vocabulary_size": size, "masked_tokens": masked,
            "special_ids": specials, "special_ids_sha256": _sha(canonical_json(specials)),
            "inventory_canonical_sha256": _sha(canonical_json(rows))}


def _prefix_observations(case, reference_ids, size, specials):
    observations = case["prefixes"]
    if type(observations) is not list or len(observations) != len(reference_ids)+1:
        raise PreflightError("all reference prefixes, including empty, must be retained")
    states = []
    for index, row in enumerate(observations):
        _keys(row, {"token_ids", "rendered_hex", "strict_utf8", "reencoded_ids"}, "prefix observation")
        ids = _ids(row["token_ids"], size, "prefix", maximum=128)
        if ids != reference_ids[:index]: raise PreflightError("reference prefix differs")
        raw = _hex(row["rendered_hex"], "prefix")
        if len(raw) > 4096: raise PreflightError("fixed prefix byte cap exceeded")
        valid = _strict_utf8(raw)
        if type(row["strict_utf8"]) is not bool or row["strict_utf8"] != valid:
            raise PreflightError("incorrect recorded strict-UTF8 predicate")
        if valid:
            encoded = _ids(row["reencoded_ids"], size, "reencoding", maximum=128)
        else:
            if row["reencoded_ids"] is not None: raise PreflightError("invalid UTF8 must not be tokenized")
            encoded = None
        states.append((raw, valid, valid and encoded == ids))
    if states[0] != (b"", True, True):
        raise PreflightError("empty codec base is not canonical")
    reasons = []
    for index, token in enumerate(reference_ids, 1):
        old, _, prior_canonical = states[index-1]
        raw, valid, canonical = states[index]
        if not prior_canonical: reason = "invalid_prefix"
        elif token in specials: reason = "special_token"
        elif not raw.startswith(old): reason = "rewrites_prior_bytes"
        elif raw == old: reason = "empty_extension"
        elif not valid: reason = "invalid_utf8"
        elif not canonical: reason = "retokenization_mismatch"
        else: reason = "accepted"
        reasons.append(reason)
    if case["gate_reasons"] != reasons:
        raise PreflightError("recorded gate reasons differ from saved tokenizer observations")
    return states, reasons


def validate_preflight_receipt(receipt, *, plan_bytes, inventory_bytes,
        expected_plan_sha256, expected_resource_identity_sha256,
        expected_native_libraries, expected_collector_source_sha256,
        expected_ready, prompt_protocol):
    """Return launch binding from complete recorded evidence, never run a probe.

    Both byte arguments are supplied by the caller after its safe-path policy.
    The inventory artifact path below is only validated, never opened. External
    activation must also pin the receipt file itself and the collector source.
    """
    _digest(expected_plan_sha256)
    _digest(expected_resource_identity_sha256)
    _digest(expected_collector_source_sha256)
    if type(plan_bytes) is not bytes or _sha(plan_bytes) != expected_plan_sha256:
        raise PreflightError("preflight plan bytes differ from frozen identity")
    plan = _load(plan_bytes, "plan")
    if (type(plan) is not dict or plan.get("schema") != PLAN_SCHEMA
            or canonical_json(plan.get("fixtures")) != canonical_json(fixed_fixtures())
            or plan.get("prompt_ids") != ["station", "library"]
            or plan.get("output_tokenize_flags") != {"add_special": False, "parse_special": False}
            or plan.get("prompt_tokenize_flags") != {"add_special": True, "parse_special": True}):
        raise PreflightError("fixed fixture list or tokenizer flags changed")
    for name in ("output_tokenize_flags", "prompt_tokenize_flags"):
        if type(plan[name]) is not dict or any(type(value) is not bool for value in plan[name].values()):
            raise PreflightError("plan tokenizer flags must be boolean")
    _keys(receipt, {"schema", "status", "plan_sha256", "resources", "collector", "ready",
                    "inventory", "cases", "prompts", "guards"}, "receipt")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["status"] != "passed" or receipt["plan_sha256"] != expected_plan_sha256:
        raise PreflightError("receipt is not a passed result of this exact plan")
    expected_resources = {"model_sha256": MODEL_SHA256, "runtime_commit": RUNTIME_COMMIT,
        "archive_sha256": ARCHIVE_SHA256, "cpu_backend_sha256": CPU_SHA256,
        "resource_identity_sha256": expected_resource_identity_sha256,
        "native_libraries_sha256": expected_native_libraries}
    if type(expected_native_libraries) is not dict or not expected_native_libraries:
        raise PreflightError("native library identities required")
    for name, digest in expected_native_libraries.items():
        if type(name) is not str or not name: raise PreflightError("invalid library identity")
        _digest(digest)
    if receipt["resources"] != expected_resources:
        raise PreflightError("foreign or unverified Qwen/runtime/resource identity")
    _keys(receipt["collector"], {"source_sha256", "mode", "complete_native_id_scan"}, "collector")
    if receipt["collector"] != {"source_sha256": expected_collector_source_sha256,
            "mode": "pinned-native-vocab-only", "complete_native_id_scan": True}:
        raise PreflightError("collector source/mode/completeness differs")
    if type(receipt["collector"]["complete_native_id_scan"]) is not bool:
        raise PreflightError("collector completeness must be boolean")
    _keys(expected_ready, {"vocabulary_size", "masked_tokens", "context_tokens"}, "expected READY")
    size = _int(expected_ready["vocabulary_size"], 56178, MAX_VOCABULARY, "production vocabulary size")
    _int(expected_ready["masked_tokens"], 1, size-1, "production masked count")
    if type(expected_ready["context_tokens"]) is not int or expected_ready["context_tokens"] != 2048:
        raise PreflightError("old context must be2048")
    if (receipt["ready"] != expected_ready or type(receipt["ready"]) is not dict
            or any(type(receipt["ready"].get(k)) is not int for k in expected_ready)):
        raise PreflightError("native READY differs")
    guards = {"forward_calls": 0, "prompt_calls": 0, "step_calls": 0, "distribution_calls": 0,
              "paid_calls": 0, "closed_test_accessed": False, "model_weights_loaded": False}
    if (receipt["guards"] != guards or type(receipt["guards"]) is not dict
            or any(type(receipt["guards"].get(k)) is not type(v) for k, v in guards.items())):
        raise PreflightError("forbidden forward/model/data activity or missing guards")
    item = receipt["inventory"]
    _keys(item, {"path", "bytes", "sha256", "canonical_sha256", "rows", "special_ids", "special_ids_sha256"}, "inventory descriptor")
    path = item["path"]
    if type(path) is not str or not path or "\\" in path:
        raise PreflightError("unsafe inventory artifact path")
    relative = PurePosixPath(path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or str(relative) != path:
        raise PreflightError("unsafe inventory artifact path")
    if type(inventory_bytes) is not bytes or len(inventory_bytes) > 256*1024*1024:
        raise PreflightError("inventory byte limit/type")
    if (type(item["bytes"]) is not int or item["bytes"] != len(inventory_bytes)
            or item["sha256"] != _sha(inventory_bytes) or type(item["rows"]) is not int or item["rows"] != size):
        raise PreflightError("inventory artifact identity/count differs")
    inventory = validate_inventory(_load(inventory_bytes, "inventory"), vocabulary_size=size,
                                   masked_tokens=expected_ready["masked_tokens"])
    if (not inventory["special_ids"] or item["canonical_sha256"] != inventory["inventory_canonical_sha256"]
            or item["special_ids"] != inventory["special_ids"]
            or item["special_ids_sha256"] != inventory["special_ids_sha256"]):
        raise PreflightError("derived complete special-ID manifest differs")
    specials = set(inventory["special_ids"])
    cases = receipt["cases"]
    if type(cases) is not list or len(cases) != len(plan["fixtures"]):
        raise PreflightError("all fixed fixtures must be retained")
    summaries = []
    for spec, case in zip(plan["fixtures"], cases):
        if type(case) is not dict or case.get("id") != spec["id"] or case.get("input_hex") != spec["input_hex"]:
            raise PreflightError("fixture identity/order differs")
        if spec["kind"] == "invalid_utf8":
            _keys(case, {"id", "input_hex", "status", "tokenizer_calls"}, "invalid-UTF8 case")
            if (_strict_utf8(_hex(case["input_hex"], "fixture")) or case["status"] != "invalid_utf8"
                    or type(case["tokenizer_calls"]) is not int or case["tokenizer_calls"] != 0):
                raise PreflightError("invalid UTF8 expectation not met")
            summaries.append({"id": case["id"], "status": "expected_invalid_utf8_refusal"})
            continue
        _keys(case, {"id", "input_hex", "status", "tokenize_flags", "reference_ids", "prefixes", "gate_reasons"}, "observed case")
        if case["status"] != "observed" or case["tokenize_flags"] != {"add_special": False, "parse_special": False}:
            raise PreflightError("output tokenizer flags/status differ")
        if any(type(v) is not bool for v in case["tokenize_flags"].values()):
            raise PreflightError("tokenizer flags must be boolean")
        ids = _ids(case["reference_ids"], size, "reference IDs", maximum=128)
        states, reasons = _prefix_observations(case, ids, size, specials)
        raw = _hex(case["input_hex"], "fixture")
        if spec["kind"] == "token_path":
            if (ids != spec["input_token_ids"] or states[1][0].hex() != spec["prefix_render_hex"]
                    or states[-1][0] != raw or case["prefixes"][-1]["reencoded_ids"] != spec["expected_full_reencoded_ids"]
                    or reasons != spec["expected_gate_reasons"]):
                raise PreflightError("known Qwen merge counterexample differs")
        else:
            if spec["require_exact_full_roundtrip"] and states[-1][0] != raw:
                raise PreflightError("required full text roundtrip failed")
            if case["id"] == "empty" and ids != []:
                raise PreflightError("empty text must tokenize to no output IDs")
            if spec["require_no_output_special_ids"] and any(token in specials for token in ids):
                raise PreflightError("literal ChatML output parsed as a control token")
        summaries.append({"id": case["id"], "status": "observed", "reference_tokens": len(ids),
                          "full_roundtrip_exact": states[-1][0] == raw,
                          "all_prefix_steps_accepted": all(reason == "accepted" for reason in reasons),
                          "gate_reasons": reasons})
    prompts = receipt["prompts"]
    _keys(prompts, {"station", "library"}, "prompt mappings")
    prompt_ids = {}
    for name in ("station", "library"):
        row = prompts[name]
        _keys(row, {"text_sha256", "tokenize_flags", "token_ids", "repeat_token_ids"}, "prompt record")
        expected_text = chatml_prompt(prompt_protocol, name).encode("utf-8")
        if row["text_sha256"] != plan["prompt_text_sha256"][name] or row["text_sha256"] != _sha(expected_text):
            raise PreflightError("public prompt text identity differs")
        if (row["tokenize_flags"] != {"add_special": True, "parse_special": True}
                or type(row["tokenize_flags"]) is not dict
                or any(type(v) is not bool for v in row["tokenize_flags"].values())):
            raise PreflightError("prompt tokenizer flags differ")
        ids = _ids(row["token_ids"], size, "prompt IDs", maximum=1536)
        repeated = _ids(row["repeat_token_ids"], size, "repeated prompt IDs", maximum=1536)
        if not ids or ids != repeated:
            raise PreflightError("prompt tokenization must be nonempty and repeat exactly")
        prompt_ids[name] = list(ids)
    normalized = {"schema": "framed-interval-llm-v1-validated-preflight-v1",
        "plan_sha256": expected_plan_sha256, "resource_identity_sha256": expected_resource_identity_sha256,
        "collector_source_sha256": expected_collector_source_sha256, "ready": dict(expected_ready),
        "inventory_canonical_sha256": inventory["inventory_canonical_sha256"],
        "special_ids": inventory["special_ids"], "special_ids_sha256": inventory["special_ids_sha256"],
        "prompt_ids": prompt_ids, "fixture_count": len(cases), "fixture_summaries": summaries,
        "receipt_canonical_sha256": _sha(canonical_json(receipt)),
        "scope": "record consistency only; provenance/resident identity requires external verified activation"}
    return {**normalized, "normalized_sha256": _sha(canonical_json(normalized))}
