"""Pure synthetic evidence checks. No tokenizer/native/model process is executed."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from synsteg_audit import framed_interval_llm_v1_preflight as p

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tiny_inventory():
    return [
        {"token_id": 0, "is_control": False, "is_eog": False, "piece_empty": False},
        {"token_id": 1, "is_control": True, "is_eog": False, "piece_empty": True},
        {"token_id": 2, "is_control": False, "is_eog": True, "piece_empty": False},
        {"token_id": 3, "is_control": False, "is_eog": False, "piece_empty": True},
    ]


@pytest.fixture(scope="module")
def synthetic_evidence():
    """Fabricated observations test a validator, never pretend to be Qwen output."""
    plan_bytes = (ROOT / "experiments/framed-interval-llm-v1/tokenizer-preflight-proposal.json").read_bytes()
    plan = json.loads(plan_bytes)
    protocol = json.loads((ROOT / "experiments/framed-interval-llm-v1/receiver-public-protocol.json").read_bytes())
    size = 56178  # Minimal production-shaped range includes the known merge ID.
    inventory = [{"token_id": i, "is_control": i == 0, "is_eog": i == 1, "piece_empty": i in {0, 2}}
                 for i in range(size)]
    inventory_bytes = p.canonical_json(inventory)
    special_ids = [0, 1]
    libraries = {"runtime/llama-b11791/libllama.so": "b"*64}
    ready = {"vocabulary_size": size, "masked_tokens": 3, "context_tokens": 2048}
    cases = []
    base = {"token_ids": [], "rendered_hex": "", "strict_utf8": True, "reencoded_ids": []}
    for i, spec in enumerate(plan["fixtures"]):
        row = {"id": spec["id"], "input_hex": spec["input_hex"]}
        if spec["kind"] == "invalid_utf8":
            row.update(status="invalid_utf8", tokenizer_calls=0)
        else:
            row.update(status="observed", tokenize_flags={"add_special": False, "parse_special": False})
            if spec["kind"] == "token_path":
                row.update(reference_ids=[334, 271], prefixes=[copy.deepcopy(base),
                    {"token_ids": [334], "rendered_hex": b"**".hex(), "strict_utf8": True, "reencoded_ids": [334]},
                    {"token_ids": [334, 271], "rendered_hex": b"**\n\n".hex(), "strict_utf8": True, "reencoded_ids": [56177]}],
                    gate_reasons=["accepted", "retokenization_mismatch"])
            elif spec["id"] == "empty":
                row.update(reference_ids=[], prefixes=[copy.deepcopy(base)], gate_reasons=[])
            else:
                token = 100+i
                row.update(reference_ids=[token], prefixes=[copy.deepcopy(base),
                    {"token_ids": [token], "rendered_hex": spec["input_hex"], "strict_utf8": True, "reencoded_ids": [token]}],
                    gate_reasons=["accepted"])
        cases.append(row)
    receipt = {
        "schema": p.RECEIPT_SCHEMA, "status": "passed", "plan_sha256": sha(plan_bytes),
        "resources": {"model_sha256": p.MODEL_SHA256, "runtime_commit": p.RUNTIME_COMMIT,
            "archive_sha256": p.ARCHIVE_SHA256, "cpu_backend_sha256": p.CPU_SHA256,
            "resource_identity_sha256": "a"*64, "native_libraries_sha256": libraries},
        "collector": {"source_sha256": "c"*64, "mode": "pinned-native-vocab-only", "complete_native_id_scan": True},
        "ready": ready,
        "inventory": {"path": "results/synthetic/inventory.json", "bytes": len(inventory_bytes),
            "sha256": sha(inventory_bytes), "canonical_sha256": sha(inventory_bytes), "rows": size,
            "special_ids": special_ids, "special_ids_sha256": sha(p.canonical_json(special_ids))},
        "cases": cases,
        "prompts": {name: {"text_sha256": sha(p.chatml_prompt(protocol, name).encode()),
            "tokenize_flags": {"add_special": True, "parse_special": True},
            "token_ids": [0, 101, 102, 1], "repeat_token_ids": [0, 101, 102, 1]}
            for name in ("station", "library")},
        "guards": {"forward_calls": 0, "prompt_calls": 0, "step_calls": 0, "distribution_calls": 0,
            "paid_calls": 0, "closed_test_accessed": False, "model_weights_loaded": False}}
    kwargs = {"plan_bytes": plan_bytes, "inventory_bytes": inventory_bytes,
        "expected_plan_sha256": sha(plan_bytes), "expected_resource_identity_sha256": "a"*64,
        "expected_native_libraries": libraries, "expected_collector_source_sha256": "c"*64,
        "expected_ready": ready, "prompt_protocol": protocol}
    return receipt, kwargs


def test_inventory_union_count_and_empty_non_special():
    result = p.validate_inventory(tiny_inventory(), vocabulary_size=4, masked_tokens=3)
    assert result["special_ids"] == [1, 2]
    assert result["masked_tokens"] == 3


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reorder", "extra", "bool_id", "int_flag", "wrong_count", "bool_size"])
def test_inventory_rejects_noncomplete_native_range(mutation):
    rows, size, count = tiny_inventory(), 4, 3
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[2]["token_id"] = 1
    elif mutation == "reorder": rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "extra": rows[0]["piece"] = "x"
    elif mutation == "bool_id": rows[0]["token_id"] = False
    elif mutation == "int_flag": rows[0]["is_control"] = 0
    elif mutation == "wrong_count": count = 2
    else: size = True
    with pytest.raises(p.PreflightError): p.validate_inventory(rows, vocabulary_size=size, masked_tokens=count)


def test_complete_synthetic_receipt_retains_all_fifteen(synthetic_evidence):
    receipt, kwargs = synthetic_evidence
    result = p.validate_preflight_receipt(receipt, **kwargs)
    assert result["fixture_count"] == 15
    assert len(result["fixture_summaries"]) == 15
    assert result["special_ids"] == [0, 1]
    assert result["fixture_summaries"][-1]["gate_reasons"] == ["accepted", "retokenization_mismatch"]
    assert sum(r["status"] == "expected_invalid_utf8_refusal" for r in result["fixture_summaries"]) == 3
    assert "record consistency only" in result["scope"]
    assert result == p.validate_preflight_receipt(receipt, **kwargs)


def test_proposal_is_fixed_and_explicitly_unexecuted(synthetic_evidence):
    _, kwargs = synthetic_evidence
    plan = json.loads(kwargs["plan_bytes"])
    assert plan["status"] == "proposed_not_frozen_not_executed"
    assert plan["real_tokenizer_executed"] is False
    assert plan["collector"]["implementation_status"] == "not_built_or_executed"
    assert plan["fixtures"] == p.fixed_fixtures() and len(plan["fixtures"]) == 15
    assert plan["execution_limits"]["attempts"] == 1 and plan["execution_limits"]["retries"] == 0
    for name in ("station", "library"):
        assert plan["prompt_text_sha256"][name] == sha(p.chatml_prompt(kwargs["prompt_protocol"], name).encode())


@pytest.mark.parametrize("field", ["forward_calls", "prompt_calls", "step_calls", "distribution_calls", "paid_calls", "closed_test_accessed", "model_weights_loaded"])
def test_any_forbidden_activity_or_bool_integer_confusion_refused(synthetic_evidence, field):
    original, kwargs = synthetic_evidence
    for replacement in (1, True) if field.endswith("calls") else (1, 0):
        receipt = copy.deepcopy(original)
        receipt["guards"][field] = replacement
        with pytest.raises(p.PreflightError): p.validate_preflight_receipt(receipt, **kwargs)


@pytest.mark.parametrize("path", ["/tmp/inventory.json", "../inventory.json", "results/../inventory.json", "x\\inventory.json", "", "results//inventory.json", "."])
def test_unsafe_inventory_paths_refused(synthetic_evidence, path):
    receipt, kwargs = synthetic_evidence
    receipt = copy.deepcopy(receipt); receipt["inventory"]["path"] = path
    with pytest.raises(p.PreflightError): p.validate_preflight_receipt(receipt, **kwargs)


@pytest.mark.parametrize("mutation", ["output_add", "output_parse", "prompt_add", "prompt_parse", "fixture_true", "fixture_false"])
def test_plan_bool_integer_confusion_refused_even_with_updated_trusted_hash(synthetic_evidence, mutation):
    original, kwargs = synthetic_evidence
    plan = json.loads(kwargs["plan_bytes"])
    if mutation == "output_add": plan["output_tokenize_flags"]["add_special"] = 0
    elif mutation == "output_parse": plan["output_tokenize_flags"]["parse_special"] = 0
    elif mutation == "prompt_add": plan["prompt_tokenize_flags"]["add_special"] = 1
    elif mutation == "prompt_parse": plan["prompt_tokenize_flags"]["parse_special"] = 1
    elif mutation == "fixture_true": plan["fixtures"][0]["require_exact_full_roundtrip"] = 1
    else: plan["fixtures"][2]["require_exact_full_roundtrip"] = 0
    raw = p.canonical_json(plan)
    receipt = copy.deepcopy(original); receipt["plan_sha256"] = sha(raw)
    with pytest.raises(p.PreflightError):
        p.validate_preflight_receipt(receipt, **{**kwargs, "plan_bytes": raw, "expected_plan_sha256": sha(raw)})


@pytest.mark.parametrize("mutation", ["missing_case", "reordered", "known_merge_pass", "bad_merge_id", "invalid_was_called", "invalid_bool_calls", "incomplete_prefix", "wrong_reason", "nonboolean_utf8", "special_chatml", "bad_ascii", "wrong_output_flags", "bad_prompt_repeat", "empty_prompt", "bad_prompt_flags", "bool_prompt_flags", "bad_prompt_hash", "extra_prompt"])
def test_incomplete_or_inconsistent_recorded_cases_refused(synthetic_evidence, mutation):
    original, kwargs = synthetic_evidence
    receipt = copy.deepcopy(original)
    cases = receipt["cases"]
    by_id = {r["id"]: r for r in cases}
    if mutation == "missing_case": cases.pop()
    elif mutation == "reordered": cases[1], cases[2] = cases[2], cases[1]
    elif mutation == "known_merge_pass": cases[-1]["gate_reasons"][-1] = "accepted"
    elif mutation == "bad_merge_id": cases[-1]["prefixes"][-1]["reencoded_ids"] = [56176]
    elif mutation == "invalid_was_called": by_id["invalid_utf8_byte"]["tokenizer_calls"] = 1
    elif mutation == "invalid_bool_calls": by_id["invalid_utf8_byte"]["tokenizer_calls"] = False
    elif mutation == "incomplete_prefix": by_id["ascii"]["prefixes"].pop(0)
    elif mutation == "wrong_reason": by_id["ascii"]["gate_reasons"] = ["empty_extension"]
    elif mutation == "nonboolean_utf8": by_id["ascii"]["prefixes"][0]["strict_utf8"] = 1
    elif mutation == "special_chatml":
        row = by_id["chatml_literal"]; row["reference_ids"] = [0]
        row["prefixes"][-1]["token_ids"] = [0]; row["prefixes"][-1]["reencoded_ids"] = [0]
        row["gate_reasons"] = ["special_token"]
    elif mutation == "bad_ascii": by_id["ascii"]["prefixes"][-1]["rendered_hex"] = b"wrong".hex()
    elif mutation == "wrong_output_flags": by_id["ascii"]["tokenize_flags"]["parse_special"] = True
    elif mutation == "bad_prompt_repeat": receipt["prompts"]["station"]["repeat_token_ids"] = [0, 102, 101, 1]
    elif mutation == "empty_prompt":
        receipt["prompts"]["station"]["token_ids"] = []
        receipt["prompts"]["station"]["repeat_token_ids"] = []
    elif mutation == "bad_prompt_flags": receipt["prompts"]["station"]["tokenize_flags"]["parse_special"] = False
    elif mutation == "bool_prompt_flags": receipt["prompts"]["station"]["tokenize_flags"]["parse_special"] = 1
    elif mutation == "bad_prompt_hash": receipt["prompts"]["station"]["text_sha256"] = "0"*64
    else: receipt["prompts"]["extra"] = copy.deepcopy(receipt["prompts"]["station"])
    with pytest.raises(p.PreflightError): p.validate_preflight_receipt(receipt, **kwargs)


@pytest.mark.parametrize("mutation", ["foreign_model", "foreign_library", "foreign_collector", "partial_scan", "bool_complete", "wrong_ready", "bool_ready", "fake_size", "wrong_inventory_sha", "wrong_inventory_size", "partial_specials", "extra_receipt", "failed_receipt", "wrong_plan_sha"])
def test_wrong_resource_and_evidence_identity_refused(synthetic_evidence, mutation):
    original, kwargs = synthetic_evidence
    receipt = copy.deepcopy(original)
    if mutation == "foreign_model": receipt["resources"]["model_sha256"] = "d"*64
    elif mutation == "foreign_library": receipt["resources"]["native_libraries_sha256"] = {"tiny": "e"*64}
    elif mutation == "foreign_collector": receipt["collector"]["source_sha256"] = "e"*64
    elif mutation == "partial_scan": receipt["collector"]["complete_native_id_scan"] = False
    elif mutation == "bool_complete": receipt["collector"]["complete_native_id_scan"] = 1
    elif mutation == "wrong_ready": receipt["ready"]["context_tokens"] = 4096
    elif mutation == "bool_ready": receipt["ready"]["masked_tokens"] = True
    elif mutation == "fake_size": receipt["inventory"]["rows"] = 32000
    elif mutation == "wrong_inventory_sha": receipt["inventory"]["sha256"] = "f"*64
    elif mutation == "wrong_inventory_size": receipt["inventory"]["bytes"] -= 1
    elif mutation == "partial_specials": receipt["inventory"]["special_ids"] = [0]
    elif mutation == "extra_receipt": receipt["success_rate"] = 1
    elif mutation == "failed_receipt": receipt["status"] = "failed"
    else: receipt["plan_sha256"] = "0"*64
    with pytest.raises(p.PreflightError): p.validate_preflight_receipt(receipt, **kwargs)


@pytest.mark.parametrize("kind", ["invalid_utf8", "empty_extension", "rewrites_prior_bytes", "retokenization_mismatch", "invalid_prefix"])
def test_permitted_prefix_refusals_are_retained_not_selected_away(synthetic_evidence, kind):
    original, kwargs = synthetic_evidence
    receipt = copy.deepcopy(original)
    case = next(row for row in receipt["cases"] if row["id"] == "emoji")
    base = case["prefixes"][0]
    if kind == "invalid_utf8":
        case["prefixes"][-1].update(rendered_hex="f0", strict_utf8=False, reencoded_ids=None)
    elif kind == "empty_extension":
        case["prefixes"][-1].update(rendered_hex="", reencoded_ids=[])
    elif kind == "retokenization_mismatch":
        case["prefixes"][-1]["reencoded_ids"] = [500]
    else:
        case["reference_ids"] = [301, 302]
        case["prefixes"] = [base,
            {"token_ids": [301], "rendered_hex": "41", "strict_utf8": True, "reencoded_ids": [301] if kind == "rewrites_prior_bytes" else [500]},
            {"token_ids": [301, 302], "rendered_hex": "42" if kind == "rewrites_prior_bytes" else "4142", "strict_utf8": True, "reencoded_ids": [301, 302]}]
    case["gate_reasons"] = (["accepted", kind] if kind == "rewrites_prior_bytes" else
        ["retokenization_mismatch", kind] if kind == "invalid_prefix" else [kind])
    result = p.validate_preflight_receipt(receipt, **kwargs)
    summary = next(row for row in result["fixture_summaries"] if row["id"] == "emoji")
    assert not summary["all_prefix_steps_accepted"] and summary["gate_reasons"][-1] == kind


def test_duplicate_inventory_json_key_and_nonfinite_plan_refused(synthetic_evidence):
    receipt, kwargs = synthetic_evidence
    bad = b'[{"token_id":0,"token_id":0}]'
    changed = copy.deepcopy(receipt)
    changed["inventory"].update(bytes=len(bad), sha256=sha(bad))
    with pytest.raises(p.PreflightError, match="duplicate"):
        p.validate_preflight_receipt(changed, **{**kwargs, "inventory_bytes": bad})
    bad_plan = b'{"x":NaN}'
    with pytest.raises(p.PreflightError, match="nonfinite"):
        p.validate_preflight_receipt(receipt, **{**kwargs, "plan_bytes": bad_plan, "expected_plan_sha256": sha(bad_plan)})
