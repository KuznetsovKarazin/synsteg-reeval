"""Independent evaluator contract tests; no natural-corpus encoding is run."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("universal_audit_independent", ROOT / "scripts/audit_universal_context_development_v1.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def copied_root(tmp_path):
    for name in (*audit.SOURCES, *audit.CODE):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.fixture
def resources():
    dictionary = base.PublicDictionary.from_mapping({"shade": ["amber", "ochre"]}, "independent-audit-fixture")
    dictionaries = {"v1": dictionary, "v2": dictionary}
    policies = {v: universal.UniversalPolicy.from_dictionaries(d, dictionary) for v, d in dictionaries.items()}
    return dictionaries, policies


def row(text="amber", state="available"):
    return {"audit_record_id": "synthetic:one", "cohort": "grounded_covers", "source_group_id": "synthetic",
            "text": text, "text_sha256": audit.sha(text.encode()) if isinstance(text, str) else None,
            "availability": state}


def update_long(root, rows):
    path = root / audit.LONG_INPUTS
    path.write_bytes(b"".join(audit.canonical(r) + b"\n" for r in rows))
    execution = audit.read(root / audit.LONG_EXECUTION)
    execution["output_sha256"]["records.jsonl"] = audit.sha(path.read_bytes())
    (root / audit.LONG_EXECUTION).write_bytes(audit.formatted(execution))


def all_arms(result):
    return [a for m in result["methods"].values() for a in m["arms"].values()]


def test_pinned_collection_keeps_all144_records_and_five_source_gates():
    rows, inventory = audit.collect_inputs(ROOT)
    assert len(rows) == 144
    assert Counter(r["cohort"] for r in rows) == Counter(audit.COUNTS)
    assert sum(r["availability"] == "source_gate_abstention" for r in rows) == 5
    assert len({r["audit_record_id"] for r in rows}) == 144
    assert set(inventory) == set(audit.SOURCES)


def test_historical_input_hash_mismatch_fails(copied_root):
    path = copied_root / audit.OLD_INPUTS
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Historical frozen input hash"):
        audit.collect_inputs(copied_root)


def test_closed_split_rejected_after_valid_parent_content_hash(copied_root):
    rows = audit.jsonl(copied_root / audit.LONG_INPUTS)
    rows[0]["split"] = "test"
    update_long(copied_root, rows)
    with pytest.raises(ValueError, match="outside development"):
        audit.collect_inputs(copied_root)


def test_source_gate_cannot_become_cover(copied_root):
    rows = audit.jsonl(copied_root / audit.LONG_INPUTS)
    item = next(r for r in rows if r["planned_action"] == "abstain_source_gate")
    item.update(cover_text="Invented cover", cover_sha256=audit.sha(b"Invented cover"), availability="available")
    update_long(copied_root, rows)
    with pytest.raises(ValueError, match="unrated abstention"):
        audit.collect_inputs(copied_root)


def test_missing_development_prompt_rejected(copied_root):
    path = copied_root / audit.PROMPTS
    path.write_bytes(b"\n".join(path.read_bytes().splitlines()[1:]) + b"\n")
    with pytest.raises(ValueError, match="all100"):
        audit.collect_inputs(copied_root)


def test_long_refusal_is_retained_in_complete_plan(copied_root):
    rows = audit.jsonl(copied_root / audit.LONG_INPUTS)
    item = next(r for r in rows if r["planned_action"] == "generate_once")
    item.update(availability="model_refusal", cover_text="I cannot answer.", cover_sha256=audit.sha(b"I cannot answer."))
    update_long(copied_root, rows)
    actual, _ = audit.collect_inputs(copied_root)
    assert len(actual) == 144
    assert sum(r["availability"] == "model_refusal" for r in actual) == 1
    assert sum(r["availability"] == "source_gate_abstention" for r in actual) == 5


@pytest.mark.parametrize("state", ["not_attempted", "model_refusal", "unknown_outcome", "source_gate_abstention", "empty_output", "failed"])
def test_unavailable_states_keep_all_six_arms_without_codec_calls(monkeypatch, resources, state):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable rows must not be analyzed or encoded")
    monkeypatch.setattr(base, "analyze_text", forbidden)
    dictionaries, policies = resources
    result = audit.evaluate(ROOT, row(None, state), dictionaries, policies)
    arms = all_arms(result)
    assert len(arms) == 6
    assert all(a["status"] == state and not a["encoding_called"] and not a["mechanical_success"] for a in arms)
    assert all(m["capacity_bits"] is None for m in result["methods"].values())


@pytest.mark.parametrize("text,capacity", [("amber", 1), ("ordinary text", 0)])
def test_small_and_zero_capacity_retained(resources, text, capacity):
    dictionaries, policies = resources
    result = audit.evaluate(ROOT, row(text), dictionaries, policies)
    assert len(all_arms(result)) == 6
    assert all(a["status"] == "insufficient_capacity" and not a["encoding_called"] for a in all_arms(result))
    assert all(m["capacity_bits"] == capacity for m in result["methods"].values())


def test_wrong_decoded_payload_never_counts_success(monkeypatch, resources):
    dictionaries, policies = resources
    monkeypatch.setattr(audit, "isolated_receive", lambda *args: (b"wrong payload", {"synthetic": True}))
    result = audit.evaluate(ROOT, row("amber " * 40), dictionaries, policies)
    arms = all_arms(result)
    assert Counter(a["status"] for a in arms) == Counter({"payload_or_protection_mismatch": 4, "insufficient_capacity": 2})
    assert not any(a["mechanical_success"] for a in arms)


def test_actual_fresh_receiver_matches_evaluator_contract(resources):
    dictionaries, policies = resources
    dictionary, policy = dictionaries["v1"], policies["v1"]
    key = bytes(range(32))
    encoded = universal.encode("amber " * 160, b"x", key, dictionary, policy)
    payload, evidence = audit.isolated_receive(ROOT, encoded.stego_text, key, dictionary, policy, "aes_siv")
    assert payload == b"x"
    assert evidence["returncode"] == 0
    assert evidence["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True}


def test_receiver_resource_mismatch_rejected(monkeypatch, resources):
    dictionaries, policies = resources
    dictionary, policy = dictionaries["v1"], policies["v1"]
    response = {"status": "decoded", "payload_hex": "78", "dictionary_sha256": dictionary.sha256,
                "policy_sha256": "0" * 64, "protocol_sha256": universal.UniversalProtocol().sha256,
                "input_keys": sorted(["key_hex", "stego_text", "public_dictionary", "public_policy", "public_protocol"]),
                "io_guard_self_checks": {"file": True, "socket": True, "subprocess": True}}
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=audit.canonical(response), stderr=b""))
    with pytest.raises(ValueError, match="resource/isolation evidence"):
        audit.isolated_receive(ROOT, "unused", bytes(32), dictionary, policy, "aes_siv")


@pytest.mark.parametrize("manifest", ["source_files", "code_files", "frozen_files"])
def test_missing_freeze_manifest_entries_rejected(copied_root, manifest):
    prepared = copied_root / "prepared"
    audit.prepare(copied_root, prepared)
    protocol = audit.read(prepared / "protocol.json")
    protocol[manifest] = {}
    (prepared / "protocol.json").write_bytes(audit.formatted(protocol))
    with pytest.raises(ValueError):
        audit.verify_freeze(copied_root, prepared)


def test_frozen_input_change_and_namespace_overwrite_rejected(copied_root):
    prepared = copied_root / "prepared"
    audit.prepare(copied_root, prepared)
    with pytest.raises(FileExistsError):
        audit.prepare(copied_root, prepared)
    path = prepared / "inputs.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Hash-bound"):
        audit.verify_freeze(copied_root, prepared)


def test_duplicate_record_cannot_replace_another():
    rows, _ = audit.collect_inputs(ROOT)
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="unique planned denominator"):
        audit.validate_rows(rows)
