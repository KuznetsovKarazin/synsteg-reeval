"""Pure activation/scope tests; fake runtime evidence, zero native/model calls."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_bounded_v1_activation as a

ROOT = Path(__file__).resolve().parents[1]


def put(root, name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": name, "sha256": a.digest(raw)}


@pytest.fixture
def environment(tmp_path):
    mapping = {}
    for name in a.REQUIRED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (ROOT / name).read_bytes() if (ROOT / name).is_file() else b"synthetic executable fixture\n"
        path.write_bytes(raw)
        mapping[name] = a.digest(raw)
    commit = "a" * 40
    freeze = {"schema": a.VERSION + "-source-freeze-v1", "commit": commit, "files_sha256": mapping}
    proof = {"schema": a.VERSION + "-source-readback-v1", "commit": commit,
             "all_files_exact": True, "files_sha256": mapping}
    protocol = json.loads((tmp_path / a.PUBLIC_PATH).read_bytes())
    obj = {"schema": a.VERSION + "-activation-v1", "experiment_id": a.VERSION, "enabled": True,
        "final_freeze_commit": commit,
        "freeze_manifest": put(tmp_path, a.FREEZE_PATH, freeze),
        "source_readback": put(tmp_path, a.READBACK_PATH, proof),
        "public_protocol": {"path": a.PUBLIC_PATH, "sha256": a.PUBLIC_SHA},
        "sender_fixtures": {"path": a.FIXTURES_PATH, "sha256": a.FIXTURES_SHA},
        "prior_runtime_activation": {"path": a.PRIOR_ACTIVATION_PATH, "sha256": a.PRIOR_ACTIVATION_SHA},
        "expected_public_runs_sha256": a.digest(a.canonical_json(protocol["runs"]))}
    calls = []
    def runtime(root, descriptor):
        calls.append((root, descriptor))
        return SimpleNamespace(public_protocol=protocol,
            preflight={"prompt_ids": {"station": [1] * 282, "library": [2] * 257},
                       "special_ids": [3, 4]},
            resource_report={"status": "fake_runtime_not_native"},
            data={"ready": {"context_tokens": 2048},
                  "qwen_binding_manifest": {"test": True}})
    def run(*, receiver=True):
        desc = put(tmp_path, a.ACTIVATION_PATH, obj)
        return a.load_activation(tmp_path, desc["path"], desc["sha256"], a.INTENT,
                                 receiver_scope=receiver, runtime_loader=runtime)
    return tmp_path, obj, freeze, proof, calls, run


def save_proofs(root, obj, freeze, proof):
    obj["freeze_manifest"] = put(root, a.FREEZE_PATH, freeze)
    obj["source_readback"] = put(root, a.READBACK_PATH, proof)


def test_receiver_can_load_with_fixture_removed_and_no_read(environment, monkeypatch):
    root, obj, freeze, proof, calls, run = environment
    fixture = root / a.FIXTURES_PATH
    fixture.unlink()
    original = Path.open
    def trap(path, *args, **kwargs):
        if str(path).endswith(a.FIXTURES_PATH):
            raise AssertionError("receiver opened/hashed private fixture")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", trap)
    activation = run(receiver=True)
    assert activation.receiver_scope
    assert len(calls) == 1
    assert activation.data["source_readback"] == obj["source_readback"]
    with pytest.raises(ValueError, match="receiver cannot"):
        a.check_fixtures(activation)


def test_parent_requires_private_fixture(environment):
    root, _, _, _, calls, run = environment
    (root / a.FIXTURES_PATH).unlink()
    with pytest.raises(FileNotFoundError):
        run(receiver=False)
    assert calls == []


def test_parent_checks_complete_fixture_matrix(environment):
    _, _, _, _, calls, run = environment
    activation = run(receiver=False)
    fixtures = a.check_fixtures(activation)
    assert len(fixtures["runs"]) == 16
    assert sum(r["payload_hex"] is not None for r in fixtures["runs"]) == 8
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [
    ("enabled", False), ("enabled", 1), ("experiment_id", "framed-interval-llm-v1"),
    ("final_freeze_commit", "short"), ("expected_public_runs_sha256", "0" * 64),
])
def test_activation_mutations_fail_before_runtime(environment, field, value):
    _, obj, _, _, calls, run = environment
    obj[field] = value
    with pytest.raises(ValueError):
        run()
    assert calls == []


@pytest.mark.parametrize("field", ["freeze_manifest", "source_readback", "public_protocol",
                                  "prior_runtime_activation", "sender_fixtures"])
def test_evidence_redirect_to_private_fails_without_read(environment, monkeypatch, field):
    root, obj, _, _, calls, run = environment
    obj[field] = {"path": a.FIXTURES_PATH, "sha256": a.FIXTURES_SHA}
    if field == "sender_fixtures":
        obj[field]["sha256"] = "0" * 64
    original = Path.open
    def trap(path, *args, **kwargs):
        if str(path).endswith(a.FIXTURES_PATH):
            raise AssertionError("redirect opened private fixture")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", trap)
    with pytest.raises(ValueError):
        run()
    assert calls == []


def test_activation_itself_cannot_be_redirected_to_fixture(environment):
    root, _, _, _, _, _ = environment
    with pytest.raises(ValueError, match="fixed public activation"):
        a.load_activation(root, a.FIXTURES_PATH, a.FIXTURES_SHA, a.INTENT, receiver_scope=True)


@pytest.mark.parametrize("mutation", ["unknown_private", "omit_source", "false_readback", "proof_mismatch",
                                     "bad_private_pin", "bad_public_pin"])
def test_closed_source_proof_mutations(environment, mutation):
    root, obj, freeze, proof, calls, run = environment
    # The JSON documents must become independent, as they are on disk.
    proof["files_sha256"] = dict(proof["files_sha256"])
    if mutation == "unknown_private":
        freeze["files_sha256"]["secrets/private.json"] = "0" * 64
        proof["files_sha256"]["secrets/private.json"] = "0" * 64
    elif mutation == "omit_source":
        freeze["files_sha256"].pop(a.NEW_SOURCES[2])
        proof["files_sha256"].pop(a.NEW_SOURCES[2])
    elif mutation == "false_readback":
        proof["all_files_exact"] = False
    elif mutation == "proof_mismatch":
        proof["files_sha256"][a.NEW_SOURCES[2]] = "0" * 64
    elif mutation == "bad_private_pin":
        freeze["files_sha256"][a.FIXTURES_PATH] = "0" * 64
        proof["files_sha256"][a.FIXTURES_PATH] = "0" * 64
    else:
        freeze["files_sha256"][a.PUBLIC_PATH] = "0" * 64
        proof["files_sha256"][a.PUBLIC_PATH] = "0" * 64
    save_proofs(root, obj, freeze, proof)
    with pytest.raises(ValueError):
        run()
    assert calls == []


def test_changed_executable_fails_before_runtime(environment):
    root, _, _, _, calls, run = environment
    (root / a.NEW_SOURCES[2]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        run()
    assert calls == []


def test_runtime_failure_is_not_suppressed(environment):
    root, obj, _, _, _, _ = environment
    desc = put(root, a.ACTIVATION_PATH, obj)
    def failure(*args):
        raise RuntimeError("runtime mismatch")
    with pytest.raises(RuntimeError, match="runtime mismatch"):
        a.load_activation(root, desc["path"], desc["sha256"], a.INTENT,
                          receiver_scope=True, runtime_loader=failure)


@pytest.mark.parametrize("mutation", ["prompt", "context", "missing_prompt"])
def test_new_prompt_or_context_fails(environment, mutation):
    root, obj, _, _, _, _ = environment
    desc = put(root, a.ACTIVATION_PATH, obj)
    protocol = json.loads((root / a.PUBLIC_PATH).read_bytes())
    prompt_ids = {"station": [1] * 282, "library": [2] * 257}
    context = 2048
    if mutation == "prompt":
        protocol["system_prompt"] += " changed"
    elif mutation == "context":
        context = 1024
    else:
        prompt_ids.pop("library")
    def runtime(*args):
        return SimpleNamespace(public_protocol=protocol, preflight={"prompt_ids": prompt_ids},
            data={"ready": {"context_tokens": context}}, resource_report={})
    with pytest.raises(ValueError):
        a.load_activation(root, desc["path"], desc["sha256"], a.INTENT,
                          receiver_scope=True, runtime_loader=runtime)


def test_old_runtime_loader_always_uses_public_receiver_scope(monkeypatch, tmp_path):
    seen = {}
    def load(*args, **kwargs):
        seen.update(kwargs)
        assert args[3] == a.prior.INTENT
        return "sentinel"
    monkeypatch.setattr(a.prior, "load_activation", load)
    descriptor = {"path": a.PRIOR_ACTIVATION_PATH, "sha256": a.PRIOR_ACTIVATION_SHA}
    assert a._load_prior_runtime(tmp_path, descriptor) == "sentinel"
    assert seen["receiver_scope"] is True
    assert seen["runtime_reader"] is a.current_runtime


def test_wrong_intent_and_nonboolean_scope_rejected(environment):
    root, obj, _, _, _, _ = environment
    desc = put(root, a.ACTIVATION_PATH, obj)
    for intent, scope in [(a.prior.INTENT, True), (a.INTENT, 1)]:
        with pytest.raises(ValueError):
            a.load_activation(root, desc["path"], desc["sha256"], intent, receiver_scope=scope)
