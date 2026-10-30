"""Preparation helper failure/success checks with fake runtime; no native calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_bounded_v1_activation as a

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bounded_activation_preparation_test",
    ROOT / "scripts/prepare_framed_interval_bounded_v1_activation.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return a.sha_file(path)


@pytest.fixture
def prepared_inputs(tmp_path):
    mapping = {}
    for name in a.REQUIRED_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        mapping[name] = a.sha_file(target)
    proof = {"schema": a.VERSION + "-source-readback-v1", "commit": "a" * 40,
             "all_files_exact": True, "files_sha256": mapping}
    sha = put(tmp_path / a.READBACK_PATH, proof)
    return tmp_path, proof, sha


def test_helper_two_role_validation_and_exact_intent(prepared_inputs, monkeypatch):
    root, _, sha = prepared_inputs
    scopes = []
    def fake_load(root, path, activation_sha, intent, *, receiver_scope):
        assert a.sha_file(root / path) == activation_sha
        assert intent == a.INTENT
        scopes.append(receiver_scope)
        return SimpleNamespace(resource_report={"verified_files": 41},
            preflight={"prompt_ids": {"station": [1] * 282, "library": [2] * 257}})
    monkeypatch.setattr(helper.act, "load_activation", fake_load)
    result = helper.prepare(root, sha)
    assert scopes == [True, False]
    assert result["status"] == "prepared_and_validated_no_model_calls"
    intent = json.loads((root / helper.PREPARATION / "execution-intent.json").read_bytes())
    assert intent["planned_generations"] == 16
    assert intent["planned_fresh_stego_receivers"] == 8
    assert intent["state"] == "prepared_not_started"
    assert intent["retry_permitted"] is False
    assert len(result["must_save_and_readback_before_launch"]) == 5
    with pytest.raises(FileExistsError):
        helper.prepare(root, sha)


@pytest.mark.parametrize("fault", ["proof_false", "source_change", "existing_output"])
def test_helper_prewrite_refusal(prepared_inputs, monkeypatch, fault):
    root, proof, sha = prepared_inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime validator must not be called")
    monkeypatch.setattr(helper.act, "load_activation", forbidden)
    if fault == "proof_false":
        proof["all_files_exact"] = False
        sha = put(root / a.READBACK_PATH, proof)
    elif fault == "source_change":
        (root / a.NEW_SOURCES[2]).write_bytes(b"changed")
    else:
        (root / helper.OUTPUT).mkdir(parents=True)
    with pytest.raises((ValueError, FileExistsError)):
        helper.prepare(root, sha)
    assert not (root / a.FREEZE_PATH).exists()
    assert not (root / a.ACTIVATION_PATH).exists()
    assert not (root / helper.PREPARATION / "execution-intent.json").exists()


def test_helper_runtime_failure_preserves_partial_evidence_without_intent(prepared_inputs, monkeypatch):
    root, _, sha = prepared_inputs
    def failure(*args, **kwargs):
        raise ValueError("changed runtime")
    monkeypatch.setattr(helper.act, "load_activation", failure)
    with pytest.raises(ValueError, match="changed runtime"):
        helper.prepare(root, sha)
    assert (root / a.FREEZE_PATH).exists()
    assert (root / a.ACTIVATION_PATH).exists()
    assert not (root / helper.PREPARATION / "activation-validation.json").exists()
    assert not (root / helper.PREPARATION / "execution-intent.json").exists()
    with pytest.raises(FileExistsError):
        helper.prepare(root, sha)
