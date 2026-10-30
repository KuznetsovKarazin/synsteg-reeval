"""Synthetic evidence/publication tests; never compile, link, or run a binary.

The tiny executable fixture is only an ELF-magic marker, not native code.
The reviewed-plan loader and activation validator are injected so these cases
test evidence binding and publication independently of real resources/runtime.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_framed_interval_llm_vocab_activation.py"
spec = importlib.util.spec_from_file_location("synsteg_test_prepare_vocab_activation", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class EvidenceTree:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.calls = []
        self.plan_name = "plan.json"
        self.build_name = "build.json"
        self.execution_name = "execution.json"
        self.proof_name = "readback.json"
        self.systems_name = "systems.json"
        self.output_name = "activation.json"
        self.binary_name = "fake/native-marker"
        self.binary = self.raw(self.binary_name, b"\x7fELFnot-a-real-executable\n")
        (root / self.binary_name).chmod(0o755)
        self.compiler = self.raw("fake/compiler-marker", b"never execute this fake compiler\n")
        self.compiler = {"path": str(root / self.compiler["path"]), "sha256": self.compiler["sha256"]}
        fake_source = self.raw("fake/source.py", b"# synthetic frozen source\n")
        fake_header = self.raw("fake/header.h", b"// synthetic header\n")
        helper = self.raw(prepare.SCRIPT_PATH, b"# synthetic helper identity\n")
        public_raw = (ROOT / prepare.PUBLIC_PATH).read_bytes()
        self.raw(prepare.PUBLIC_PATH, public_raw)
        self.plan = {
            "sources_sha256": {fake_source["path"]: fake_source["sha256"]},
            "headers": {fake_header["path"]: fake_header["sha256"]},
            "source_bundle_sha256": sha(prepare.canonical_json({fake_source["path"]: fake_source["sha256"]})),
            "binary_path": self.binary_name,
            "compiler": self.compiler, "compiler_subtools": {},
            "build_command": [self.compiler["path"], "fake/source.py", "-o", self.binary_name],
            "build_environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            "compile_timeout_seconds": 120,
            "header_provenance": self.doc("header-provenance.json", {"fake": True}),
            "upstream_license": self.raw("LICENSE.fake", b"synthetic fixture only\n"),
            "preparation_script": helper,
            "tokenizer_plan": self.doc("tokenizer-plan.json", {"fake": "plan"}),
            "resource_identity_sha256": "a" * 64,
        }
        system_file = self.raw("fake/libexample.so", b"fake current system library\n")
        self.systems = {"schema": "framed-interval-llm-v1-vocab-system-libraries-v1",
            "files_sha256": {str(root / system_file["path"]): system_file["sha256"]}}
        self.plan["system_library_manifest"] = self.doc(self.systems_name, self.systems)
        self.plan_entry = self.doc(self.plan_name, self.plan)
        self.stdout = self.raw("compiler.stdout", b"synthetic build log\n")
        self.stderr = self.raw("compiler.stderr", b"")
        self.execution = {"schema": "framed-interval-llm-v1-vocab-build-execution-v1",
            "plan_sha256": self.plan_entry["sha256"], "build_command": self.plan["build_command"],
            "build_environment": self.plan["build_environment"], "working_directory": "repository_root",
            "exit_code": 0, "compile_attempts": 1, "native_model_executions": 0,
            "wall_seconds": 0.25, "source_freeze_commit": "b" * 40,
            "stdout": self.stdout, "stderr": self.stderr, "binary": self.binary}
        self.execution_entry = self.doc(self.execution_name, self.execution)
        self.build = {"schema": "framed-interval-llm-v1-vocab-build-v1",
            "source_bundle_sha256": self.plan["source_bundle_sha256"], "runtime_commit": prepare.RUNTIME_COMMIT,
            "binary_sha256": self.binary["sha256"], "native_model_executions": 0,
            "compiler": self.compiler, "headers": self.plan["headers"], "build_command": self.plan["build_command"],
            "build_plan": self.plan_entry, "execution_evidence": self.execution_entry,
            "compiler_subtools": {}, "header_provenance": self.plan["header_provenance"]}
        self.build_entry = self.doc(self.build_name, self.build)
        self.seal_proof()
        monkeypatch.setattr(prepare, "load_plan", self.fake_load_plan)
        monkeypatch.setattr(prepare, "current_runtime", lambda: {"fake_runtime": True})

    def raw(self, name, raw):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {"path": name, "sha256": sha(raw)}

    def doc(self, name, obj):
        return self.raw(name, prepare.canonical_json(obj))

    def fake_load_plan(self, root, path, expected):
        assert root == self.root and path == self.plan_name and expected == self.plan_entry["sha256"]
        self.calls.append("reviewed-plan")
        return copy.deepcopy(self.plan)

    def seal_proof(self):
        required = {**self.plan["sources_sha256"], **self.plan["headers"],
            self.plan_name: self.plan_entry["sha256"], self.build_name: self.build_entry["sha256"],
            self.binary_name: self.binary["sha256"],
            self.plan["header_provenance"]["path"]: self.plan["header_provenance"]["sha256"],
            self.plan["upstream_license"]["path"]: self.plan["upstream_license"]["sha256"],
            prepare.SCRIPT_PATH: self.plan["preparation_script"]["sha256"],
            self.plan["tokenizer_plan"]["path"]: self.plan["tokenizer_plan"]["sha256"],
            self.systems_name: self.plan["system_library_manifest"]["sha256"],
            self.execution_name: self.execution_entry["sha256"],
            self.stdout["path"]: self.stdout["sha256"], self.stderr["path"]: self.stderr["sha256"],
            prepare.PUBLIC_PATH: prepare.PUBLIC_SHA}
        self.proof = {"schema": "framed-interval-llm-v1-vocab-build-git-readback-v1",
            "commit": "c" * 40, "all_files_exact": True, "files_sha256": required}
        self.proof_entry = self.doc(self.proof_name, self.proof)

    def make_receipt(self):
        self.execution_entry = self.doc(self.execution_name, self.execution)
        return prepare.make_build_receipt(self.root, self.plan_name, self.plan_entry["sha256"],
            self.execution_name, self.execution_entry["sha256"], "new-build-receipt.json")

    def activate(self, *, enable=False, validator=None):
        self.build_entry = self.doc(self.build_name, self.build)
        self.proof_entry = self.doc(self.proof_name, self.proof)
        return prepare.make_activation(self.root, self.plan_name, self.plan_entry["sha256"],
            self.build_name, self.build_entry["sha256"], self.proof_name, self.proof_entry["sha256"],
            self.systems_name, self.plan["system_library_manifest"]["sha256"], self.output_name,
            enable=enable, validator=validator or self.validator)

    def validator(self, root, path, expected, intent):
        self.calls.append("validator")
        assert not (root / self.output_name).exists()
        candidate = (root / path).read_bytes()
        assert sha(candidate) == expected and json.loads(candidate)["enabled"] is True
        assert intent == "collect-frozen-qwen-vocabulary-only-preflight"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    return EvidenceTree(tmp_path, monkeypatch)


def test_synthetic_successful_build_evidence_creates_receipt_without_execution(tree):
    result = tree.make_receipt()
    receipt = json.loads((tree.root / result["path"]).read_bytes())
    assert receipt["binary_sha256"] == tree.binary["sha256"]
    assert receipt["build_command"] == tree.plan["build_command"]
    assert receipt["build_plan"] == tree.plan_entry
    assert receipt["native_model_executions"] == 0
    assert tree.calls == ["reviewed-plan"]


@pytest.mark.parametrize("change", [
    {"exit_code": 1}, {"exit_code": False}, {"compile_attempts": 2}, {"compile_attempts": True},
    {"native_model_executions": 1}, {"native_model_executions": False},
    {"wall_seconds": -1}, {"wall_seconds": 121}, {"wall_seconds": True},
    {"plan_sha256": "0" * 64}, {"source_freeze_commit": "bad"},
    {"build_command": ["unexpected-compiler"]}, {"build_environment": {"LD_PRELOAD": "bad"}},
])
def test_failed_or_foreign_build_evidence_never_publishes_receipt(tree, change):
    tree.execution.update(change)
    with pytest.raises(ValueError):
        tree.make_receipt()
    assert not (tree.root / "new-build-receipt.json").exists()


@pytest.mark.parametrize("change", ["magic", "mode", "bytes"])
def test_binary_marker_must_match_recorded_bytes_and_executable_elf_guard(tree, change):
    binary = tree.root / tree.binary_name
    if change == "mode": binary.chmod(0o644)
    elif change == "magic": binary.write_bytes(b"not an ELF marker")
    else: binary.write_bytes(binary.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        tree.make_receipt()
    assert not (tree.root / "new-build-receipt.json").exists()


def test_default_activation_remains_disabled_and_never_calls_enabled_validator(tree):
    result = tree.activate()
    saved = json.loads((tree.root / result["path"]).read_bytes())
    assert saved["enabled"] is False
    assert saved["native_system_libraries_sha256"] == tree.systems["files_sha256"]
    assert tree.calls == ["reviewed-plan"]


def test_enabled_activation_is_published_only_after_successful_readonly_validation(tree):
    result = tree.activate(enable=True)
    assert tree.calls == ["reviewed-plan", "validator"]
    assert json.loads((tree.root / result["path"]).read_bytes())["enabled"] is True
    assert not (tree.root / (tree.output_name + ".validation-candidate.json")).exists()


def test_failed_validation_leaves_final_activation_absent(tree):
    def fail(*args):
        raise ValueError("synthetic validator refusal")
    with pytest.raises(ValueError, match="synthetic validator refusal"):
        tree.activate(enable=True, validator=fail)
    assert not (tree.root / tree.output_name).exists()
    assert (tree.root / (tree.output_name + ".validation-candidate.json")).exists()


@pytest.mark.parametrize("name", ["fake/source.py", "fake/header.h", "plan.json", "build.json",
    "fake/native-marker", "systems.json", "execution.json", "compiler.stdout", "compiler.stderr",
    prepare.SCRIPT_PATH, prepare.PUBLIC_PATH])
def test_exact_readback_must_cover_every_required_publication_artifact(tree, name):
    del tree.proof["files_sha256"][name]
    with pytest.raises(ValueError, match="readback omits"):
        tree.activate(enable=True)
    assert not (tree.root / tree.output_name).exists()
    assert "validator" not in tree.calls


def test_failed_build_command_binding_is_rejected_before_enabled_validation(tree):
    tree.build["build_command"] = ["different"]
    with pytest.raises(ValueError, match="build receipt differs"):
        tree.activate(enable=True)
    assert "validator" not in tree.calls


@pytest.mark.parametrize("field, value", [
    ("build_plan", {"path": "different-plan.json", "sha256": "0" * 64}),
    ("compiler_subtools", {"different": {"path": "/fake", "sha256": "0" * 64}}),
    ("header_provenance", {"path": "different-headers.json", "sha256": "0" * 64}),
])
def test_build_provenance_is_rebound_to_reviewed_plan_before_activation(tree, field, value):
    tree.build[field] = value
    with pytest.raises(ValueError, match="build receipt differs"):
        tree.activate(enable=True)
    assert "validator" not in tree.calls


def test_activation_revalidates_failed_execution_even_with_consistent_build_reference(tree):
    tree.execution["exit_code"] = 1
    tree.execution_entry = tree.doc(tree.execution_name, tree.execution)
    tree.build["execution_evidence"] = tree.execution_entry
    with pytest.raises(ValueError, match="one-attempt build evidence"):
        tree.activate(enable=True)
    assert "validator" not in tree.calls


def test_modified_compiler_logs_rejected_before_enabled_activation(tree):
    (tree.root / tree.stdout["path"]).write_bytes(b"changed compiler log")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        tree.activate(enable=True)
    assert "validator" not in tree.calls


@pytest.mark.parametrize("enable", [1, "false", None])
def test_enable_requires_explicit_boolean_not_truthiness(tree, enable):
    with pytest.raises(ValueError, match="explicit boolean"):
        tree.activate(enable=enable)
    assert tree.calls == []


def test_existing_final_activation_is_never_overwritten(tree):
    target = tree.root / tree.output_name
    target.write_bytes(b"existing activation evidence")
    with pytest.raises(FileExistsError):
        tree.activate()
    assert target.read_bytes() == b"existing activation evidence"
