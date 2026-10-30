"""Activation and resident-map contracts using tiny, explicitly fake resources.

No model/tokenizer/native bridge, process, network request, or resource restorer
is started. The public checkpoint files and Python sources are read as bytes;
resource/runtime/preflight providers are injected into the pure activation gate.
These tests attest validation behavior, never a real native execution.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_llm_v1_activation as activation
from synsteg_audit.framed_interval_llm_v1 import canonical_json
from synsteg_audit.framed_interval_llm_v1_qwen import (
    CPU_SHA256, MODEL_SHA256, PARAMETERS, RUNTIME_COMMIT,
)


REPO = Path(__file__).resolve().parents[1]
BRIDGE = "seeded_pair_llama_bridge"
HASWELL = "runtime/llama-b11791/libggml-cpu-haswell.so"
LLAMA = "runtime/llama-b11791/libllama.so.0.0.11791"


class FakeActivationTree:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.preflight_calls = []
        sources = set(activation.REQUIRED_SOURCES) | {activation.EXECUTION_PATH}
        for name in sources:
            self.write(name, (REPO / name).read_bytes())
        self.protocol = self.read_json(activation.PUBLIC_PATH)
        old = self.read_json(activation.EXECUTION_PATH)["channel_resources"]
        self.runtime = {name: copy.deepcopy(old[name])
                        for name in ("python", "platform", "machine", "libc")}
        self.runtime.update(cryptography="fake-test-version",
            file_sha256={"python": {"path": "/synthetic/python", "sha256": "a" * 64}},
            historical_system_library_byte_hashes_available=False)
        identities = {}
        names = [BRIDGE, HASWELL, LLAMA] + [f"fake-resource-{i:02}" for i in range(38)]
        for index, name in enumerate(names):
            raw = f"synthetic resource {index}\n".encode()
            self.write(f"{activation.RESOURCE_ROOT}/{name}", raw)
            identities[name] = {"bytes": len(raw), "sha256": activation.digest(raw)}
        self.report = {"status": "verified", "expected_files": 41, "verified_files": 41,
            "verified_file_identities": identities,
            "frozen_resource_identity_sha256": activation.digest(canonical_json(identities)),
            "bridge_executable_verified": True, "manifest_sha256": activation.EXECUTION_SHA,
            "downloads": 0, "model_calls": 0, "compile_calls": 0}
        self.system_path = self.write("fake-system/libc.so.6", b"synthetic system library\n")
        libraries = {name: row["sha256"] for name, row in identities.items()
                     if name.startswith("runtime/")}
        self.preflight = {"special_ids": [999],
            "special_ids_sha256": activation.digest(canonical_json([999])),
            "prompt_ids": {"station": [7, 8], "library": [9, 10]}}
        qwen = {"model_sha256": MODEL_SHA256, "runtime_commit": RUNTIME_COMMIT,
            "cpu_backend_sha256": CPU_SHA256,
            "bridge_sha256": identities[BRIDGE]["sha256"],
            "shared_libraries_sha256": libraries,
            "python_runtime_sha256": activation.digest(canonical_json(self.runtime)),
            "quantizer_sha256": activation.sha_file(self.root / "src/synsteg_audit/seeded_pair_llm.py"),
            "special_ids_sha256": self.preflight["special_ids_sha256"],
            "vocabulary_size": 1024, "masked_tokens": 3, "parameters": copy.deepcopy(PARAMETERS)}
        self.freeze = {"commit": "a" * 40,
            "files_sha256": {name: activation.sha_file(self.root / name)
                             for name in activation.REQUIRED_SOURCES}}
        self.obj = {"schema": "framed-interval-llm-v1-activation-v1",
            "experiment_id": "framed-interval-llm-v1", "enabled": True,
            "final_freeze_commit": self.freeze["commit"], "coordinator_readback": {},
            "freeze_manifest": {}, "public_protocol": {"path": activation.PUBLIC_PATH,
                "sha256": activation.PUBLIC_SHA},
            "expected_public_runs_sha256": activation.digest(canonical_json(self.protocol["runs"])),
            "resource_receipt": self.artifact("fake-evidence/resources.json", self.report),
            "runtime": copy.deepcopy(self.runtime), "qwen_binding_manifest": qwen,
            "ready": {"vocabulary_size": 1024, "masked_tokens": 3, "context_tokens": 2048},
            "preflight_plan": self.artifact("fake-evidence/plan.json", {"fake": "plan"}),
            "preflight_receipt": self.artifact("fake-evidence/preflight.json", {"fake": "receipt"}),
            "preflight_inventory": self.artifact("fake-evidence/inventory.json", [{"fake": "inventory"}]),
            "collector_source_sha256": "c" * 64,
            "native_system_libraries_sha256": {str(self.system_path): activation.sha_file(self.system_path)}}
        self.seal_freeze()

    def write(self, name, raw):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return target

    def read_json(self, name):
        import json
        return json.loads((self.root / name).read_bytes())

    def artifact(self, name, obj):
        raw = canonical_json(obj)
        self.write(name, raw)
        return {"path": name, "sha256": activation.digest(raw)}

    def seal_freeze(self):
        self.obj["freeze_manifest"] = self.artifact("fake-evidence/freeze.json", self.freeze)
        self.obj["coordinator_readback"] = {"commit": self.obj["final_freeze_commit"],
            "all_files_exact": True, "freeze_manifest_sha256": self.obj["freeze_manifest"]["sha256"]}

    def resource_verifier(self, root):
        assert root == self.root.resolve()
        self.calls.append("resources")
        return copy.deepcopy(self.report)

    def runtime_reader(self):
        self.calls.append("runtime")
        return copy.deepcopy(self.runtime)

    def preflight_validator(self, receipt, **kwargs):
        self.calls.append("preflight")
        self.preflight_calls.append((receipt, kwargs))
        return copy.deepcopy(self.preflight)

    def load(self, *, intent=activation.INTENT, expected_sha=None, receiver_scope=False):
        descriptor = self.artifact("fake-evidence/activation.json", self.obj)
        return activation.load_activation(self.root, descriptor["path"],
            expected_sha or descriptor["sha256"], intent,
            resource_verifier=self.resource_verifier, runtime_reader=self.runtime_reader,
            preflight_validator=self.preflight_validator, receiver_scope=receiver_scope)


@pytest.fixture
def fake_tree(tmp_path):
    return FakeActivationTree(tmp_path)


def test_synthetic_activation_checks_sources_resources_runtime_and_preflight(fake_tree):
    result = fake_tree.load()
    assert result.root == fake_tree.root
    assert tuple(row["run_id"] for row in result.public_protocol["runs"]) == activation.EXPECTED_RUNS
    assert fake_tree.calls == ["resources", "runtime", "preflight"]
    assert len(result.resource_report["verified_file_identities"]) == 41
    receipt, passed = fake_tree.preflight_calls[0]
    assert receipt == {"fake": "receipt"}
    for name in ("plan", "inventory"):
        assert passed[name + "_bytes"] == (fake_tree.root / fake_tree.obj["preflight_" + name]["path"]).read_bytes()
    assert passed["expected_ready"] == fake_tree.obj["ready"]
    assert passed["expected_collector_source_sha256"] == fake_tree.obj["collector_source_sha256"]
    assert passed["expected_native_libraries"] == fake_tree.obj["qwen_binding_manifest"]["shared_libraries_sha256"]
    assert passed["expected_resource_identity_sha256"] == fake_tree.report["frozen_resource_identity_sha256"]
    assert passed["prompt_protocol"] == fake_tree.protocol


@pytest.mark.parametrize("kwargs", [
    {"intent": "run"}, {"expected_sha": "0" * 64}, {"expected_sha": "A" * 64},
])
def test_intent_and_trusted_sha_fail_before_any_resource_checks(fake_tree, kwargs):
    with pytest.raises(ValueError):
        fake_tree.load(**kwargs)
    assert fake_tree.calls == []


@pytest.mark.parametrize("mutation", [
    lambda obj: obj.update(enabled=False),
    lambda obj: obj.update(enabled=1),
    lambda obj: obj.update(unexpected=True),
    lambda obj: obj.update(final_freeze_commit="a" * 39),
    lambda obj: obj["coordinator_readback"].update(all_files_exact=1),
    lambda obj: obj["coordinator_readback"].update(commit="b" * 40),
    lambda obj: obj["coordinator_readback"].update(freeze_manifest_sha256="0" * 64),
])
def test_disabled_or_incomplete_coordinator_evidence_prevents_resource_checks(fake_tree, mutation):
    mutation(fake_tree.obj)
    with pytest.raises(ValueError):
        fake_tree.load()
    assert fake_tree.calls == []


@pytest.mark.parametrize("name", activation.REQUIRED_SOURCES)
def test_every_executable_source_dependency_must_be_in_freeze(fake_tree, name):
    del fake_tree.freeze["files_sha256"][name]
    fake_tree.seal_freeze()
    with pytest.raises(ValueError, match="required executable source"):
        fake_tree.load()
    assert fake_tree.calls == []


def test_changed_source_or_changed_freeze_bytes_prevents_resources(fake_tree):
    name = "src/synsteg_audit/framed_interval_llm_v1.py"
    with (fake_tree.root / name).open("ab") as stream:
        stream.write(b"\n# synthetic mutation\n")
    with pytest.raises(ValueError, match="frozen source changed"):
        fake_tree.load()
    assert fake_tree.calls == []


@pytest.mark.parametrize("name", [activation.PROPOSAL_PATH, activation.PUBLIC_PATH,
    activation.EXECUTION_PATH, activation.RESTORER_PATH, *activation.FROZEN_COMPONENTS])
def test_resealed_freeze_cannot_redefine_frozen_checkpoint_components(fake_tree, name):
    path = fake_tree.root / name
    path.write_bytes(path.read_bytes() + b"\n")
    if name in fake_tree.freeze["files_sha256"]:
        fake_tree.freeze["files_sha256"][name] = activation.sha_file(path)
    fake_tree.seal_freeze()
    with pytest.raises(ValueError, match="fixed checkpoint resource changed"):
        fake_tree.load()
    assert fake_tree.calls == []


def test_changed_freeze_artifact_bytes_cannot_reuse_its_old_digest(fake_tree):
    path = fake_tree.root / fake_tree.obj["freeze_manifest"]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        fake_tree.load()
    assert fake_tree.calls == []


def test_freeze_manifest_commit_is_bound_to_coordinator_readback(fake_tree):
    fake_tree.freeze["commit"] = "b" * 40
    fake_tree.seal_freeze()
    with pytest.raises(ValueError, match="freeze manifest commit"):
        fake_tree.load()
    assert fake_tree.calls == []


@pytest.mark.parametrize("changes", [
    {"expected_public_runs_sha256": "0" * 64},
    {"public_protocol": {"path": "fake-evidence/other.json", "sha256": activation.PUBLIC_SHA}},
])
def test_public_four_case_protocol_and_run_hash_are_fixed(fake_tree, changes):
    fake_tree.obj.update(changes)
    with pytest.raises(ValueError):
        fake_tree.load()
    assert fake_tree.calls == []


def test_receiver_scope_never_opens_private_payload_proposal(fake_tree, monkeypatch):
    original_open = Path.open
    private_path = fake_tree.root / activation.PROPOSAL_PATH
    def guarded_open(path, *args, **kwargs):
        if path == private_path:
            raise AssertionError("receiver attempted to read private payload proposal")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = fake_tree.load(receiver_scope=True)
    assert result.public_protocol == fake_tree.protocol
    assert fake_tree.calls == ["resources", "runtime", "preflight"]


def test_receiver_can_validate_without_private_proposal_but_parent_cannot(fake_tree):
    (fake_tree.root / activation.PROPOSAL_PATH).unlink()
    assert fake_tree.load(receiver_scope=True).sha256
    with pytest.raises(FileNotFoundError):
        fake_tree.load()


def test_receiver_scope_still_requires_pinned_private_proposal_digest(fake_tree):
    fake_tree.freeze["files_sha256"][activation.PROPOSAL_PATH] = "0" * 64
    fake_tree.seal_freeze()
    with pytest.raises(ValueError, match="private proposal freeze identity"):
        fake_tree.load(receiver_scope=True)
    assert fake_tree.calls == []


@pytest.mark.parametrize("name", ["private/expected-payloads.json", "private/sender-trace.json"])
def test_receiver_skips_optional_private_freeze_artifacts_while_parent_verifies(fake_tree, monkeypatch, name):
    private = fake_tree.write(name, b'{"fake_private_evidence":true}\n')
    fake_tree.freeze["files_sha256"][name] = activation.sha_file(private)
    fake_tree.seal_freeze()
    original_open = Path.open
    forbidden = {private, fake_tree.root / activation.PROPOSAL_PATH}
    def guarded_open(path, *args, **kwargs):
        if path in forbidden:
            raise AssertionError("receiver attempted to open private parent evidence")
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        assert fake_tree.load(receiver_scope=True).sha256
    private.write_bytes(b"changed private parent evidence")
    assert fake_tree.load(receiver_scope=True).sha256
    with pytest.raises(ValueError, match="frozen source changed"):
        fake_tree.load()


@pytest.mark.parametrize("path", ["", ".", "..", "../outside", "/absolute", "a/../b", "a//b", "a/./b", "a\\b"])
def test_safe_path_rejects_noncanonical_or_escaping_names(tmp_path, path):
    with pytest.raises(ValueError):
        activation.safe_path(tmp_path, path)


def test_safe_path_and_checked_bytes_reject_symlinks_and_wrong_hash(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real/file").write_bytes(b"fake")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        activation.safe_path(tmp_path, "link/file")
    with pytest.raises(ValueError, match="bytes differ"):
        activation.checked_bytes(tmp_path, {"path": "real/file", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="descriptor"):
        activation.checked_bytes(tmp_path, {"path": "real/file", "sha256": "0" * 64, "extra": 1})


@pytest.mark.parametrize("name", ["status", "expected_files", "verified_files", "verified_file_identities",
    "frozen_resource_identity_sha256", "bridge_executable_verified", "manifest_sha256"])
def test_saved_resource_receipt_must_match_fresh_verification(fake_tree, name):
    fake_tree.report[name] = None
    with pytest.raises(ValueError, match="restoration receipt differs"):
        fake_tree.load()
    assert fake_tree.calls == ["resources"]


def test_current_runtime_change_rejected_before_preflight(fake_tree):
    fake_tree.runtime["cryptography"] = "different-current-version"
    with pytest.raises(ValueError, match="current Python/math/crypto runtime"):
        fake_tree.load()
    assert fake_tree.calls == ["resources", "runtime"]


@pytest.mark.parametrize("name", ["python", "platform", "machine", "libc"])
def test_consistently_redeclared_runtime_must_still_match_historical_metadata(fake_tree, name):
    value = ["fake", "0"] if name == "libc" else "synthetic incompatible runtime"
    fake_tree.runtime[name] = value
    fake_tree.obj["runtime"][name] = value
    with pytest.raises(ValueError, match="historical runtime metadata differs"):
        fake_tree.load()
    assert fake_tree.calls == ["resources", "runtime"]


@pytest.mark.parametrize("changes", [
    {"python_runtime_sha256": "0" * 64}, {"shared_libraries_sha256": {}},
    {"special_ids_sha256": "0" * 64}, {"vocabulary_size": 1025}, {"masked_tokens": 4},
    {"model_sha256": "0" * 64}, {"cpu_backend_sha256": "0" * 64},
    {"runtime_commit": "0" * 40}, {"parameters": {**PARAMETERS, "threads": 3}},
    {"bridge_sha256": "0" * 64}, {"quantizer_sha256": "0" * 64},
    {"vocabulary_size": 1024.0}, {"masked_tokens": 3.0},
    {"extra": "not part of the frozen Qwen binding"},
    {"parameters": {**PARAMETERS, "repetition_penalty": True}},
    {"parameters": {**PARAMETERS, "threads": 4.0}},
])
def test_invalid_qwen_declarations_reject_in_activation_before_native_launch(fake_tree, changes):
    fake_tree.obj["qwen_binding_manifest"].update(changes)
    with pytest.raises(ValueError):
        fake_tree.load()


def test_preflight_failure_propagates_before_native_launch(fake_tree):
    def reject(*args, **kwargs):
        raise ValueError("synthetic failed preflight")
    fake_tree.preflight_validator = reject
    with pytest.raises(ValueError, match="synthetic failed preflight"):
        fake_tree.load()


@pytest.mark.parametrize("changes", [{"context_tokens": 2048.0}, {"context_tokens": 1024},
    {"masked_tokens": 3.0}, {"vocabulary_size": 1024.0}, {"extra": 0}])
def test_ready_metadata_schema_is_rejected_before_preflight(fake_tree, changes):
    fake_tree.obj["ready"].update(changes)
    with pytest.raises(ValueError, match="READY"):
        fake_tree.load()
    assert "preflight" not in fake_tree.calls


@pytest.mark.parametrize("bad", [{}, {"relative/lib.so": "0" * 64}])
def test_native_system_identity_requires_nonempty_absolute_paths(fake_tree, bad):
    fake_tree.obj["native_system_libraries_sha256"] = bad
    with pytest.raises(ValueError):
        fake_tree.load()


def test_changed_native_system_file_rejected_before_launch(fake_tree):
    fake_tree.system_path.write_bytes(b"changed fake system file")
    with pytest.raises(ValueError, match="current native system-library identity"):
        fake_tree.load()


def fake_maps(tree, *, omitted=(), extra=()):
    base = tree.root / activation.RESOURCE_ROOT
    rows = [(base / name, "r-xp") for name in (BRIDGE, HASWELL, LLAMA) if name not in omitted]
    rows.append((tree.system_path, "r-xp"))
    rows.extend(extra)
    return "\n".join(f"1000-2000 {perms} 00000000 00:01 123 {path}" for path, perms in rows)


def test_resident_map_validation_hashes_tiny_files_after_fake_ready(fake_tree):
    act = fake_tree.load()
    calls = []
    def hasher(path):
        calls.append(Path(path))
        return activation.sha_file(path)
    text = fake_maps(fake_tree) + "\n2000-3000 rw-p 0 00:00 0 [heap]\n"
    result = activation.verify_resident_maps(act, 12345, text, file_hasher=hasher)
    assert result["status"] == "passed" and result["pid"] == 12345
    assert len(calls) == 4
    assert len(result["mapped_files_sha256"]) == 4
    assert "not historical" in result["scope"]


def test_resident_duplicate_segments_hash_each_current_file_only_once(fake_tree):
    calls = []
    def hasher(path):
        calls.append(Path(path))
        return activation.sha_file(path)
    text = fake_maps(fake_tree)
    text = text.replace("r-xp", "r--p") + "\n" + text + "\n" + text
    result = activation.verify_resident_maps(fake_tree.load(), 12345, text, file_hasher=hasher)
    assert result["status"] == "passed"
    assert len(calls) == len(set(calls)) == 4


@pytest.mark.parametrize("non_executable", [BRIDGE, HASWELL, LLAMA])
def test_required_resident_component_needs_executable_segment(fake_tree, non_executable):
    base = fake_tree.root / activation.RESOURCE_ROOT
    text = fake_maps(fake_tree, omitted=(non_executable,), extra=[(base / non_executable, "r--p")])
    with pytest.raises(ValueError, match="evidence incomplete"):
        activation.verify_resident_maps(fake_tree.load(), 12345, text)


@pytest.mark.parametrize("pid", [0, -1, True, "123"])
def test_resident_map_check_requires_explicit_positive_owned_pid(fake_tree, pid):
    with pytest.raises(ValueError, match="owned native PID"):
        activation.verify_resident_maps(fake_tree.load(), pid, "")


@pytest.mark.parametrize("omitted", [(BRIDGE,), (HASWELL,), (LLAMA,)])
def test_missing_resident_bridge_backend_or_llama_refuses(fake_tree, omitted):
    with pytest.raises(ValueError, match="evidence incomplete"):
        activation.verify_resident_maps(fake_tree.load(), 12345, fake_maps(fake_tree, omitted=omitted))


@pytest.mark.parametrize("kind", ["unknown_resource", "alternate_cpu", "unknown_system", "deleted", "hash_changed"])
def test_resident_map_refuses_unregistered_changed_deleted_and_other_backend(fake_tree, kind):
    act = fake_tree.load()
    base = fake_tree.root / activation.RESOURCE_ROOT
    if kind == "unknown_resource":
        text = fake_maps(fake_tree, extra=[(base / "unregistered.so", "r--p")])
    elif kind == "alternate_cpu":
        name = "runtime/llama-b11791/libggml-cpu-skylakex.so"
        path = fake_tree.write(f"{activation.RESOURCE_ROOT}/{name}", b"other fake backend")
        act.resource_report["verified_file_identities"][name] = {"sha256": activation.sha_file(path)}
        text = fake_maps(fake_tree, extra=[(path, "r-xp")])
    elif kind == "unknown_system":
        text = fake_maps(fake_tree, extra=[(fake_tree.root / "fake-system/unregistered.so", "r-xp")])
    elif kind == "deleted":
        text = fake_maps(fake_tree).replace(str(base / BRIDGE), str(base / BRIDGE) + " (deleted)")
    else:
        (base / HASWELL).write_bytes(b"changed after activation")
        text = fake_maps(fake_tree)
    with pytest.raises(ValueError):
        activation.verify_resident_maps(act, 12345, text)


def test_non_executable_external_data_is_outside_resident_library_claim(fake_tree):
    text = fake_maps(fake_tree, extra=[(fake_tree.root / "outside-readonly-data", "r--p")])
    result = activation.verify_resident_maps(fake_tree.load(), 12345, text)
    assert result["status"] == "passed"
    assert len(result["mapped_files_sha256"]) == 4


@pytest.mark.parametrize("changes", [None, {"status": "missing_resources_no_changes"},
    {"expected_files": 40}, {"verified_files": 40}, {"bridge_executable_verified": False},
    {"downloads": 1}, {"model_calls": 1}])
def test_default_resource_wrapper_checks_all41_and_disables_downloads(fake_tree, monkeypatch, changes):
    """Replace only module loading with a fake; never execute the real restorer."""
    calls = []
    report = copy.deepcopy(fake_tree.report)
    if changes:
        report.update(changes)
    sentinel = object()
    def load_spec(root):
        calls.append(("load_spec", root))
        return sentinel
    def restore(root, spec, *, allow_download):
        assert spec is sentinel and allow_download is False
        calls.append(("restore", root, allow_download))
        return report
    module = SimpleNamespace(load_spec=load_spec, restore=restore)
    name = "synsteg_fake_activation_restorer_test"
    spec = SimpleNamespace(name=name, loader=SimpleNamespace(exec_module=lambda actual: calls.append(("exec_fake", actual is module))))
    monkeypatch.setattr(activation.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(activation.importlib.util, "module_from_spec", lambda actual: module)
    monkeypatch.setitem(activation.sys.modules, name, module)
    if changes is None:
        assert activation.verify_resource_files(fake_tree.root) == report
    else:
        with pytest.raises(ValueError, match="all41 exact resource"):
            activation.verify_resource_files(fake_tree.root)
    assert calls == [("exec_fake", True), ("load_spec", fake_tree.root),
        ("restore", fake_tree.root / activation.RESOURCE_ROOT, False)]
