"""Vocabulary collector activation/resident checks with tiny fake artifacts only."""
import copy
import json
from pathlib import Path

import pytest

from synsteg_audit import framed_interval_llm_v1_vocab_collect as mod
from synsteg_audit.framed_interval_llm_v1_preflight import canonical_json

REPO = Path(__file__).resolve().parents[1]


class FakeTree:
    def __init__(self, root):
        self.root, self.calls = root, []
        for name in mod.SOURCE_PATHS: self.write(name, b"synthetic source fixture\n")
        self.write(mod.PUBLIC_PATH, (REPO / mod.PUBLIC_PATH).read_bytes())
        self.plan_name = "experiments/framed-interval-llm-v1/tokenizer-preflight-proposal.json"
        self.write(self.plan_name, (REPO / self.plan_name).read_bytes())
        self.runtime = {"synthetic": True}
        self.binary = self.write("fake-resources/vocab-bridge", b"fake executable, never executed\n")
        self.binary.chmod(0o700)
        self.compiler = self.write("fake-resources/compiler", b"fake compiler, never invoked\n")
        self.header = self.write("fake-resources/llama.h", b"fake header, never compiled\n")
        self.system = self.write("fake-resources/libc.so", b"fake libc mapping\n")
        lib = self.write(mod.RESOURCE_ROOT+"/runtime/llama-b11791/libllama.so.0", b"fake llama mapping\n")
        self.report = {"frozen_resource_identity_sha256": "a"*64,
            "verified_file_identities": {"runtime/llama-b11791/libllama.so.0": {"sha256": mod.sha_file(lib)}}}
        self.sources = {name: mod.sha_file(root / name) for name in mod.SOURCE_PATHS}
        build = {"schema": "framed-interval-llm-v1-vocab-build-v1", "source_bundle_sha256": mod.digest(canonical_json(self.sources)),
            "runtime_commit": mod.RUNTIME_COMMIT, "binary_sha256": mod.sha_file(self.binary), "native_model_executions": 0,
            "compiler": {"path": str(self.compiler), "sha256": mod.sha_file(self.compiler)},
            "headers": {self.header.relative_to(root).as_posix(): mod.sha_file(self.header)},
            "build_command": ["synthetic-not-executed"]}
        self.obj = {"schema": "framed-interval-llm-v1-vocab-activation-v1", "enabled": True,
            "freeze_commit": "b"*40, "coordinator_readback": {"commit": "b"*40, "all_files_exact": True},
            "sources_sha256": self.sources, "plan": self.descriptor(self.plan_name),
            "public_protocol": {"path": mod.PUBLIC_PATH, "sha256": mod.PUBLIC_SHA},
            "binary": self.descriptor(self.binary.relative_to(root).as_posix()),
            "build_receipt": self.artifact("fake-evidence/build.json", build),
            "resource_identity_sha256": "a"*64, "runtime": self.runtime,
            "native_system_libraries_sha256": {str(self.system): mod.sha_file(self.system)}, "timeout_seconds": 120}

    def write(self, name, raw):
        path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); return path
    def descriptor(self, name): return {"path": name, "sha256": mod.sha_file(self.root / name)}
    def artifact(self, name, obj): self.write(name, canonical_json(obj)); return self.descriptor(name)
    def resource(self, root): self.calls.append("read-only-resource-verification"); return self.report
    def load(self, *, intent=mod.INTENT):
        descriptor = self.artifact("fake-evidence/activation.json", self.obj)
        return mod.load_collection_activation(self.root, descriptor["path"], descriptor["sha256"], intent,
            resource_verifier=self.resource, runtime_reader=lambda: self.runtime)


def test_full_fake_activation_opens_no_process(tmp_path, monkeypatch):
    tree = FakeTree(tmp_path)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: pytest.fail("native process forbidden"))
    obj, report, plan, protocol, bundle = tree.load()
    assert obj == tree.obj and report == tree.report
    assert len(json.loads(plan)["fixtures"]) == 15
    assert len(protocol["runs"]) == 4
    assert bundle == mod.digest(canonical_json(tree.sources))
    assert tree.calls == ["read-only-resource-verification"]


@pytest.mark.parametrize("change", ["disabled", "commit", "readback", "readbackbool", "source", "public", "runtime", "resource", "timeout", "timeoutbool", "systems", "binarymode", "headers", "build", "compiler", "emptyargv", "boolargv", "emptyarg", "plan"])
def test_activation_failures_before_any_native(tmp_path, monkeypatch, change):
    tree = FakeTree(tmp_path)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: pytest.fail("native process forbidden"))
    if change == "disabled": tree.obj["enabled"] = False
    elif change == "commit": tree.obj["freeze_commit"] = "bad"
    elif change == "readback": tree.obj["coordinator_readback"]["all_files_exact"] = False
    elif change == "readbackbool": tree.obj["coordinator_readback"]["all_files_exact"] = 1
    elif change == "source": tree.obj["sources_sha256"].pop(mod.SOURCE_PATHS[0])
    elif change == "public": tree.obj["public_protocol"]["sha256"] = "0"*64
    elif change == "runtime": tree.obj["runtime"] = {"different": True}
    elif change == "resource": tree.obj["resource_identity_sha256"] = "0"*64
    elif change == "timeout": tree.obj["timeout_seconds"] = 121
    elif change == "timeoutbool": tree.obj["timeout_seconds"] = True
    elif change == "systems": tree.obj["native_system_libraries_sha256"] = {}
    elif change == "binarymode": tree.binary.chmod(0o600)
    elif change == "plan":
        plan = json.loads((tmp_path / tree.plan_name).read_bytes()); plan["output_tokenize_flags"]["parse_special"] = True
        tree.obj["plan"] = tree.artifact(tree.plan_name, plan)
    else:
        build = json.loads((tmp_path / tree.obj["build_receipt"]["path"]).read_bytes())
        if change == "headers": build["headers"] = {}
        elif change == "build": build["native_model_executions"] = True
        elif change == "emptyargv": build["build_command"] = []
        elif change == "boolargv": build["build_command"] = [True]
        elif change == "emptyarg": build["build_command"] = [""]
        else: build["compiler"]["sha256"] = "0"*64
        tree.obj["build_receipt"] = tree.artifact("fake-evidence/build.json", build)
    with pytest.raises((ValueError, KeyError)): tree.load()


def test_wrong_intent_rejects_before_readonly_resource_check(tmp_path):
    tree = FakeTree(tmp_path)
    with pytest.raises(ValueError, match="intent"): tree.load(intent="generate-model-output")
    assert not tree.calls


def test_resident_collector_and_llama_only_fake_maps(tmp_path):
    tree = FakeTree(tmp_path)
    paths = [tree.binary, tmp_path / mod.RESOURCE_ROOT / "runtime/llama-b11791/libllama.so.0", tree.system]
    maps = "\n".join(f"1000-2000 r-xp 0000 00:00 1 {path}" for path in paths)
    result = mod.verify_collection_resident(tmp_path, tree.obj, tree.report, 123, maps_text=maps)
    assert result["status"] == "passed" and len(result["mapped_files_sha256"]) == 3
    for corrupted in (maps.replace("r-xp", "r--p"), maps+"\n1000-2000 r-xp 0000 00:00 1 /foreign/lib.so", maps+" (deleted)"):
        with pytest.raises(ValueError): mod.verify_collection_resident(tmp_path, tree.obj, tree.report, 123, maps_text=corrupted)
    model = tmp_path / mod.RESOURCE_ROOT / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    model.write_bytes(b"fake tensor file must never be mapped")
    tree.report["verified_file_identities"][model.name] = {"sha256": mod.sha_file(model)}
    with pytest.raises(ValueError, match="tensor"):
        mod.verify_collection_resident(tmp_path, tree.obj, tree.report, 123, maps_text=maps+f"\n1000-2000 r--p 0000 00:00 1 {model}")
