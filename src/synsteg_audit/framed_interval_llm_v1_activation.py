"""Fail-closed activation and pre-PROMPT checks; no inference at import.

An activation SHA and explicit intent must be supplied by the coordinator after
Git readback. This validates that supplied evidence; it does not cryptographically
prove the human identity of a coordinator. No downloader or compiler is called.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys

from .framed_interval_llm_v1 import canonical_json
from .framed_interval_llm_v1_preflight import validate_preflight_receipt

INTENT = "execute-frozen-framed-interval-llm-v1-four-cases"
PROPOSAL_PATH = "experiments/framed-interval-llm-v1/proposed-protocol.json"
PROPOSAL_SHA = "d15f11b36bd64fc126e2ada41f1ca30fef5f641ed44311c01852de84f1522efc"
PUBLIC_PATH = "experiments/framed-interval-llm-v1/receiver-public-protocol.json"
PUBLIC_SHA = "4cab7dd3035ac1ad298952814c12b12ecbe2d021f4b9cd14b35692a2c300d6ff"
RESTORER_PATH = "scripts/restore_framed_interval_llm_v1_resources.py"
RESTORER_SHA = "284904969e2663472bfcb8caa19b776801771120b4156911b8dafb77c01c97f9"
EXECUTION_PATH = "experiments/seeded_pair_llm_v1/execution_manifest.json"
EXECUTION_SHA = "c877bfdd27a64abd126d6d22580277b3163159bc35a515679a5423e412f453e6"
RESOURCE_ROOT = "resources/seeded-pair-llm-v1"
EXPECTED_RUNS = ("station-stego", "station-ordinary", "library-stego", "library-ordinary")
REQUIRED_SOURCES = (
    "src/synsteg_audit/__init__.py",
    "src/synsteg_audit/framed_interval_llm_v1.py",
    "src/synsteg_audit/framed_interval_llm_v1_qwen.py",
    "src/synsteg_audit/framed_interval_llm_v1_journal.py",
    "src/synsteg_audit/framed_interval_llm_v1_activation.py",
    "src/synsteg_audit/framed_interval_llm_v1_orchestration.py",
    "src/synsteg_audit/framed_interval_llm_v1_preflight.py",
    "src/synsteg_audit/canonical_channel_v2.py",
    "src/synsteg_audit/integer_interval_channel_v1.py",
    "src/synsteg_audit/seeded_pair_llm.py",
    "experiments/framed-interval-llm-v1/run_fixed512.py", RESTORER_PATH, PROPOSAL_PATH, PUBLIC_PATH,
)
FROZEN_COMPONENTS = {
    "src/synsteg_audit/framed_interval_llm_v1.py": "64ebbfe07b526d62789ae27a4f55feb717e83abf9e0a2ca72bb593ff9f30ff00",
    "src/synsteg_audit/framed_interval_llm_v1_qwen.py": "5ced911eb6929785d2602bebf19d1be298b8faaa6aec5f4900ee102f98fd364d",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024*1024): h.update(block)
    return h.hexdigest()


def _hex(value, length=64):
    if type(value) is not str or re.fullmatch("[0-9a-f]{"+str(length)+"}", value) is None:
        raise ValueError("expected exact lowercase identity digest")


def safe_path(root, relative):
    if type(relative) is not str or "\\" in relative:
        raise ValueError("relative artifact path required")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or not relative:
        raise ValueError("unsafe artifact path")
    root = Path(root).resolve()
    target = root / relative
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink(): raise ValueError("symlinked activation/source artifact")
    if root not in target.resolve().parents:
        raise ValueError("artifact escapes repository root")
    return target


def checked_bytes(root, entry):
    if type(entry) is not dict or set(entry) != {"path", "sha256"}:
        raise ValueError("artifact identity descriptor required")
    _hex(entry["sha256"])
    raw = safe_path(root, entry["path"]).read_bytes()
    if digest(raw) != entry["sha256"]: raise ValueError("artifact bytes differ: "+entry["path"])
    return raw


def current_runtime():
    import math
    import cryptography
    from cryptography.hazmat.bindings import _rust
    files = {"python": str(Path(sys.executable).resolve()), "math": str(Path(math.__file__).resolve()),
             "cryptography_rust": str(Path(_rust.__file__).resolve())}
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "cryptography": cryptography.__version__,
            "file_sha256": {name: {"path": path, "sha256": sha_file(path)} for name, path in files.items()},
            "historical_system_library_byte_hashes_available": False}


def verify_resource_files(root):
    """Calls frozen restorer in read-only mode; never permits downloads."""
    path = safe_path(root, RESTORER_PATH)
    if sha_file(path) != RESTORER_SHA: raise ValueError("frozen restorer source differs")
    spec = importlib.util.spec_from_file_location("synsteg_fixed512_scoped_restorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    resources = module.load_spec(root)
    report = module.restore(safe_path(root, RESOURCE_ROOT), resources, allow_download=False)
    if (report.get("status") != "verified" or report.get("verified_files") != 41
            or report.get("expected_files") != 41 or report.get("bridge_executable_verified") is not True
            or report.get("downloads") != 0 or report.get("model_calls") != 0):
        raise ValueError("all41 exact resource files and executable bridge are required")
    return report


@dataclass(frozen=True)
class Activation:
    root: Path
    sha256: str
    data: dict
    public_protocol: dict
    preflight: dict
    resource_report: dict


def load_activation(root, activation_path, expected_sha, intent, *,
                    resource_verifier=verify_resource_files, runtime_reader=current_runtime,
                    preflight_validator=validate_preflight_receipt, receiver_scope=False):
    """Production CLI uses the default checks; dependency injection is for fake tests."""
    if intent != INTENT: raise ValueError("explicit frozen-four-case execution intent required")
    _hex(expected_sha)
    root = Path(root).resolve()
    raw = safe_path(root, activation_path).read_bytes()
    if digest(raw) != expected_sha: raise ValueError("activation differs from trusted coordinator SHA")
    obj = json.loads(raw)
    fields = {"schema", "experiment_id", "enabled", "final_freeze_commit", "coordinator_readback",
        "freeze_manifest", "public_protocol", "expected_public_runs_sha256", "resource_receipt",
        "runtime", "qwen_binding_manifest", "ready", "preflight_plan", "preflight_receipt",
        "preflight_inventory", "collector_source_sha256", "native_system_libraries_sha256"}
    if (type(obj) is not dict or set(obj) != fields or obj["schema"] != "framed-interval-llm-v1-activation-v1"
            or obj["experiment_id"] != "framed-interval-llm-v1" or obj["enabled"] is not True):
        raise ValueError("complete enabled production activation required")
    _hex(obj["final_freeze_commit"], 40)
    proof = obj["coordinator_readback"]
    if (type(proof) is not dict or set(proof) != {"commit", "all_files_exact", "freeze_manifest_sha256"}
            or proof["commit"] != obj["final_freeze_commit"] or proof["all_files_exact"] is not True
            or proof["freeze_manifest_sha256"] != obj["freeze_manifest"]["sha256"]):
        raise ValueError("coordinator's exact Git readback evidence is missing")
    freeze = json.loads(checked_bytes(root, obj["freeze_manifest"]))
    if freeze.get("commit") != obj["final_freeze_commit"] or type(freeze.get("files_sha256")) is not dict:
        raise ValueError("freeze manifest commit/source mapping differs")
    if not set(REQUIRED_SOURCES).issubset(freeze["files_sha256"]):
        raise ValueError("activation omits required executable source dependencies")
    for name, expected in freeze["files_sha256"].items():
        _hex(expected)
        if receiver_scope and name == PROPOSAL_PATH:
            if expected != PROPOSAL_SHA: raise ValueError("private proposal freeze identity differs")
            continue  # Parent verified these bytes; receiver does not open plaintext source.
        if receiver_scope and name not in REQUIRED_SOURCES:
            continue  # Optional parent artifacts may include private expectations/traces.
        if sha_file(safe_path(root, name)) != expected: raise ValueError("frozen source changed: "+name)
    for name, expected in [(PROPOSAL_PATH, PROPOSAL_SHA), (PUBLIC_PATH, PUBLIC_SHA),
                           (EXECUTION_PATH, EXECUTION_SHA), (RESTORER_PATH, RESTORER_SHA),
                           *FROZEN_COMPONENTS.items()]:
        if receiver_scope and name == PROPOSAL_PATH: continue
        if sha_file(safe_path(root, name)) != expected: raise ValueError("fixed checkpoint resource changed: "+name)
    if obj["public_protocol"] != {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA}:
        raise ValueError("fixed payload-free public protocol required")
    protocol = json.loads(checked_bytes(root, obj["public_protocol"]))
    if (protocol["source_proposal_sha256"] != PROPOSAL_SHA
            or tuple(r["run_id"] for r in protocol["runs"]) != EXPECTED_RUNS
            or any("payload_hex" in r or r["output_tokens"] != 512 for r in protocol["runs"])
            or digest(canonical_json(protocol["runs"])) != obj["expected_public_runs_sha256"]):
        raise ValueError("complete fixed four-case public manifest differs")
    report = resource_verifier(root)
    saved_report = json.loads(checked_bytes(root, obj["resource_receipt"]))
    identity_fields = ("status", "expected_files", "verified_files", "verified_file_identities",
                       "frozen_resource_identity_sha256", "bridge_executable_verified", "manifest_sha256")
    if any(saved_report.get(k) != report.get(k) for k in identity_fields):
        raise ValueError("resource restoration receipt differs from fresh on-disk verification")
    runtime = runtime_reader()
    if runtime != obj["runtime"]: raise ValueError("current Python/math/crypto runtime differs from activation")
    old = json.loads(safe_path(root, EXECUTION_PATH).read_bytes())
    for name in ("python", "platform", "machine", "libc"):
        if runtime[name] != old["channel_resources"][name]:
            raise ValueError("historical runtime metadata differs: "+name)
    qwen = obj["qwen_binding_manifest"]
    qwen_fields = {"model_sha256", "runtime_commit", "cpu_backend_sha256", "bridge_sha256",
        "shared_libraries_sha256", "python_runtime_sha256", "quantizer_sha256", "special_ids_sha256",
        "vocabulary_size", "masked_tokens", "parameters"}
    if type(qwen) is not dict or set(qwen) != qwen_fields:
        raise ValueError("exact complete Qwen binding declaration required before launch")
    from .framed_interval_llm_v1_qwen import MODEL_SHA256, CPU_SHA256, RUNTIME_COMMIT, PARAMETERS
    if (type(qwen["parameters"]) is not dict or set(qwen["parameters"]) != set(PARAMETERS)
            or any(type(qwen["parameters"][k]) is not type(v) for k, v in PARAMETERS.items())):
        raise ValueError("exact Qwen parameter types required before launch")
    for name in qwen_fields - {"runtime_commit", "shared_libraries_sha256", "vocabulary_size", "masked_tokens", "parameters"}:
        _hex(qwen[name])
    if (type(obj["ready"]) is not dict or set(obj["ready"]) != {"vocabulary_size", "masked_tokens", "context_tokens"}
            or any(type(value) is not int for value in obj["ready"].values())
            or obj["ready"]["context_tokens"] != 2048
            or type(qwen["vocabulary_size"]) is not int or type(qwen["masked_tokens"]) is not int):
        raise ValueError("exact integral READY metadata required before launch")
    if qwen.get("python_runtime_sha256") != digest(canonical_json(runtime)):
        raise ValueError("Qwen runtime declaration differs from current fingerprint")
    libraries = {name: row["sha256"] for name, row in report["verified_file_identities"].items()
                 if name.startswith("runtime/")}
    if qwen.get("shared_libraries_sha256") != libraries:
        raise ValueError("Qwen library declaration omits or changes frozen resources")
    preflight = preflight_validator(json.loads(checked_bytes(root, obj["preflight_receipt"])),
        plan_bytes=checked_bytes(root, obj["preflight_plan"]),
        inventory_bytes=checked_bytes(root, obj["preflight_inventory"]),
        expected_plan_sha256=obj["preflight_plan"]["sha256"],
        expected_resource_identity_sha256=report["frozen_resource_identity_sha256"],
        expected_native_libraries=libraries, expected_collector_source_sha256=obj["collector_source_sha256"],
        expected_ready=obj["ready"], prompt_protocol=protocol)
    if qwen.get("special_ids_sha256") != preflight["special_ids_sha256"]:
        raise ValueError("Qwen complete special-ID declaration differs from preflight")
    if (qwen.get("vocabulary_size") != obj["ready"]["vocabulary_size"]
            or qwen.get("masked_tokens") != obj["ready"]["masked_tokens"]):
        raise ValueError("Qwen READY declarations disagree before process launch")
    if (qwen.get("model_sha256") != MODEL_SHA256 or qwen.get("cpu_backend_sha256") != CPU_SHA256
            or qwen.get("runtime_commit") != RUNTIME_COMMIT or qwen.get("parameters") != PARAMETERS
            or qwen.get("bridge_sha256") != report["verified_file_identities"]["seeded_pair_llama_bridge"]["sha256"]
            or qwen.get("quantizer_sha256") != sha_file(safe_path(root, "src/synsteg_audit/seeded_pair_llm.py"))):
        raise ValueError("Qwen binding declarations differ before process launch")
    system = obj["native_system_libraries_sha256"]
    if type(system) is not dict or not system:
        raise ValueError("expected current native system-library identities required")
    for path, expected in system.items():
        _hex(expected)
        if not Path(path).is_absolute() or sha_file(path) != expected:
            raise ValueError("current native system-library identity differs")
    return Activation(root, expected_sha, obj, protocol, preflight, report)


def verify_resident_maps(activation, pid, maps_text=None, *, file_hasher=sha_file):
    """After READY, before PROMPT; inspect only this explicitly owned native PID."""
    if type(pid) is not int or pid <= 0: raise ValueError("owned native PID required")
    if maps_text is None: maps_text = Path(f"/proc/{pid}/maps").read_text()
    resource_root = safe_path(activation.root, RESOURCE_ROOT).resolve()
    allowed = activation.resource_report["verified_file_identities"]
    system = activation.data["native_system_libraries_sha256"]
    observed, native = {}, set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"): continue
        text = fields[5]
        if text.endswith(" (deleted)"): raise ValueError("mapped file deleted after load")
        path = Path(text).resolve()
        under = resource_root in path.parents
        if not under and "x" not in fields[1]: continue
        if under:
            relative = path.relative_to(resource_root).as_posix()
            if relative not in allowed: raise ValueError("mapped resource outside exact41 allowlist")
            if "libggml-cpu-" in path.name and path.name != "libggml-cpu-haswell.so":
                raise ValueError("unexpected CPU backend mapped")
            expected = allowed[relative]["sha256"]
            if "x" in fields[1]: native.add(relative)
        else:
            expected = system.get(str(path))
            if expected is None: raise ValueError("unregistered native system library mapped: "+str(path))
        if str(path) not in observed and file_hasher(path) != expected:
            raise ValueError("resident mapped-file hash differs")
        observed[str(path)] = expected
    required = {"seeded_pair_llama_bridge", "runtime/llama-b11791/libggml-cpu-haswell.so"}
    if not required.issubset(native) or not any(Path(p).name.startswith("libllama.so") for p in native):
        raise ValueError("resident bridge/Haswell/libllama evidence incomplete")
    return {"status": "passed", "pid": pid, "mapped_files_sha256": observed,
            "scope": "mapped paths and current file bytes after READY, before PROMPT; not historical system-byte attestation"}
