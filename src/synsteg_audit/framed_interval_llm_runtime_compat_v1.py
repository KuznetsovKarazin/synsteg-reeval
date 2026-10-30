"""runtime-identity-compat-v1: explicit identities for built-in Python math.

This additive provider does not change frozen codec/native behavior. A built-in
module has no separate .so to hash: its origin is recorded explicitly alongside
the Python executable identity. This is current-runtime evidence, not a claim
that historical math implementation bytes were separately archived.
"""
import importlib.machinery
import json
from pathlib import Path
import platform
import re
import sys

from .framed_interval_llm_v1_activation import checked_bytes, sha_file

SCHEMA = "runtime-identity-compat-v1"
MODULE_PATH = "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py"
ENTRY_PATH = "experiments/framed-interval-llm-v1/vocab_runtime_compat_v1.py"
HELPER_PATH = "scripts/prepare_framed_interval_llm_vocab_activation.py"
PINNED = {
    HELPER_PATH: "b4d92d91acaf29b8a7d150739cad232d3b0e5c7f8d3cc0d540de382951706fdb",
    "src/synsteg_audit/framed_interval_llm_v1_vocab_collect.py": "0ee0aa5b4631cb6750f7a9084cba81234626d9de42be9d3a14efe66814863ab4",
}


def current_runtime():
    import math
    import cryptography
    from cryptography.hazmat.bindings import _rust
    python_path = str(Path(sys.executable).resolve(strict=True))
    files = {"python": python_path,
             "cryptography_rust": str(Path(_rust.__file__).resolve(strict=True))}
    origin = getattr(getattr(math, "__spec__", None), "origin", None)
    module_file = getattr(math, "__file__", None)
    if (origin == "built-in" and module_file is None and "math" in sys.builtin_module_names
            and math.__spec__.loader is importlib.machinery.BuiltinImporter):
        math_identity = {"kind": "builtin", "origin": "built-in", "separate_module_file": None,
            "python_identity_reference": "file_sha256.python",
            "scope": "registered built-in module; no independently addressable math module file"}
    elif type(module_file) is str and origin == module_file:
        files["math"] = str(Path(module_file).resolve(strict=True))
        math_identity = {"kind": "file", "origin": origin,
                         "separate_module_file": files["math"]}
    else:
        raise ValueError("unsupported or inconsistent math module provenance")
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "libc": list(platform.libc_ver()), "cryptography": cryptography.__version__,
        "file_sha256": {name: {"path": path, "sha256": sha_file(path)} for name, path in files.items()},
        "historical_system_library_byte_hashes_available": False,
        "runtime_identity_schema": SCHEMA, "module_identities": {"math": math_identity}}


def verify_adapter_readback(root, path, expected_sha):
    proof = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if (proof.get("schema") != "runtime-identity-compat-v1-git-readback"
            or type(proof.get("commit")) is not str
            or re.fullmatch("[0-9a-f]{40}", proof["commit"]) is None
            or proof.get("all_files_exact") is not True or type(proof.get("files_sha256")) is not dict):
        raise ValueError("exact coordinator adapter Git readback required")
    required = {MODULE_PATH, ENTRY_PATH, *PINNED}
    if not required.issubset(proof["files_sha256"]):
        raise ValueError("adapter readback omits executable dependency")
    for name, expected in proof["files_sha256"].items():
        checked_bytes(root, {"path": name, "sha256": expected})
    for name, expected in PINNED.items():
        if proof["files_sha256"][name] != expected:
            raise ValueError("frozen dependency identity differs")
    return proof


def load_vocab_activation(root, path, expected_sha, intent):
    from .framed_interval_llm_v1_vocab_collect import load_collection_activation
    return load_collection_activation(root, path, expected_sha, intent, runtime_reader=current_runtime)


def load_run_activation(root, path, expected_sha, intent, *, receiver_scope=False):
    from .framed_interval_llm_v1_activation import load_activation
    return load_activation(root, path, expected_sha, intent,
                           receiver_scope=receiver_scope, runtime_reader=current_runtime)
