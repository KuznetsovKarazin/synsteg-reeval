"""Public native resource binding, independent of experiment fixtures/provenance.

Import and profile validation do not start a subprocess or tokenize. A native
child starts ONLY from PublicNativeService.prepare, called by the explicit
worker. The public profile contains algorithms/resources/runtime metadata only.
"""
from dataclasses import replace
import hashlib
import importlib.machinery
import platform
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

from .canonical_channel_v2 import CanonicalGate, ContractError
from .framed_interval_llm_v1 import Endpoint, canonical_json
from .framed_interval_llm_v1_qwen import (
    prepare_qwen_endpoint, MODEL_SHA256, CPU_SHA256, RUNTIME_COMMIT, PARAMETERS,
)
from .framed_interval_llm_v1_journal import atomic_json
from .proc_namespace_identity_v2 import capture_owned_child
from .seeded_pair_llm import NativeBridge
from .framed_interval_public_v1 import (
    VERSION, TOKEN_CAP, public_digest, validate_public_context, validate_public_spec,
)

PROFILE_SCHEMA = VERSION + "-qwen-native-profile"
RESOURCE_ID = "qwen2.5-1.5b-q4km-b11791-public-profile-v1"
RESOURCE_ROOT = "resources/seeded-pair-llm-v1"
BRIDGE = "seeded_pair_llama_bridge"
MODEL = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
CPU = "runtime/llama-b11791/libggml-cpu-haswell.so"
PROFILE_FIELDS = {"schema", "resource_files", "runtime", "system_libraries_sha256",
                  "ready", "qwen_binding", "special_ids"}
QWEN_FIELDS = {"model_sha256", "runtime_commit", "cpu_backend_sha256", "bridge_sha256",
    "shared_libraries_sha256", "python_runtime_sha256", "quantizer_sha256",
    "special_ids_sha256", "vocabulary_size", "masked_tokens", "parameters"}
RUNTIME_FIELDS = {"python", "platform", "machine", "libc", "cryptography", "file_sha256",
    "historical_system_library_byte_hashes_available", "runtime_identity_schema", "module_identities"}
RUNTIME_SOURCE = Path(__file__).with_name("seeded_pair_llm.py")
RESOURCE_IDENTITIES = {'llama-b11791-bin-ubuntu-x64.tar.gz': {'bytes': 22302292, 'sha256': 'ac75208179d00d86f378d5f10c65f9785028ce877cf57279f1a7b8dac54a25ff'}, 'qwen2.5-1.5b-instruct-q4_k_m.gguf': {'bytes': 1117320736, 'sha256': '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'}, 'runtime/llama-b11791/libggml-base.so': {'bytes': 897792, 'sha256': '34ae7cd0afb50808e867d44f2cced9b68aa01c176ff23cd602c468e97f9d4f7c'}, 'runtime/llama-b11791/libggml-base.so.0': {'bytes': 897792, 'sha256': '34ae7cd0afb50808e867d44f2cced9b68aa01c176ff23cd602c468e97f9d4f7c'}, 'runtime/llama-b11791/libggml-base.so.0.27.0': {'bytes': 897792, 'sha256': '34ae7cd0afb50808e867d44f2cced9b68aa01c176ff23cd602c468e97f9d4f7c'}, 'runtime/llama-b11791/libggml-cpu-alderlake.so': {'bytes': 1416424, 'sha256': '7758a8c74953999a664bc88549e9ed0c2e387f23f839f85272630eb376d00a86'}, 'runtime/llama-b11791/libggml-cpu-cannonlake.so': {'bytes': 1561704, 'sha256': '36d6c8f21f3b68f80a7bddfe8efbfc18a1d57aad631014b4bfa8e0d5695e7e8f'}, 'runtime/llama-b11791/libggml-cpu-cascadelake.so': {'bytes': 1577792, 'sha256': 'c4e0622bfe8966ef1ee303f0ec11fb49a33d12adfcc97f55bae06951e956097f'}, 'runtime/llama-b11791/libggml-cpu-cooperlake.so': {'bytes': 1577864, 'sha256': '2f954ab8e105425de866b25cec9b94a5b6c3c3439e36133e9d7278fd4fd037f8'}, 'runtime/llama-b11791/libggml-cpu-haswell.so': {'bytes': 1416424, 'sha256': '1bc37cef126b01fe5a15222c07d20d29931936bfaa909a97052f6662f4c9d97c'}, 'runtime/llama-b11791/libggml-cpu-icelake.so': {'bytes': 1577792, 'sha256': 'b0c79233cc397c6f60a0d097629c645e1216d431b4477c9f092a4727d746b59d'}, 'runtime/llama-b11791/libggml-cpu-ivybridge.so': {'bytes': 1352040, 'sha256': 'a71f2adfa51ce175082eefef528a9bc57c141cd7864be1bdcbf2b9248f4fc807'}, 'runtime/llama-b11791/libggml-cpu-piledriver.so': {'bytes': 1347944, 'sha256': '5c4bde3757542a078ff5c66d290ce981655549583b3873c53b4d35b17d6e30a2'}, 'runtime/llama-b11791/libggml-cpu-sandybridge.so': {'bytes': 1350992, 'sha256': 'e927919ce87edb4cc2ed9bdde5bd3599ca696482a43b0fcb21175713b05aace2'}, 'runtime/llama-b11791/libggml-cpu-sapphirerapids.so': {'bytes': 1844160, 'sha256': '5cb3d534592a69bd41abbdcdf1bd5e2347ef7b3a4b7529dacac39c93ebfaa8ea'}, 'runtime/llama-b11791/libggml-cpu-skylakex.so': {'bytes': 1561704, 'sha256': '4eeaa0c0953865ec7d5992878af899045bab059f06ba7e0d233b062e1757b328'}, 'runtime/llama-b11791/libggml-cpu-sse42.so': {'bytes': 954536, 'sha256': '2b3a4aa00fb42097553710c217264f453ef519c88ef062294c8416b090b9a000'}, 'runtime/llama-b11791/libggml-cpu-x64.so': {'bytes': 950616, 'sha256': '90e347de07371df6610bee0dbfa1c13f93da43d52d7c1617bbab40f4ed1333b6'}, 'runtime/llama-b11791/libggml-cpu-zen4.so': {'bytes': 1577864, 'sha256': 'e0c2eea9629810ab8f6d17ee00819d61419323f1e1c12ef0605cc55a5ff80626'}, 'runtime/llama-b11791/libggml-rpc.so': {'bytes': 319776, 'sha256': 'e28139e2d982fb057b88d243e5b2e9d72c52f8b98ae910b1cf751240ef1f9803'}, 'runtime/llama-b11791/libggml.so': {'bytes': 51040, 'sha256': '8643a58bb37af6fb153e2ea98b48efa227855725ee51c05f4ca76b9867956463'}, 'runtime/llama-b11791/libggml.so.0': {'bytes': 51040, 'sha256': '8643a58bb37af6fb153e2ea98b48efa227855725ee51c05f4ca76b9867956463'}, 'runtime/llama-b11791/libggml.so.0.27.0': {'bytes': 51040, 'sha256': '8643a58bb37af6fb153e2ea98b48efa227855725ee51c05f4ca76b9867956463'}, 'runtime/llama-b11791/libllama-batched-bench-impl.so': {'bytes': 49064, 'sha256': 'a69a1abbca090c7ab3fa286adcb9fa51b710730c7950c02796a9a84197ef00c7'}, 'runtime/llama-b11791/libllama-bench-impl.so': {'bytes': 490560, 'sha256': 'e4ca59722bb4c5567fde2d5e07deb2116211e88d6d1afc43deabc5278a575ddb'}, 'runtime/llama-b11791/libllama-cli-impl.so': {'bytes': 277120, 'sha256': '085ffb43408aac8202d35dcd24e0fb3800e1d952925f862c0de3d6fd2f13bb82'}, 'runtime/llama-b11791/libllama-common.so': {'bytes': 6646064, 'sha256': '3bce6d6da2e589f86c8e533a2c0e5f84f0881d1b306b2b4f2de9c61033adcde4'}, 'runtime/llama-b11791/libllama-common.so.0': {'bytes': 6646064, 'sha256': '3bce6d6da2e589f86c8e533a2c0e5f84f0881d1b306b2b4f2de9c61033adcde4'}, 'runtime/llama-b11791/libllama-common.so.0.7.0': {'bytes': 6646064, 'sha256': '3bce6d6da2e589f86c8e533a2c0e5f84f0881d1b306b2b4f2de9c61033adcde4'}, 'runtime/llama-b11791/libllama-completion-impl.so': {'bytes': 145168, 'sha256': '0d3d7ae4cdeafbce5d765f9027371506f5205eeb3203553e463035bc4babae23'}, 'runtime/llama-b11791/libllama-fit-params-impl.so': {'bytes': 39920, 'sha256': '397ce456786d0c161b14dd0ee225378d1f1ff875cbc428d32fda7dcaf15e306b'}, 'runtime/llama-b11791/libllama-perplexity-impl.so': {'bytes': 171912, 'sha256': '357dc3ef0ebe6fe69b35c965db701452f44cc08a01f77a22c609b4aa96afbca9'}, 'runtime/llama-b11791/libllama-quantize-impl.so': {'bytes': 81840, 'sha256': '8dd43c1fb15f2e6096237a5cc3aa52694b04978c176f10db28c27d15d6c1adcc'}, 'runtime/llama-b11791/libllama-server-impl.so': {'bytes': 7818920, 'sha256': '79aa8e038998d07b0868c7f98f0261f37b91cb4cc30fe60c42aa69beda01f581'}, 'runtime/llama-b11791/libllama.so': {'bytes': 5156072, 'sha256': '4c054daf65e763f4563b1dc702141e6937ba2d4636a80ac13eb1f9f31a1c6731'}, 'runtime/llama-b11791/libllama.so.0': {'bytes': 5156072, 'sha256': '4c054daf65e763f4563b1dc702141e6937ba2d4636a80ac13eb1f9f31a1c6731'}, 'runtime/llama-b11791/libllama.so.0.7.0': {'bytes': 5156072, 'sha256': '4c054daf65e763f4563b1dc702141e6937ba2d4636a80ac13eb1f9f31a1c6731'}, 'runtime/llama-b11791/libmtmd.so': {'bytes': 1997136, 'sha256': 'f1a081d4a53dba5f4e01e5c05471cdf0770809fddeb4fadcf0e8709764eeb00a'}, 'runtime/llama-b11791/libmtmd.so.0': {'bytes': 1997136, 'sha256': 'f1a081d4a53dba5f4e01e5c05471cdf0770809fddeb4fadcf0e8709764eeb00a'}, 'runtime/llama-b11791/libmtmd.so.0.7.0': {'bytes': 1997136, 'sha256': 'f1a081d4a53dba5f4e01e5c05471cdf0770809fddeb4fadcf0e8709764eeb00a'}, 'seeded_pair_llama_bridge': {'bytes': 50976, 'sha256': '7b37da550a10d8f0c457e838e574bcf8c97159e9717ba88bd536e7e208eefe77'}}
SYSTEM_PATHS = frozenset(['/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2', '/usr/lib/x86_64-linux-gnu/libc.so.6', '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1', '/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0', '/usr/lib/x86_64-linux-gnu/libm.so.6', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33'])
SPECIAL_IDS = [128247, 151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656, 151659, 151660, 151661, 151662, 151663, 151664]



def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


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
        "runtime_identity_schema": "runtime-identity-compat-v1", "module_identities": {"math": math_identity}}


def exact_digest(value):
    if type(value) is not str or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ContractError("exact public SHA256 required")


def relative_path(value):
    if type(value) is not str or not value or "\\" in value:
        raise ContractError("safe relative resource path required")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or str(p) != value:
        raise ContractError("unsafe public resource path")
    return p


def safe_file(base, relative):
    base = Path(base).resolve(strict=True)
    rel = relative_path(relative)
    p = base
    for part in rel.parts:
        p = p / part
        if p.is_symlink(): raise ContractError("symlinked resource refused")
    if not p.is_file() or base not in p.resolve(strict=True).parents:
        raise ContractError("resource is not an owned regular file")
    return p


def validate_profile(profile):
    if type(profile) is not dict or set(profile) != PROFILE_FIELDS or profile["schema"] != PROFILE_SCHEMA:
        raise ContractError("strict message-independent native profile required")
    files = profile["resource_files"]
    if type(files) is not dict or files != RESOURCE_IDENTITIES:
        raise ContractError("exact41 public resource identities required")
    for name, item in files.items():
        relative_path(name)
        if type(item) is not dict or set(item) != {"sha256", "bytes"}:
            raise ContractError("strict resource identity required")
        exact_digest(item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise ContractError("positive resource size required")
    if not {BRIDGE, MODEL, CPU}.issubset(files):
        raise ContractError("required bridge/model/CPU missing")
    q = profile["qwen_binding"]
    if type(q) is not dict or set(q) != QWEN_FIELDS:
        raise ContractError("strict Qwen binding required")
    if (type(q["parameters"]) is not dict or q["parameters"] != PARAMETERS or
        any(type(q["parameters"].get(k)) is not type(v) for k,v in PARAMETERS.items()) or
        q["model_sha256"] != MODEL_SHA256 or q["cpu_backend_sha256"] != CPU_SHA256 or
        q["runtime_commit"] != RUNTIME_COMMIT):
        raise ContractError("pinned Qwen algorithms/resources differ")
    for key in QWEN_FIELDS - {"parameters","shared_libraries_sha256","runtime_commit","vocabulary_size","masked_tokens"}:
        exact_digest(q[key])
    ready = profile["ready"]
    if (type(ready) is not dict or set(ready) != {"vocabulary_size","masked_tokens","context_tokens"} or
        any(type(v) is not int for v in ready.values()) or ready["context_tokens"] != 2048 or
        ready["vocabulary_size"] != 151936 or ready["masked_tokens"] != 292 or
        q["vocabulary_size"] != ready["vocabulary_size"] or q["masked_tokens"] != ready["masked_tokens"]):
        raise ContractError("strict pinned READY metadata required")
    ids = profile["special_ids"]
    if (type(ids) is not list or ids != SPECIAL_IDS or
        any(type(i) is not int or not 0 <= i < ready["vocabulary_size"] for i in ids) or
        sorted(set(ids)) != ids or public_digest(ids) != q["special_ids_sha256"]):
        raise ContractError("public special-ID identity differs")
    expected_libraries = {n:v["sha256"] for n,v in files.items() if n.startswith("runtime/")}
    if q["shared_libraries_sha256"] != expected_libraries:
        raise ContractError("Qwen library binding differs")
    if any(files[n]["sha256"] != q[k] for n,k in ((BRIDGE,"bridge_sha256"),(MODEL,"model_sha256"),(CPU,"cpu_backend_sha256"))):
        raise ContractError("model/bridge/CPU identity differs")
    runtime = profile["runtime"]
    if type(runtime) is not dict or set(runtime) != RUNTIME_FIELDS or public_digest(runtime) != q["python_runtime_sha256"]:
        raise ContractError("public runtime fingerprint differs")
    system = profile["system_libraries_sha256"]
    if type(system) is not dict or set(system) != SYSTEM_PATHS:
        raise ContractError("public native system identities required")
    for name, sha in system.items():
        if type(name) is not str or not Path(name).is_absolute():
            raise ContractError("absolute public system-library path required")
        exact_digest(sha)
    canonical_json(profile)
    return profile


def resource_descriptor(profile):
    validate_profile(profile)
    digest = public_digest(profile)
    return {"resource_id": RESOURCE_ID, "resource_sha256": digest,
        "provider_sha256": public_digest({"resources":digest,"law":"old-masked-top50-f32-temp0.8-quantize-2^24-canonical-v2"}),
        "codec_sha256": public_digest({"resources":digest,"special_ids":profile["special_ids"],
            "encode":"TOKENIZE add_special=0 parse_special=0","render":"TEXT all-output-IDs"}),
        "context_tokens":2048}


def verify_files(root, profile):
    """Fresh bytes/runtime verification only; no activation/provenance file access."""
    validate_profile(profile)
    root = Path(root).resolve(strict=True)
    base = root / RESOURCE_ROOT
    verified = {}
    for name,item in sorted(profile["resource_files"].items()):
        p = safe_file(base,name)
        if p.stat().st_size != item["bytes"] or sha_file(p) != item["sha256"]:
            raise ContractError("public resource bytes differ: "+name)
        verified[name] = item
    if not os.access(safe_file(base, BRIDGE), os.X_OK):
        raise ContractError("bridge executable bit missing")
    if current_runtime() != profile["runtime"]:
        raise ContractError("runtime differs from static public profile")
    if sha_file(RUNTIME_SOURCE) != profile["qwen_binding"]["quantizer_sha256"]:
        raise ContractError("quantizer source differs")
    for name,digest in profile["system_libraries_sha256"].items():
        if sha_file(name) != digest: raise ContractError("system-library bytes differ")
    return {"schema":VERSION+"-native-resource-check","verified_files":len(verified),
        "public_profile_sha256":public_digest(profile),"model_calls":0,"tokenizer_calls":0,
        "scope":"current files/runtime only; no historical or resident attestation"}


class PublicOwnedBridge(NativeBridge):
    """Capture owned child before READY; metadata is a static resource identity."""
    def __init__(self,binary,model,cpu_backend,log_path,deadline,*,expected_sha256,profile_sha256):
        self.ticket = None
        self._capture_attempted = False
        self._terminal = False
        self._close_complete = False
        self._expected = Path(binary)
        self._expected_sha256 = expected_sha256
        self._profile_sha256 = profile_sha256
        self._capture_path = Path(log_path).parent/"proc-identity-capture.json"
        try: super().__init__(binary,model,cpu_backend,log_path,deadline)
        except BaseException:
            self.close()
            raise

    def _line(self):
        if self._terminal: raise ContractError("terminal native bridge; no retry")
        if not self._capture_attempted:
            self._capture_attempted = True
            self._terminal = True
            self.ticket = capture_owned_child(self.process,self._expected,self._expected_sha256)
            atomic_json(self._capture_path,{"schema":VERSION+"-owned-process",
                "popen_pid":self.process.pid,"parent":self.ticket.parent,"child":self.ticket.child,
                "executable":self.ticket.expected_executable,"public_profile_sha256":self._profile_sha256})
            self._terminal = False
        try: return super()._line()
        except BaseException:
            self._terminal = True
            raise

    def _guard_protocol(self, call):
        if self._terminal: raise ContractError("terminal native bridge; no retry")
        try: return call()
        except BaseException:
            self._terminal=True
            raise

    def command(self, line):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).command(line))

    def tokenize(self, text, add_special=False, parse_special=False):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).tokenize(text,add_special,parse_special))

    def render(self, ids):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).render(ids))

    def start(self, ids):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).start(ids))

    def step(self, token):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).step(token))

    def distribution(self):
        return self._guard_protocol(lambda: super(PublicOwnedBridge,self).distribution())

    def close(self):
        if self._close_complete: return
        self._terminal = True
        original_error = sys.exception()
        errors=[]
        try: super().close()
        except Exception as exc: errors.append(exc)
        process=getattr(self,"process",None)
        for stream in (getattr(process,"stdin",None),getattr(process,"stdout",None),getattr(self,"log",None)):
            if stream is not None:
                try: stream.close()
                except Exception as exc: errors.append(exc)
        if self.ticket is not None:
            try:self.ticket.close()
            except Exception as exc:errors.append(exc)
        self._close_complete = not errors
        if errors:
            detail="; ".join(type(e).__name__+": "+str(e)[:256] for e in errors)
            if original_error is not None: original_error.add_note("native cleanup: "+detail)
            else: raise RuntimeError("native cleanup failed: "+detail)


def verify_resident(root, profile, bridge):
    ticket=bridge.ticket
    if ticket is None or ticket.process is not bridge.process:
        raise ContractError("owned child identity required")
    maps,identity=ticket.maps_snapshot()
    base=(Path(root)/RESOURCE_ROOT).resolve(strict=True)
    observed,native={},set()
    for line in maps.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or not fields[5].startswith("/"):continue
        if fields[5].endswith(" (deleted)"):raise ContractError("deleted mapped file")
        p=Path(fields[5]).resolve(strict=True); under=base in p.parents
        if not under and "x" not in fields[1]:continue
        if under:
            relative=p.relative_to(base).as_posix()
            if relative not in profile["resource_files"]:raise ContractError("unregistered mapped resource")
            if "libggml-cpu-" in p.name and p.name!="libggml-cpu-haswell.so":
                raise ContractError("unexpected mapped CPU backend")
            expected=profile["resource_files"][relative]["sha256"]
            if "x" in fields[1]:native.add(relative)
        else:
            expected=profile["system_libraries_sha256"].get(str(p))
            if expected is None:raise ContractError("unregistered mapped system library")
        if str(p) not in observed and sha_file(p)!=expected:raise ContractError("mapped bytes differ")
        observed[str(p)]=expected
    if not {BRIDGE,CPU}.issubset(native) or not any(Path(p).name.startswith("libllama.so") for p in native):
        raise ContractError("resident native evidence incomplete")
    if ticket.snapshot()!=identity["after"]:raise ContractError("owned process changed during map verification")
    return {"status":"passed","pid":bridge.process.pid,"mapped_files_sha256":observed,
        "identity":identity,"scope":"owned live maps and current bytes; no hardware attestation"}


class PublicNativeService:
    kind="native"
    def __init__(self, root, profile, public_spec, public_context, directory, deadline):
        self.root=Path(root); self.profile=profile; self.spec=public_spec; self.context=public_context
        self.directory=Path(directory);self.deadline=deadline;self.bridge=None;self.process_attempted=False;self.prepare_attempted=False

    def prepare(self):
        if self.prepare_attempted: raise ContractError("native service is one-attempt; no retry")
        self.prepare_attempted=True
        validate_profile(self.profile)
        validate_public_spec(self.spec)
        ids=validate_public_context(self.context,self.spec)
        if self.spec["resources"]!=resource_descriptor(self.profile):
            raise ContractError("spec does not bind this complete public profile")
        if any(t>=self.profile["ready"]["vocabulary_size"] for t in ids):
            raise ContractError("public context has out-of-vocabulary ID")
        resource_check=verify_files(self.root,self.profile)
        atomic_json(self.directory/"native-resource-verification.json",resource_check)
        if time.monotonic()>=self.deadline:raise TimeoutError("prelaunch deadline reached")
        base=self.root/RESOURCE_ROOT
        self.process_attempted=True
        self.bridge=PublicOwnedBridge(base/BRIDGE,base/MODEL,base/CPU,self.directory/"native.stderr.log",self.deadline,
            expected_sha256=self.profile["resource_files"][BRIDGE]["sha256"],profile_sha256=public_digest(self.profile))
        ready={"vocabulary_size":self.bridge.vocabulary_size,"masked_tokens":self.bridge.masked_tokens,
            "context_tokens":self.bridge.context_tokens}
        if ready!=self.profile["ready"]:raise ContractError("actual READY differs")
        resident=verify_resident(self.root,self.profile,self.bridge)
        old=prepare_qwen_endpoint(self.bridge,self.profile["qwen_binding"],special_ids=frozenset(self.profile["special_ids"]))
        desc=resource_descriptor(self.profile)
        codec=replace(old.gate.codec,resource_digest=desc["codec_sha256"])
        endpoint=Endpoint(CanonicalGate(codec),resource_id=desc["resource_id"],resource_digest=desc["resource_sha256"],
            provider_digest=desc["provider_sha256"],distribution_callback=old._distribution,
            start_callback=old._start,observe_callback=old._observe,context_tokens=2048)
        validate_public_spec(self.spec,endpoint)
        atomic_json(self.directory/"native-preprompt-verification.json",{
            "schema":VERSION+"-native-preprompt","status":"passed","ready":ready,"resident":resident,
            "public_profile_sha256":public_digest(self.profile),"public_spec_sha256":public_digest(self.spec),
            "public_context_sha256":public_digest(self.context),"prompt_ids":list(ids),
            "context_token_count":len(ids),"model_process_loaded":True,"prompt_calls_so_far":0,
            "distribution_calls_so_far":0,"step_calls_so_far":0,"tokenizer_preflight_repeated":False})
        return endpoint

    def close(self):
        if self.bridge is not None:self.bridge.close()
