"""Bounded-v1 activation with a closed public receiver read scope.

No subprocess or model operation occurs here. The existing, pinned Qwen runtime
activation is validated read-only in receiver scope, then bound to the new
sixteen-case protocol. The sender fixture is opened only by the coordinator /
sender activation path, never by receiver scope.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import framed_interval_llm_v1_activation as prior
from .framed_interval_llm_runtime_compat_v1 import current_runtime
from .framed_interval_llm_v1 import canonical_json

VERSION = "framed-interval-bounded-v1"
INTENT = "execute-frozen-framed-interval-bounded-v1-sixteen-cases"
PUBLIC_PATH = "experiments/framed-interval-bounded-v1/public-protocol.json"
PUBLIC_SHA = "e6fa062b75c9bdc5f8a9887b97bf968493c96858735f11e8f05187ef8751eec2"
FIXTURES_PATH = "experiments/framed-interval-bounded-v1/sender-fixtures.json"
FIXTURES_SHA = "c54586084ad39c0fa745202254bb30f8be715b5035d5b49165c55af2ae295ffb"
ACTIVATION_PATH = "experiments/framed-interval-bounded-v1/activation-v1.json"
FREEZE_PATH = "results/framed-interval-bounded-v1-preparation/freeze-manifest.json"
READBACK_PATH = "results/framed-interval-bounded-v1-preparation/source-readback.json"
PRIOR_ACTIVATION_PATH = "experiments/framed-interval-llm-v1/activation-after-preflight-proc-v1.json"
PRIOR_ACTIVATION_SHA = "1978302778134592a7835c63be363932cc8b9c353500873b9e7c7aaadc57911b"
CONFIG_IDS = ("c00", "c01", "c02", "c03")
EXPECTED_RUNS = tuple(f"{prompt}-{config}-{arm}"
    for prompt in ("station", "library") for config in CONFIG_IDS
    for arm in ("stego", "ordinary"))
NEW_SOURCES = (
    "src/synsteg_audit/framed_interval_bounded_v1.py",
    "src/synsteg_audit/framed_interval_bounded_v1_activation.py",
    "src/synsteg_audit/framed_interval_bounded_v1_journal.py",
    "src/synsteg_audit/framed_interval_bounded_v1_orchestration.py",
    "src/synsteg_audit/framed_interval_bounded_v1_qwen.py",
    "experiments/framed-interval-bounded-v1/run_bounded.py",
)
ADAPTER_PINS = {
    "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py":
        "6ef38d3b3d88da78e3ed587f0582aa5f31d11fa26973260c841ef8843c4fd938",
    "src/synsteg_audit/framed_interval_llm_proc_service_v1.py":
        "7b6e2545fe5ca766ee3ef927302ad117a34ef71497b09f8904f6e69101b532da",
    "src/synsteg_audit/proc_namespace_identity_v1.py":
        "30ced033d3f5186e41f1b99720f2057287ec2ed6172483fc46750ad797aaea94",
    "src/synsteg_audit/proc_namespace_identity_v2.py":
        "ee5714058205b295c50fe8cb3cce499d1a3b6fa4527f333fd1205c5cf5c52a6d",
}
PUBLIC_SOURCES = frozenset((
    *[name for name in prior.REQUIRED_SOURCES if name != prior.PROPOSAL_PATH],
    prior.EXECUTION_PATH, PRIOR_ACTIVATION_PATH, PUBLIC_PATH,
    *NEW_SOURCES, *ADAPTER_PINS,
))
REQUIRED_SOURCES = PUBLIC_SOURCES | {FIXTURES_PATH}
PINNED = {
    PUBLIC_PATH: PUBLIC_SHA, FIXTURES_PATH: FIXTURES_SHA,
    PRIOR_ACTIVATION_PATH: PRIOR_ACTIVATION_SHA,
    "src/synsteg_audit/framed_interval_bounded_v1.py":
        "b5aeaed1ba6f07c8d1e795b41955d7b22baf2ae7146e122e4adb74ac381c7f83",
    **ADAPTER_PINS,
}
safe_path, checked_bytes, sha_file, digest = (
    prior.safe_path, prior.checked_bytes, prior.sha_file, prior.digest)


def expected_key(config_id):
    if config_id not in CONFIG_IDS:
        raise ValueError("configuration outside fixed four-key matrix")
    return hashlib.sha256(canonical_json([VERSION, "fixture-key", config_id])).hexdigest()


def _validate_protocol(protocol):
    if (protocol.get("schema") != VERSION + "-public-protocol-v1"
            or protocol.get("experiment_id") != VERSION
            or tuple(row["run_id"] for row in protocol["runs"]) != EXPECTED_RUNS
            or protocol["planned_cases"] != 16 or protocol["planned_fresh_stego_receivers"] != 8
            or protocol["planned_stego_cases"] != 8 or protocol["planned_ordinary_cases"] != 8
            or protocol["retries"] != 0 or protocol["per_endpoint_timeout_seconds"] != 1350
            or protocol["termination"]["resolved_bits_threshold"] != 272
            or protocol["termination"]["maximum_output_tokens"] != 768
            or protocol["interval_config"] != {"state_bits": 32, "max_symbols": 768,
                "max_pending_bits": 4096, "max_stream_bits": 65536}):
        raise ValueError("complete fixed bounded protocol differs")
    for run in protocol["runs"]:
        if set(run) != {"run_id", "prompt_id", "config_id", "arm", "nonce_hex", "ordinary_seed_hex"}:
            raise ValueError("public run contains private or unexpected fields")
        if run["run_id"] != f'{run["prompt_id"]}-{run["config_id"]}-{run["arm"]}':
            raise ValueError("run identity mismatch")
        expected_nonce = hashlib.sha256(canonical_json([
            VERSION, "nonce", run["config_id"], run["prompt_id"], run["arm"]])).digest()[:12].hex()
        expected_seed = hashlib.sha256(canonical_json([
            VERSION, "ordinary-seed", run["config_id"], run["prompt_id"]])).hexdigest()
        if run["nonce_hex"] != expected_nonce or run["ordinary_seed_hex"] != expected_seed:
            raise ValueError("public fixture nonce/seed differs")
    return protocol


@dataclass(frozen=True)
class Activation:
    root: Path
    sha256: str
    data: dict
    public_protocol: dict
    preflight: dict
    resource_report: dict
    receiver_scope: bool


def check_fixtures(activation):
    """Coordinator/sender-only file read, never required by fresh receiver."""
    if activation.receiver_scope:
        raise ValueError("receiver cannot open or hash sender fixtures")
    raw = checked_bytes(activation.root, activation.data["sender_fixtures"])
    obj = json.loads(raw)
    protocol = activation.public_protocol
    if (obj.get("schema") != VERSION + "-sender-fixtures-v1"
            or obj.get("experiment_id") != VERSION or obj.get("public_protocol_sha256") != PUBLIC_SHA
            or obj.get("case_count") != 16 or obj.get("stego_payload_count") != 8
            or obj.get("retries") != 0
            or tuple(row["run_id"] for row in obj["runs"]) != EXPECTED_RUNS
            or obj["configuration_keys"] != {key: expected_key(key) for key in CONFIG_IDS}):
        raise ValueError("complete fixed private fixture matrix differs")
    for run, public in zip(obj["runs"], protocol["runs"], strict=True):
        if set(run) != set(public) | {"key_hex", "payload_hex"}:
            raise ValueError("fixture fields differ")
        if any(run[name] != value for name, value in public.items()):
            raise ValueError("private/public run differs")
        if run["key_hex"] != expected_key(run["config_id"]):
            raise ValueError("private key fixture differs")
        payload = (hashlib.sha256(("seeded-pair-llm-v1:payload:" + run["prompt_id"]).encode()).digest()[:16].hex()
            if run["arm"] == "stego" else None)
        if run["payload_hex"] != payload:
            raise ValueError("private payload fixture differs")
    return obj


def _load_prior_runtime(root, descriptor):
    # Deliberately use receiver scope even for the new parent: the old private
    # proposal is not a dependency of this new method. This does not run a probe.
    return prior.load_activation(root, descriptor["path"], descriptor["sha256"], prior.INTENT,
        receiver_scope=True, runtime_reader=current_runtime)


def load_activation(root, activation_path, expected_sha, intent, *,
                    receiver_scope=False, runtime_loader=_load_prior_runtime):
    """Explicit SHA/intent gate. Injection is only for pure activation tests.

    The two evidence documents are coordinator assertions checked against actual
    local bytes. Their existence is not a cryptographic proof of a Git read.
    The coordinator must actually fetch and compare the frozen Git contents.
    """
    if intent != INTENT:
        raise ValueError("explicit frozen-sixteen-case execution intent required")
    if type(receiver_scope) is not bool:
        raise ValueError("receiver_scope must be a boolean")
    prior._hex(expected_sha)
    if activation_path != ACTIVATION_PATH:
        raise ValueError("bounded activation must use fixed public activation path")
    root = Path(root).resolve()
    raw = safe_path(root, activation_path).read_bytes()
    if digest(raw) != expected_sha:
        raise ValueError("activation differs from coordinator SHA")
    obj = json.loads(raw)
    fields = {"schema", "experiment_id", "enabled", "final_freeze_commit",
        "freeze_manifest", "source_readback", "public_protocol", "sender_fixtures",
        "expected_public_runs_sha256", "prior_runtime_activation"}
    if (type(obj) is not dict or set(obj) != fields
            or obj["schema"] != VERSION + "-activation-v1"
            or obj["experiment_id"] != VERSION or obj["enabled"] is not True):
        raise ValueError("complete enabled bounded activation required")
    prior._hex(obj["final_freeze_commit"], 40)
    for name, expected in (
        ("public_protocol", {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA}),
        ("sender_fixtures", {"path": FIXTURES_PATH, "sha256": FIXTURES_SHA}),
        ("prior_runtime_activation", {"path": PRIOR_ACTIVATION_PATH, "sha256": PRIOR_ACTIVATION_SHA}),
    ):
        if obj[name] != expected:
            raise ValueError("fixed " + name + " descriptor differs")
    for name, path in (("freeze_manifest", FREEZE_PATH), ("source_readback", READBACK_PATH)):
        descriptor = obj[name]
        if (type(descriptor) is not dict or set(descriptor) != {"path", "sha256"}
                or descriptor["path"] != path):
            raise ValueError("bounded public evidence descriptor path differs")
        prior._hex(descriptor["sha256"])
    freeze = json.loads(checked_bytes(root, obj["freeze_manifest"]))
    proof = json.loads(checked_bytes(root, obj["source_readback"]))
    if (type(freeze) is not dict or set(freeze) != {"schema", "commit", "files_sha256"}
            or freeze["schema"] != VERSION + "-source-freeze-v1"
            or freeze["commit"] != obj["final_freeze_commit"]
            or type(freeze["files_sha256"]) is not dict
            or set(freeze["files_sha256"]) != REQUIRED_SOURCES):
        raise ValueError("exact closed bounded source freeze required")
    if (type(proof) is not dict
            or set(proof) != {"schema", "commit", "all_files_exact", "files_sha256"}
            or proof["schema"] != VERSION + "-source-readback-v1"
            or proof["commit"] != obj["final_freeze_commit"]
            or proof["all_files_exact"] is not True
            or proof["files_sha256"] != freeze["files_sha256"]):
        raise ValueError("coordinator exact complete source Git readback required")
    # Validate the complete path allowlist and pins BEFORE opening mapped paths.
    for name, expected in freeze["files_sha256"].items():
        prior._hex(expected)
        if name in PINNED and PINNED[name] != expected:
            raise ValueError("frozen source identity differs: " + name)
    for name, expected in freeze["files_sha256"].items():
        if receiver_scope and name == FIXTURES_PATH:
            continue
        checked_bytes(root, {"path": name, "sha256": expected})
    protocol = _validate_protocol(json.loads(checked_bytes(root, obj["public_protocol"])))
    if digest(canonical_json(protocol["runs"])) != obj["expected_public_runs_sha256"]:
        raise ValueError("public full-run digest differs")
    old = runtime_loader(root, obj["prior_runtime_activation"])
    if (protocol["system_prompt"] != old.public_protocol["system_prompt"]
            or protocol["prompts"] != old.public_protocol["prompts"]):
        raise ValueError("bounded public prompts differ from validated tokenizer evidence")
    if (set(old.preflight["prompt_ids"]) != {"station", "library"}
            or any(len(ids) + 768 > old.data["ready"]["context_tokens"]
                for ids in old.preflight["prompt_ids"].values())):
        raise ValueError("prompt plus bounded cap exceeds validated context")
    data = {**old.data, **obj}
    activation = Activation(root, expected_sha, data, protocol, old.preflight,
                            old.resource_report, receiver_scope)
    if not receiver_scope:
        check_fixtures(activation)
    return activation
