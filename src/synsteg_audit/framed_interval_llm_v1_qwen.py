"""Binding for an EXPLICITLY supplied, already started old NativeBridge.

Importing this module does not create a process, compile, load a model, read a
model file, or perform inference. Manifest values are declarations from an
external verified restoration; this binding cannot attest resident binaries.
There is intentionally no model-launch function in this module.
"""
import hashlib
from pathlib import Path

from . import seeded_pair_llm
from .canonical_channel_v2 import CanonicalGate, ContractError, PublicCodec
from .framed_interval_llm_v1 import Distribution, Endpoint, canonical_json

MODEL_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
RUNTIME_COMMIT = "ea9e3446b5c84a7f972cbbdea9df615fc4ff9159"
CPU_SHA256 = "1bc37cef126b01fe5a15222c07d20d29931936bfaa909a97052f6662f4c9d97c"
PARAMETERS = {"threads": 4, "context_tokens": 2048, "prefill_batch_tokens": 512,
              "micro_batch_tokens": 512, "gpu_layers": 0, "kv_types": "f16",
              "flash_attention": "disabled", "temperature": 0.8, "top_k": 50,
              "repetition_penalty": 1, "probability_integer_total": 1 << 24}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_qwen_endpoint(bridge, manifest: dict, *, special_ids: frozenset[int]):
    """Attach old line protocol without silently changing its masks or sampler.

    The caller must have verified all declared files/loaded backend identities
    and received coordinator clearance for any real model process beforehand.
    Tests may supply a fake bridge; that is not verification of real weights.
    """
    fields = {"model_sha256", "runtime_commit", "cpu_backend_sha256", "bridge_sha256",
              "shared_libraries_sha256", "python_runtime_sha256", "quantizer_sha256",
              "special_ids_sha256", "vocabulary_size", "masked_tokens", "parameters"}
    if type(manifest) is not dict or set(manifest) != fields:
        raise ContractError("require the complete declared old-backend manifest")
    if (type(manifest["parameters"]) is not dict
            or manifest["model_sha256"] != MODEL_SHA256 or manifest["runtime_commit"] != RUNTIME_COMMIT
            or manifest["cpu_backend_sha256"] != CPU_SHA256 or manifest["parameters"] != PARAMETERS
            or any(type(manifest["parameters"][k]) is not type(v) for k, v in PARAMETERS.items())):
        raise ContractError("old model/runtime/sampler declaration differs")
    from .framed_interval_llm_v1 import _digest
    for name in fields - {"runtime_commit", "shared_libraries_sha256", "vocabulary_size", "masked_tokens", "parameters"}:
        _digest(manifest[name])
    libraries = manifest["shared_libraries_sha256"]
    if type(libraries) is not dict or not libraries or any(type(name) is not str or not name for name in libraries):
        raise ContractError("declared native library fingerprints are required")
    for value in libraries.values(): _digest(value)
    if type(special_ids) is not frozenset or not special_ids or any(type(i) is not int or i < 0 for i in special_ids):
        raise ContractError("pinned output special-ID set required")
    if _sha(canonical_json(sorted(special_ids))) != manifest["special_ids_sha256"]:
        raise ContractError("special-ID declaration differs")
    if _sha(Path(seeded_pair_llm.__file__).read_bytes()) != manifest["quantizer_sha256"]:
        raise ContractError("frozen old quantizer source changed")
    if (type(manifest["vocabulary_size"]) is not int or manifest["vocabulary_size"] < 50
            or type(manifest["masked_tokens"]) is not int
            or not len(special_ids) <= manifest["masked_tokens"] < manifest["vocabulary_size"]
            or any(i >= manifest["vocabulary_size"] for i in special_ids)
            or bridge.vocabulary_size != manifest["vocabulary_size"]
            or bridge.masked_tokens != manifest["masked_tokens"] or bridge.context_tokens != 2048):
        raise ContractError("native READY metadata differs")
    resource_digest = _sha(canonical_json(manifest))
    codec_digest = _sha(canonical_json({"resources": resource_digest, "special_ids": sorted(special_ids),
        "encode": "TOKENIZE add_special=0 parse_special=0", "render": "TEXT all-output-IDs"}))
    def primitive(call):
        before = getattr(bridge, "evaluated", None)
        result = call()
        if getattr(bridge, "evaluated", None) != before:
            raise ContractError("tokenizer/render primitive mutated model step count")
        return result
    codec = PublicCodec(lambda text: primitive(lambda: bridge.tokenize(text, False, False)),
                        lambda ids: primitive(lambda: bridge.render(list(ids))), special_ids, codec_digest)
    def distribution(model_prefix):
        if bridge.evaluated != len(model_prefix):
            raise ContractError("native evaluated history differs from observed prefix")
        logits = bridge.distribution()
        if bridge.evaluated != len(model_prefix) or len(logits) != 50:
            raise ContractError("DIST changed history or top-50 contract")
        rows = seeded_pair_llm.quantize(logits, .8, 1 << 24)
        return Distribution(tuple((row["token_id"], row["weight"]) for row in rows),
            {"source": "old-masked-top50-logits", "float32_logits_and_integer_weights": rows,
             "base_weight_total": 1 << 24, "full_softmax_mask_mass": None,
             "full_softmax_topk_mass": None, "parameters": PARAMETERS})
    return Endpoint(CanonicalGate(codec), resource_id="qwen2.5-1.5b-q4km-b11791-fixed512",
        resource_digest=resource_digest,
        provider_digest=_sha(canonical_json({"resources": resource_digest, "law": "old-top50-then-v2-gate"})),
        distribution_callback=distribution, start_callback=bridge.start, observe_callback=bridge.step,
        context_tokens=2048)
