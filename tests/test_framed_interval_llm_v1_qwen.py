"""The Qwen binding is exercised only through a byte-based fake line bridge."""
import copy
import hashlib
from pathlib import Path
import struct

import pytest

from synsteg_audit import seeded_pair_llm
from synsteg_audit.canonical_channel_v2 import ContractError
from synsteg_audit.framed_interval_llm_v1 import canonical_json, generate, make_envelope, receive
from synsteg_audit.framed_interval_llm_v1_qwen import (
    CPU_SHA256, MODEL_SHA256, PARAMETERS, RUNTIME_COMMIT, prepare_qwen_endpoint,
)

PUBLIC_KEY = bytes(range(32))
SPECIALS = frozenset({999})


class FakeBridge:
    vocabulary_size, masked_tokens, context_tokens = 1024, 3, 2048

    def __init__(self):
        self.calls = []

    def tokenize(self, text, add_special, parse_special):
        self.calls.append(("tokenize", add_special, parse_special))
        assert add_special is False and parse_special is False
        return list(text.encode())

    def render(self, ids):
        self.calls.append(("render", len(ids)))
        return bytes(ids)

    def start(self, ids):
        self.calls.append(("start", tuple(ids)))
        self.evaluated = len(ids)

    def step(self, token):
        self.calls.append(("step", token))
        self.evaluated += 1

    def distribution(self):
        self.calls.append(("distribution", self.evaluated))
        return [(token, struct.pack(">f", (token % 7)*0.125).hex()) for token in range(65, 115)]


def manifest():
    # SHA declarations below intentionally describe no real restored model.
    # The binding's documentation explicitly says it cannot attest loaded files.
    return {"model_sha256": MODEL_SHA256, "runtime_commit": RUNTIME_COMMIT,
        "cpu_backend_sha256": CPU_SHA256, "bridge_sha256": "a"*64,
        "shared_libraries_sha256": {"fake-library": "b"*64}, "python_runtime_sha256": "c"*64,
        "quantizer_sha256": hashlib.sha256(Path(seeded_pair_llm.__file__).read_bytes()).hexdigest(),
        "special_ids_sha256": hashlib.sha256(canonical_json(sorted(SPECIALS))).hexdigest(),
        "vocabulary_size": 1024, "masked_tokens": 3, "parameters": copy.deepcopy(PARAMETERS)}


def test_preparation_uses_only_fake_tokenizer_primitives_without_begin_or_distribution():
    bridge = FakeBridge()
    prepare_qwen_endpoint(bridge, manifest(), special_ids=SPECIALS)
    assert bridge.calls and all(call[0] in {"tokenize", "render"} for call in bridge.calls)
    assert not hasattr(bridge, "evaluated")


def test_fake_native_top50_quantizer_and_exact_single_token_schedule():
    left = FakeBridge()
    endpoint = prepare_qwen_endpoint(left, manifest(), special_ids=SPECIALS)
    env = make_envelope(endpoint, bytes(range(12)), arm="stego", prompt_id="fake-qwen-unit",
        protocol_digest="d"*64, prompt_ids=(1000, 1001))
    sent = generate(PUBLIC_KEY, env, endpoint, payload=b"fake bridge")
    assert sent.status == "fixed512_pending_receiver" and left.evaluated == 514
    assert sum(call[0] == "step" for call in left.calls) == 512
    assert sum(call[0] == "distribution" for call in left.calls) == 512
    assert sum(call[0] == "start" for call in left.calls) == 1
    right = FakeBridge()
    result = receive(sent.wire, PUBLIC_KEY, env, prepare_qwen_endpoint(right, manifest(), special_ids=SPECIALS))
    assert result.status == "authenticated_delivery" and result.payload == b"fake bridge"
    assert result.traces == sent.traces and right.evaluated == 514
    assert all(t["admission"]["input_weight_total"] == 1 << 24 for t in sent.traces)
    assert sent.traces[0]["base_law_evidence"]["full_softmax_topk_mass"] is None


@pytest.mark.parametrize("change", [{"model_sha256": "0"*64}, {"cpu_backend_sha256": "0"*64},
    {"runtime_commit": "0"*40}, {"quantizer_sha256": "0"*64}, {"special_ids_sha256": "0"*64},
    {"shared_libraries_sha256": {}}, {"vocabulary_size": True}, {"masked_tokens": 2000}])
def test_invalid_old_resource_declarations_rejected_before_primitives(change):
    bridge = FakeBridge()
    with pytest.raises(ContractError): prepare_qwen_endpoint(bridge, {**manifest(), **change}, special_ids=SPECIALS)
    assert bridge.calls == []


def test_model_history_mutating_tokenizer_primitive_is_rejected():
    class BadBridge(FakeBridge):
        def tokenize(self, text, add_special, parse_special):
            self.evaluated = 1
            return super().tokenize(text, add_special, parse_special)
    with pytest.raises(ContractError, match="mutated"):
        prepare_qwen_endpoint(BadBridge(), manifest(), special_ids=SPECIALS)
