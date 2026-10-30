"""Synthetic cross-implementation checks; these perform no neural inference."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "independent_llm_audit", ROOT / "scripts/audit_seeded_pair_llm_v1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

from synsteg_audit.seeded_pair_llm import generate, receive


class FakeBridge:
    def tokenize(self, text, add_special=False, parse_special=False):
        return [100, 101] if add_special else [ord(c) - 33 for c in text]

    def start(self, ids):
        assert ids == [100, 101]

    def distribution(self):
        return [(i, "00000000") for i in range(50)]

    def render(self, ids):
        return bytes(33 + i for i in ids)

    def step(self, token):
        assert 0 <= token < 50


def fixture():
    protocol = json.loads((ROOT / "experiments/seeded_pair_llm_v1/protocol.json").read_text())
    envelope = {"version": audit.VERSION, "prompt_id": "station", "arm": "stego",
                "protocol_sha256": audit.sha(protocol), "resources_sha256": "a" * 64,
                "nonce_hex": bytes(16).hex(), "units": 320}
    key = bytes(range(32))
    payload = __import__("hashlib").sha256(
        (audit.VERSION + ":payload:station").encode()).digest()[:16]
    trace = io.StringIO()
    sent = generate(FakeBridge(), protocol, envelope, key, payload, trace=trace)
    sent["prompt_token_ids"] = [100, 101]
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    return protocol, envelope, key, sent, rows


def test_reference_quantization_equal_tie_vector():
    rows = [{"token_id": i, "logit_hex": "00000000",
             "weight": 335545 if i < 16 else 335544} for i in range(50)]
    assert audit.independent_weights(rows) == rows
    rows[0]["weight"] -= 1
    with pytest.raises(AssertionError):
        audit.independent_weights(rows)


def test_independent_sender_receiver_and_wrong_key():
    protocol, envelope, key, sent, rows = fixture()
    checked = audit.audit_sender(rows, sent, protocol)
    assert checked["tokens"] == 320
    stream = io.StringIO()
    received = receive(FakeBridge(), protocol, envelope, key, sent["stego_text"], trace=stream)
    received["prompt_token_ids"] = [100, 101]
    receiver_rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert received["status"] == "ok"
    assert audit.audit_receiver(receiver_rows, received, envelope, protocol)["payload_hex"] == received["payload_hex"]
    assert audit.audit_receiver(receiver_rows, received, envelope, protocol,
                                bytes([7]) * 32)["status"] != "ok"
    assert audit.audit_sender(rows, sent, protocol)["capacity_bits"] >= 272


def test_tampered_trace_is_rejected():
    protocol, envelope, key, sent, rows = fixture()
    rows[17]["selected_token"] = (rows[17]["selected_token"] + 1) % 50
    with pytest.raises(AssertionError):
        audit.audit_sender(rows, sent, protocol)
