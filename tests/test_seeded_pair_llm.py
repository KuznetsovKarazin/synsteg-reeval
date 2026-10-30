import importlib.util
import io
import json
from pathlib import Path
import struct
import pytest
from synsteg_audit import seeded_pair_llm as c

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("seeded_llm_runner", ROOT / "scripts/run_seeded_pair_llm_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def bits(value):
    return struct.pack(">f", value).hex()


class FakeBridge:
    def __init__(self, *, collapse=False, timeout_after=None):
        self.generated = []
        self.collapse = collapse
        self.timeout_after = timeout_after
    def tokenize(self, text, add_special=False, parse_special=False):
        if add_special:
            assert parse_special
            return [900]
        ids = list(map(ord, text))
        return ids[:-1] if self.collapse and ids else ids
    def render(self, ids):
        return bytes(ids)
    def start(self, ids):
        assert ids == [900]
        self.generated = []
    def distribution(self):
        # Truly history-dependent finite fixture, with enough entropy.
        bonus = self.generated[-1] % 4 if self.generated else 0
        return [(65+i, bits(.25 if i == bonus else 0.)) for i in range(4)]
    def step(self, token):
        self.generated.append(token)
        if self.timeout_after == len(self.generated):
            raise TimeoutError("synthetic timeout")


def fixture(arm="stego"):
    p = json.loads((ROOT / "experiments/seeded_pair_llm_v1/protocol.json").read_text())
    e = {"version": c.VERSION, "prompt_id": "station", "arm": arm, "nonce_hex": "00"*16,
         "units": 512, "protocol_sha256": "11"*32, "resources_sha256": "22"*32}
    return p, e


def test_integer_quantization_and_exact_float32_ties():
    logits = [(i, bits(0.)) for i in range(50, 0, -1)]
    out = c.quantize(logits)
    assert [x["token_id"] for x in out] == list(range(1, 51))
    assert sum(x["weight"] for x in out) == 2**24
    base, residual = divmod(2**24, 50)
    assert [x["weight"] for x in out] == [base+1]*residual+[base]*(50-residual)
    with pytest.raises(ValueError):
        c.quantize([(1, "7f800000")])
    with pytest.raises(ValueError):
        c.quantize([(1, bits(0)), (1, bits(1))])


def test_full_frame_text_only_receiver_and_separate_traces():
    p, e = fixture()
    sender_trace, receiver_trace = io.StringIO(), io.StringIO()
    payload = bytes(range(16))
    sent = c.generate(FakeBridge(), p, e, c.DEMO_KEY, payload, trace=sender_trace)
    got = c.receive(FakeBridge(), p, e, c.DEMO_KEY, sent["stego_text"], trace=receiver_trace)
    assert sent["generated_tokens"] == 512
    assert sent["frame_bits"] == 272
    assert sent["text_roundtrip_exact"] and sent["frame_completed"]
    assert got["status"] == "ok" and got["payload_hex"] == payload.hex()
    s = [json.loads(x) for x in sender_trace.getvalue().splitlines()]
    r = [json.loads(x) for x in receiver_trace.getvalue().splitlines()]
    assert len(s) == len(r) == 512
    assert [x["weights_sha256"] for x in s] == [x["weights_sha256"] for x in r]
    assert [x["prefix_sha256"] for x in s] == [x["prefix_sha256"] for x in r]
    assert sent["prompt_token_ids"] == got["prompt_token_ids"] == [900]


def test_wrong_key_and_envelope_authentication():
    p, e = fixture()
    sent = c.generate(FakeBridge(), p, e, c.DEMO_KEY, b"0123456789abcdef")
    got = c.receive(FakeBridge(), p, e, bytes([5])*32, sent["stego_text"])
    assert got["status"] != "ok"
    framed = c.frame_bits(b"test", c.DEMO_KEY, e)
    assert c.read_frame(framed, c.DEMO_KEY, {**e, "nonce_hex": "01"*16})["status"] == "authentication_failed"


def test_noncanonical_text_is_retained_and_receiver_ids_forbidden():
    p, e = fixture()
    sent = c.generate(FakeBridge(collapse=True), p, e, c.DEMO_KEY, b"0123456789abcdef")
    assert sent["status"] == "tokenization_mismatch" and len(sent["stego_text"]) == 512
    got = c.receive(FakeBridge(collapse=True), p, e, c.DEMO_KEY, sent["stego_text"])
    assert got["status"] == "token_count_mismatch"
    with pytest.raises(ValueError):
        runner.validate_receiver_input({"envelope": e, "key_hex": c.DEMO_KEY.hex(),
                                        "stego_text": sent["stego_text"], "token_ids": sent["generated_token_ids"]})


def test_fixed_budget_control_and_collision_refusal():
    p, e = fixture("ordinary")
    ordinary = c.generate(FakeBridge(), p, e, c.DEMO_KEY)
    assert ordinary["generated_tokens"] == 512 and ordinary["sender_categorical_draws"] == 512
    assert ordinary["frame_bits"] is None
    class Constant(FakeBridge):
        def distribution(self):
            return [(65, bits(0))]
    p, e = fixture()
    sent = c.generate(Constant(), p, e, c.DEMO_KEY, bytes(16))
    assert sent["generated_tokens"] == 512 and sent["capacity_bits"] == 0
    assert sent["status"] == "insufficient_capacity"
    assert c.receive(Constant(), p, e, c.DEMO_KEY, sent["stego_text"])["status"] == "insufficient_capacity"


def test_partial_output_and_trace_survive_endpoint_error():
    p, e = fixture()
    trace, raw = io.StringIO(), io.BytesIO()
    with pytest.raises(TimeoutError):
        c.generate(FakeBridge(timeout_after=4), p, e, c.DEMO_KEY, bytes(16),
                   trace=trace, raw_stream=raw)
    assert len(trace.getvalue().splitlines()) == len(raw.getvalue()) == 4


def test_invalid_utf8_is_failure_not_replacement_decoding():
    class InvalidUTF8(FakeBridge):
        def render(self, ids):
            return b"\xff"
    p, e = fixture()
    sent = c.generate(InvalidUTF8(), p, e, c.DEMO_KEY, bytes(16))
    assert sent["status"] == "invalid_utf8" and sent["stego_text"] is None
    assert sent["generated_tokens"] == 512


def test_frozen_plan_exact_four_rows_and_payload():
    p, _ = fixture()
    assert p["prompt_ids"] == ["station", "library"]
    assert p["arms"] == ["stego", "ordinary"]
    assert p["output_tokens"] == 512 and p["payload_bytes"] == 16
    assert p["frame_bits"] == p["payload_bytes"]*8 + p["frame_overhead_bits"] == 272
    assert p["expected_generated_tokens"] == 2*2*512
    assert p["runtime"]["forced_cpu_backend"] == "libggml-cpu-haswell.so"


def test_prepare_sha_streams_complete_file(tmp_path):
    import hashlib
    data = b"x" * (1024 * 1024 + 1)
    path = tmp_path / "resource.bin"
    path.write_bytes(data)
    assert runner.sha(path) == hashlib.sha256(data).hexdigest()
    path.write_bytes(b"")
    assert runner.sha(path) == hashlib.sha256(b"").hexdigest()
