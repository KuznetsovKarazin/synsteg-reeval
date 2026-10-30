"""Synthetic unit tests with explicitly public disposable test keys/nonces.

No predeclared production matrix or fresh-process diagnostic is executed here.
"""
from dataclasses import replace
import copy
import hashlib

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from synsteg_audit.canonical_channel_v2 import ContractError, Prefix
from synsteg_audit.framed_interval_packet_v1 import (
    PublicBundle, _aad, _input_bits, _law, decode, derive_keys, encode,
    make_envelope, open_frame, seal_frame, validate_envelope,
)
from synsteg_audit.framed_interval_packet_v1_fixtures import PROMPT_IDS, public_bundle
from synsteg_audit.integer_interval_channel_v1 import Config, Receiver

PUBLIC_TEST_KEY = bytes(range(32))
PUBLIC_TEST_NONCE = bytes(range(12))


def setup(profile="silent16", *, maximum=32, config=None, max_wire=65536):
    bundle = public_bundle(profile)
    envelope = make_envelope(bundle, PUBLIC_TEST_NONCE,
                             config=config or Config(16, 512, 512, 8192),
                             prompt_ids=PROMPT_IDS, max_payload_bytes=maximum,
                             max_wire_bytes=max_wire)
    return bundle, envelope


@pytest.mark.parametrize("payload", [b"", b"x", bytes(range(16)), bytes(range(32))])
def test_fixed_slot_roundtrip(payload):
    bundle, envelope = setup()
    sent = encode(payload, PUBLIC_TEST_KEY, envelope, bundle)
    received = decode(sent.wire, PUBLIC_TEST_KEY, envelope, public_bundle("silent16"))
    assert sent.status == "frame_resolved_pending_receiver"
    assert received.status == "authenticated_delivery"
    assert received.payload == payload
    assert sent.required_frame_bits == received.required_frame_bits == 400
    assert sent.packet_count == received.packet_count == len(sent.wire) == 100
    assert sent.resolved_bits == received.resolved_bits == 400
    assert sent.traces == received.traces


def test_adaptive_public_law_reconstructed_without_sender_trace():
    bundle, envelope = setup("adaptive4", maximum=16)
    sent = encode(b"prefix-dependent", PUBLIC_TEST_KEY, envelope, bundle)
    received = decode(sent.wire, PUBLIC_TEST_KEY, copy.deepcopy(envelope), public_bundle("adaptive4"))
    assert sent.status == "frame_resolved_pending_receiver"
    assert received.status == "authenticated_delivery"
    assert received.payload == b"prefix-dependent"
    assert sent.traces == received.traces
    assert len({tuple(tuple(x) for x in t["ordered_integer_law"]) for t in sent.traces}) > 1


def test_ciphertext_length_is_fixed_and_plain_length_is_encrypted():
    _, envelope = setup()
    left = seal_frame(b"a", PUBLIC_TEST_KEY, envelope)
    second_envelope = {**envelope, "nonce_hex": "01" * 12}
    right = seal_frame(b"ab", PUBLIC_TEST_KEY, second_envelope)
    assert len(left) == len(right) == 50
    assert open_frame(left, PUBLIC_TEST_KEY, envelope) == b"a"
    assert open_frame(right, PUBLIC_TEST_KEY, second_envelope) == b"ab"


def test_aead_rejects_ciphertext_key_nonce_and_aad_changes():
    _, envelope = setup(maximum=1)
    frame = seal_frame(b"a", PUBLIC_TEST_KEY, envelope)
    mutations = [
        (bytes([frame[0] ^ 1]) + frame[1:], PUBLIC_TEST_KEY, envelope),
        (frame, bytes(reversed(PUBLIC_TEST_KEY)), envelope),
        (frame, PUBLIC_TEST_KEY, {**envelope, "nonce_hex": "ff" * 12}),
        (frame, PUBLIC_TEST_KEY, {**envelope, "max_wire_bytes": 65535}),
    ]
    for ciphertext, key, descriptor in mutations:
        with pytest.raises(InvalidTag):
            open_frame(ciphertext, key, descriptor)


def test_authenticated_malformed_internal_length_and_padding_refused():
    _, envelope = setup(maximum=2)
    enc_key, _, _ = derive_keys(PUBLIC_TEST_KEY, envelope)
    for plaintext in (b"\x00\x03ab", b"\x00\x01ab"):
        frame = ChaCha20Poly1305(enc_key).encrypt(PUBLIC_TEST_NONCE, plaintext, _aad(envelope))
        with pytest.raises(ContractError):
            open_frame(frame, PUBLIC_TEST_KEY, envelope)


def test_wrong_key_and_valid_envelope_changes_never_return_plaintext():
    bundle, envelope = setup(maximum=1)
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    wrong_key = decode(sent.wire, bytes(reversed(PUBLIC_TEST_KEY)), envelope, bundle)
    wrong_nonce = decode(sent.wire, PUBLIC_TEST_KEY, {**envelope, "nonce_hex": "ff" * 12}, bundle)
    wrong_aad = decode(sent.wire, PUBLIC_TEST_KEY, {**envelope, "max_wire_bytes": 65535}, bundle)
    assert all(result.status != "authenticated_delivery" and result.payload is None
               for result in (wrong_key, wrong_nonce, wrong_aad))


def test_tamper_truncation_and_append_never_return_plaintext():
    bundle, envelope = setup(maximum=1)
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    altered = bytes([65 + (sent.wire[0] - 65 + 1) % 16]) + sent.wire[1:]
    for wire in (altered, sent.wire[:-1], sent.wire + b"A", sent.wire + b"!", b"\xff"):
        received = decode(wire, PUBLIC_TEST_KEY, envelope, bundle)
        assert received.status != "authenticated_delivery"
        assert received.payload is None
    assert decode(sent.wire + b"A", PUBLIC_TEST_KEY, envelope, bundle).reason == "trailing_after_frame"


def test_same_authenticated_frame_different_excess_tail_wire_is_rejected():
    base = public_bundle("silent16")
    bundle = PublicBundle(base.gate, lambda _: tuple(((0, i), 1) for i in range(1, 9)),
                          "unit-uniform8", base.resource_digest, hashlib.sha256(b"unit-uniform8").hexdigest())
    envelope = make_envelope(bundle, PUBLIC_TEST_NONCE, config=Config(16, 128, 128, 4096),
                             prompt_ids=PROMPT_IDS, max_payload_bytes=1)
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    assert sent.required_frame_bits == 152 and sent.resolved_bits == 153
    assert sent.packet_count == 51
    final = sent.traces[-1]
    rows = final["ordered_integer_law"]
    rank = next(i for i, row in enumerate(rows) if row[0] == final["symbol_id"])
    alternative_symbol = rows[rank ^ 1][0]
    # Three resolved bits in the final packet; adjacent rank changes only bit153.
    alternative_packet = (0, alternative_symbol + 1)
    altered_wire = sent.wire[:-1] + base.gate.codec.render(alternative_packet)
    received = decode(altered_wire, PUBLIC_TEST_KEY, envelope, bundle)
    assert received.status == "invalid_wire"
    assert received.reason == "noncanonical_embedding"
    assert received.payload is None


@pytest.mark.parametrize("profile,reason", [("singleton", "max_symbols"), ("empty_support", "empty_support"),
                                           ("empty_input", "empty_input_distribution")])
def test_zero_capacity_and_empty_support_are_retained(profile, reason):
    bundle, envelope = setup(profile, maximum=1, config=Config(16, 12, 12, 512))
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    received = decode(sent.wire, PUBLIC_TEST_KEY, envelope, bundle)
    assert sent.status == "insufficient_capacity" and sent.reason == reason
    assert sent.resolved_bits == 0
    assert received.status == "insufficient_capacity"
    assert received.payload is None
    assert sent.packet_count == (12 if profile == "singleton" else 0)


def test_construction_refusal_retained():
    bundle, envelope = setup("construction_refusal")
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    assert sent.status == "construction_refusal"
    assert sent.wire == b"" and sent.packet_count == 0


def test_symbol_stream_and_wire_budgets_account_for_transmitted_prefix_only():
    for config, expected_packets, expected_bits, reason in (
        (Config(16, 2, 20, 512), 2, 8, "max_symbols"),
        (Config(16, 20, 20, 16), 0, 0, "max_stream_bits"),
    ):
        bundle, envelope = setup(maximum=1, config=config)
        sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
        assert sent.status == "insufficient_capacity" and sent.reason == reason
        assert sent.packet_count == expected_packets and sent.resolved_bits == expected_bits
        assert len(sent.wire) == expected_packets
    bundle, envelope = setup(maximum=1, max_wire=1)
    sent = encode(b"a", PUBLIC_TEST_KEY, envelope, bundle)
    receiver = decode(sent.wire, PUBLIC_TEST_KEY, envelope, bundle)
    assert sent.status == "wire_limit_exceeded"
    assert sent.packet_count == receiver.packet_count == 1
    assert sent.resolved_bits == receiver.resolved_bits == 4
    assert sent.consumed_bits == 20
    assert sent.traces[-1]["transmitted"] is False


def test_tail_is_explicit_and_subkeys_are_distinct():
    _, envelope = setup(maximum=1)
    config = Config(**envelope["interval_config"])
    keys = derive_keys(PUBLIC_TEST_KEY, envelope)
    assert len(set(keys)) == 3
    frame = seal_frame(b"a", PUBLIC_TEST_KEY, envelope)
    bits = _input_bits(frame, keys[2], envelope, config)
    assert len(bits) == config.max_stream_bits
    assert bits[152:] != (0,) * (len(bits) - 152)
    assert bits == _input_bits(frame, keys[2], envelope, config)


def test_contract_rejects_bad_types_extra_fields_and_resource_mismatch():
    bundle, envelope = setup()
    for modified in ({**envelope, "nonce_hex": "0" * 23}, {**envelope, "max_payload_bytes": True},
                     {**envelope, "provider_sha256": "0" * 64}, {**envelope, "extra": 1},
                     {**envelope, "prompt_ids": [True]}, {**envelope, "max_payload_bytes": 65536}):
        with pytest.raises(ContractError):
            validate_envelope(modified, bundle)
    with pytest.raises(ContractError):
        encode(b"a", b"short", envelope, bundle)
    with pytest.raises(ContractError):
        encode(bytes(33), PUBLIC_TEST_KEY, envelope, bundle)


def test_runner_failure_accounting_with_fake_processes_only(tmp_path, monkeypatch):
    """A separate two-case stub plan; never executes the proposed 12-case matrix."""
    import importlib.util
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from synsteg_audit.framed_interval_packet_v1 import EncodeResult
    path = Path(__file__).resolve().parents[1] / "experiments/framed-interval-packet-v1/run_synthetic.py"
    spec = importlib.util.spec_from_file_location("unit_framed_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = tmp_path / "experiments/framed-interval-packet-v1"
    directory.mkdir(parents=True)
    variants = ["original", "wrong_key", "wrong_nonce", "wrong_aad",
                "tamper_first_or_insert_if_empty", "truncate_last_byte", "append_A"]
    base = {"payload_hex": "61", "max_payload_bytes": 1, "max_wire_bytes": 16,
            "interval_config": {"state_bits": 16, "max_symbols": 64, "max_pending_bits": 64, "max_stream_bits": 512}}
    plan = {"public_test_key_hex": PUBLIC_TEST_KEY.hex(), "cryptography_version": "50.0.2",
            "receiver_variants": variants, "cases": [
                {**base, "id": "unit_sender_error", "profile": "unknown", "nonce_hex": "01" * 12},
                {**base, "id": "unit_fake_receiver_errors", "profile": "silent16", "nonce_hex": "02" * 12}]}
    (directory / "proposed-protocol.json").write_text(json.dumps(plan))
    (directory / "freeze-manifest.json").write_text(json.dumps({"files_sha256": {}}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.platform, "platform", lambda: "unit-platform-no-subprocess")
    monkeypatch.setattr(runner, "encode", lambda *args: EncodeResult("insufficient_capacity", "unit_stub",
                        b"A", 152, 4, 20, 0, 1, "0" * 64, ()))
    count = [0]
    def fake_process(*args, **kwargs):
        index = count[0]
        count[0] += 1
        if index == 0:
            raise subprocess.TimeoutExpired("unit", 30, output=b'{"partial"', stderr=b"unit timeout")
        if index == 1:
            return subprocess.CompletedProcess([], 0, b"not json", b"")
        if index == 2:
            raise OSError("unit spawn failure")
        if index == 4:
            return subprocess.CompletedProcess([], 1, b"", b"unit child failure")
        report = {"status": "insufficient_capacity", "reason": "unit_stub", "payload_hex": None,
                  "required_frame_bits": 152, "resolved_bits": 4, "packet_count": 1, "traces": []}
        if index == 6:
            report.update(status="authenticated_delivery", reason=None, payload_hex="61")
        payload = {"schema": "framed-interval-packet-v1-fresh-receiver", "pid": os.getpid() + 1,
                   "input_fields": ["envelope", "key_hex", "wire_hex"], "cryptography_version": "50.0.2",
                   "denied_events": ["unit_guard"] if index == 3 else [], "forbidden_modules_loaded": [], "result": report}
        return subprocess.CompletedProcess([], 0, json.dumps(payload).encode(), b"")
    monkeypatch.setattr(runner.subprocess, "run", fake_process)
    monkeypatch.setattr(sys, "argv", [str(path), "--freeze-commit", "1" * 40, "--output", "unit-output"])
    runner.main()
    summary = json.loads((tmp_path / "unit-output/summary.json").read_text())
    assert summary["case_denominator"] == 2 and summary["receiver_run_denominator"] == 14
    assert summary["receiver_not_attempted"] == 7 and summary["receiver_process_attempts"] == 7
    assert summary["receiver_processes_spawned"] == 6 and summary["validated_receiver_reports"] == 2
    assert summary["unassessed_negative_variants"] == 10
    assert summary["verified_negative_rejections_all_cases"] == 1
    assert summary["unexpected_negative_authentication_all_cases"] == 1
    assert count[0] == 7
