"""Fake-backend unit mechanics only; no real model, process launcher, or weights."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from synsteg_audit.canonical_channel_v2 import ContractError
from synsteg_audit.framed_interval_llm_v1 import (
    DEFAULT_CONFIG, FRAME_BITS, Distribution, _aad, derive_keys, generate, input_bits,
    make_envelope, open_frame, ordered_law, ordering_context_digest, ordinary_bits, receive,
    seal_frame, validate_envelope,
)
from synsteg_audit.framed_interval_llm_v1_fixtures import PROMPT_IDS, fake_endpoint
from synsteg_audit.integer_interval_channel_v1 import Config

KEY = bytes(range(32))  # Public unit-only fixture; no secret retrieval.
NONCE = bytes(range(12))
PROTOCOL = hashlib.sha256(b"fixed512 synthetic unit protocol, not LLM freeze").hexdigest()


def setup(profile="uniform16", arm="stego", **kwargs):
    endpoint = fake_endpoint(profile)
    env = make_envelope(endpoint, NONCE, arm=arm, prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS, **kwargs)
    return endpoint, env


@pytest.mark.parametrize("payload", [b"", b"A", bytes(range(16))])
def test_complete_fixed512_roundtrip_with_early_frame_and_long_tail(payload):
    sender, env = setup()
    sent = generate(KEY, env, sender, payload=payload)
    assert sent.status == "fixed512_pending_receiver" and not sent.authenticated_delivery
    assert sent.emitted_tokens == len(sent.wire) == 512
    assert sent.required_frame_bits == 272 and sent.first_frame_resolved_token == 68
    assert sent.resolved_bits == 2048 and sent.consumed_bits == 2080
    result = receive(sent.wire, KEY, copy.deepcopy(env), fake_endpoint())
    assert result.status == "authenticated_delivery" and result.payload == payload
    assert result.processed_tokens == result.wire_tokens == 512
    assert result.first_frame_resolved_token == 68 and result.authenticated_delivery
    assert result.traces == sent.traces


def test_prefix_adaptive_law_replayed_by_fresh_endpoint():
    sender, env = setup("adaptive4")
    sent = generate(KEY, env, sender, payload=b"adaptive laws")
    result = receive(sent.wire, KEY, env, fake_endpoint("adaptive4"))
    assert result.status == "authenticated_delivery" and result.payload == b"adaptive laws"
    assert result.traces == sent.traces
    assert len({tuple(tuple(row) for row in t["ordered_integer_law"]) for t in sent.traces}) > 1
    assert any(t["bin_law_vs_weights_TV"]["numerator"] > 0 for t in sent.traces)


def test_ordinary_uses_identical_integer_transform_but_never_claims_payload():
    endpoint, env = setup(arm="ordinary")
    sent = generate(KEY, env, endpoint)
    verified = receive(sent.wire, KEY, env, fake_endpoint())
    assert sent.status == "ordinary_generated" and sent.emitted_tokens == 512
    assert sent.required_frame_bits is None and sent.frame_sha256 is None
    assert sent.first_frame_resolved_token is None and not sent.authenticated_delivery
    assert verified.status == "ordinary_replay_verified" and verified.payload is None
    assert verified.traces == sent.traces and not verified.authenticated_delivery
    assert len(ordinary_bits(env, DEFAULT_CONFIG)) == DEFAULT_CONFIG.max_stream_bits


def test_fake_backend_prompt_and_observe_schedule_includes_final_token():
    starts, observations = [], []
    endpoint = fake_endpoint(start_callback=lambda ids: starts.append(ids), observe_callback=observations.append)
    env = make_envelope(endpoint, NONCE, arm="stego", prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS)
    sent = generate(KEY, env, endpoint, payload=b"schedule")
    assert starts == [PROMPT_IDS] and observations == list(sent.wire)
    assert len(observations) == 512
    with pytest.raises(ContractError, match="one-use"):
        endpoint.begin(PROMPT_IDS)


def test_fixed512_requires_tail_even_after_valid_authenticated_frame():
    endpoint, env = setup()
    sent = generate(KEY, env, endpoint, payload=b"hello")
    # The first68 bytes already resolve all272 ciphertext bits. It is still incomplete.
    shortened = receive(sent.wire[:68], KEY, env, fake_endpoint())
    assert shortened.status == "incomplete_wire" and shortened.payload is None
    assert shortened.resolved_bits == 272 and shortened.first_frame_resolved_token == 68
    # Change only the final tail token: frame remains identical and tag-valid,
    # but deterministic replay must reject all512-token embedding alteration.
    other = 65 + ((sent.wire[-1]-65+1) % 16)
    altered = receive(sent.wire[:-1]+bytes([other]), KEY, env, fake_endpoint())
    assert altered.status == "invalid_wire" and altered.reason == "noncanonical_fullwire_embedding"
    assert altered.payload is None and altered.processed_tokens == 512
    assert altered.traces[:68] == sent.traces[:68]


def test_change_only_extra_bit_of_first_complete_frame_symbol_is_rejected():
    endpoint, env = setup("uniform8")
    sent = generate(KEY, env, endpoint, payload=b"extra bit")
    assert sent.first_frame_resolved_token == 91  # 91*3=273, one bit beyond the272-bit frame.
    trace = sent.traces[90]
    tokens = [row[0] for row in trace["ordered_integer_law"]]
    rank = tokens.index(trace["observed_token"])
    replacement = tokens[rank ^ 1]  # Changes only bit273 in the first completed frame symbol.
    changed = sent.wire[:90]+bytes([replacement])+sent.wire[91:]
    result = receive(changed, KEY, env, fake_endpoint("uniform8"))
    assert result.status == "invalid_wire" and result.reason == "noncanonical_fullwire_embedding"
    assert result.payload is None and result.processed_tokens == 512


@pytest.mark.parametrize("mutator", [
    lambda wire: wire[:-1], lambda wire: wire+b"A", lambda wire: b"\xff"+wire[1:],
    lambda wire: bytes([65+(wire[0]-65+1)%16])+wire[1:],
])
def test_wire_changes_never_return_payload(mutator):
    endpoint, env = setup()
    sent = generate(KEY, env, endpoint, payload=b"protected")
    result = receive(mutator(sent.wire), KEY, env, fake_endpoint())
    assert not result.authenticated_delivery and result.payload is None


def test_key_nonce_or_valid_aad_change_never_returns_payload():
    endpoint, env = setup()
    sent = generate(KEY, env, endpoint, payload=b"protected")
    for key, descriptor in [(KEY[::-1], env), (KEY, {**env, "nonce_hex": "ff"*12}),
                            (KEY, {**env, "max_wire_bytes": 65535})]:
        result = receive(sent.wire, key, descriptor, fake_endpoint())
        assert not result.authenticated_delivery and result.payload is None


def test_aead_fixed_slot_length_and_zero_padding_validation():
    _, env = setup()
    for payload in [b"", b"x", bytes(16)]:
        frame = seal_frame(payload, KEY, env)
        assert len(frame) == 34 and open_frame(frame, KEY, env) == payload
    encryption, _, _ = derive_keys(KEY, env)
    for plain in [b"\x00\x11"+bytes(16), b"\x00\x01"+b"a"+bytes(14)+b"x"]:
        frame = ChaCha20Poly1305(encryption).encrypt(NONCE, plain, _aad(env))
        with pytest.raises(ContractError): open_frame(frame, KEY, env)
    frame = seal_frame(b"a", KEY, env)
    with pytest.raises(InvalidTag): open_frame(bytes([frame[0]^1])+frame[1:], KEY, env)


@pytest.mark.parametrize("profile,reason", [("empty_support", "empty_support"),
    ("empty_input", "empty_input_distribution"), ("partial_utf8", "empty_support")])
def test_all_support_refusals_preserved(profile, reason):
    endpoint, env = setup(profile)
    sent = generate(KEY, env, endpoint, payload=b"x")
    assert sent.status == "support_refusal" and sent.reason == reason
    assert sent.emitted_tokens == sent.resolved_bits == 0 and sent.wire == b""
    assert sent.traces[0]["transmitted"] is False


def test_singleton_zero_capacity_keeps_all512_emitted_tokens():
    endpoint, env = setup("singleton")
    sent = generate(KEY, env, endpoint, payload=b"x")
    result = receive(sent.wire, KEY, env, fake_endpoint("singleton"))
    assert sent.status == result.status == "insufficient_capacity"
    assert sent.emitted_tokens == result.processed_tokens == 512
    assert sent.resolved_bits == result.resolved_bits == 0 and result.payload is None
    assert sent.consumed_bits == 32 and sent.first_frame_resolved_token is None


def test_wire_budget_accounts_only_for_committed_output():
    endpoint, env = setup(max_wire_bytes=3)
    sent = generate(KEY, env, endpoint, payload=b"x")
    assert sent.status == "wire_limit_exceeded" and sent.emitted_tokens == len(sent.wire) == 3
    assert sent.resolved_bits == 12 and sent.consumed_bits == 44
    assert sent.traces[-1]["transmitted"] is False
    receiver = receive(sent.wire, KEY, env, fake_endpoint())
    assert receiver.processed_tokens == 3 and receiver.resolved_bits == 12 and receiver.payload is None


def test_read_limit_refuses_before_transmitted_bits_without_zero_fallback():
    endpoint, env = setup(config=Config(32, 512, 4096, 32))
    sent = generate(KEY, env, endpoint, payload=b"x")
    assert sent.status == "capacity_limit" and sent.reason == "max_stream_bits"
    assert sent.emitted_tokens == sent.resolved_bits == 0 and sent.consumed_bits == 32


def test_backend_observation_failure_retains_only_previous_committed_prefix():
    observations = []
    def observe(token):
        if len(observations) == 6: raise RuntimeError("synthetic backend failure")
        observations.append(token)
    endpoint = fake_endpoint(observe_callback=observe)
    env = make_envelope(endpoint, NONCE, arm="stego", prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS)
    sent = generate(KEY, env, endpoint, payload=b"x")
    assert sent.status == "backend_error" and sent.reason == "observe:RuntimeError"
    assert sent.emitted_tokens == 6 and sent.resolved_bits == 24 and len(sent.traces) == 6
    assert bytes(observations) == sent.wire


def test_invalid_provider_law_is_contract_error_not_zero_capacity():
    endpoint = fake_endpoint(distribution_callback=lambda ids: Distribution(((65, True),)))
    env = make_envelope(endpoint, NONCE, arm="stego", prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS)
    result = generate(KEY, env, endpoint, payload=b"x")
    assert result.status == "contract_error" and result.wire == b"" and result.resolved_bits == 0


@pytest.mark.parametrize("bad", [{"x": {1, 2}}, {"x": float("nan")}])
def test_bad_provider_evidence_is_explicit_contract_error_without_commit(bad):
    seen = []
    endpoint = fake_endpoint(distribution_callback=lambda ids: Distribution(((65, 1), (66, 1)), bad),
                             observe_callback=seen.append)
    env = make_envelope(endpoint, NONCE, arm="stego", prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS)
    result = generate(KEY, env, endpoint, payload=b"x")
    assert result.status == "contract_error" and result.emitted_tokens == 0 and seen == []


def test_observe_cannot_mutate_the_saved_distribution_evidence():
    shared = {}
    def provider(ids):
        shared.clear(); shared.update({"step": len(ids)-len(PROMPT_IDS)})
        return Distribution(tuple((t, 1) for t in range(65, 81)), shared)
    def observe(token):
        shared["after_observe"] = {"not-json"}
    endpoint = fake_endpoint(distribution_callback=provider, observe_callback=observe)
    env = make_envelope(endpoint, NONCE, arm="stego", prompt_id="unit", protocol_digest=PROTOCOL,
                        prompt_ids=PROMPT_IDS)
    result = generate(KEY, env, endpoint, payload=b"x")
    assert result.status == "fixed512_pending_receiver" and result.emitted_tokens == 512
    assert all(t["base_law_evidence"] == {"step": i} for i, t in enumerate(result.traces))


def test_explicit_full_budget_tail_and_domain_separated_subkeys():
    _, env = setup()
    keys = derive_keys(KEY, env)
    frame = seal_frame(b"x", KEY, env)
    stream = input_bits(frame, keys[2], env, DEFAULT_CONFIG)
    assert len(stream) == 65536 and len(set(keys)) == 3
    assert stream[FRAME_BITS:] != (0,)*(65536-FRAME_BITS)
    assert stream != ordinary_bits(env, DEFAULT_CONFIG)


def test_arms_share_exact_conditional_order_and_bins_with_independent_transport_keys():
    from synsteg_audit.canonical_channel_v2 import Prefix
    from synsteg_audit.integer_interval_channel_v1 import Receiver, integer_bins
    stego, env_s = setup("adaptive4")
    ordinary, env_o = setup("adaptive4", arm="ordinary", ordinary_seed=b"z"*32)
    env_o = {**env_o, "nonce_hex": "ff"*12}
    keys_s, keys_o = derive_keys(KEY, env_s), derive_keys(KEY, env_o)
    assert ordering_context_digest(env_s) == ordering_context_digest(env_o)
    assert keys_s[1] == keys_o[1] and keys_s[0] != keys_o[0] and keys_s[2] != keys_o[2]
    stego.begin(PROMPT_IDS); ordinary.begin(PROMPT_IDS)
    prefix = Prefix(); arithmetic = Receiver(DEFAULT_CONFIG)
    for token in [65, 67, 66]:
        a = stego.gate.admit(prefix, stego.distribution(prefix).weights)
        b = ordinary.gate.admit(prefix, ordinary.distribution(prefix).weights)
        law_a = ordered_law(a, keys_s[1], env_s, prefix)
        law_b = ordered_law(b, keys_o[1], env_o, prefix)
        assert law_a == law_b
        assert integer_bins(arithmetic.state, law_a, DEFAULT_CONFIG) == integer_bins(arithmetic.state, law_b, DEFAULT_CONFIG)
        arithmetic.take(law_a, token)
        stego.observe(token); ordinary.observe(token)
        prefix = stego.gate.take(prefix, a, token)


@pytest.mark.parametrize("change", [{"output_tokens": 511}, {"output_tokens": True},
    {"max_payload_bytes": 17}, {"stop": "first frame"}, {"extra": 1}, {"prompt_ids": [True]},
    {"nonce_hex": "ff"*16}, {"ordinary_seed_hex": "bad"}, {"arm": "unknown"}])
def test_fixed_rule_envelope_changes_fail_before_backend_start(change):
    endpoint, env = setup()
    with pytest.raises(ContractError): validate_envelope({**env, **change}, endpoint)


def test_ordinary_rejects_payload_and_stego_rejects_oversized_slot():
    endpoint, env = setup(arm="ordinary")
    with pytest.raises(ContractError): generate(KEY, env, endpoint, payload=b"x")
    endpoint, env = setup()
    with pytest.raises(ContractError): generate(KEY, env, endpoint, payload=bytes(17))


def test_fresh_process_receiver_only_receives_wire_key_envelope_and_public_fixture():
    endpoint, env = setup()
    sent = generate(KEY, env, endpoint, payload=b"fresh process")
    script = """
import json,sys
from synsteg_audit.framed_interval_llm_v1 import receive
from synsteg_audit.framed_interval_llm_v1_fixtures import fake_endpoint
x=json.load(sys.stdin)
r=receive(bytes.fromhex(x['wire_hex']),bytes.fromhex(x['key_hex']),x['envelope'],fake_endpoint())
print(json.dumps({'status':r.status,'payload_hex':None if r.payload is None else r.payload.hex(),'tokens':r.processed_tokens}))
"""
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps({"wire_hex": sent.wire.hex(),
        "key_hex": KEY.hex(), "envelope": env}), text=True, capture_output=True, timeout=30, check=True)
    decoded = json.loads(result.stdout)
    assert decoded == {"status": "authenticated_delivery", "payload_hex": b"fresh process".hex(), "tokens": 512}
