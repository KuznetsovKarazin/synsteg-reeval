"""Synthetic exact-stop composition checks. No process, tokenizer or model."""
from dataclasses import replace
import copy

import pytest

from synsteg_audit import framed_interval_bounded_v1 as bounded
from synsteg_audit import framed_interval_llm_v1 as fixed
from synsteg_audit.canonical_channel_v2 import CanonicalGate, ContractError, Prefix, PublicCodec
from synsteg_audit.integer_interval_channel_v1 import Config, Sender, integer_bins

KEY = bytes(range(32))
PAYLOAD = b"0123456789abcdef"


def endpoint(n=8, *, provider=None, observe_failure=None, context_tokens=2048):
    events = {"begin": [], "distribution": [], "observe": []}
    codec = PublicCodec(lambda text: tuple(ord(c)-97 for c in text),
        lambda ids: bytes(t+97 for t in ids), frozenset(), "1"*64)
    def distribution(ids):
        events["distribution"].append(tuple(ids))
        return bounded.Distribution(tuple((t, 1) for t in range(n))) if provider is None else provider(ids)
    def observe(token):
        events["observe"].append(token)
        if observe_failure == len(events["observe"]): raise OSError("synthetic observe failure")
    ep = bounded.Endpoint(CanonicalGate(codec), resource_id="synthetic-ascii",
        resource_digest="2"*64, provider_digest="3"*64, distribution_callback=distribution,
        start_callback=lambda ids: events["begin"].append(ids), observe_callback=observe,
        context_tokens=context_tokens)
    return ep, events


def envelope(ep, arm="stego", **kwargs):
    return bounded.make_envelope(ep, bytes(12), arm=arm, prompt_id="fake",
        protocol_digest="4"*64, **kwargs)


def send(n=8, arm="stego", **kwargs):
    ep, events = endpoint(n, **kwargs)
    env = envelope(ep, arm)
    result = bounded.generate(KEY, env, ep, payload=PAYLOAD if arm == "stego" else None)
    return result, env, events


@pytest.mark.parametrize("n,tokens,resolved", [(2, 272, 272), (4, 136, 272), (8, 91, 273), (16, 68, 272)])
@pytest.mark.parametrize("arm", ["stego", "ordinary"])
def test_exact_same_stop_rule_fresh_receivers(n, tokens, resolved, arm):
    result, env, sender_events = send(n, arm)
    assert result.emitted_tokens == result.first_threshold_token == tokens
    assert result.resolved_bits == resolved and result.consumed_bits == resolved + 32
    assert len(sender_events["observe"]) == len(sender_events["distribution"]) == tokens
    assert result.authenticated_delivery is False
    assert sum(len(t["emitted_bits"]) for t in result.traces[:-1]) < 272
    receiver, receiver_events = endpoint(n)
    recovered = bounded.receive(result.wire, KEY, env, receiver)
    assert recovered.processed_tokens == recovered.wire_tokens == tokens
    assert recovered.first_threshold_token == tokens and recovered.traces == result.traces
    assert len(receiver_events["observe"]) == tokens
    if arm == "stego":
        assert recovered.authenticated_delivery and recovered.payload == PAYLOAD
    else:
        assert recovered.status == "ordinary_threshold_replay_verified"
        assert not recovered.authenticated_delivery and recovered.payload is None


@pytest.mark.parametrize("arm", ["stego", "ordinary"])
def test_singleton_keeps_full_cap_and_zero_capacity(arm):
    result, env, events = send(1, arm)
    assert result.emitted_tokens == 768 and len(events["observe"]) == 768
    assert (result.resolved_bits, result.consumed_bits, result.pending_bits) == (0, 32, 0)
    assert result.first_threshold_token is None
    assert result.status == ("insufficient_capacity" if arm == "stego" else "ordinary_cap_generated")
    recovered = bounded.receive(result.wire, KEY, env, endpoint(1)[0])
    assert recovered.status == ("insufficient_capacity" if arm == "stego" else "ordinary_cap_replay_verified")
    assert recovered.payload is None and not recovered.authenticated_delivery


def test_threshold_exactly_at_cap_takes_priority_over_capacity_failure():
    def provider(ids):
        return bounded.Distribution(((0, 1),) if len(ids) < 496 else ((0, 1), (1, 1)))
    result, env, _ = send(2, provider=provider)
    assert result.emitted_tokens == result.first_threshold_token == 768
    assert result.resolved_bits == 272 and result.status == "threshold_pending_receiver"
    recovered = bounded.receive(result.wire, KEY, env, endpoint(2, provider=provider)[0])
    assert recovered.authenticated_delivery and recovered.payload == PAYLOAD


@pytest.mark.parametrize("payload", [b"", b"x", PAYLOAD])
def test_fixed272_frame_empty_short_full_payload(payload):
    ep, _ = endpoint(16); env = envelope(ep)
    result = bounded.generate(KEY, env, ep, payload=payload)
    recovered = bounded.receive(result.wire, KEY, env, endpoint(16)[0])
    assert result.emitted_tokens == 68 and recovered.payload == payload and recovered.authenticated_delivery


def wire_from_stream(stream, env):
    ep, _ = endpoint(8); ep.begin(())
    sender = Sender(stream, bounded.DEFAULT_CONFIG)
    prefix = Prefix(); order = bounded.derive_keys(KEY, env)[1]
    while len(sender.resolved_bits) < 272:
        decision = ep.gate.admit(prefix, ep.distribution(prefix).weights)
        law = bounded.ordered_law(decision, order, env, prefix)
        step = sender.take(law)
        prefix = ep.gate.take(prefix, decision, step.symbol); ep.observe(step.symbol)
    return prefix.raw, sender.resolved_bits


def test_surplus_bit_tamper_preserves_AEAD_but_fullwire_replay_rejects():
    ep, _ = endpoint(8); env = envelope(ep)
    frame = bounded.seal_frame(PAYLOAD, KEY, env)
    stream = list(bounded.input_bits(frame, bounded.derive_keys(KEY, env)[2], env, bounded.DEFAULT_CONFIG))
    stream[272] ^= 1
    wire, resolved = wire_from_stream(stream, env)
    assert resolved[:272] == bounded.bits_from_bytes(frame)
    assert bounded.open_frame(frame, KEY, env) == PAYLOAD
    recovered = bounded.receive(wire, KEY, env, endpoint(8)[0])
    assert recovered.status == "invalid_wire" and recovered.reason == "noncanonical_fullwire_embedding"
    assert recovered.payload is None and not recovered.authenticated_delivery


@pytest.mark.parametrize("mutation", ["truncate", "extra", "wrong-key", "wrong-nonce", "frame-bit"])
def test_modified_wire_key_and_envelope_never_return_plaintext(mutation):
    result, env, _ = send(8); wire, key = result.wire, KEY
    if mutation == "truncate": wire = wire[:-1]
    elif mutation == "extra": wire += b"a"
    elif mutation == "wrong-key": key = bytes(32)
    elif mutation == "wrong-nonce": env = dict(env, nonce_hex="01"*12)
    else: wire = bytes([97+(wire[0]-97+1)%8]) + wire[1:]
    ep, events = endpoint(8)
    recovered = bounded.receive(wire, key, env, ep)
    assert recovered.payload is None and not recovered.authenticated_delivery
    if mutation == "extra":
        assert recovered.reason == "trailing_tokens_after_first_threshold"
        assert len(events["observe"]) == 91  # The forbidden next token is never observed.
    if mutation == "truncate": assert recovered.status == "incomplete_wire"


def test_byte_limit_reports_only_committed_prefix_counters():
    ep, events = endpoint(8); env = envelope(ep, max_wire_bytes=2)
    result = bounded.generate(KEY, env, ep, payload=PAYLOAD)
    assert result.status == "wire_limit_exceeded" and result.emitted_tokens == 2
    assert len(result.wire) == 2 and len(events["observe"]) == 2
    assert (result.resolved_bits, result.consumed_bits) == (6, 38)
    assert result.traces[-1]["transmitted"] is False


def test_backend_error_does_not_commit_tentative_counters():
    result, _, events = send(8, observe_failure=3)
    assert result.status == "backend_error" and result.emitted_tokens == 2
    assert (result.resolved_bits, result.consumed_bits) == (6, 38)
    assert len(events["observe"]) == 3  # Native outcome unknown; no retry is performed.
    assert len(result.traces) == 2


@pytest.mark.parametrize("weights,reason", [((), "empty_input_distribution"), (((0, 0),), "empty_input_distribution")])
def test_zero_support_preserved_without_fallback(weights, reason):
    result, _, events = send(provider=lambda ids: bounded.Distribution(weights))
    assert result.status == "support_refusal" and result.reason == reason
    assert result.emitted_tokens == 0 and result.resolved_bits == 0
    assert result.wire == b"" and not events["observe"]


def test_lookahead_read_budget_does_not_create_capacity():
    ep, events = endpoint(8); env = envelope(ep, config=Config(32, 768, 4096, 32))
    result = bounded.generate(KEY, env, ep, payload=PAYLOAD)
    assert result.status == "capacity_limit" and result.reason == "max_stream_bits"
    assert result.emitted_tokens == 0 and not events["observe"] and result.consumed_bits == 32


def test_context_cap_and_endpoint_one_use():
    ep, events = endpoint(context_tokens=700)
    with pytest.raises(ContractError, match="bounded output"): envelope(ep)
    assert not events["begin"]
    ep, _ = endpoint(); env = envelope(ep)
    bounded.generate(KEY, env, ep, payload=PAYLOAD)
    second = bounded.generate(KEY, env, ep, payload=PAYLOAD)
    assert second.status == "contract_error" and second.emitted_tokens == 0


@pytest.mark.parametrize("field,value", [("threshold_bits", 271), ("token_cap", 512),
    ("experiment_id", fixed.VERSION), ("max_wire_bytes", True), ("stop", "arbitrary")])
def test_public_stop_and_domain_are_bound(field, value):
    ep, events = endpoint(); env = envelope(ep); env[field] = value
    with pytest.raises(ContractError): bounded.generate(KEY, env, ep, payload=PAYLOAD)
    assert not events["begin"]


def test_domains_differ_from_fixed512_and_arm_order_is_exactly_matched():
    ep, _ = endpoint(8); stego = envelope(ep)
    ordinary = dict(stego, arm="ordinary", nonce_hex="11"*12, ordinary_seed_hex="22"*32)
    s_keys, o_keys = bounded.derive_keys(KEY, stego), bounded.derive_keys(KEY, ordinary)
    assert s_keys[1] == o_keys[1] and s_keys[0] != o_keys[0] and s_keys[2] != o_keys[2]
    assert bounded.DOMAIN != fixed.DOMAIN
    old = fixed.make_envelope(ep, bytes(12), arm="stego", prompt_id="fake", protocol_digest="4"*64)
    assert s_keys != fixed.derive_keys(KEY, old)
    prefix = Prefix(); decision = ep.gate.admit(prefix, tuple((t, t+1) for t in range(8)))
    slaw = bounded.ordered_law(decision, s_keys[1], stego, prefix)
    olaw = bounded.ordered_law(decision, o_keys[1], ordinary, prefix)
    assert slaw == olaw
    state = Sender((0,)*65536, bounded.DEFAULT_CONFIG).state
    assert integer_bins(state, slaw, bounded.DEFAULT_CONFIG) == integer_bins(state, olaw, bounded.DEFAULT_CONFIG)


def test_non_JSON_backend_evidence_is_contract_failure_before_observe():
    result, _, events = send(provider=lambda ids: bounded.Distribution(((0, 1),), {"bad": {1}}))
    assert result.status == "contract_error" and not events["observe"]
