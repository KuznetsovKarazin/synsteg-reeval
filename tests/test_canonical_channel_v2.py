from fractions import Fraction
import hashlib
import math

import pytest

from synsteg_audit.canonical_channel_v2 import (
    Admission, CanonicalGate, ContractError, MAX_WEIGHT_TOTAL, Prefix, PublicCodec, Verdict, recover_path,
)

DIGEST = hashlib.sha256(b"synthetic public tokenizer; not Qwen validation").hexdigest()


def byte_codec(*, encode=None, render=None, specials=frozenset({999})):
    return PublicCodec(encode or (lambda text: list(text.encode())),
                       render or (lambda ids: bytes(ids)), specials, DIGEST)


def test_empty_base_is_explicitly_checked():
    with pytest.raises(ContractError, match="canonical"):
        CanonicalGate(byte_codec(encode=lambda text: [1] + list(text.encode())))


def test_qwen_counterexample_shape_is_rejected_without_claiming_real_tokenizer():
    # Minimal fixture reproduces the actual observed token IDs/merge relation.
    pieces = {334: b"**", 271: b"\n\n", 56177: b"**\n\n", 65: b"A"}
    def encode(text):
        if text == "": return []
        if text == "**": return [334]
        if text == "**\n\n": return [56177]
        if text == "**A": return [334, 65]
        raise ValueError("fixture outside support")
    gate = CanonicalGate(PublicCodec(encode, lambda ids: b"".join(pieces[t] for t in ids), frozenset(), DIGEST))
    prefix = Prefix((334,), b"**")
    decision = gate.admit(prefix, [(271, 9), (65, 1)])
    assert decision.weights == ((65, 1),)
    assert {r.token_id: r.reason for r in decision.verdicts}[271] == "retokenization_mismatch"
    assert decision.summary()["removed_mass"] == {"numerator": 9, "denominator": 10}


def test_integer_mass_and_conditional_law_are_exact():
    gate = CanonicalGate(byte_codec())
    decision = gate.admit(Prefix(), [(65, 7), (66, 2), (999, 1), (67, 0)])
    assert decision.weights == ((65, 7), (66, 2))
    assert [decision.choose(i) for i in range(9)] == [65] * 7 + [66] * 2
    p = {65: Fraction(7, 10), 66: Fraction(2, 10), 999: Fraction(1, 10)}
    q = {65: Fraction(7, 9), 66: Fraction(2, 9), 999: Fraction(0)}
    assert sum(abs(p[t] - q[t]) for t in p) / 2 == Fraction(1, 10)
    assert decision.input_total == 10 and decision.retained_total == 9


def test_empty_support_preserves_denominator_and_refuses_sampling():
    decision = CanonicalGate(byte_codec()).admit(Prefix(), [(999, 8), (0xC3, 2)])
    assert decision.status == "empty_support"
    assert decision.summary()["removed_mass"] == {"numerator": 10, "denominator": 10}
    with pytest.raises(ContractError): decision.choose(0)


def test_zero_input_is_not_a_probability_distribution():
    gate = CanonicalGate(byte_codec())
    for candidates in [[], [(65, 0)]]:
        decision = gate.admit(Prefix(), candidates)
        assert decision.status == "empty_input_distribution"
        assert decision.summary()["removed_mass"] is None


def test_partial_utf8_tokens_cannot_be_completed_later():
    gate = CanonicalGate(byte_codec())
    assert gate.codec.encode(b"\xc3\xa9") == (195, 169)
    # Final roundtrip exists, but either individual byte is invalid UTF-8.
    result = recover_path(gate, b"\xc3\xa9", lambda prefix: [(195, 1), (169, 1)])
    assert result["status"] == "ineligible_token"
    assert result["steps"][0]["status"] == "empty_support"


def test_full_prefix_render_supports_contextual_spacing():
    def render(ids):
        return b" ".join({1: b"one", 2: b"two"}[t] for t in ids)
    def encode(text):
        return [] if not text else [{"one": 1, "two": 2}[word] for word in text.split(" ")]
    gate = CanonicalGate(PublicCodec(encode, render, frozenset(), DIGEST))
    prefix = Prefix((1,), b"one")
    decision = gate.admit(prefix, [(2, 1)])
    assert decision.verdicts[0].suffix == b" two"
    assert gate.take(prefix, decision, 2).raw == b"one two"
    assert render((2,)) == b"two"  # Single-token decode is not the append suffix.


def test_canonical_but_rewriting_renderer_is_rejected():
    rendered = {(): b"", (1,): b" A", (1, 2): b"A!"}
    encoded = {v.decode(): list(k) for k, v in rendered.items()}
    gate = CanonicalGate(PublicCodec(encoded.__getitem__, rendered.__getitem__, frozenset(), DIGEST))
    decision = gate.admit(Prefix((1,), b" A"), [(2, 1)])
    assert decision.verdicts[0].reason == "rewrites_prior_bytes"


def test_empty_render_special_eos_and_bos_are_not_hidden_output_symbols():
    gate = CanonicalGate(byte_codec(render=lambda ids: bytes(t for t in ids if t != 998)))
    decision = gate.admit(Prefix(), [(998, 1), (999, 1), (65, 1)])
    assert [r.reason for r in decision.verdicts] == ["accepted", "empty_extension", "special_token"]
    result = recover_path(gate, b"A", lambda ids: [(65, 1)], prompt_ids=(999, 998))
    assert result["status"] == "canonical_path_verified"  # Prompt specials are public model context only.


def test_receiver_reconstructs_all_public_prefix_admissions_without_sender_ids():
    gate = CanonicalGate(byte_codec())
    prompt, seen_sender, seen_receiver = (999, 1000), [], []
    def candidates(prefix):
        return [(65, 2), (66, 1), (999, 1)] if len(prefix) % 2 else [(65, 1), (66, 3), (0xC3, 1)]
    state, send_summaries = Prefix(), []
    for rank in [0, 1, 3, 2, 1, 0]:
        model_prefix = prompt + state.ids
        seen_sender.append(model_prefix)
        decision = gate.admit(state, candidates(model_prefix))
        send_summaries.append(decision.summary())
        state = gate.take(state, decision, decision.choose(rank % decision.retained_total))
    def public_receiver_provider(prefix):
        seen_receiver.append(prefix)
        return candidates(prefix)
    fresh_gate = CanonicalGate(byte_codec())
    recovered = recover_path(fresh_gate, state.raw, public_receiver_provider,
                             prompt_ids=prompt, expected_output_tokens=6)
    assert recovered["status"] == "canonical_path_verified"
    assert seen_receiver == seen_sender
    assert recovered["token_ids"] == list(state.ids)
    assert [{k: v for k, v in row.items() if k not in {"step", "observed_token"}}
            for row in recovered["steps"]] == send_summaries
    assert recovered["payload_authenticated"] is False


def test_receiver_does_not_silently_accept_normalized_wire_alias():
    gate = CanonicalGate(byte_codec(encode=lambda text: list(text.replace("\r\n", "\n").encode())))
    result = recover_path(gate, b"A\r\nB", lambda prefix: [(65, 1), (10, 1), (66, 1)])
    assert result["status"] == "noncanonical_wire_bytes"


@pytest.mark.parametrize("candidates", [[(True, 1)], [(1, True)], [(1, -1)], [(-1, 1)],
                                      [(1, 1), (1, 2)], [(1, MAX_WEIGHT_TOTAL + 1)], [(1,)]])
def test_malformed_integer_candidate_laws_fail_closed(candidates):
    with pytest.raises(ContractError): CanonicalGate(byte_codec()).admit(Prefix(), candidates)


def test_backend_failure_is_not_reported_as_lost_candidate_mass():
    def render(ids):
        if ids: raise RuntimeError("backend failure")
        return b""
    with pytest.raises(ContractError, match="render backend failed"):
        CanonicalGate(byte_codec(render=render)).admit(Prefix(), [(65, 1)])


def test_stale_admission_cannot_be_applied_to_another_prefix():
    gate = CanonicalGate(byte_codec())
    first = gate.admit(Prefix(), [(65, 1)])
    state = gate.take(Prefix(), first, 65)
    with pytest.raises(ContractError, match="does not belong"):
        gate.take(state, first, 65)


def test_filter_can_raise_conditional_pair_rate_and_entropy():
    decision = CanonicalGate(byte_codec()).admit(Prefix(), [(999, 198), (65, 1), (66, 1)])
    p, q = [198 / 200, 1 / 200, 1 / 200], [1 / 2, 1 / 2]
    assert decision.summary()["removed_weight_total"] == 198
    assert 1 - sum(x*x for x in q) > 1 - sum(x*x for x in p)
    assert -sum(x*math.log2(x) for x in q) > -sum(x*math.log2(x) for x in p)


@pytest.mark.parametrize("weight", [-1, True, 0])
def test_forged_admission_cannot_inject_nonpositive_or_boolean_weight(weight):
    with pytest.raises(ContractError):
        Admission(DIGEST, Prefix(), (Verdict(65, weight, "accepted", b"A"),))


def test_arbitrary_object_cannot_be_used_as_an_admission():
    with pytest.raises(ContractError):
        CanonicalGate(byte_codec()).take(Prefix(), object(), 65)
