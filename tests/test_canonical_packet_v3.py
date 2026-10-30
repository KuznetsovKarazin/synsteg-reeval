from fractions import Fraction
import hashlib
from itertools import product

import pytest

from synsteg_audit.canonical_channel_v2 import ContractError, MAX_WEIGHT_TOTAL, Prefix, PublicCodec
from synsteg_audit.canonical_packet_v3 import (
    CanonicalPacketGate, ConstructionRefusal, PacketAdmission, PacketVerdict,
    first_canonical_frontier, recover_packets,
)

DIGEST = hashlib.sha256(b"synthetic codec only").hexdigest()
POLICY = hashlib.sha256(b"synthetic public packet policy").hexdigest()


def gate(codec=None, limit=5, policy=POLICY):
    codec = codec or PublicCodec(lambda text: list(text.encode()), lambda ids: bytes(ids),
                                frozenset({255}), DIGEST)
    return CanonicalPacketGate(codec, max_packet_tokens=limit, policy_digest=policy)


def silent_codec():
    # Every character has a real leading silent token 29871 in canonical IDs.
    # This is a synthetic fixture; it does not claim to emulate TinyLlama.
    return PublicCodec(lambda text: [token for char in text for token in (29871, *char.encode())],
        lambda ids: bytes(token for token in ids if token != 29871), frozenset({999}), DIGEST)


def test_packet_completes_silent_first_token_without_deleting_its_id():
    g = gate(silent_codec())
    decision = g.admit(Prefix(), [((29871, 65), 2), ((29871, 66), 1)])
    assert decision.retained_total == 3
    state = g.take(Prefix(), decision, (29871, 65))
    assert state == Prefix((29871, 65), b"A")
    received = recover_packets(g, b"AB", lambda prefix: [((29871, 65), 2), ((29871, 66), 1)])
    assert received["status"] == "canonical_packet_path_verified"
    assert received["packets"] == [[29871, 65], [29871, 66]]
    assert received["token_ids"] == [29871, 65, 29871, 66]
    assert not received["payload_authenticated"] and not received["payload_delivered"]


def test_packet_can_span_internal_invalid_utf8():
    g = gate()
    decision = g.admit(Prefix(), [((195, 169), 1)])
    assert decision.weights == (((195, 169), 1),)
    assert g.take(Prefix(), decision, (195, 169)).raw == "é".encode()
    assert recover_packets(g, "é".encode(), lambda p: [((195, 169), 1)])["packets"] == [[195, 169]]


def test_internal_rewrite_is_not_published_but_endpoint_must_append():
    rendered = {(): b"", (1,): b"A", (1, 2): b"X", (1, 2, 3): b"AB"}
    encoded = {"": [], "A": [1], "X": [4], "AB": [1, 2, 3]}
    g = gate(PublicCodec(encoded.__getitem__, rendered.__getitem__, frozenset(), DIGEST))
    state = Prefix((1,), b"A")
    row = g.admit(state, [((2, 3), 1)]).verdicts[0]
    assert row.reason == "accepted" and row.suffix == b"B"


def test_joint_merge_is_still_rejected_at_packet_boundary():
    pieces = {334: b"**", 271: b"\n\n", 56177: b"**\n\n"}
    encoded = {"": [], "**": [334], "\n\n": [271], "**\n\n": [56177]}
    g = gate(PublicCodec(encoded.__getitem__, lambda ids: b"".join(pieces[i] for i in ids),
                         frozenset(), DIGEST))
    decision = g.admit(Prefix(), [((334, 271), 1), ((56177,), 1)])
    assert [r.reason for r in decision.verdicts] == ["retokenization_mismatch", "accepted"]


@pytest.mark.parametrize("rows", [
    [((65,), 1), ((65, 66), 1)],  # 'AB' otherwise has one- or two-packet parse.
    [((65, 66), 1), ((65,), 0)],  # Stronger contract includes zero-weight rows.
    [((65,), 1), ((65,), 1)],
])
def test_ambiguous_or_duplicate_raw_packet_set_is_rejected(rows):
    with pytest.raises(ContractError, match="prefix-free"):
        gate().admit(Prefix(), rows)


def test_same_token_prefix_is_allowed_when_neither_packet_contains_the_other():
    d = gate().admit(Prefix(), [((65, 66), 1), ((65, 67), 1)])
    assert d.retained_total == 2
    assert recover_packets(gate(), b"ABAC", lambda p: [((65, 66), 1), ((65, 67), 1)])["packet_count"] == 2


def test_receiver_rejects_leftover_partial_packet():
    result = recover_packets(gate(), b"A", lambda prefix: [((65, 66), 1)])
    assert result["status"] == "truncated_packet"


def test_boundary_truncation_is_undetectable_without_external_count():
    provider = lambda prefix: [((65,), 1), ((66,), 1)]
    assert recover_packets(gate(), b"A", provider)["status"] == "canonical_packet_path_verified"
    assert recover_packets(gate(), b"A", provider, expected_packets=2)["status"] == "packet_count_mismatch"
    assert recover_packets(gate(), b"A", provider, expected_output_tokens=2)["status"] == "token_count_mismatch"


def test_receiver_uses_public_boundary_model_context_not_prompt_text_reencoding():
    calls = []
    def provider(ids):
        calls.append(ids)
        return [((65, 66), 1)] if ids == (999, 1000) else [((67,), 1)]
    result = recover_packets(gate(), b"ABC", provider, prompt_ids=(999, 1000), expected_packets=2)
    assert result["status"] == "canonical_packet_path_verified"
    assert calls == [(999, 1000), (999, 1000, 65, 66)]


def test_exact_law_mass_and_rank_mapping_for_variable_length_packets():
    d = gate().admit(Prefix(), [((65, 66), 6), ((67,), 3), ((255,), 1)])
    assert [d.choose(i) for i in range(9)] == [(65, 66)] * 6 + [(67,)] * 3
    p, q = [Fraction(6, 10), Fraction(3, 10), Fraction(1, 10)], [Fraction(6, 9), Fraction(3, 9), 0]
    assert sum(abs(a-b) for a, b in zip(p, q)) / 2 == Fraction(1, 10)
    assert d.summary()["removed_mass"] == {"numerator": 1, "denominator": 10}


def test_empty_support_and_zero_input_are_distinct():
    g = gate()
    assert g.admit(Prefix(), []).status == "empty_input_distribution"
    assert g.admit(Prefix(), [((65,), 0)]).status == "empty_input_distribution"
    d = g.admit(Prefix(), [((255,), 1), ((195,), 2)])
    assert d.status == "empty_support" and d.retained_total == 0
    with pytest.raises(ContractError): d.choose(0)


def test_empty_wire_has_zero_packets_and_is_not_payload_delivery():
    def unused(_):
        raise AssertionError("No packet should be requested for an empty wire")
    result = recover_packets(gate(), b"", unused, expected_packets=0)
    assert result["status"] == "canonical_packet_path_verified" and result["packet_count"] == 0
    assert result["payload_delivered"] is False
    assert recover_packets(gate(), b"", unused, expected_packets=1)["status"] == "packet_count_mismatch"


def test_invalid_utf8_wire_and_output_special_packet_are_explicit_refusals():
    assert recover_packets(gate(), b"\xc3", lambda p: [((195,), 1)])["status"] == "invalid_utf8"
    codec = PublicCodec(lambda s: list(s.encode()), lambda ids: bytes(ids), frozenset({65}), DIGEST)
    result = recover_packets(gate(codec), b"A", lambda p: [((65,), 1)])
    assert result["status"] == "rejected_packet" and result["reason"] == "special_token"


def test_builder_backend_error_is_not_terminal_refusal_or_probability_mass():
    def render(ids):
        if ids: raise RuntimeError("synthetic backend error")
        return b""
    g = gate(PublicCodec(lambda text: list(text.encode()), render, frozenset(), DIGEST))
    with pytest.raises(ContractError, match="render backend failed"):
        first_canonical_frontier(g, Prefix(), [(65, 66)], node_budget=5)


@pytest.mark.parametrize("rows", [
    [((True,), 1)], [((-1,), 1)], [((), 1)], [((65,), True)], [((65,), -1)],
    [((65,), MAX_WEIGHT_TOTAL+1)], [((65,)*6, 1)], [(65, 1)], [((),)],
])
def test_invalid_packet_laws_fail_closed(rows):
    with pytest.raises(ContractError): gate().admit(Prefix(), rows)


@pytest.mark.parametrize("weight", [-1, True, 0])
def test_forged_admission_rejects_invalid_positive_support(weight):
    with pytest.raises(ContractError):
        PacketAdmission(DIGEST, POLICY, 5, Prefix(), (PacketVerdict((65,), weight, "accepted", b"A"),))


def test_decision_bound_to_exact_state_policy_and_packet_bound():
    g = gate(); d = g.admit(Prefix(), [((65,), 1)]); advanced = g.take(Prefix(), d, (65,))
    for other, prefix in [(g, advanced), (gate(limit=4), Prefix()),
                          (gate(policy="1"*64), Prefix())]:
        with pytest.raises(ContractError): other.take(prefix, d, (65,))


def test_mutable_backend_change_is_detected_before_packet_commit():
    altered = [False]
    def render(ids):
        return bytes(ids) + (b"!" if ids and altered[0] else b"")
    g = gate(PublicCodec(lambda s: list(s.encode()), render, frozenset(), DIGEST))
    d = g.admit(Prefix(), [((65,), 1)]); altered[0] = True
    with pytest.raises(ContractError, match="changed"): g.take(Prefix(), d, (65,))


def test_wire_normalization_alias_does_not_verify():
    codec = PublicCodec(lambda s: list(s.replace("\r\n", "\n").encode()),
                        lambda ids: bytes(ids), frozenset(), DIGEST)
    assert recover_packets(gate(codec), b"A\r\n", lambda p: [((65,), 1)])["status"] == "noncanonical_wire_bytes"


def test_builder_stops_at_first_canonical_node_and_leaves_are_prefix_free():
    g = gate(silent_codec())
    paths = [(29871, 65), (29871, 65, 29871, 66), (29871, 195, 169), (999, 65)]
    result = first_canonical_frontier(g, Prefix(), paths, node_budget=20)
    assert result.status == "ok"
    assert result.leaves == ((999,), (29871, 65), (29871, 195, 169))
    assert (29871, 65, 29871) not in {r.packet for r in result.inspections}
    reasons = {r.packet: r.reason for r in result.inspections}
    assert reasons[(29871,)] == "empty_extension"
    assert reasons[(29871, 195)] == "invalid_utf8"
    assert g.admit(Prefix(), result.candidates()).retained_total == 2


def test_builder_bound_preserves_dead_end_in_frontier_denominator():
    g = gate(silent_codec())
    result = first_canonical_frontier(g, Prefix(), [(29871, 195), (29871, 65)], node_budget=10)
    assert result.leaves == ((29871, 65), (29871, 195))
    assert g.admit(Prefix(), result.candidates()).summary()["removed_mass"] == {"numerator": 1, "denominator": 2}
    assert result.summary()["mass_outside_public_trie"] is None


def test_budget_refuses_entire_construction_before_any_node_inspection():
    g = gate()
    result = first_canonical_frontier(g, Prefix(), [(65, 66), (67,)], node_budget=2)
    assert result.status == "construction_node_budget_exceeded" and result.possible_trie_nodes == 3
    assert result.inspections == () and result.leaves == ()
    with pytest.raises(ConstructionRefusal): result.candidates()
    assert recover_packets(g, b"A", lambda p: result.candidates())["status"] == "construction_refusal"


def test_sender_receiver_replay_all_short_synthetic_packet_histories():
    g = gate(silent_codec())
    pool = [(29871, 65), (29871, 66), (29871, 195, 169), (999,)]
    for length in range(5):
        for text_parts in product(["A", "B", "é"], repeat=length):
            text = "".join(text_parts); state = Prefix(); summaries = []; packets = []
            for char in text_parts:
                frontier = first_canonical_frontier(g, state, pool, node_budget=20)
                decision = g.admit(state, frontier.candidates()); summaries.append(decision.summary())
                packet = tuple(g.codec.encode(char.encode()))
                state = g.take(state, decision, packet); packets.append(list(packet))
            fresh = gate(silent_codec())
            def provider(ids):
                boundary = Prefix(ids, fresh.codec.render(ids))
                return first_canonical_frontier(fresh, boundary, pool, node_budget=20).candidates()
            result = recover_packets(fresh, text.encode(), provider, expected_packets=length)
            assert result["status"] == "canonical_packet_path_verified"
            assert result["packets"] == packets and state.raw == text.encode()
            assert [{k: v for k, v in row.items() if k not in {"packet_index", "token_offset", "observed_packet"}}
                    for row in result["steps"]] == summaries
