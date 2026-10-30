"""Synthetic mechanical tests; no natural-language efficacy evidence."""
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synsteg_audit import invariant_context as context
from synsteg_audit import shared_dictionary as base

ROOT = Path(__file__).resolve().parents[1]
WORDS = {
    "brightness": ["bright", "clear", "lucid", "vivid"],
    "darkness": ["dark", "dim"],
    "matter": ["affair", "matter"],
    "pace": ["fast", "quick", "rapid", "swift"],
    "time": ["clock", "time"],
}
KEY = bytes(range(32))  # Public nonproduction fixture.


@pytest.fixture
def dictionary():
    return base.PublicDictionary.from_mapping(WORDS, "synthetic-invariant-context-tests-v1")


@pytest.fixture
def policy(dictionary):
    return context.ProtectedPhrasePolicy.from_phrases(
        dictionary, ["over time", "dark matter"], version="posthoc-development-fixture-v1")


def carrier(n=180):
    return 'Report: over "Time"; DARK (matter).\n' + "\t ".join(["quick,"] * n) + "\nEnd."


@pytest.mark.parametrize("framing,protocol_hash,stego_hash,frame_bits,changed", [
    ("aes_siv", "d1498d7a4658e8f933c6f873c1ac51d9516e461e02c10c2c69ad2f03d9fa301c",
     "1164faea6906972f433a267fb0cbaa63e60767a60f9b7237d904d3f12da10f15", 160, 105),
    ("plain_length16", "24cad6f551cd1e04870c3faa23177d90e18a062199142dcaf6d7ba57e5635e21",
     "45cb41b6385bbb8349ca3d4f669ca0a337dcdfd92ad06beaf91cef50aed941dc", 32, 18),
])
def test_fixed_synthetic_protocol_vectors(dictionary, policy, framing, protocol_hash, stego_hash, frame_bits, changed):
    # Fixed expectations were recorded when this separate protocol was created;
    # encoder+decoder drifting together must not silently retain this version.
    protocol = context.ContextProtocol(framing=framing)
    encoded = context.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    assert dictionary.sha256 == "729d88747d5e5dc349f886c1695deeefc30b9c8c08aec1afb6a8c0f2c230e2a7"
    assert policy.sha256 == "9d3db899f149ed51e98620153a1062ca7b3858ea0c2b2f82d9c0a9cbae843df6"
    assert protocol.sha256 == protocol_hash
    assert hashlib.sha256(encoded.stego_text.encode()).hexdigest() == stego_hash
    assert encoded.skeleton_sha256 == "e1728232a9bec198252211ffe9ba8c116952e5602f8818ca8f4545fde7b40b5c"
    assert encoded.virtual_normalization_sha256 == "208d278acdab4dcde8bc1d8c8657cd6106fe313da37a7b066af63b7fe10f8198"
    assert encoded.capacity_bits == 180 and encoded.frame_bits == frame_bits and encoded.changed_slots == changed
    assert context.decode(encoded.stego_text, KEY, dictionary, policy, protocol) == b"xy"


@pytest.mark.parametrize("framing", ["plain_length16", "aes_siv"])
@pytest.mark.parametrize("min_group,max_bits", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_roundtrip_preserves_protected_surface_and_all_public_invariants(dictionary, policy, framing, min_group, max_bits):
    protocol = context.ContextProtocol(min_group, max_bits, framing)
    original = carrier()
    before = context.analyze_text(original, dictionary, policy, protocol)
    encoded = context.encode(original, b"xy", KEY, dictionary, policy, protocol)
    assert context.decode(encoded.stego_text, KEY, dictionary, policy, protocol) == b"xy"
    assert encoded.stego_text.startswith('Report: over "Time"; DARK (matter).\n')
    assert context.analyze_text(encoded.stego_text, dictionary, policy, protocol) == before
    assert encoded.capacity_bits <= encoded.raw_capacity_bits
    assert encoded.protected_slot_count == 3
    assert before.protected_token_indices == (2, 3, 4)
    assert not set(before.protected_token_indices) & set(before.eligible_token_indices)
    assert encoded.dictionary_sha256 == dictionary.sha256
    assert encoded.policy_sha256 == policy.sha256
    assert encoded.protocol_sha256 == protocol.sha256


def test_all_class_members_preserve_skeleton_and_eligibility_before_filtering(dictionary, policy):
    reference = None
    # Include changes to protected and unprotected slots. Eligibility must be
    # invariant even when an adversary changes a protected word's class alias.
    for time_word, dark_word, matter_word, pace_word in itertools.product(
            WORDS["time"], WORDS["darkness"], WORDS["matter"], WORDS["pace"]):
        text = f"over {time_word}; {dark_word} {matter_word}. {pace_word}\t{time_word}"
        value = context.analyze_text(text, dictionary, policy)
        signature = (value.skeleton_sha256, value.protected_token_indices,
                     value.eligible_token_indices, value.capacity_bits, value.raw_capacity_bits)
        if reference is None:
            reference = signature
        assert signature == reference
        assert value.protected_token_indices == (1, 2, 3)
        assert value.eligible_token_indices == (4, 5)


def test_sender_only_surface_rule_counterexample_is_closed_by_class_compilation(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["over time"], version="counterexample-v1")
    original, alias = "over time quick", "over clock quick"
    assert ("over time" in original) != ("over time" in alias)  # Invalid sender-only literal gate.
    a, b = [context.analyze_text(text, dictionary, policy) for text in (original, alias)]
    assert a.protected_token_indices == b.protected_token_indices == (1,)
    assert a.eligible_token_indices == b.eligible_token_indices == (2,)
    assert a.skeleton_sha256 == b.skeleton_sha256
    # Exact protected spelling is separately authenticated, not erased.
    assert a.virtual_normalization_sha256 != b.virtual_normalization_sha256


def test_overlapping_adjacent_and_repeated_class_patterns_protect_union(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary,
        ["time time", "clock time clock", "time clock", "bright time"], version="overlaps-v1")
    # The first and third phrases compile to the same pattern and are deduplicated.
    assert len(policy.patterns) == 3
    result = context.analyze_text("clock time clock vivid time quick", dictionary, policy)
    assert result.protected_token_indices == (0, 1, 2, 3, 4)
    assert result.eligible_token_indices == (5,)


def test_variable_length_replacements_do_not_shift_protected_token_restoration(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["over time"], version="offsets-v1")
    text = "quick " * 90 + "over time " + "quick " * 90 + "over CLOCK."
    result = context.encode(text, b"xy", KEY, dictionary, policy)
    tokens = result.stego_text.split()
    assert tokens[90:92] == ["over", "time"]
    assert tokens[182:184] == ["over", "CLOCK."]
    assert context.decode(result.stego_text, KEY, dictionary, policy) == b"xy"


def test_reserved_external_marker_domain_is_rejected(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["over time"], version="markers-v1")
    for reserved in (context._marker(2, "time"), "\ue000", "\ue001", "\ue000deadbeef", "deadbeef\ue001"):
        text = reserved + " over time " + "quick " * 180
        with pytest.raises(context.ContextError, match="reserved"):
            context.analyze_text(text, dictionary, policy)
        with pytest.raises(context.ContextError, match="reserved"):
            context.encode(text, b"xy", KEY, dictionary, policy)
        with pytest.raises(context.ContextError, match="reserved"):
            context.decode(text, KEY, dictionary, policy)


def test_protected_word_cannot_be_replaced_by_its_virtual_marker(dictionary):
    # This exact substitution used to recreate the same virtual carrier while
    # deleting a protected word. External-domain validation must reject it.
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["over time"], version="alias-regression-v1")
    cover = "over time. " + "quick bright " * 100
    encoded = context.encode(cover, b"a", KEY, dictionary, policy)
    assert encoded.stego_text.startswith("over time. ")
    tampered = encoded.stego_text.replace("time.", context._marker(1, "time") + ".", 1)
    with pytest.raises(context.ContextError, match="reserved"):
        context.decode(tampered, KEY, dictionary, policy)


def test_pua_and_compound_affixes_are_opaque_barriers_not_new_slots(dictionary, policy):
    # Other PUA characters are allowed opaque data; only the two public marker
    # delimiters are excluded from the accepted external-text domain.
    text = "over \ue002time\ue003 dark_matter over time-based \ue002deadbeef\ue003 quick"
    analysis = context.analyze_text(text, dictionary, policy)
    assert analysis.protected_slot_count == 0
    assert analysis.candidate_slot_count == analysis.eligible_slot_count == 1
    assert analysis.capacity_bits == analysis.raw_capacity_bits == 1


def test_capacity_monotone_and_exact_frame_failure(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["time"], version="capacity-v1")
    protocol = context.ContextProtocol(framing="plain_length16")
    short = "time " + "quick " * 23
    a = context.analyze_text(short, dictionary, policy, protocol)
    assert a.raw_capacity_bits == 24 and a.capacity_bits == 23
    assert a.max_payload_bytes == 0
    with pytest.raises(base.CapacityError):
        context.encode(short, b"x", KEY, dictionary, policy, protocol)
    exact = "time " + "quick " * 24
    encoded = context.encode(exact, b"x", KEY, dictionary, policy, protocol)
    assert context.decode(encoded.stego_text, KEY, dictionary, policy, protocol) == b"x"
    assert encoded.frame_bits == encoded.capacity_bits == 24


def test_empty_policy_keeps_capacity_but_has_distinct_protocol_domain(dictionary):
    policy = context.ProtectedPhrasePolicy.from_phrases(dictionary, [], version="empty-control-v1")
    protocol = context.ContextProtocol()
    text = carrier()
    result = context.encode(text, b"xy", KEY, dictionary, policy, protocol)
    assert result.capacity_bits == base.analyze_text(text, dictionary, protocol.base_protocol()).capacity_bits
    assert context.decode(result.stego_text, KEY, dictionary, policy, protocol) == b"xy"
    with pytest.raises(base.CodecError):
        base.decode(result.stego_text, KEY, dictionary, protocol.base_protocol())


def test_authenticated_protected_word_tampering_rejected_with_same_eligibility(dictionary, policy):
    text = "over time dark matter " + "quick " * 180
    result = context.encode(text, b"xy", KEY, dictionary, policy)
    tampered = result.stego_text.replace("over time", "over clock", 1)
    assert tampered != result.stego_text
    a = context.analyze_text(result.stego_text, dictionary, policy)
    b = context.analyze_text(tampered, dictionary, policy)
    assert a.skeleton_sha256 == b.skeleton_sha256
    assert a.protected_token_indices == b.protected_token_indices
    with pytest.raises(base.CodecError):
        context.decode(tampered, KEY, dictionary, policy)


def test_policy_version_and_canonicalization_are_bound(dictionary, policy):
    result = context.encode(carrier(), b"xy", KEY, dictionary, policy)
    renamed = replace(policy, version="different-public-policy-version")
    assert renamed.patterns == policy.patterns and renamed.sha256 != policy.sha256
    with pytest.raises(base.CodecError):
        context.decode(result.stego_text, KEY, dictionary, renamed)
    for field in ("canonicalization_version", "schema"):
        data = policy.to_dict()
        data[field] += "-changed"
        with pytest.raises(context.ContextError):
            context.ProtectedPhrasePolicy.from_dict(data, dictionary)
    data = context.ContextProtocol().to_dict()
    data["virtualization_version"] += "-changed"
    with pytest.raises(context.ContextError):
        context.ContextProtocol.from_dict(data)
    data = context.ContextProtocol().to_dict()
    data.pop("external_text_version")
    with pytest.raises(context.ContextError):
        context.ContextProtocol.from_dict(data)


def test_policy_canonical_dedup_order_and_dictionary_binding(dictionary):
    a = context.ProtectedPhrasePolicy.from_phrases(dictionary, [" OVER  Time ", "dark matter", "over time"], version="canonical-v1")
    b = context.ProtectedPhrasePolicy.from_phrases(dictionary, ["dark matter", "over time"], version="canonical-v1")
    assert a == b and a.sha256 == b.sha256
    assert context.ProtectedPhrasePolicy.from_dict(a.to_dict(), dictionary) == a
    other = base.PublicDictionary(dictionary.classes, version="different-dictionary")
    with pytest.raises(context.ContextError, match="different"):
        context.analyze_text(carrier(), other, a)
    invalid = a.to_dict()
    invalid["phrases"] = list(reversed(invalid["phrases"]))
    with pytest.raises(context.ContextError, match="canonical"):
        context.ProtectedPhrasePolicy.from_dict(invalid, dictionary)


def test_observed_word_literal_cannot_override_class_policy(dictionary, policy):
    data = policy.to_dict()
    # An attacker cannot change an atom to a particular observed spelling while
    # retaining the canonical public phrase, even if it is syntactically valid.
    pattern = next(p for p in data["patterns"] if p[0]["kind"] == "literal")
    pattern[1] = {"kind": "literal", "value": "time"}
    with pytest.raises(context.ContextError):
        context.ProtectedPhrasePolicy.from_dict(data, dictionary)


def test_quality_selector_rejected_instead_of_misusing_virtual_offsets(dictionary, policy):
    with pytest.raises(context.ContextError, match="canonical"):
        context.encode(carrier(), b"xy", KEY, dictionary, policy, selector="quality")


def _receiver(request, tmp_path):
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP") if name in os.environ}
    environment.update(PATH=os.defpath, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-I", str(ROOT / "src/synsteg_audit/invariant_context.py"), "receiver"],
        input=json.dumps(request).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmp_path, env=environment, timeout=30, check=False)


def test_independent_receiver_needs_no_original_cover_mask_or_payload(dictionary, policy, tmp_path):
    protocol = context.ContextProtocol()
    encoded = context.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    request = {"key_hex": KEY.hex(), "stego_text": encoded.stego_text, "public_dictionary": dictionary.to_dict(),
               "public_policy": policy.to_dict(), "public_protocol": protocol.to_dict()}
    result = _receiver(request, tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    response = json.loads(result.stdout)
    assert response["payload_hex"] == b"xy".hex()
    assert response["input_keys"] == sorted(request)
    assert response["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True}
    assert response["policy_sha256"] == policy.sha256
    assert response["protocol_sha256"] == protocol.sha256
    # Additional sender-only state is refused, rather than silently consumed.
    request["cover_text"] = carrier()
    rejected = _receiver(request, tmp_path)
    assert rejected.returncode != 0


def test_policy_and_protocol_do_not_mutate_while_encoding(dictionary, policy):
    protocol = context.ContextProtocol()
    before = json.dumps([policy.to_dict(), protocol.to_dict()], sort_keys=True)
    context.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    assert json.dumps([policy.to_dict(), protocol.to_dict()], sort_keys=True) == before
