"""Synthetic protocol checks; these texts are not a natural-language experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synsteg_audit.shared_dictionary import (
    AuthenticationError,
    CapacityError,
    CodecError,
    DecodeError,
    Protocol,
    PublicDictionary,
    analyze_text,
    decode,
    encode,
)


WORDS = {
    "size": ["compact", "dinky", "little", "miniature", "petite", "small", "tiny", "wee"],
    "pace": ["fast", "quick", "rapid", "swift"],
    "calm": ["calm", "placid", "serene", "tranquil"],
    "pair": ["huge", "large"],
}
# Public test fixture, never a production secret.
KEY = bytes(range(32))


@pytest.mark.parametrize("framing", ["aes_siv", "plain_length16"])
def test_fixed_v1_protocol_reference_vector(framing):
    # Stored expectations, not values produced by the current encoder during
    # this test: encoder+decoder drift under the same v1 must fail together.
    fixture_path = Path(__file__).parent / "fixtures" / "shared_dictionary_v1_vectors.json"
    fixture = json.loads(fixture_path.read_text())
    vector = next(item for item in fixture["vectors"] if item["protocol"]["framing"] == framing)
    dictionary = PublicDictionary.from_dict(fixture["dictionary"])
    protocol = Protocol(min_group_size=vector["protocol"]["min_group_size"],
                        max_bits_per_slot=vector["protocol"]["max_bits_per_slot"], framing=framing)
    key = bytes.fromhex(fixture["key_hex"])
    payload = bytes.fromhex(fixture["payload_hex"])
    assert dictionary.sha256 == fixture["dictionary_sha256"]
    assert protocol.to_dict() == vector["protocol"]
    assert protocol.sha256 == vector["protocol_sha256"]
    analysis = analyze_text(fixture["cover_text"], dictionary, protocol)
    assert analysis.normalization_sha256 == vector["normalization_sha256"]
    assert analysis.capacity_bits == vector["capacity_bits"]
    result = encode(fixture["cover_text"], payload, key, dictionary, protocol)
    assert result.stego_text == vector["stego_text"]
    for field in ("frame_bits", "consumed_slots", "padding_bits", "changed_slots"):
        assert getattr(result, field) == vector[field]
    assert decode(vector["stego_text"], key, dictionary, protocol) == payload
    recorded_frame = bytes.fromhex(vector["frame_hex"])
    assert int.from_bytes(recorded_frame[:2], "big") == len(payload)
    assert len(recorded_frame) * 8 == vector["frame_bits"]
    if framing == "plain_length16":
        assert recorded_frame[2:] == payload


@pytest.fixture
def dictionary():
    return PublicDictionary.from_mapping(WORDS, version="test-fixture-v1")


def carrier(n=160):
    return "Report:\n" + "\t ".join(["small,"] * n) + " END."


def replace_slot(text, dictionary, index, word, protocol=Protocol()):
    slot = analyze_text(text, dictionary, protocol=protocol).slots[index]
    return text[:slot.start] + word + text[slot.end:]


def accepted_alternatives(text, dictionary, index, expected, protocol=Protocol()):
    """Classify substitutions using only the public receiver interface."""
    slot = analyze_text(text, dictionary, protocol=protocol).slots[index]
    accepted, rejected = [], []
    for word in WORDS[slot.class_id]:
        if word == slot.surface:
            continue
        changed = replace_slot(text, dictionary, index, word, protocol)
        try:
            output = decode(changed, KEY, dictionary, protocol=protocol)
        except DecodeError as exc:
            rejected.append(exc)
        else:
            assert output == expected
            accepted.append(changed)
    return accepted, rejected


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x80\x01", bytes(range(32))])
@pytest.mark.parametrize("selector", ["canonical", "quality"])
def test_authenticated_complete_roundtrip(dictionary, payload, selector):
    kwargs = {"candidate_cost": lambda slot, candidate: -float(len(candidate))} if selector == "quality" else {}
    result = encode(carrier(240), payload, KEY, dictionary, selector=selector, **kwargs)
    assert decode(result.stego_text, KEY, dictionary) == payload
    assert result.payload_bytes == len(payload)
    assert result.frame_bits == 144 + 8 * len(payload)
    assert result.capacity_bits == 480
    assert result.consumed_slots == result.frame_bits // 2
    assert result.padding_bits == 0
    assert result.dictionary_sha256 == dictionary.sha256
    assert result.protocol_sha256 == Protocol().sha256
    assert result.normalization_sha256 == analyze_text(carrier(240), dictionary).normalization_sha256
    assert analyze_text(result.stego_text, dictionary).normalization_sha256 == result.normalization_sha256


def test_default_widths_and_exact_capacity(dictionary):
    analysis = analyze_text("small fast calm huge absent", dictionary)
    assert [slot.width for slot in analysis.slots] == [2, 1, 1]
    assert analysis.capacity_bits == 4
    assert analysis.max_payload_bytes is None
    assert analyze_text(carrier(72), dictionary).max_payload_bytes == 0
    assert analyze_text(carrier(76), dictionary).max_payload_bytes == 1
    assert analyze_text(carrier(80), dictionary).max_payload_bytes == 2
    diagnostic = analyze_text("small fast calm huge", dictionary, protocol=Protocol(min_group_size=1))
    assert [slot.width for slot in diagnostic.slots] == [2, 2, 2, 1]


def test_odd_class_sizes_keep_minimum_group_choice_and_balanced_labels():
    odd = PublicDictionary.from_mapping({
        "five": ["alpha", "bravo", "charlie", "delta", "echo"],
        "nine": ["able", "baker", "cable", "daring", "eager", "fable", "giant", "happy", "ideal"],
    }, version="odd-size-fixture")
    analysis = analyze_text("alpha able", odd)
    assert [slot.width for slot in analysis.slots] == [1, 2]
    diagnostic = Protocol(framing="plain_length16")
    text = encode(" ".join(["able"] * 24), b"xy", KEY, odd, protocol=diagnostic).stego_text
    slot = analyze_text(text, odd, protocol=diagnostic).slots[10]
    messages = {}
    for word in ["able", "baker", "cable", "daring", "eager", "fable", "giant", "happy", "ideal"]:
        variant = text[:slot.start] + word + text[slot.end:]
        message = decode(variant, KEY, odd, protocol=diagnostic)
        messages[message] = messages.get(message, 0) + 1
    assert sorted(messages.values()) == [2, 2, 2, 3]


def test_exact_fit_and_short_capacity_raise_without_partial_output(dictionary):
    assert decode(encode(carrier(76), b"x", KEY, dictionary).stego_text, KEY, dictionary) == b"x"
    with pytest.raises(CapacityError):
        encode(carrier(75), b"x", KEY, dictionary)
    with pytest.raises(CapacityError):
        encode(carrier(71), b"", KEY, dictionary)
    with pytest.raises(DecodeError):
        decode("small fast", KEY, dictionary)


def test_odd_last_slot_is_zero_padded_and_checked(dictionary):
    text = "fast " + carrier(90)
    result = encode(text, b"xy", KEY, dictionary)
    assert result.frame_bits == 160
    assert result.consumed_slots == 81
    assert result.padding_bits == 1
    assert decode(result.stego_text, KEY, dictionary) == b"xy"
    # A different label that changes only the spare final bit must not be
    # accepted. Exactly the other member of the original label is valid.
    accepted, rejected = accepted_alternatives(result.stego_text, dictionary, 80, b"xy")
    assert len(accepted) == 1
    assert len(rejected) == 6


def test_case_punctuation_and_ineligible_compounds_are_receiver_public(dictionary):
    text = '"small,"\tSmall!\nSMALL? sMaLl small-fast small\'s xsmall small2 _small яsmall smallя huge.'
    analysis = analyze_text(text, dictionary)
    assert [slot.surface for slot in analysis.slots] == ["small", "Small", "SMALL"]
    assert [slot.case_style for slot in analysis.slots] == ["lower", "title", "upper"]
    assert [text[slot.start:slot.end] for slot in analysis.slots] == ["small", "Small", "SMALL"]
    long_text = (text + "\n") * 40
    result = encode(long_text, b"\x00", KEY, dictionary)
    assert decode(result.stego_text, KEY, dictionary) == b"\x00"
    assert analyze_text(long_text, dictionary).normalization_sha256 == analyze_text(result.stego_text, dictionary).normalization_sha256
    assert result.stego_text.count("small-fast") == 40
    assert result.stego_text.count("small's") == 40
    assert result.stego_text.count("яsmall") == 40
    assert result.stego_text.count("smallя") == 40
    assert result.stego_text.count("\t") == long_text.count("\t")
    assert result.stego_text.count("\n") == long_text.count("\n")


def test_only_punctuation_can_surround_ascii_dictionary_core(dictionary):
    text = "small\u0301 \u0301small small\u200d \u200dsmall $small small+ 😀small small😀 «small» “Small” —SMALL—"
    slots = analyze_text(text, dictionary).slots
    assert [slot.surface for slot in slots] == ["small", "Small", "SMALL"]
    assert [slot.token_index for slot in slots] == [8, 9, 10]


def test_quality_changes_realization_but_not_receiver_requirements(dictionary):
    seen = []

    def reverse_lexical(slot, candidate):
        seen.append((slot.index, candidate))
        return -float(sum(ord(char) * 256 ** (20 - i) for i, char in enumerate(candidate)))

    canonical = encode(carrier(), b"xy", KEY, dictionary)
    quality = encode(carrier(), b"xy", KEY, dictionary, selector="quality", candidate_cost=reverse_lexical)
    assert seen
    assert quality.stego_text != canonical.stego_text
    assert decode(canonical.stego_text, KEY, dictionary) == decode(quality.stego_text, KEY, dictionary) == b"xy"
    assert canonical.normalization_sha256 == quality.normalization_sha256


def test_wrong_key_and_changed_noneligible_text_reject(dictionary):
    text = encode(carrier(), b"xy", KEY, dictionary).stego_text
    with pytest.raises(DecodeError):
        decode(text, bytes(reversed(KEY)), dictionary)
    with pytest.raises(DecodeError):
        decode(text.replace("Report:", "Report!", 1), KEY, dictionary)
    with pytest.raises(DecodeError):
        decode(text.replace("\t ", " ", 1), KEY, dictionary)


def test_header_and_ciphertext_bit_changes_reject_but_same_label_is_allowed(dictionary):
    text = encode(carrier(), b"xy", KEY, dictionary).stego_text
    header_accepted, header_rejected = accepted_alternatives(text, dictionary, 0, b"xy")
    body_accepted, body_rejected = accepted_alternatives(text, dictionary, 20, b"xy")
    assert len(header_accepted) == len(body_accepted) == 1
    assert len(header_rejected) == len(body_rejected) == 6
    assert all(isinstance(exc, AuthenticationError) for exc in body_rejected)
    # Within-class edits after the frame are also outside byte-level text
    # authentication; they do not change the invariant or extracted frame.
    unused = replace_slot(text, dictionary, 120, "tiny")
    assert decode(unused, KEY, dictionary) == b"xy"


def test_truncated_frame_and_changed_public_configuration_reject(dictionary):
    text = encode(carrier(), b"xy", KEY, dictionary).stego_text
    prefix = " ".join(text.split()[:50])
    with pytest.raises(DecodeError):
        decode(prefix, KEY, dictionary)
    different = PublicDictionary.from_mapping({**WORDS, "other": ["happy", "joyful"]}, version="test-fixture-v1")
    with pytest.raises(DecodeError):
        decode(text, KEY, different)
    with pytest.raises(DecodeError):
        decode(text, KEY, dictionary, protocol=Protocol(max_bits_per_slot=1))


def test_deterministic_siv_explicitly_repeats_same_context_and_payload(dictionary):
    first = encode(carrier(), b"xy", KEY, dictionary)
    second = encode(carrier(), b"xy", KEY, dictionary)
    assert first.stego_text == second.stego_text
    assert encode(carrier(), b"xz", KEY, dictionary).stego_text != first.stego_text


def test_plain_diagnostic_has_only_length_overhead_and_separate_protocol(dictionary):
    diagnostic = Protocol(framing="plain_length16")
    assert diagnostic.frame_overhead_bits == 16
    assert Protocol().frame_overhead_bits == 144
    assert diagnostic.sha256 != Protocol().sha256
    analysis = analyze_text(carrier(16), dictionary, protocol=diagnostic)
    assert analysis.max_payload_bytes == 2
    result = encode(carrier(16), b"\x00\xff", KEY, dictionary, protocol=diagnostic)
    assert result.frame_bits == 32
    assert decode(result.stego_text, KEY, dictionary, protocol=diagnostic) == b"\x00\xff"
    with pytest.raises(CapacityError):
        encode(carrier(15), b"\x00\xff", KEY, dictionary, protocol=diagnostic)
    with pytest.raises(DecodeError):
        decode(result.stego_text, KEY, dictionary)


def test_plain_diagnostic_silently_accepts_payload_bit_substitution(dictionary):
    diagnostic = Protocol(framing="plain_length16")
    result = encode(carrier(16), b"xy", KEY, dictionary, protocol=diagnostic)
    slot = analyze_text(result.stego_text, dictionary, protocol=diagnostic).slots[10]
    outputs = []
    for word in WORDS[slot.class_id]:
        if word != slot.surface:
            changed = replace_slot(result.stego_text, dictionary, 10, word, diagnostic)
            outputs.append(decode(changed, KEY, dictionary, protocol=diagnostic))
    assert sum(output == b"xy" for output in outputs) == 1
    assert sum(output != b"xy" for output in outputs) == 6


@pytest.mark.parametrize("kwargs", [
    {"min_group_size": 0},
    {"min_group_size": True},
    {"max_bits_per_slot": 0},
    {"max_bits_per_slot": 3},
    {"max_bits_per_slot": True},
    {"framing": "automatic"},
])
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(CodecError):
        Protocol(**kwargs)


def test_dictionary_fingerprint_is_canonical_and_metadata_is_not_secret_state(dictionary):
    reverse = {key: list(reversed(value)) for key, value in reversed(list(WORDS.items()))}
    assert PublicDictionary.from_mapping(reverse, version="test-fixture-v1").sha256 == dictionary.sha256
    data = {"schema": "synsteg-public-dictionary-v1", "version": "test-fixture-v1", "source": "public fixture", "classes": [{"id": key, "words": value, "comment": "not used by codec"} for key, value in WORDS.items()]}
    assert PublicDictionary.from_dict(data).sha256 == dictionary.sha256
    assert PublicDictionary.from_mapping(WORDS, version="test-fixture-v2").sha256 != dictionary.sha256


@pytest.mark.parametrize("mapping", [
    {"a": ["small", "tiny"], "b": ["small", "little"]},
    {"a": ["small", "small"]},
    {"a": ["small"]},
    {"a": ["small", "Tiny"]},
    {"a": ["small", "two words"]},
    {"a": ["small", "naïve"]},
    {"a": ["small", "small-word"]},
])
def test_invalid_or_overlapping_public_dictionary_rejected(mapping):
    with pytest.raises(CodecError):
        PublicDictionary.from_mapping(mapping, version="invalid-fixture")


def test_single_letter_members_rejected_before_titlecase_can_change_invariant():
    # "Small" -> "A" would change the public case marker from title to upper.
    # Reject this dictionary at construction, rather than after partial work.
    with pytest.raises(CodecError, match="at least two letters"):
        PublicDictionary.from_mapping({
            "fixture": ["a", "able", "brief", "short", "small", "tiny", "little", "petite"]
        })


def test_two_letter_members_preserve_title_and_upper_case_roundtrip():
    dictionary = PublicDictionary.from_mapping({"fixture": ["at", "be", "do", "go"]})
    cover = " ".join(["At", "AT"] * 90)
    result = encode(cover, b"x", KEY, dictionary)
    assert decode(result.stego_text, KEY, dictionary) == b"x"
    original = analyze_text(cover, dictionary)
    rendered = analyze_text(result.stego_text, dictionary)
    assert original.normalization_sha256 == rendered.normalization_sha256
    assert [slot.case_style for slot in rendered.slots] == ["title", "upper"] * 90


@pytest.mark.parametrize("bad_key", [b"", b"x" * 31, b"x" * 33, "x" * 32])
def test_invalid_key_rejected_before_encoding(dictionary, bad_key):
    with pytest.raises(CodecError):
        encode(carrier(), b"x", bad_key, dictionary)


def test_invalid_payload_selector_and_scorer_rejected(dictionary):
    with pytest.raises(CodecError):
        encode(carrier(), "x", KEY, dictionary)
    with pytest.raises(CodecError):
        encode(carrier(), b"x" * 65536, KEY, dictionary)
    with pytest.raises(CodecError):
        encode(carrier(), b"x", KEY, dictionary, selector="unknown")
    with pytest.raises(CodecError):
        encode(carrier(), b"x", KEY, dictionary, selector="quality")
    with pytest.raises(CodecError):
        encode(carrier(), b"x", KEY, dictionary, selector="quality", candidate_cost=lambda slot, word: float("nan"))


def test_receiver_in_fresh_process_needs_no_cover_or_document_codebook(dictionary):
    encoded = encode(carrier(), b"\x00\xff", KEY, dictionary)
    request = {"key_hex": KEY.hex(), "stego_text": encoded.stego_text, "dictionary": WORDS, "dictionary_version": "test-fixture-v1"}
    script = """
import json, sys
from synsteg_audit.shared_dictionary import PublicDictionary, decode
r = json.load(sys.stdin)
d = PublicDictionary.from_mapping(r['dictionary'], version=r['dictionary_version'])
print(decode(r['stego_text'], bytes.fromhex(r['key_hex']), d).hex())
"""
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src)
    proc = subprocess.run([sys.executable, "-c", script], input=json.dumps(request), text=True, capture_output=True, env=env, check=True, timeout=30)
    assert proc.stdout.strip() == "00ff"
