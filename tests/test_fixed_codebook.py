import json
from dataclasses import FrozenInstanceError

import pytest

from synsteg_audit.fixed_codebook import (
    CapacityError, CodecError, FrozenCodebook, candidates_from_legacy_key,
    decode_bits, encode_bits, freeze_codebook,
)


def make_book(n=100, **kwargs):
    cover = " ".join(["small"] * n)
    return freeze_codebook(cover, [["tiny", "little", "compact", "short"]] * n, **kwargs)


@pytest.mark.parametrize("bits", ["", "0", "1", "00000000", "01011110110110101"])
@pytest.mark.parametrize("selection", ["canonical", "quality"])
def test_full_frame_roundtrip(bits, selection):
    book = make_book()
    result = encode_bits(book, bits, selection=selection)
    assert decode_bits(book, result.text) == bits
    assert result.embedded_bits == len(bits)
    assert result.frame_bits == len(bits) + 64
    assert result.external_metadata_bits == 0
    assert result.codebook_sha256 == book.sha256


def test_mapping_is_frozen_before_payload_and_not_mutated():
    calls = []

    def scorer(context):
        calls.append(context)
        assert context.original == "small"
        assert context.cover.count("small") == 100
        assert len(context.candidate_text.split()) == 100
        return float(len(context.candidate))

    book = make_book(scorer=scorer, scorer_id="length-demo-not-linguistic-quality")
    before, count = book.to_json(), len(calls)
    for bits in ["0" * 16, "1" * 16, "10100101"]:
        for variant in ["canonical", "quality"]:
            assert decode_bits(book, encode_bits(book, bits, selection=variant).text) == bits
    assert book.to_json() == before
    assert len(calls) == count
    with pytest.raises(FrozenInstanceError):
        book.seed = "changed"


def test_scoring_changes_choices_only_inside_the_same_bit_group():
    book = make_book(scorer=lambda c: len(c.candidate), scorer_id="word-length-diagnostic")
    bits = "101001100101"
    a = encode_bits(book, bits, selection="canonical")
    b = encode_bits(book, bits, selection="quality")
    assert a.codebook_sha256 == b.codebook_sha256
    assert decode_bits(book, a.text) == decode_bits(book, b.text) == bits
    for position, ca, cb in zip(book.positions, a.text.split(), b.text.split()):
        if ca == position.original and cb == position.original:
            continue
        group_a = next(i for i, group in enumerate(position.groups) if ca in group)
        group_b = next(i for i, group in enumerate(position.groups) if cb in group)
        assert group_a == group_b
        assert len(cb) <= len(ca)


def test_non_power_of_two_candidates_are_balanced_not_dropped():
    options = ["tiny", "little", "compact", "short", "brief", "minor"]
    book = freeze_codebook("small", [options], max_bits_per_position=3)
    groups = book.positions[0].groups
    assert list(map(len, groups)) == [4, 3]
    assert {s for g in groups for s in g} == set(options + ["small"])
    other = freeze_codebook("small", [list(reversed(options))], max_bits_per_position=3)
    assert book.sha256 == other.sha256


def test_singletons_explicitly_reported_for_low_capacity_variant():
    book = freeze_codebook("small small", [["tiny"], ["tiny", "little"]], min_group_size=1)
    assert book.capacity_bits == 2
    assert book.eligibility == {"positions": 2, "eligible_positions": 2, "groups": 4,
                                "singleton_groups": 3, "groups_with_quality_choice": 1}
    result = encode_bits(book, "10", framing="external_length")
    assert decode_bits(book, result.text, framing="external_length", expected_bits=2) == "10"
    assert result.external_metadata_bits == 32
    assert result.frame_bits == 2


def test_external_length_is_mandatory_and_cannot_be_implicit():
    book = make_book(8, min_group_size=1)
    result = encode_bits(book, "101", framing="external_length")
    with pytest.raises(CodecError, match="separate"):
        decode_bits(book, result.text, framing="external_length")
    with pytest.raises(CodecError, match="capacity"):
        decode_bits(book, result.text, framing="external_length", expected_bits=20)
    with pytest.raises(CodecError, match="own length"):
        decode_bits(book, result.text, expected_bits=3)
    assert result.framing == "external_length"


def test_empty_external_payload_leaves_cover_unchanged():
    book = make_book(3)
    result = encode_bits(book, "", framing="external_length")
    assert result.text == book.cover
    assert result.used_positions == 0
    assert decode_bits(book, result.text, framing="external_length", expected_bits=0) == ""


def test_capacity_shortfall_is_not_reported_as_prefix_success():
    book = make_book(70)
    before = book.to_json()
    with pytest.raises(CapacityError, match="Need 71"):
        encode_bits(book, "1" * 7)
    with pytest.raises(CapacityError, match="Need 71"):
        encode_bits(book, "1" * 71, framing="external_length")
    assert book.to_json() == before


def test_punctuation_whitespace_literals_and_unused_tail_are_exact():
    cover = "  [small],\t" + "small\n" * 80 + "(small)!  "
    book = freeze_codebook(cover, [["tiny", "little", "compact"]] * 82)
    result = encode_bits(book, "10101")
    assert result.text.startswith("  [") and result.text.endswith("(small)!  ")
    assert "\t" in result.text and result.text.count("\n") == 80
    assert decode_bits(book, result.text) == "10101"
    with pytest.raises(CodecError, match="layout"):
        decode_bits(book, result.text.replace("\t", " "))
    with pytest.raises(CodecError, match="tail"):
        decode_bits(book, result.text.removesuffix("(small)!  ") + "(tiny)!  ")


def test_multiword_and_punctuation_candidates_cannot_cross_positions():
    options = [["by means", "by means of", "tiny,", "tiny-small", "tiny", "little", "compact"]]
    book = freeze_codebook("small,", options)
    assert {f for group in book.positions[0].groups for f in group} == {"small,", "tiny,", "little,", "compact,"}
    with pytest.raises(CodecError, match="layout"):
        decode_bits(book, "by means of,", framing="external_length", expected_bits=1)


def test_case_collisions_are_removed_before_capacity_calculation():
    book = freeze_codebook("small", [["SMALL", "Small", "tiny", "TINY"]])
    assert book.capacity_bits == 0
    book = freeze_codebook("small", [["SMALL", "Small", "tiny", "TINY"]], min_group_size=1)
    assert book.capacity_bits == 1
    assert sorted(s for group in book.positions[0].groups for s in group) == ["small", "tiny"]


def test_capitalized_and_nonlexical_source_tokens_are_protected():
    book = freeze_codebook("London NASA iPhone can't café small", [["tiny", "little", "compact"]] * 6)
    assert [p.width for p in book.positions] == [0, 0, 0, 0, 0, 1]
    book = freeze_codebook("Small NASA", [["tiny", "little", "compact"]] * 2, protect_capitalized=False)
    assert all(s[0].isupper() for g in book.positions[0].groups for s in g)
    assert all(s.isupper() for g in book.positions[1].groups for s in g)


def test_context_filter_is_applied_before_mapping():
    book = freeze_codebook("small", [["tiny", "little", "compact", "short"]],
                           candidate_filter=lambda c: c.candidate != "short", candidate_filter_id="reject-short-v1")
    assert "short" not in {s for g in book.positions[0].groups for s in g}
    assert book.capacity_bits == 1


def test_crc_detects_payload_corruption_without_claiming_authentication():
    book = make_book()
    tokens = encode_bits(book, "010101010101").text.split()
    position = book.positions[32]
    current = next(i for i, g in enumerate(position.groups) if tokens[32] in g)
    tokens[32] = position.groups[1 - current][0]
    with pytest.raises(CodecError, match="checksum"):
        decode_bits(book, " ".join(tokens))


def test_zero_padding_is_checked_at_partial_final_group():
    book = make_book(40, max_bits_per_position=2, min_group_size=1)
    result = encode_bits(book, "1")
    assert result.padding_bits == 1
    assert decode_bits(book, result.text) == "1"
    tokens = result.text.split()
    pos = book.positions[32]
    current = next(i for i, g in enumerate(pos.groups) if tokens[32] in g)
    tokens[32] = pos.groups[current | 1][0]
    with pytest.raises(CodecError, match="padding"):
        decode_bits(book, " ".join(tokens))


def test_roundtrip_json_and_corrupt_codebook_rejection():
    book = make_book()
    restored = FrozenCodebook.from_dict(json.loads(book.to_json()))
    assert restored.sha256 == book.sha256
    result = encode_bits(book, "10110")
    assert decode_bits(restored, result.text) == "10110"
    malformed = json.loads(book.to_json())
    malformed["positions"][0]["groups"][1][0] = malformed["positions"][0]["groups"][0][0]
    with pytest.raises(CodecError, match="collision"):
        FrozenCodebook.from_dict(malformed)
    malformed = json.loads(book.to_json())
    malformed["positions"][0]["groups"][0][0] = "two words"
    with pytest.raises(CodecError, match="Whitespace"):
        FrozenCodebook.from_dict(malformed)
    malformed = json.loads(book.to_json())
    malformed["positions"][0]["groups"][0][0] = "tiny!"
    with pytest.raises(CodecError, match="punctuation"):
        FrozenCodebook.from_dict(malformed)
    malformed = json.loads(book.to_json())
    malformed["positions"][0]["original"] = "large"
    with pytest.raises(CodecError, match="hash"):
        FrozenCodebook.from_dict(malformed)


def test_legacy_adapter_discards_codes_and_checks_alignment():
    key = [{"Small": {"000": "tiny", "001": "little"}}, {"word": {"0": "term"}}]
    assert candidates_from_legacy_key("Small, word.", key) == [["tiny", "little"], ["term"]]
    changed_codes = [{"Small": {"111": "tiny", "101": "little"}}, key[1]]
    assert candidates_from_legacy_key("Small, word.", changed_codes) == candidates_from_legacy_key("Small, word.", key)
    with pytest.raises(CodecError, match="align"):
        candidates_from_legacy_key("Small", key)
    with pytest.raises(CodecError, match="original"):
        candidates_from_legacy_key("Large, word.", key)
    options = candidates_from_legacy_key("well-being. 2018,", [{"well-being": {"0": "welfare"}}, {"2018": {"0": "2018"}}])
    book = freeze_codebook("well-being. 2018,", options)
    assert book.capacity_bits == 0
    assert book.cover == "well-being. 2018,"


@pytest.mark.parametrize("bad", ["02", "hello", None, 10])
def test_nonbinary_payload_rejected(bad):
    with pytest.raises(CodecError, match="binary"):
        encode_bits(make_book(), bad)


def test_callbacks_require_identity_and_finite_scores():
    with pytest.raises(CodecError, match="identity"):
        make_book(scorer=lambda c: 1.0)
    with pytest.raises(CodecError, match="identity"):
        make_book(candidate_filter=lambda c: True)
    with pytest.raises(CodecError, match="non-finite"):
        make_book(scorer=lambda c: float("nan"), scorer_id="broken")
