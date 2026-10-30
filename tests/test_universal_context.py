"""Mechanical protocol tests; successful roundtrips are not semantic evidence."""
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as uc

ROOT = Path(__file__).resolve().parents[1]
KEY = bytes(range(32))  # Public nonproduction fixture.
WORDS = {"pace": ["quick", "rapid"], "brightness": ["bright", "clear", "lucid", "vivid"],
         "matter": ["affair", "matter"], "sex": ["sportsman", "sportswoman"],
         "process": ["conduction", "conductivity"]}
REFERENCE = {key: value for key, value in WORDS.items() if key != "matter"}


@pytest.fixture
def dictionary():
    return base.PublicDictionary.from_mapping(WORDS, "universal-synthetic-carrier-v1")


@pytest.fixture
def policy(dictionary):
    return uc.UniversalPolicy.from_dictionaries(dictionary,
        base.PublicDictionary.from_mapping(REFERENCE, "universal-synthetic-reference-v1"))


def carrier():
    return 'Intro: an "quick" matter QUICK.\n' + "quick bright\t" * 180 + "\nEnd."


def signature(analysis):
    return (analysis.candidate_slot_count, analysis.protected_slot_count, analysis.eligible_slot_count,
            analysis.raw_capacity_bits, analysis.capacity_bits, analysis.protected_token_indices,
            analysis.eligible_token_indices, analysis.skeleton_sha256, analysis.slot_decisions)


def test_joint_orbit_closes_single_substitution_counterexample():
    safe = lambda row: row != ("bb", "yy")
    assert all(safe(row) for row in [("aa", "xx"), ("bb", "xx"), ("aa", "yy")])
    result = uc.local_orbit_acceptance((("aa", "bb"), ("xx", "yy")), safe, cap=4)
    assert result == uc.OrbitResult(4, 4, False, "predicate_rejected")
    # Reordering or duplicating a coordinate does not alter the canonical orbit.
    assert uc.local_orbit_acceptance((("bb", "aa", "aa"), ("yy", "xx")), safe, cap=4) == result


def test_orbit_acceptance_visits_all_independent_repeated_coordinates():
    seen = []
    options = (("aa", "bb"),) * 3
    result = uc.local_orbit_acceptance(options, lambda row: seen.append(row) is None, cap=8)
    assert result == uc.OrbitResult(8, 8, True, "accepted")
    assert seen == list(itertools.product(*options))
    assert ("aa", "bb", "aa") in seen  # Repeated class occurrences are not tied.


def test_cap_checked_before_any_predicate_call():
    calls = []
    pred = lambda row: calls.append(row) is None
    assert uc.local_orbit_acceptance((("aa", "bb"),) * 3, pred, cap=7) == uc.OrbitResult(8, 0, False, "orbit_cap_exceeded")
    assert calls == []
    assert uc.local_orbit_acceptance((("aa", "bb"),) * 3, pred, cap=8).accepted
    assert len(calls) == 8


@pytest.mark.parametrize("verdict", [None, 1, 0, float("nan"), "yes", [], {}])
def test_non_boolean_judge_results_fail_closed(verdict):
    result = uc.local_orbit_acceptance((("aa", "bb"),), lambda row: verdict, cap=2)
    assert result == uc.OrbitResult(2, 1, False, "predicate_error")


def test_judge_exception_fails_closed():
    def failing(row):
        raise RuntimeError("unavailable")
    assert uc.local_orbit_acceptance((("aa", "bb"),), failing, cap=2).reason == "predicate_error"


@pytest.mark.parametrize("options,cap", [((), 1), (((),), 1), ((("",),), 1), (((42,),), 1),
    ((("aa",),), 0), ((("aa",),), True), ((("aa",),), 1_000_001), ("aa", 1)])
def test_malformed_orbit_rejected(options, cap):
    with pytest.raises(uc.UniversalContextError):
        uc.local_orbit_acceptance(options, lambda row: True, cap=cap)


def test_all_class_aliases_keep_skeleton_mask_and_decisions(dictionary, policy):
    reference = None
    for first, middle, last in itertools.product(WORDS["pace"], WORDS["matter"], WORDS["brightness"]):
        text = f'an "{first}" {middle}. {last}\tquick'
        value = uc.analyze_text(text, dictionary, policy)
        if reference is None:
            reference = signature(value)
        assert signature(value) == reference
        assert value.protected_token_indices == (1, 2)
        assert value.eligible_token_indices == (3, 4)


def test_reference_requires_whole_class_and_id(dictionary):
    altered = dict(REFERENCE)
    altered["brightness"] = ["bright", "clear"]  # A subset cannot be silently used.
    altered["different-id"] = altered.pop("pace")
    policy = uc.UniversalPolicy.from_dictionaries(dictionary,
        base.PublicDictionary.from_mapping(altered, "subset-reference"))
    a = uc.analyze_text("quick bright conduction", dictionary, policy)
    assert a.protected_token_indices == (0, 1)
    assert a.eligible_token_indices == (2,)
    assert [d.reason for d in a.slot_decisions] == ["class_not_exact_reference", "class_not_exact_reference", "accepted"]


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_article_rule_cannot_be_disabled_by_radius(dictionary, policy, radius):
    policy = replace(policy, radius=radius)
    a = uc.analyze_text('a quick an "bright" the rapid', dictionary, policy)
    assert a.protected_token_indices == (1, 3)
    assert a.eligible_token_indices == (5,)
    assert a.slot_decisions[0].reason == "adjacent_indefinite_article"


def test_neighbor_orbit_includes_rejected_classes_and_every_alias():
    d = base.PublicDictionary.from_mapping({"article": ["an", "some"], "pace": ["quick", "rapid"]}, "neighbor-carrier")
    ref = base.PublicDictionary.from_mapping({"pace": ["quick", "rapid"]}, "neighbor-reference")
    p = uc.UniversalPolicy.from_dictionaries(d, ref)
    values = [uc.analyze_text(f"{article} {pace}", d, p) for article, pace in itertools.product(["an", "some"], ["quick", "rapid"])]
    assert all(signature(value) == signature(values[0]) for value in values)
    assert values[0].eligible_slot_count == 0
    assert values[0].slot_decisions[1].reason == "adjacent_indefinite_article"
    assert values[0].slot_decisions[1].orbit_size == 4


def test_case_punctuation_and_opaque_token_rules(dictionary, policy):
    text = 'Quick QUICK quick, (rapid) quick-based bright_clear quIck \ue002quick\ue003'
    a = uc.analyze_text(text, dictionary, policy)
    assert a.candidate_slot_count == 4
    assert a.protected_token_indices == (0, 1)
    assert a.eligible_token_indices == (2, 3)
    assert [d.reason for d in a.slot_decisions[:2]] == ["case_not_lowercase"] * 2


def test_actual_orbit_cap_abstention_invariant(dictionary, policy):
    p = replace(policy, orbit_cap=7)
    # Each interior occurrence has 2^3 variants; each boundary has 2^2.
    a = uc.analyze_text("quick quick quick", dictionary, p)
    b = uc.analyze_text("rapid quick rapid", dictionary, p)
    assert signature(a) == signature(b)
    assert a.protected_token_indices == (1,)
    assert a.slot_decisions[1] == uc.SlotDecision(1, "pace", 8, 0, False, "orbit_cap_exceeded")
    assert a.eligible_token_indices == (0, 2)


@pytest.mark.parametrize("framing", ["plain_length16", "aes_siv"])
@pytest.mark.parametrize("min_group,max_bits", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_roundtrip_variable_lengths_all_protocol_widths(dictionary, policy, framing, min_group, max_bits):
    protocol = uc.UniversalProtocol(min_group, max_bits, framing)
    before = uc.analyze_text(carrier(), dictionary, policy, protocol)
    encoded = uc.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    assert encoded.stego_text.startswith('Intro: an "quick" matter QUICK.\n')
    assert uc.decode(encoded.stego_text, KEY, dictionary, policy, protocol) == b"xy"
    assert uc.analyze_text(encoded.stego_text, dictionary, policy, protocol) == before
    assert encoded.capacity_bits <= encoded.raw_capacity_bits
    assert len(before.slot_decisions) == before.candidate_slot_count
    # Potential slots remain in decisions even when final framing excludes them.
    assert before.candidate_slot_count == 363


def test_exact_plain_frame_boundary_and_zero_capacity(dictionary, policy):
    protocol = uc.UniversalProtocol(framing="plain_length16")
    a = uc.analyze_text("quick " * 23, dictionary, policy, protocol)
    assert a.capacity_bits == 23 and a.max_payload_bytes == 0
    with pytest.raises(base.CapacityError):
        uc.encode("quick " * 23, b"x", KEY, dictionary, policy, protocol)
    encoded = uc.encode("quick " * 24, b"x", KEY, dictionary, policy, protocol)
    assert encoded.frame_bits == encoded.capacity_bits == 24
    assert uc.decode(encoded.stego_text, KEY, dictionary, policy, protocol) == b"x"
    zero = uc.analyze_text("an quick MATTER", dictionary, policy, protocol)
    assert zero.capacity_bits == 0 and zero.max_payload_bytes is None


def test_aes_binds_protected_spellings_policy_and_key(dictionary, policy):
    result = uc.encode(carrier(), b"xy", KEY, dictionary, policy)
    for tampered in [result.stego_text.replace("matter", "affair", 1),
                     result.stego_text.replace('an "quick"', 'an "rapid"', 1)]:
        assert signature(uc.analyze_text(tampered, dictionary, policy)) == signature(uc.analyze_text(result.stego_text, dictionary, policy))
        with pytest.raises(base.CodecError):
            uc.decode(tampered, KEY, dictionary, policy)
    with pytest.raises(base.CodecError):
        uc.decode(result.stego_text, bytes(reversed(KEY)), dictionary, policy)
    for changed in [replace(policy, version="changed"), replace(policy, orbit_cap=4095), replace(policy, radius=2)]:
        with pytest.raises(base.CodecError):
            uc.decode(result.stego_text, KEY, dictionary, changed)
    with pytest.raises(base.CodecError):
        base.decode(result.stego_text, KEY, dictionary, base.Protocol(1, 1, "aes_siv"))


@pytest.mark.parametrize("reserved", ["\ue000", "\ue001", uc._marker(2, "quick")])
def test_reserved_external_marker_rejected_on_all_paths(dictionary, policy, reserved):
    text = reserved + carrier()
    for call in [lambda: uc.analyze_text(text, dictionary, policy),
                 lambda: uc.encode(text, b"x", KEY, dictionary, policy),
                 lambda: uc.decode(text, KEY, dictionary, policy)]:
        with pytest.raises(uc.UniversalContextError, match="reserved"):
            call()


def test_policy_protocol_canonical_validation_and_bindings(dictionary, policy):
    assert uc.UniversalPolicy.from_dict(policy.to_dict(), dictionary) == policy
    protocol = uc.UniversalProtocol()
    assert uc.UniversalProtocol.from_dict(protocol.to_dict()) == protocol
    for mutation in [lambda p: p.update(extra=True), lambda p: p["predicate"].update(scope="semantic-guarantee"),
                     lambda p: p.update(reference_dictionary_sha256="0" * 64),
                     lambda p: p.update(radius=True), lambda p: p.update(radius=3),
                     lambda p: p.update(orbit_cap=0), lambda p: p.update(orbit_cap=True),
                     lambda p: p["strict_dictionary"]["classes"].reverse()]:
        data = policy.to_dict()
        mutation(data)
        with pytest.raises(uc.UniversalContextError):
            uc.UniversalPolicy.from_dict(data, dictionary)
    other = base.PublicDictionary(dictionary.classes, "other-version")
    with pytest.raises(uc.UniversalContextError):
        uc.analyze_text("quick", other, policy)
    data = protocol.to_dict()
    data["version"] = "old"
    with pytest.raises(uc.UniversalContextError):
        uc.UniversalProtocol.from_dict(data)
    with pytest.raises(uc.UniversalContextError):
        uc.encode(carrier(), b"x", KEY, dictionary, policy, selector="quality")


def test_known_semantic_counterexamples_are_not_claimed_fixed(dictionary, policy):
    for text in ["the sportsman won", "the sportswoman won", "thermal conduction is a process", "thermal conductivity is a process"]:
        assert uc.analyze_text(text, dictionary, policy).eligible_slot_count == 1
    assert "not-semantic" in policy.to_dict()["predicate"]["scope"]


def test_pinned_real_resources_reference_counts_and_counterexamples():
    d = base.PublicDictionary.from_dict(json.loads((ROOT / "data/shared_dictionary_recovery_v1.json").read_text()))
    ref = base.PublicDictionary.from_dict(json.loads((ROOT / "data/shared_dictionary_recovery_v2.json").read_text()))
    p = uc.UniversalPolicy.from_dictionaries(d, ref)
    assert len(p.admitted_class_ids(d)) == 174
    assert sum(len(entry.words) for entry in d.classes if entry.class_id in p.admitted_class_ids(d)) == 360
    assert len(uc.UniversalPolicy.from_dictionaries(ref, ref).admitted_class_ids(ref)) == 239
    for text, surface in [("the sportsman won", "sportsman"), ("thermal conduction is a process", "conduction")]:
        raw = base.analyze_text(text, d, base.Protocol(1, 1, "plain_length16"))
        index = next(slot.token_index for slot in raw.slots if slot.surface == surface)
        assert index in uc.analyze_text(text, d, p).eligible_token_indices


def _receiver(request, tmp_path):
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP") if name in os.environ}
    env.update(PATH=os.defpath, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-I", str(ROOT / "src/synsteg_audit/universal_context.py"), "receiver"],
        input=json.dumps(request).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmp_path, env=env, timeout=30, check=False)


def test_fresh_receiver_public_resources_only(dictionary, policy, tmp_path):
    protocol = uc.UniversalProtocol()
    result = uc.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    request = {"key_hex": KEY.hex(), "stego_text": result.stego_text,
        "public_dictionary": dictionary.to_dict(), "public_policy": policy.to_dict(), "public_protocol": protocol.to_dict()}
    process = _receiver(request, tmp_path)
    assert process.returncode == 0, process.stderr.decode()
    response = json.loads(process.stdout)
    assert response["status"] == "decoded"
    assert response["dictionary_sha256"] == dictionary.sha256
    assert response["payload_hex"] == b"xy".hex()
    assert response["input_keys"] == sorted(request)
    assert response["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True}
    assert response["policy_sha256"] == policy.sha256
    assert response["protocol_sha256"] == protocol.sha256
    request["cover_text"] = carrier()
    assert _receiver(request, tmp_path).returncode != 0


def test_input_resources_do_not_mutate(dictionary, policy):
    protocol = uc.UniversalProtocol()
    before = json.dumps([dictionary.to_dict(), policy.to_dict(), protocol.to_dict()], sort_keys=True)
    uc.encode(carrier(), b"xy", KEY, dictionary, policy, protocol)
    assert json.dumps([dictionary.to_dict(), policy.to_dict(), protocol.to_dict()], sort_keys=True) == before
