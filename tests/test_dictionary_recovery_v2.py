"""Structural/source tests, not evidence of contextual synonymy or security."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synsteg_audit.shared_dictionary import PublicDictionary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_shared_dictionary_recovery_v2.py"
SPEC = importlib.util.spec_from_file_location("dictionary_recovery_v2", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def source_members():
    members = {name: b"" for name in builder.REQUIRED_MEMBERS}
    members["wordnet/LICENSE"] = b"WordNet Release 3.0\nPrinceton University\nSynthetic fixture only.\n"
    senses, records = [], {name: [] for name in builder.POS_NAMES.values()}

    def add(pos, offset, words, ss_type=None):
        ss_type = ss_type or pos
        type_id = {"n": 1, "v": 2, "a": 3, "r": 4, "s": 5}[ss_type]
        text = " ".join(f"{word} 0" for word, number, count in words)
        records[builder.POS_NAMES[pos]].append(f"{offset:08d} 00 {ss_type} {len(words):02x} {text} 000 | fixture {offset}")
        for word, number, count in words:
            lemma = word.removesuffix("(a)").lower()
            tail = "bright:00" if ss_type == "s" else ":"
            senses.append(f"{lemma}%{type_id}:00:00:{tail} {offset:08d} {number} {count}")

    add("n", 1, [("alpha", 1, 10), ("beta", 1, 20), ("poly", 1, 100)])
    add("v", 2, [("gamma", 1, 10), ("delta", 1, 9), ("crosspos", 1, 100)])
    add("a", 3, [("sunny", 1, 2), ("cheery", 1, 2), ("quick(a)", 1, 99)], "s")
    # Ambiguity exists even though these extra senses were never tagged.
    add("n", 4, [("poly", 2, 0), ("crosspos", 1, 0)])
    add("a", 5, [("drawn", 1, 20), ("careworn", 1, 3)])
    add("n", 6, [("terms", 1, 20), ("footing", 1, 20), ("solid", 1, 3), ("stable", 1, 4)])
    add("n", 7, [("term", 1, 0)])
    add("v", 8, [("foot", 1, 0)])
    add("a", 9, [("happier", 1, 4), ("happiest", 1, 5), ("joyous", 1, 3), ("gleeful", 1, 3)])
    add("a", 10, [("happy", 1, 0)])
    add("n", 11, [("can", 1, 999), ("will", 1, 999), ("light", 2, 99), ("dark", 1, 0), ("multi_word", 1, 99), ("London", 1, 99), ("x", 1, 99)])
    members["wordnet/index.sense"] = ("\n".join(senses) + "\n").encode()
    for name, lines in records.items():
        members[f"wordnet/data.{name}"] = ("\n".join(lines) + "\n").encode()
    # A source exception excludes even if its base has no indexed entry.
    members["wordnet/verb.exc"] = b"drawn draw\n"
    members["wordnet/adj.exc"] = b"happier happy\nhappiest happy\n"
    return members


def test_global_ambiguity_and_morphology_are_excluded(source_members):
    dictionary, audit = builder.build(source_members)
    actual = {entry["id"]: entry["words"] for entry in dictionary["classes"]}
    assert actual == {
        "wn30-a-00000003": ["cheery", "sunny"],
        "wn30-a-00000009": ["gleeful", "joyous"],
        "wn30-n-00000001": ["alpha", "beta"],
        "wn30-n-00000006": ["solid", "stable"],
        "wn30-v-00000002": ["delta", "gamma"],
    }
    assert audit["counts"]["strict_reason_not_exactly_one_pos"] == 1
    assert audit["counts"]["strict_reason_not_exactly_one_synset"] == 2
    assert audit["counts"]["strict_reason_potential_inflection"] == 5
    assert PublicDictionary.from_dict(dictionary).sha256 == audit["dictionary"]["canonical_sha256"]


def test_untagged_senses_are_part_of_ambiguity_check(source_members):
    senses = builder.parse_senses(source_members["wordnet/index.sense"].decode())
    memberships, lexicon = builder.source_lexicon(senses)
    exclusions = builder.parse_exceptions(source_members)
    assert builder.static_exclusions("poly", memberships, lexicon, exclusions)["reasons"] == ["not_exactly_one_synset"]
    assert builder.static_exclusions("crosspos", memberships, lexicon, exclusions)["reasons"] == ["not_exactly_one_pos", "not_exactly_one_synset"]


def test_duplicate_lexical_ids_in_one_synset_do_not_invent_polysemy():
    senses = builder.parse_senses("alpha%1:00:00:: 00000001 1 3\nalpha%1:00:01:: 00000001 1 2\n")
    memberships, _ = builder.source_lexicon(senses)
    assert memberships["alpha"] == {("n", 1)}


@pytest.mark.parametrize("word,pos,base,rule", [
    ("leaves", "n", "leaf", "ves->f"), ("studies", "n", "study", "ies->y"),
    ("talked", "v", "talk", "ed->"), ("making", "v", "make", "ing->e"),
    ("happier", "a", "happy", "exception"), ("bigger", "a", "big", "exception"),
    ("best", "r", "well", "exception"),
])
def test_explicit_morphology_routes(word, pos, base, rule):
    lexicon = {p: set() for p in builder.POS_NAMES}
    exceptions = {p: {} for p in builder.POS_NAMES}
    lexicon[pos].add(base)
    if rule == "exception":
        exceptions[pos][word] = (base,)
    evidence = builder.morphology_evidence(word, lexicon, exceptions)
    assert any(item["pos"] == pos and item["base"] == base and (item["kind"] == "exception" if rule == "exception" else item["rule"] == rule) for item in evidence)


def test_suffix_requires_source_base_and_self_exception_is_ignored():
    lexicon = {p: set() for p in builder.POS_NAMES}
    exceptions = {p: {} for p in builder.POS_NAMES}
    exceptions["a"]["after"] = ("after",)
    assert builder.morphology_evidence("after", lexicon, exceptions) == []
    assert builder.morphology_evidence("talked", lexicon, exceptions) == []


def test_exception_base_need_not_be_indexed():
    lexicon = {p: set() for p in builder.POS_NAMES}
    exceptions = {p: {} for p in builder.POS_NAMES}
    exceptions["v"]["drawn"] = ("draw",)
    assert builder.morphology_evidence("drawn", lexicon, exceptions) == [{"pos": "v", "base": "draw", "kind": "exception"}]


def test_no_recursive_stemming_or_morphological_additions(source_members):
    lexicon = {p: set() for p in builder.POS_NAMES}
    exceptions = {p: {} for p in builder.POS_NAMES}
    lexicon["n"].add("word")
    assert builder.morphology_evidence("wordss", lexicon, exceptions) == []
    dictionary, _ = builder.build(source_members)
    words = {word for entry in dictionary["classes"] for word in entry["words"]}
    assert not {"draw", "foot", "term", "happy"} & words


def test_source_order_and_hash_seed_do_not_change_output(source_members, tmp_path):
    dictionary, audit = builder.build(source_members)
    reverse = {name: (b"\n".join(reversed(raw.splitlines())) + (b"\n" if raw else b"") if name != "wordnet/LICENSE" else raw) for name, raw in source_members.items()}
    other, other_audit = builder.build(reverse)
    assert builder.json_bytes(dictionary) == builder.json_bytes(other)
    assert audit["selected_classes"] == other_audit["selected_classes"]
    fixture = tmp_path / "source.json"
    fixture.write_text(json.dumps({name: raw.decode() for name, raw in source_members.items()}))
    code = """
import importlib.util,json,pathlib,sys
s=importlib.util.spec_from_file_location('b',sys.argv[1]); b=importlib.util.module_from_spec(s); s.loader.exec_module(b)
m={k:v.encode() for k,v in json.loads(pathlib.Path(sys.argv[2]).read_text()).items()}
sys.stdout.buffer.write(b.json_bytes(b.build(m)))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", code, str(SCRIPT), str(fixture)], env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ("1", "777")]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("member", builder.REQUIRED_MEMBERS)
def test_missing_source_member_is_rejected(source_members, member):
    del source_members[member]
    with pytest.raises(builder.BuildError, match="source member"):
        builder.build(source_members)


def test_malformed_exception_and_sense_records_fail(source_members):
    source_members["wordnet/noun.exc"] = b"missingbase\n"
    with pytest.raises(builder.BuildError, match="Malformed noun.exc"):
        builder.build(source_members)
    with pytest.raises(builder.BuildError, match="Malformed index.sense"):
        builder.parse_senses("word%1:00:00:: 00000001 1 -1\n")


def test_archive_identity_mismatch_creates_no_dictionary(tmp_path, capsys):
    archive, out = tmp_path / "wordnet.zip", tmp_path / "dictionary.json"
    archive.write_bytes(b"untrusted archive")
    assert builder.main(["--wordnet-zip", str(archive), "--output", str(out), "--provenance-dir", str(tmp_path / "audit")]) == 2
    assert "identity mismatch" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("name", ["shared_dictionary_ai_v1.json", "shared_dictionary_ai_v2.json", "shared_dictionary_v1.json", "shared_dictionary_recovery_v1.json"])
def test_v1_and_historical_filenames_are_protected(source_members, tmp_path, name):
    with pytest.raises(builder.BuildError, match="historical dictionary"):
        builder.write_outputs(*builder.build(source_members), tmp_path / name, tmp_path / "audit")
    assert not (tmp_path / name).exists()


def test_output_collision_and_partial_overwrite_protection(source_members, tmp_path):
    dictionary, audit = builder.build(source_members)
    with pytest.raises(builder.BuildError, match="paths overlap"):
        builder.write_outputs(dictionary, audit, tmp_path / "provenance.json", tmp_path)
    folder, output = tmp_path / "audit", tmp_path / "dictionary.json"
    builder.write_outputs(dictionary, audit, output, folder)
    builder.write_outputs(dictionary, audit, output, folder)
    output.unlink()
    (folder / "provenance.json").write_text("different evidence")
    with pytest.raises(builder.BuildError, match="Refusing to overwrite"):
        builder.write_outputs(dictionary, audit, output, folder)
    assert not output.exists()


def test_shipped_resource_identity_license_and_v1_preservation():
    output = ROOT / "data/shared_dictionary_recovery_v2.json"
    folder = ROOT / "results/dictionary-recovery-v2"
    raw = output.read_bytes()
    dictionary, audit = json.loads(raw), json.loads((folder / "provenance.json").read_bytes())
    parsed = PublicDictionary.from_dict(dictionary)
    assert dictionary["version"] == "wordnet-recovery-v2"
    assert parsed.sha256 == audit["dictionary"]["canonical_sha256"]
    assert builder.sha256(raw) == audit["dictionary"]["file_sha256"]
    assert len(raw) == audit["dictionary"]["file_bytes"]
    assert builder.builder_source_sha256() == audit["builder_sha256"]
    assert audit["rules"] == builder.rules()
    assert dictionary["license"]["text"].encode() == (folder / "WORDNET-LICENSE.txt").read_bytes()
    assert builder.sha256((ROOT / "data/shared_dictionary_recovery_v1.json").read_bytes()) == "a6266ac3c120a081970e3b74024b8618f2d69c0c1ad98e812135f30d92343239"
    audit_classes = {entry["id"]: entry for entry in audit["selected_classes"]}
    for entry in parsed.classes:
        members = audit_classes[entry.class_id]["members"]
        assert tuple(item["word"] for item in members) == entry.words
        assert all(item["sense_number"] == 1 and item["tag_count"] >= 1 for item in members)
        assert not set(entry.words) & builder.DENYLIST
    assert len(parsed.classes) == audit["counts"]["classes"]
    assert sum(len(entry.words) for entry in parsed.classes) == audit["counts"]["words"]


def test_known_development_failure_tokens_are_general_rule_regressions():
    # Diagnostic regression only: these words do not occur in any builder ban list.
    dictionary = json.loads((ROOT / "data/shared_dictionary_recovery_v2.json").read_bytes())
    words = {word for entry in dictionary["classes"] for word in entry["words"]}
    examples = {"time", "clock", "study", "analyse", "terms", "footing", "matter", "affair", "drawn", "haggard"}
    assert not examples & words
    assert not examples & builder.DENYLIST
