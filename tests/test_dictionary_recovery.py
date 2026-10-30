"""Construction/audit tests; synthetic inputs are not language-quality evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synsteg_audit.shared_dictionary import PublicDictionary


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_shared_dictionary_recovery.py"
SPEC = importlib.util.spec_from_file_location("dictionary_recovery", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def source_members():
    members = {name: b"" for name in builder.REQUIRED_MEMBERS}
    members["wordnet/LICENSE"] = b"WordNet Release 3.0\nPrinceton University\nSynthetic test license.\n"
    senses = []
    records = {name: [] for name in builder.POS_NAMES.values()}

    def add(pos, offset, words, *, ss_type=None):
        ss_type = ss_type or pos
        type_id = {"n": 1, "v": 2, "a": 3, "r": 4, "s": 5}[ss_type]
        lexical = " ".join(f"{word} 0" for word, _number, _count in words)
        records[builder.POS_NAMES[pos]].append(f"{offset:08d} 00 {ss_type} {len(words):02x} {lexical} 000 | fixture synset {offset}")
        for word, number, count in words:
            sense_word = word.removesuffix("(a)").lower()
            tail = "bright:00" if ss_type == "s" else ":"
            senses.append(f"{sense_word}%{type_id}:00:00:{tail} {offset:08d} {number} {count}")

    add("n", 1, [("glow", 1, 10), ("luster", 1, 20)])
    add("v", 2, [("glow", 1, 10), ("shine", 1, 9), ("spark", 1, 8)])
    add("n", 3, [("can", 1, 999), ("will", 1, 999), ("light", 2, 99), ("dark", 1, 0), ("multi_word", 1, 99), ("London", 1, 99), ("x", 1, 99)])
    add("a", 4, [("fast", 1, 2), ("rapid", 1, 2), ("quick(a)", 1, 99)], ss_type="s")
    # This class loses one member and is discarded; it must not reserve calm.
    add("r", 5, [("glow", 1, 3), ("calm", 1, 3)])
    add("a", 6, [("calm", 1, 2), ("placid", 1, 2)])
    members["wordnet/index.sense"] = ("\n".join(senses) + "\n").encode()
    for name, lines in records.items():
        members[f"wordnet/data.{name}"] = ("\n".join(lines) + "\n").encode()
    return members


def test_selection_disjointness_and_first_sense_rules(source_members):
    dictionary, audit = builder.build(source_members)
    actual = {item["id"]: item["words"] for item in dictionary["classes"]}
    assert actual == {
        "wn30-a-00000004": ["fast", "rapid"],
        "wn30-a-00000006": ["calm", "placid"],
        "wn30-n-00000001": ["glow", "luster"],
        "wn30-v-00000002": ["shine", "spark"],
    }
    codec_dictionary = PublicDictionary.from_dict(dictionary)
    assert codec_dictionary.sha256 == audit["dictionary"]["canonical_sha256"]
    assert audit["counts"]["classes_discarded_after_disjointness"] == 1
    assert audit["counts"]["overlapping_member_occurrences_removed"] == 2
    assert audit["counts"]["rejected_nonfirst_sense"] == 1
    assert audit["counts"]["rejected_zero_tag_count"] == 1
    assert audit["counts"]["rejected_denylist"] == 2
    assert audit["counts"]["rejected_form"] == 4


def test_build_is_independent_of_input_record_order(source_members):
    original = builder.build(source_members)
    reordered = {name: (b"\n".join(reversed(raw.splitlines())) + b"\n" if name != "wordnet/LICENSE" else raw) for name, raw in source_members.items()}
    other = builder.build(reordered)
    assert builder.json_bytes(original[0]) == builder.json_bytes(other[0])
    # Source member hashes should, correctly, change when source order changes.
    assert original[1]["selected_classes"] == other[1]["selected_classes"]
    assert original[1]["counts"] == other[1]["counts"]


def test_repeated_subprocess_builds_ignore_python_hash_seed(source_members, tmp_path):
    fixture = tmp_path / "source.json"
    fixture.write_text(json.dumps({name: raw.decode() for name, raw in source_members.items()}))
    code = """
import importlib.util,json,pathlib,sys
spec=importlib.util.spec_from_file_location('recovery',sys.argv[1]); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
members={k:v.encode() for k,v in json.loads(pathlib.Path(sys.argv[2]).read_text()).items()}
d,p=b.build(members); sys.stdout.buffer.write(b.json_bytes([d,p]))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", code, str(SCRIPT), str(fixture)], env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ("1", "777")]
    assert outputs[0] == outputs[1]


def test_tie_breaking_prefers_eligible_class_size(source_members):
    source_members["wordnet/index.sense"] = source_members["wordnet/index.sense"].replace(b"glow%2:00:00:: 00000002 1 10", b"glow%2:00:00:: 00000002 1 13")
    dictionary, _ = builder.build(source_members)
    # Both sums equal 30; verb has three eligible members, so wins size tie.
    actual = {item["id"]: item["words"] for item in dictionary["classes"]}
    assert actual["wn30-v-00000002"] == ["glow", "shine", "spark"]
    assert "wn30-n-00000001" not in actual


def test_tie_breaking_then_uses_fixed_pos_order(source_members):
    source_members["wordnet/data.verb"] = source_members["wordnet/data.verb"].replace(b"v 03 glow 0 shine 0 spark 0", b"v 02 glow 0 shine 0")
    source_members["wordnet/index.sense"] = source_members["wordnet/index.sense"].replace(b"shine%2:00:00:: 00000002 1 9", b"shine%2:00:00:: 00000002 1 20")
    dictionary, _ = builder.build(source_members)
    actual = {item["id"]: item["words"] for item in dictionary["classes"]}
    assert actual["wn30-n-00000001"] == ["glow", "luster"]
    assert "wn30-v-00000002" not in actual


@pytest.mark.parametrize("bad", ["nonsense", "glow%1:00:00:: 00000001 0 2", "glow%1:00:00:: 00000001 1 -2", "glow%6:00:00:: 00000001 1 2"])
def test_malformed_sense_record_is_an_error(bad):
    with pytest.raises(builder.BuildError, match="Malformed index.sense"):
        builder.parse_senses(bad)


def test_inconsistent_data_record_is_an_error(source_members):
    source_members["wordnet/data.noun"] = source_members["wordnet/data.noun"].replace(b"00000001 00 n", b"00000001 01 n")
    with pytest.raises(builder.BuildError, match="inconsistent sense-index"):
        builder.build(source_members)


def test_missing_exact_index_variant_is_excluded_not_case_folded(source_members):
    source_members["wordnet/data.noun"] = source_members["wordnet/data.noun"].replace(b"glow 0", b"glow 1")
    dictionary, provenance = builder.build(source_members)
    assert provenance["counts"]["rejected_missing_exact_sense_index"] == 1
    assert "wn30-n-00000001" not in {item["id"] for item in dictionary["classes"]}


@pytest.mark.parametrize("member", ["wordnet/LICENSE", "wordnet/data.noun", "wordnet/index.sense"])
def test_missing_source_members_fail(source_members, member):
    del source_members[member]
    with pytest.raises(builder.BuildError, match="source member"):
        builder.build(source_members)


def test_empty_result_and_missing_license_fail(source_members):
    with pytest.raises(builder.BuildError, match="No classes survive"):
        builder.build({**source_members, **{f"wordnet/data.{name}": b"" for name in builder.POS_NAMES.values()}})
    with pytest.raises(builder.BuildError, match="license notice"):
        builder.build({**source_members, "wordnet/LICENSE": b"wrong source"})


def test_archive_hash_mismatch_never_builds_or_overwrites(tmp_path, capsys):
    archive = tmp_path / "wordnet.zip"
    archive.write_bytes(b"different corpus or corrupt download")
    out = tmp_path / "dictionary.json"
    provenance = tmp_path / "audit"
    assert builder.main(["--wordnet-zip", str(archive), "--output", str(out), "--provenance-dir", str(provenance)]) == 2
    assert "identity mismatch" in capsys.readouterr().err
    assert not out.exists() and not provenance.exists()


def test_missing_archive_has_actionable_error(tmp_path):
    with pytest.raises(builder.BuildError, match="--download or --wordnet-zip"):
        builder.load_source(tmp_path / "missing.zip")


def test_failed_download_is_not_saved(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b"untrusted download"

    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    output = tmp_path / "wordnet.zip"
    with pytest.raises(builder.BuildError, match="fixed size/SHA256"):
        builder.download_source(output)
    assert not output.exists()


def test_idempotent_outputs_and_no_partial_replace(source_members, tmp_path):
    dictionary, provenance = builder.build(source_members)
    out, folder = tmp_path / "dictionary.json", tmp_path / "audit"
    builder.write_outputs(dictionary, provenance, out, folder)
    before = {p: p.read_bytes() for p in (out, folder / "provenance.json", folder / "WORDNET-LICENSE.txt")}
    builder.write_outputs(dictionary, provenance, out, folder)
    assert before == {p: p.read_bytes() for p in before}
    out.unlink()
    (folder / "provenance.json").write_text("different existing provenance")
    with pytest.raises(builder.BuildError, match="Refusing to overwrite"):
        builder.write_outputs(dictionary, provenance, out, folder)
    assert not out.exists()


@pytest.mark.parametrize("name", ["shared_dictionary_ai_v1.json", "shared_dictionary_ai_v2.json", "shared_dictionary_v1.json"])
def test_historical_output_names_are_protected(source_members, tmp_path, name):
    with pytest.raises(builder.BuildError, match="historical dictionary"):
        builder.write_outputs(*builder.build(source_members), tmp_path / name, tmp_path / "audit")
    assert not (tmp_path / name).exists()


def test_output_path_collision_fails_before_write(source_members, tmp_path):
    with pytest.raises(builder.BuildError, match="paths overlap"):
        builder.write_outputs(*builder.build(source_members), tmp_path / "provenance.json", tmp_path)
    assert not (tmp_path / "provenance.json").exists()


def test_checked_in_dictionary_and_provenance_are_consistent():
    """Validates shipped resource identity independently through the frozen codec."""
    output = ROOT / "data/shared_dictionary_recovery_v1.json"
    provenance_dir = ROOT / "results/dictionary-recovery-v1"
    raw = output.read_bytes()
    dictionary = json.loads(raw)
    provenance = json.loads((provenance_dir / "provenance.json").read_bytes())
    parsed = PublicDictionary.from_dict(dictionary)
    assert dictionary["version"] == builder.VERSION
    assert parsed.sha256 == provenance["dictionary"]["canonical_sha256"]
    assert builder.sha256(raw) == provenance["dictionary"]["file_sha256"]
    assert len(raw) == provenance["dictionary"]["file_bytes"]
    assert builder.builder_source_sha256() == provenance["builder_sha256"]
    assert provenance["builder_hash_format"] == "source bytes with CRLF normalized to LF"
    assert provenance["rules"] == builder.rules()
    assert dictionary["license"]["text"].encode() == (provenance_dir / "WORDNET-LICENSE.txt").read_bytes()
    audit_classes = {item["id"]: item for item in provenance["selected_classes"]}
    for entry in parsed.classes:
        members = audit_classes[entry.class_id]["members"]
        assert tuple(item["word"] for item in members) == entry.words
        assert all(item["sense_number"] == 1 and item["tag_count"] >= 1 for item in members)
        assert not (set(entry.words) & builder.DENYLIST)
    assert len(parsed.classes) == provenance["counts"]["classes"]
    assert sum(len(entry.words) for entry in parsed.classes) == provenance["counts"]["words"]
