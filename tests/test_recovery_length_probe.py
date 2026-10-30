import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_recovery_length_probe as probe


@pytest.fixture
def inputs(tmp_path):
    rows = [{"record_id": "dev-" + str(i), "source_group_id": "source-" + str(i),
             "split": "development", "topic": "astronomy", "genre": "explanation",
             "target_words": size, "prompt": "Source facts remain quoted data " + str(i),
             "attribution": "Fixture source " + str(i), "excerpt": "Fact " + str(i)}
            for i, size in enumerate((500, 250, 500))]
    source = tmp_path / "development-prompts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, tmp_path / "probe", rows


def test_exact_first_two_sources_retained_and_original_untouched(inputs):
    source, output, original = inputs
    before = source.read_bytes()
    result = probe.prepare(source, output)
    assert result["status"] == "prepared_no_generation"
    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    assert [row["source_group_id"] for row in rows] == [row["source_group_id"] for row in original[:2]]
    for old, new in zip(original[:2], rows):
        assert new["record_id"] == old["record_id"]
        assert new["prompt"].startswith(old["prompt"] + "\n\n")
        assert new["attribution"] == old["attribution"] and new["excerpt"] == old["excerpt"]
        assert new["parent_prompt_sha256"] == hashlib.sha256(old["prompt"].encode()).hexdigest()
        assert new["original_prompt_sha256"] == new["parent_prompt_sha256"]
    assert source.read_bytes() == before
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["generator"]["max_output_tokens"] == 1024
    assert protocol["generator"]["seed"] == 20261024
    assert protocol["development_only"] and not protocol["independent_new_sample"]
    assert "seeds match baseline" in protocol["sampling_seed_pairing"]
    assert protocol["prompts_sha256"] == hashlib.sha256((output / "prompts.jsonl").read_bytes()).hexdigest()


def test_length_bands_and_paragraph_counts(inputs):
    source, output, _ = inputs
    probe.prepare(source, output)
    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    for row, expected in zip(rows, ((450, 550, 10), (225, 275, 5))):
        settings = row["length_requirement"]
        assert (settings["minimum_words"], settings["maximum_words"], settings["paragraph_count"]) == expected
        assert f"exactly {expected[2]} substantial paragraphs" in row["prompt"]
        assert f"between {expected[0]} and {expected[1]} words" in row["prompt"]
        assert "do not introduce outside facts" in row["prompt"]


def test_closed_split_anywhere_in_file_rejected_before_output(inputs):
    source, output, rows = inputs
    rows[2]["split"] = "test"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="development-only"):
        probe.prepare(source, output)
    assert not output.exists()


def test_exact_idempotence_and_existing_output_never_overwritten(inputs):
    source, output, _ = inputs
    probe.prepare(source, output)
    saved = {name: (output / name).read_bytes() for name in ("prompts.jsonl", "protocol.json")}
    assert probe.prepare(source, output)["status"] == "already_prepared_identical"
    assert all((output / name).read_bytes() == content for name, content in saved.items())
    (output / "protocol.json").write_bytes(b"existing-result-do-not-change")
    with pytest.raises(FileExistsError, match="no file will be overwritten"):
        probe.prepare(source, output)
    assert (output / "protocol.json").read_bytes() == b"existing-result-do-not-change"
    assert (output / "prompts.jsonl").read_bytes() == saved["prompts.jsonl"]


def test_incomplete_existing_directory_refused(inputs):
    source, output, _ = inputs
    output.mkdir()
    with pytest.raises(FileExistsError):
        probe.prepare(source, output)
    assert list(output.iterdir()) == []


def test_source_prompt_hash_mismatch_rejected(inputs):
    source, output, rows = inputs
    rows[0]["prompt_sha256"] = "0" * 64
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="hash"):
        probe.prepare(source, output)
    assert not output.exists()


def test_duplicate_source_groups_cannot_create_extra_independent_n(inputs):
    source, output, rows = inputs
    rows[1]["source_group_id"] = rows[0]["source_group_id"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Duplicate"):
        probe.prepare(source, output)
