"""Archived corpus identity, exclusion and immutable-export safeguards."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("neural_corpus", Path(__file__).resolve().parents[1] / "scripts/build_neural_corpus.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_identity_retains_duplicates_as_distinct_source_units(tmp_path):
    inv = builder.Inventory(tmp_path)
    inv.add("dataset", "first", "cover", "same text", "source.json")
    inv.add("dataset", "second", "stego", "same text", "source.json")
    rows = list(inv.records.values())
    assert len(rows) == 2 and rows[0]["id"] != rows[1]["id"]
    assert rows[0]["text_sha256"] == rows[1]["text_sha256"]
    with pytest.raises(ValueError, match="Conflicting"):
        inv.add("dataset", "first", "cover", "altered text", "source.json")


def test_missing_text_attempt_has_explicit_exclusion_record(tmp_path):
    inv = builder.Inventory(tmp_path)
    inv.add("dataset", "attempt1", "stego", None, "attempts.jsonl")
    assert not inv.records
    assert inv.excluded_records[0]["reason"] == "no_nonempty_stored_text"


def test_frozen_export_preserves_baseline_retries_and_cannot_overwrite(tmp_path):
    source = tmp_path / "source"
    directory = source / "experiments/conversation_bench/results"
    directory.mkdir(parents=True)
    path = directory / "corpus_default.jsonl"
    path.write_text('\n'.join(json.dumps(x) for x in [
        {"Key": "codec/128/0", "L": 128, "stegotext": "first text", "old_score": .9},
        {"Key": "codec/128/0", "L": 128, "stegotext": "second text", "old_score": .1},
        {"Key": "codec/128/1", "L": 128, "stegotext": "first text", "old_score": .5},
        {"Key": "codec/128/2", "L": 128, "encode_error": "failed"},
    ]) + '\n')
    out = tmp_path / "frozen"
    manifest = builder.build(source, out, "baseline")
    records = [json.loads(line) for line in (out / "corpus.jsonl").read_text().splitlines()]
    assert manifest["n_records"] == 3 and manifest["n_unique_texts"] == 2
    assert len({r["id"] for r in records}) == 3
    assert sum(r["uuid"] == "codec/128/0" for r in records) == 2
    assert len(manifest["excluded_records"]) == 1
    assert all("old_score" not in r for r in records)
    assert all(r["paired_with_cover"] is False for r in records)
    assert manifest["corpus_sha256"] == builder.sha256_bytes((out / "corpus.jsonl").read_bytes())
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.build(source, out, "baseline")


def test_build_failure_does_not_publish_partial_output(tmp_path):
    out = tmp_path / "frozen"
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.build(tmp_path / "empty", out, "primary")
    assert not out.exists()
