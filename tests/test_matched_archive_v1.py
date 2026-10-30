"""Synthetic transport and accounting checks; never evaluates natural archive."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matched_archive", ROOT / "scripts/run_matched_archive_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fixed_configuration_is_explicit_and_ceiling_not_v3():
    assert runner.FIXED_CONFIGS == (("strict_b1_g2", 1, 2), ("strict_b2_g2", 2, 2),
                                  ("lowrate_b1_g1", 1, 1), ("ceiling_b5_g1", 5, 1))


def test_runtime_matches_all_frozen_configuration():
    protocol = runner.validate_protocol(ROOT / "experiments/matched_archive_v1/protocol.json")
    assert protocol["public_common"]["fixture_key_hex"] == bytes(range(32)).hex()
    assert protocol["denominators"]["total_rows"] == 6375


def test_runtime_key_mismatch_rejected_before_run(monkeypatch):
    monkeypatch.setattr(runner, "FIXTURE_KEY", b"x" * 32)
    with pytest.raises(ValueError, match="public_common"):
        runner.validate_protocol(ROOT / "experiments/matched_archive_v1/protocol.json")


def test_runtime_configuration_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(runner, "FIXED_CONFIGS", (("strict_b1_g2", 3, 2),))
    with pytest.raises(ValueError, match="fixed_configurations"):
        runner.validate_protocol(ROOT / "experiments/matched_archive_v1/protocol.json")


def test_frozen_protocol_bytes_cannot_drift(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text((ROOT / "experiments/matched_archive_v1/protocol.json").read_text() + " ")
    with pytest.raises(ValueError, match="identity"):
        runner.validate_protocol(path)


def test_complementary_payloads_do_not_mutate_saved_book():
    cover = " ".join(["calm"] * 40)
    candidates = [["quiet", "still", "gentle", "mild"] for _ in range(40)]
    book = runner.fixed.freeze_codebook(cover, candidates, seed=runner.SEED, max_bits_per_position=2, min_group_size=1)
    saved = book.to_json()
    for bits in ("01" * 8, "10" * 8):
        stego, result = runner.encode_fixed({"container": cover}, bits, "fixture", book)
        assert runner.fixed.decode_bits(book, stego, framing="external_length", expected_bits=16) == bits
        assert result["frame_bits"] == 16
        assert book.to_json() == saved


def test_capacity_rejection_is_not_partial_delivery():
    book = runner.fixed.freeze_codebook("calm", [["still"]], min_group_size=1)
    with pytest.raises(runner.fixed.CapacityError):
        runner.encode_fixed({"container": "calm"}, "01" * 8, "fixture", book)


def test_gzip_is_deterministic_and_lossless(tmp_path):
    rows = [{"a": 1}, {"unicode": "смысл", "bits": "001001"}]
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    runner.write_rows(first, rows)
    runner.write_rows(second, rows)
    assert first.read_bytes() == second.read_bytes()
    assert runner.read_rows(first) == rows


def test_book_accounting_is_exact_and_roundtrips():
    books = []
    value = {"positions": [{"calm": {"0": "quiet", "1": "still"}}]}
    cost = runner.book_resource(books, "fixture", "fixture", value)
    raw = runner.canonical(value)
    assert cost["book_json_bytes"] == len(raw)
    assert cost["book_zlib9_bytes"] == len(runner.zlib.compress(raw, 9))
    assert books[0]["value"] == value


def test_summary_keeps_zero_and_refusal_denominators():
    rows = []
    for capacity, success in ((0, False), (16, True), (8, False)):
        rows.append({"grid": "secondary16", "series": "fixture", "arm": "a", "archived_payload_bits": 128,
                     "capacity_bits": capacity, "book_zlib9_bytes": 100, "status": "full_exact" if success else "insufficient_capacity",
                     "full_exact": success, "requested_payload_bits": 16, "delivered_payload_bits": 16 if success else 0,
                     "cover_words": 20})
    groups = runner.summarize(rows)["groups"]
    assert len(groups) == 2
    for group in groups:
        assert group["denominator"] == 3 and group["full_exact"] == 1
        assert group["zero_capacity"] == 1 and group["delivered_bits"] == 16
        assert group["unconditional_delivered_bits_per_source_word"] == 16 / 60


def test_receiver_public_recovers_without_cover_or_payload(tmp_path, monkeypatch):
    resources = tmp_path / "recipient_resources"
    resources.mkdir()
    dictionary = runner.shared.PublicDictionary.from_mapping({"calm": ("calm", "quiet", "still", "mild")})
    for version in ("v1", "v2"):
        runner.write_json(resources / f"dictionary_{version}.json", dictionary.to_dict())
    # Pure toy originals are never invoked in this public-resource test.
    bundle = {"constants": {}, "modules": {}}
    runner.write_json(resources / "original_functions.json", bundle)
    monkeypatch.setattr(runner, "load_original", lambda _: None)
    runner.write_rows(tmp_path / "books.jsonl.gz", [])
    protocol = runner.shared.Protocol(framing="plain_length16")
    encoded = runner.shared.encode(" ".join(["calm"] * 40), b"\x00\x01", runner.FIXTURE_KEY, dictionary, protocol)
    job = {"row_id": "fixture", "kind": "public", "dictionary_version": "v1", "stego_text": encoded.stego_text,
           "fixture_key_hex": runner.FIXTURE_KEY.hex(),
           "protocol": {"min_group_size": 2, "max_bits_per_slot": 2, "framing": "plain_length16"}}
    assert "payload" not in job and "cover" not in job
    runner.write_rows(tmp_path / "receiver_jobs.jsonl.gz", [job])
    runner.receiver(tmp_path)
    assert runner.read_rows(tmp_path / "receiver_results.jsonl.gz")[0]["decoded_bits"] == "0000000000000001"


def test_run_requires_real_freeze_and_preserves_existing_path(tmp_path):
    with pytest.raises(ValueError, match="freeze"):
        runner.run(tmp_path / "absent.zip", tmp_path / "out", "not-a-commit", None)
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "absent.zip", tmp_path, "a" * 40, None)
