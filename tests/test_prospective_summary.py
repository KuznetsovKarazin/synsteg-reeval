"""Completion gates, retained-record identity, and all-attempt denominators."""
import importlib.util
import json
from pathlib import Path
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location("prospective_summary", Path(__file__).resolve().parents[1] /
                                            "scripts/summarize_prospective_results.py")
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(s.canonical(value) + b"\n")


def complete_fixture(run):
    variants = ["fixed_canonical", "fixed_quality"]
    write(run / "protocol.json", {"variants": variants, "payload_bits": [1]})
    samples = [{"id": "success", "topic_id": "a", "genre_id": "explanation", "payload_bits": 1},
               {"id": "failed", "topic_id": "b", "genre_id": "narrative", "payload_bits": 1}]
    write(run / "manifests/test.json", samples)
    frozen = {"protocol_sha256": s.file_hash(run / "protocol.json"),
              "manifest_sha256": {"test": s.file_hash(run / "manifests/test.json")}}
    frozen["freeze_id"] = s.digest(frozen)
    write(run / "freeze.json", frozen)
    lock = {"freeze_id": frozen["freeze_id"], "attacks": {"fixed_canonical__word": {}}}
    lock["lock_id"] = s.digest(lock)
    write(run / "attack_lock.json", lock)
    key = {"positions": [{"original": "alpha", "groups": [["alpha", "aleph"], ["beta"]], "scores": [[2., 1.], [3.]]},
                         {"original": "omega", "groups": [], "scores": []}]}
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
    outputs = {name: {"stego_text": "alpha omega" if name.endswith("canonical") else "aleph omega",
        "decoded_payload": "0", "key": key,
        "metrics": {"full_message_success": True, "requested_bits": 1, "stego_words": 2,
                    "key_json_bytes": len(raw), "key_zlib9_bytes": len(zlib.compress(raw, 9))},
        "metadata": {"framing": "external_length", "external_metadata_bits": 32, "used_positions": 1}}
        for name in variants}
    records = [{"sample": samples[0], "freeze_id": frozen["freeze_id"], "cover_text": "alpha omega",
                "synthetic_payload": "0", "variants": outputs,
                "generation_metadata": {"generated_tokens": 3, "hit_token_limit": False}},
               {"sample": samples[1], "freeze_id": frozen["freeze_id"], "cover_text": "one two three four",
                "variants": {name: {"metrics": None, "error": "CapacityError"} for name in variants},
                "generation_metadata": {"generated_tokens": 6, "hit_token_limit": True}}]
    for row in records:
        write(run / "records/test" / (row["sample"]["id"] + ".json"), row)
    h = s.digest(records)
    write(run / "test_generation.json", {"complete": True, "records_sha256": h})
    opening = {"freeze_id": frozen["freeze_id"], "lock_id": lock["lock_id"], "test_records_sha256": h}
    write(run / "test_opened.json", opening)
    write(run / "evaluation.json", {**opening, "attacks": {"fixed_canonical__word": {"status": "complete"}}})
    events = [{"event": "test_opened_once", "lock_id": lock["lock_id"], "time_unix": 1},
              {"event": "test_evaluation_completed", "time_unix": 2, "evaluation_sha256": s.file_hash(run / "evaluation.json")}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(x) for x in events) + "\n")


def test_refuses_before_test_access_when_evaluation_missing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No JSON should be read without the completion files")
    monkeypatch.setattr(s, "read_json", forbidden)
    with pytest.raises(ValueError, match="completed evaluation required"):
        s.load_completed_test(tmp_path)


def test_denominators_failures_cost_and_used_choice(tmp_path):
    complete_fixture(tmp_path)
    report = s.summarize(tmp_path)
    row = report["variants"]["fixed_quality"]["overall"]
    assert row["attempted_samples"] == 2 and row["full_success_samples"] == 1
    assert row["delivered_fraction_of_attempted_bits"] == .5
    assert row["successful_text_payload_bpw_weighted"] == .5
    assert row["full_message_goodput_per_all_generated_cover_word"] == pytest.approx(1 / 6)
    assert row["hit_token_limit_samples"] == 1
    assert row["successful_transmission_costs"]["external_metadata_bits"]["sum"] == 32
    choice = row["choice_freedom_on_successful_used_positions"]
    assert choice["positions_with_choice"] == 1 and choice["used_positions"] == 1
    assert choice["edited_positions"] == 1
    assert choice["selected_minus_canonical_local_proxy_nll_sum"] == -1
    assert report["selector_comparison"]["same_codebook_pairs"] == 1
    assert report["selector_comparison"]["identical_selector_output_pairs"] == 0


def test_rejects_mutated_evaluated_records(tmp_path):
    complete_fixture(tmp_path)
    path = tmp_path / "records/test/success.json"
    record = s.read_json(path)
    record["cover_text"] = "changed after evaluation"
    write(path, record)
    with pytest.raises(ValueError, match="records differ from evaluated"):
        s.load_completed_test(tmp_path)


def test_missing_completion_event_blocks_test_record_read(tmp_path, monkeypatch):
    complete_fixture(tmp_path)
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    (tmp_path / "events.jsonl").write_text(events[0] + "\n")
    real_read = s.read_json
    def guarded(path):
        assert "records" not in Path(path).parts
        return real_read(path)
    monkeypatch.setattr(s, "read_json", guarded)
    with pytest.raises(ValueError, match="exactly one opening"):
        s.load_completed_test(tmp_path)


def test_generation_failure_keeps_attempt_denominator():
    row = {"full_message_success": False, "delivered_bits": 0, "payload_bits": 16,
           "cover_words": None, "generated_tokens": None, "hit_token_limit": None}
    result = s.aggregate([row])
    assert result["mean_delivered_bits_per_planned_attempt"] == 0
    assert result["attempts_without_generated_cover_word_denominator"] == 1
    assert result["full_message_goodput_per_all_generated_cover_word"] is None


def test_quality_limitation_follows_declared_adapter_identity():
    root = Path(__file__).resolve().parents[1]
    corrected = s.read_json(root / "configs/prospective_pilot_cpu_v3.json")
    legacy = s.read_json(root / "configs/prospective_pilot_cpu_v2.json")
    current_warning = s.quality_proxy_limitation(corrected)
    assert "verifies rendered candidate tokens and causal prefixes" in current_warning
    assert "mismatch newline/quote" not in current_warning
    assert "original cover" in current_warning and "interactions" in current_warning
    historical_warning = s.quality_proxy_limitation(legacy)
    assert "Legacy artificial-leading-space" in historical_warning
    assert "do not identify the rendered-token correction" in historical_warning
    assert "verifies rendered" not in historical_warning


def test_alignment_counts_deduplicate_shared_states_and_preserve_unavailable():
    counts = {"candidate_pairs_considered": 10, "candidate_pairs_admitted": 7, "candidate_pairs_rejected": 3}
    variants = ["fixed_canonical", "fixed_quality"]
    records = [{"sample": {"id": "new"}, "prepared_methods": {v: {"token_alignment": dict(counts)} for v in variants}},
               {"sample": {"id": "legacy"}, "prepared_methods": {v: {} for v in variants}}]
    summary = s.token_alignment_summary(records, variants)
    assert summary["records_with_counts"] == 1 and summary["records_without_counts"] == 1
    assert summary["totals_over_available_records"] == counts
    assert summary["admitted_fraction_of_considered"] == .7
    assert summary["records"][1]["candidate_pairs_considered"] is None
    assert s.token_alignment_summary(records[1:], variants)["totals_over_available_records"]["candidate_pairs_considered"] is None
    records[0]["prepared_methods"]["fixed_quality"]["token_alignment"]["candidate_pairs_admitted"] = 8
    with pytest.raises(ValueError, match="counts differ across shared selectors"):
        s.token_alignment_summary(records, variants)
