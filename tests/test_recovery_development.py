"""Development audit checks with synthetic covers, never empirical evidence."""
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_development_audit_test", ROOT / "scripts/audit_recovery_development.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def dictionary(tmp_path):
    value = audit.codec.PublicDictionary.from_mapping({"test-only": ["bright", "clear"]}, "synthetic-audit-tests-v1")
    path = tmp_path / "dictionary.json"
    save(path, value.to_dict())
    return path, value


def make_run(tmp_path, outcomes, target_words=None):
    run = tmp_path / "run"
    records = [{"record_id": f"record-{i}", "source_group_id": f"source-{i}", "split": "development",
                "topic": "test", "genre": "test", "target_words": target_words[i] if target_words else 250,
                "prompt_sha256": audit.digest(["fixture-prompt", i])} for i in range(len(outcomes))]
    manifest = {"schema": "synsteg-budget-generation-run-v1", "provider": "local", "model": "test/model",
                "revision": "a" * 40, "dtype": "float32", "settings": {}, "prompts_sha256": "b" * 64,
                "split": "development", "selected_records_count": len(records), "selected_records": records}
    save(run / "run.json", manifest)
    for i, result in enumerate(outcomes):
        if result is None:
            continue
        if isinstance(result, str):
            result = {"cover_text": result, "stop_reason": "end_turn", "hit_output_limit": False,
                      "usage": {"input_tokens": 1, "output_tokens": len(result.split())}}
        attempt = {"schema": "synsteg-budget-generation-attempt-v1", "identity": str(i),
                   "record": records[i], "provider": "local", "model": "test/model",
                   "status": result.pop("attempt_status", "completed"), "result": result}
        save(run / "attempts" / f"{i}.json", attempt)
    return run


def carrier(n):
    return " ".join(["bright"] * n)


def test_independent_receiver_and_all_planned_failure_denominator(tmp_path, dictionary):
    run = make_run(tmp_path, [carrier(200), carrier(10), {"attempt_status": "failed_or_unknown_no_retry"}, None,
                              {"cover_text": "I cannot comply", "stop_reason": "refusal"},
                              {"cover_text": carrier(200), "stop_reason": "max_tokens", "hit_output_limit": True}])
    result = audit.audit([run], dictionary[0], tmp_path / "audit")
    model = result["models"][0]
    assert model["planned_source_groups"] == 6
    assert model["generation_outcomes"] == {"completed": 2, "failed_or_unknown_no_retry": 1,
        "not_attempted": 1, "refusal": 1, "completed_at_output_limit": 1}
    for name in ("plain_length16__canonical__8", "plain_length16__canonical__16", "aes_siv__canonical__8"):
        arm = model["single_text_arms"][name]
        assert arm["planned"] == 6
        assert arm["full_success"] == 2
        assert arm["insufficient_capacity"] == 1
        assert arm["unavailable_cover"] == 3
        assert arm["full_success_fraction_all_planned"] == 2 / 6
    rows = [json.loads(line) for line in (tmp_path / "audit/model-01-records.jsonl").read_text().splitlines()]
    passed = rows[0]["arms"]["aes_siv__canonical__8"]
    assert passed["receiver"]["input_keys"] == ["key_hex", "public_dictionary", "public_protocol", "stego_text"]
    assert passed["receiver"]["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True}
    assert passed["receiver"]["actual_returncode"] == 0
    assert result["quality_metrics"]["status"] == "not_computed"
    assert result["test_split_opened"] is False
    assert model["four_source_bundles"]["arms"]["aes_siv__canonical__8"]["unavailable_cover"] == 1
    assert model["four_source_bundles"]["unbundled_tail_record_ids"] == ["record-4", "record-5"]


def test_fixed_four_source_bundle_succeeds_without_cherry_picking(tmp_path, dictionary):
    run = make_run(tmp_path, [carrier(45)] * 4)
    result = audit.audit([run], dictionary[0], tmp_path / "audit")
    model = result["models"][0]
    assert model["single_text_arms"]["aes_siv__canonical__8"]["full_success"] == 0
    assert model["single_text_arms"]["aes_siv__canonical__8"]["insufficient_capacity"] == 4
    bundles = model["four_source_bundles"]
    assert bundles["planned_complete_bundles"] == 1
    assert bundles["arms"]["aes_siv__canonical__8"]["full_success"] == 1
    assert bundles["arms"]["aes_siv__canonical__32"]["full_success"] == 1
    assert bundles["arms"]["aes_siv__canonical__128"]["insufficient_capacity"] == 1
    row = json.loads((tmp_path / "audit/model-01-bundles.jsonl").read_text())
    assert row["record_ids"] == [f"record-{i}" for i in range(4)]
    assert row["cover_text"] == "\n\n".join([carrier(45)] * 4)


@pytest.mark.parametrize("field", ["manifest", "record"])
def test_rejects_test_split_without_writing(tmp_path, dictionary, field):
    run = make_run(tmp_path, [None])
    manifest = audit.load_json(run / "run.json")
    if field == "manifest":
        manifest["split"] = "test"
    else:
        manifest["selected_records"][0]["split"] = "test"
    save(run / "run.json", manifest)
    with pytest.raises(ValueError, match="development"):
        audit.audit([run], dictionary[0], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_planned_attempt_identity_cannot_be_replaced(tmp_path):
    run = make_run(tmp_path, [carrier(1)])
    attempt = audit.load_json(run / "attempts/0.json")
    attempt["record"]["source_group_id"] = "wrong-source"
    save(run / "attempts/0.json", attempt)
    with pytest.raises(ValueError, match="frozen planned record"):
        audit.read_run(run)


def test_windows_snapshot_uses_portable_attempt_keys(monkeypatch):
    class WindowsFixturePath(PureWindowsPath):
        def glob(self, pattern):
            assert self.name == "attempts" and pattern == "*.json"
            return [self / "attempt.json"]

    monkeypatch.setattr(audit, "file_hash", lambda path: "fixture-hash")
    result = audit.snapshot(WindowsFixturePath("E:/synsteg/run"))
    assert set(result) == {"run.json", "attempts/attempt.json"}
    assert len([name for name in result if name.startswith("attempts/")]) == 1


def test_rejects_duplicate_source_groups(tmp_path):
    run = make_run(tmp_path, [None, None])
    manifest = audit.load_json(run / "run.json")
    manifest["selected_records"][1]["source_group_id"] = "source-0"
    save(run / "run.json", manifest)
    with pytest.raises(ValueError, match="distinct source"):
        audit.read_run(run)


def test_no_output_overwrite(tmp_path, dictionary):
    run = make_run(tmp_path, [None])
    audit.audit([run], dictionary[0], tmp_path / "audit")
    before = (tmp_path / "audit/summary.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        audit.audit([run], dictionary[0], tmp_path / "audit")
    assert (tmp_path / "audit/summary.json").read_bytes() == before


def test_archived_dictionary_keeps_exact_license_and_provenance_bytes(tmp_path, dictionary):
    run = make_run(tmp_path, [None])
    source = audit.load_json(dictionary[0])
    source.update(license={"notice": "Example license notice; retain on redistribution."},
                  provenance={"source_url": "https://example.test/dictionary", "source_sha256": "a" * 64})
    raw = (json.dumps(source, indent=3) + "\n\n").encode("utf-8")
    dictionary[0].write_bytes(raw)
    report = audit.audit([run], dictionary[0], tmp_path / "audit")
    assert (tmp_path / "audit/public_dictionary.json").read_bytes() == raw
    assert report["dictionary_file_sha256"] == audit.file_hash(tmp_path / "audit/public_dictionary.json")
    assert report["public_dictionary_sha256"] == dictionary[1].sha256


def test_running_generation_lock_blocks_audit(tmp_path, dictionary):
    from synsteg_audit.budget_ledger import BudgetError
    run = make_run(tmp_path, [None])
    with audit.CampaignLedger(run / "run_guard", "20").locked():
        with pytest.raises(BudgetError, match="locked"):
            audit.audit([run], dictionary[0], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
    # The same persistent lock file is reusable after the OS lock is released.
    assert audit.audit([run], dictionary[0], tmp_path / "audit")["models"][0]["planned_source_groups"] == 1


def test_fixed_bundle_partition_stays_within_requested_length(tmp_path, dictionary):
    run = make_run(tmp_path, [None] * 8, [500, 250, 500, 250, 500, 250, 500, 250])
    audit.audit([run], dictionary[0], tmp_path / "audit")
    rows = [json.loads(line) for line in (tmp_path / "audit/model-01-bundles.jsonl").read_text().splitlines()]
    assert rows[0]["record_ids"] == ["record-1", "record-3", "record-5", "record-7"]
    assert rows[1]["record_ids"] == ["record-0", "record-2", "record-4", "record-6"]


def test_paired_public_payloads_are_domain_separated_and_prefix_nested():
    ids = ["same-source"]
    assert audit.public_fixture("payload", ids, 1) == audit.public_fixture("payload", ids, 2)[:1]
    assert audit.public_fixture("key", ids) != audit.public_fixture("payload", ids)


def test_input_mutation_during_audit_does_not_write_report(tmp_path, dictionary, monkeypatch):
    run = make_run(tmp_path, [carrier(32)])
    original = audit.isolated_decode
    changed = False

    def mutate(*args):
        nonlocal changed
        if not changed:
            manifest = audit.load_json(run / "run.json")
            manifest["mutation"] = "changed during audit"
            save(run / "run.json", manifest)
            changed = True
        return original(*args)

    monkeypatch.setattr(audit, "isolated_decode", mutate)
    with pytest.raises(ValueError, match="changed during audit"):
        audit.audit([run], dictionary[0], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
