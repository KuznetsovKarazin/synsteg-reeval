"""Synthetic integration fixtures, never natural-language study evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("invariant_context_development_test", ROOT / "scripts/audit_invariant_context_development.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "project"
    for name in audit.CODE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    dictionary = audit.base.PublicDictionary.from_mapping(
        {"time": ["clock", "time"], "quality": ["fine", "good"]}, "synthetic-audit-integration-v1")
    source = dictionary.to_dict()
    source["license"] = "Synthetic fixture notice retained byte-for-byte."
    write(root / audit.DICTIONARY, source)
    records = [{"record_id": f"r{i}", "source_group_id": f"g{i}", "split": "development",
                "excerpt": "Synthetic fixture source text.", "excerpt_sha256": audit.sha(b"Synthetic fixture source text."),
                "topic": "fixture", "genre": "fixture", "target_words": 250, "prompt_sha256": "a" * 64,
                "title": f"Fixture {i}", "attribution": "Synthetic test input"} for i in range(100)]
    source_path = root / audit.SOURCE_PROMPTS
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"".join(audit.canonical(row) + b"\n" for row in records))
    for run_name, n, count in [(audit.BASELINE_RUN, 30, 2), (audit.LENGTH_RUN, 2, 2), (audit.GROUNDED_RUN, 20, 20)]:
        metadata = [{k: row[k] for k in ("record_id", "source_group_id", "split", "topic", "genre", "target_words", "prompt_sha256")}
                    for row in records[:n]]
        plan = {"schema": "synsteg-budget-generation-run-v1", "split": "development", "provider": "local",
                "model": "synthetic/model", "selected_records": metadata, "selected_records_count": n}
        write(root / run_name / "run.json", plan)
        for i in range(count):
            if run_name == audit.GROUNDED_RUN and i == 19:
                continue  # Never-attempted row must remain in the denominator.
            text = "over time " + "good " * 34 if run_name == audit.LENGTH_RUN and i == 1 else "Tiny synthetic input."
            status = "failed_or_unknown_no_retry" if run_name == audit.GROUNDED_RUN and i == 18 else "completed"
            attempt = {"schema": "synsteg-budget-generation-attempt-v1", "record": metadata[i], "provider": "local",
                       "model": "synthetic/model", "status": status,
                       "result": {"cover_text": text, "stop_reason": "end_turn"} if status == "completed" else None}
            write(root / run_name / "attempts" / f"{i}.json", attempt)
    return root, tmp_path / "audit"


def test_prepare_freezes_without_codec_calls_and_changed_code_blocks_execution(workspace, monkeypatch):
    root, output = workspace
    monkeypatch.setattr(audit.base, "encode", lambda *a, **k: pytest.fail("Preparation encoded a frame"))
    monkeypatch.setattr(audit.guarded, "encode", lambda *a, **k: pytest.fail("Preparation encoded a frame"))
    result = audit.prepare(root, output)
    assert result["status"] == "frozen_before_encoding"
    protocol = audit.read(output / "protocol.json")
    assert protocol["actual_codec_encoding_calls_during_preparation"] == 0
    assert protocol["cohorts"] == {
        "source_excerpts": {"planned_records": 100, "distinct_source_groups": 100},
        "historical_covers": {"planned_records": 4, "distinct_source_groups": 2},
        "grounded_covers": {"planned_records": 20, "distinct_source_groups": 20}}
    assert (output / "public_dictionary.json").read_bytes() == (root / audit.DICTIONARY).read_bytes()
    changed = root / "src/synsteg_audit/invariant_context.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed after freeze\n")
    with pytest.raises(ValueError, match="Frozen file changed"):
        audit.execute(root, output)
    assert not (output / "evaluation").exists()


def test_actual_codec_receiver_preservation_and_separate_failure_denominators(workspace):
    root, output = workspace
    audit.prepare(root, output)
    before = (output / "protocol.json").read_bytes()
    report = audit.execute(root, output)
    assert (output / "protocol.json").read_bytes() == before
    historical = report["cohorts"]["historical_covers"]
    assert historical["planned_records"] == 4 and historical["distinct_source_groups"] == 2
    for method in audit.METHODS:
        arm = historical["methods"][method]["arms"]["plain_length16__8"]
        assert arm["planned"] == 4 and arm["full_success"] == 1
        assert arm["statuses"] == {"insufficient_capacity": 3, "decoded": 1}
        unavailable = report["cohorts"]["grounded_covers"]["methods"][method]["arms"]["plain_length16__8"]
        assert unavailable["planned"] == 20 and unavailable["statuses"]["unavailable_input"] == 2
    examples = report["mechanical_protection_examples"]
    assert len(examples) == 1 and examples[0]["protected_tokens"] == [{"token_index": 1, "token": "time"}]
    rows = [json.loads(line) for line in (output / "evaluation/records.jsonl").read_text().splitlines()]
    passed = next(row for row in rows if row["audit_record_id"] == "length_probe:r1")
    assert passed["methods"]["base_v1"]["capacity_bits"] == passed["methods"]["wrapper_empty"]["capacity_bits"] == 35
    assert passed["methods"]["wrapper_guard"]["capacity_bits"] == 34
    success = passed["methods"]["wrapper_guard"]["arms"]["plain_length16__8"]
    assert success["stego_text"].startswith("over time ") and success["protected_spellings_preserved"]
    assert success["receiver"]["actual_returncode"] == 0
    assert success["receiver"]["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True}
    assert report["cohorts_must_not_be_pooled_as_independent_n"] and not report["quality_effect_attribution"]
    with pytest.raises(FileExistsError):
        audit.execute(root, output)


def test_closed_source_record_rejected_before_freeze(workspace):
    root, output = workspace
    source = root / audit.SOURCE_PROMPTS
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[99]["split"] = "test"
    source.write_bytes(b"".join(audit.canonical(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="development-only"):
        audit.prepare(root, output)
    assert not output.exists()
