"""Workflow controls use synthetic fixtures, never a hidden experiment corpus."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prospective_experiment", ROOT / "scripts/prospective_experiment.py")
prospective = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prospective
SPEC.loader.exec_module(prospective)


class FixtureAdapter:
    """Intentionally trivial detector fixtures, not generated language evidence."""
    def __init__(self):
        self.order = []

    def generate_cover(self, sample, generation):
        self.order.append("cover")
        return {"text": f"Ordinary writing introduces {sample['id']} with clear familiar expressions.",
                "metadata": {"fixture": True}}

    def prepare_methods(self, cover, sample, method_config):
        self.order.append("prepare")
        assert "synthetic_payload" not in sample and "payload_seed" not in sample
        return {"fixed_canonical": {"original": cover}, "fixed_quality": {"original": cover}}

    def encode_decode(self, variant, prepared, payload_bits, cover, sample, method_config):
        self.order.append("encode")
        return {"stego_text": "Unusual extraordinary transformative phrasing. " + cover,
                "decoded_payload": payload_bits, "embedded_bits": len(payload_bits),
                "key": prepared, "metadata": {"fixture": True}}


def fixture_factory():
    return FixtureAdapter()


@pytest.fixture
def setup_run(tmp_path):
    protocol = json.loads((ROOT / "configs/prospective_protocol_v1.json").read_text())
    protocol.update(execution_enabled=True, fixture_only=True,
                    adapter_file="tests/test_prospective_experiment.py", adapter_factory="fixture_factory")
    protocol["model"] = {"backend": "fixture"}
    protocol["genres"] = protocol["genres"][:1]
    protocol["topics"] = {s: rows[:2] for s, rows in protocol["topics"].items()}
    protocol["statistics"]["bootstrap_resamples"] = 5
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    run = tmp_path / "run"
    run.mkdir()
    prospective.prepare(path, run)
    prospective.freeze(run)
    return protocol, run


def test_sample_design_is_disjoint_and_balanced():
    protocol = json.loads((ROOT / "configs/prospective_protocol_v1.json").read_text())
    manifest = prospective.make_manifest(protocol)
    assert {s: len(rows) for s, rows in manifest.items()} == {"train": 24, "development": 12, "test": 24}
    topics = {s: {r["topic_id"] for r in rows} for s, rows in manifest.items()}
    prompts = {s: {r["prompt_family_id"] for r in rows} for s, rows in manifest.items()}
    for a, b in (("train", "development"), ("train", "test"), ("development", "test")):
        assert not topics[a] & topics[b]
        assert not prompts[a] & prompts[b]
    assert prospective.make_manifest(protocol) == manifest


def test_topic_or_prompt_family_reuse_rejected():
    p = json.loads((ROOT / "configs/prospective_protocol_v1.json").read_text())
    p["topics"]["test"][0] = copy.deepcopy(p["topics"]["train"][0])
    with pytest.raises(ValueError, match="Topic"):
        prospective.make_manifest(p)
    p = json.loads((ROOT / "configs/prospective_protocol_v1.json").read_text())
    p["prompts"]["test"]["explanation"] = copy.deepcopy(p["prompts"]["train"]["explanation"])
    with pytest.raises(ValueError, match="Prompt"):
        prospective.make_manifest(p)


def test_test_generation_requires_locked_attacks(setup_run):
    _, run = setup_run
    with pytest.raises(ValueError, match="locked"):
        prospective.generate(run, "test", adapter=FixtureAdapter())
    assert not (run / "records" / "test").exists()


def test_frozen_protocol_mutation_rejected(setup_run):
    _, run = setup_run
    p = prospective.read_json(run / "protocol.json")
    p["payload_seed"] += 1
    prospective.atomic_json(run / "protocol.json", p)
    with pytest.raises(ValueError, match="Frozen protocol"):
        prospective.generate(run, "train", adapter=FixtureAdapter())


def test_frozen_manifest_mutation_rejected(setup_run):
    _, run = setup_run
    p = prospective.read_json(run / "manifests/train.json")
    p[0]["topic"] = "changed topic"
    prospective.atomic_json(run / "manifests/train.json", p)
    with pytest.raises(ValueError, match="manifest"):
        prospective.generate(run, "train", adapter=FixtureAdapter())


def test_adapter_prepares_before_payload_and_preserves_record(setup_run):
    protocol, run = setup_run
    adapter = FixtureAdapter()
    report = prospective.generate(run, "train", adapter=adapter, limit=1)
    assert report["new_attempts"] == 1
    assert adapter.order == ["cover", "prepare", "encode", "encode"]
    row = prospective.read_records(run, "train", require_complete=False)[0]
    assert row["synthetic_payload"] == prospective.payload_for(protocol, row["sample"])
    assert row["prepared_methods"]["fixed_canonical"]["original"] == row["cover_text"]
    assert row["variants"]["fixed_quality"]["metrics"]["full_message_success"]


def test_payload_dependent_prepared_state_mutation_fails(setup_run):
    _, run = setup_run
    class BadAdapter(FixtureAdapter):
        def encode_decode(self, variant, prepared, payload_bits, *args):
            prepared["payload"] = payload_bits
            return super().encode_decode(variant, prepared, payload_bits, *args)
    prospective.generate(run, "train", adapter=BadAdapter(), limit=1)
    row = prospective.read_records(run, "train", require_complete=False)[0]
    assert all("changed after payload" in v["error"] for v in row["variants"].values())
    assert not prospective.common_success([row], ["fixed_canonical", "fixed_quality"])


def test_interrupted_attempt_is_not_silently_regenerated(setup_run):
    _, run = setup_run
    sample = prospective.read_json(run / "manifests/train.json")[0]
    prospective.atomic_json(run / "attempt_started/train" / (sample["id"] + ".json"), {"time_unix": 0})
    adapter = FixtureAdapter()
    prospective.generate(run, "train", adapter=adapter, limit=1)
    assert adapter.order == []
    row = prospective.read_records(run, "train", require_complete=False)[0]
    assert row["failure"] == "interrupted_attempt_not_retried"


def test_capacity_failure_stays_in_reliability_denominator(setup_run):
    _, run = setup_run
    class ShortAdapter(FixtureAdapter):
        def encode_decode(self, *args):
            raise ValueError("capacity shortfall")
    prospective.generate(run, "train", adapter=ShortAdapter(), limit=1)
    rows = prospective.read_records(run, "train", require_complete=False)
    result = prospective.reliability(rows, ["fixed_canonical"])["fixed_canonical"]
    assert result["attempted_samples"] == 1
    assert result["full_success_rate"] == 0
    assert result["failed_samples"] == 1


def test_duplicate_covers_quarantined_without_replacement(setup_run):
    _, run = setup_run
    class DuplicateAdapter(FixtureAdapter):
        def generate_cover(self, *args):
            return {"text": "Exactly repeated cover"}
    prospective.generate(run, "train", adapter=DuplicateAdapter())
    rows = prospective.read_records(run, "train")
    assert sum(bool(r["failure"]) for r in rows) == len(rows) - 1
    assert all("quarantined_duplicate" in r["failure"] for r in rows[1:])


def test_conservative_threshold_handles_ties():
    scores = [.8, .8, .2, .1]
    threshold = prospective.conservative_fpr_threshold(scores, .25)
    assert sum(p >= threshold for p in scores) / len(scores) <= .25
    assert threshold > .8


def test_full_fixture_workflow_locks_models_and_opens_test_once(setup_run):
    protocol, run = setup_run
    for split in ("train", "development"):
        prospective.generate(run, split, adapter=FixtureAdapter())
    lock = prospective.train_attacks(run)
    assert len(lock["attacks"]) == 4
    with pytest.raises(ValueError, match="cannot change"):
        prospective.generate(run, "train", adapter=FixtureAdapter())
    with pytest.raises(ValueError, match="Complete test"):
        prospective.evaluate(run)
    assert not (run / "test_opened.json").exists()
    prospective.generate(run, "test", adapter=FixtureAdapter())
    evaluation = prospective.evaluate(run)
    assert evaluation["fixture_only"] is True
    assert evaluation["attack_cohort"]["pairs"] == 4
    assert all(v["full_success_samples"] == 4 for v in evaluation["reliability"].values())
    assert (run / "test_opened.json").exists()
    with pytest.raises(ValueError, match="already evaluated"):
        prospective.evaluate(run)
    with pytest.raises(ValueError, match="Opened test"):
        prospective.generate(run, "test", adapter=FixtureAdapter())


def test_train_data_changes_after_lock_rejected(setup_run):
    protocol, run = setup_run
    for split in ("train", "development"):
        prospective.generate(run, split, adapter=FixtureAdapter())
    prospective.train_attacks(run)
    path = next((run / "records/train").glob("*.json"))
    row = prospective.read_json(path)
    row["cover_text"] += " altered"
    prospective.atomic_json(path, row)
    with pytest.raises(ValueError, match="data changed"):
        prospective.generate(run, "test", adapter=FixtureAdapter())


def test_common_success_preserves_shared_comparison_cohort():
    records = [
        {"variants": {"a": {"metrics": {"full_message_success": True}}, "b": {"metrics": {"full_message_success": True}}}},
        {"variants": {"a": {"metrics": {"full_message_success": True}}, "b": {"metrics": {"full_message_success": False}}}},
        {"failure": "generator failed", "variants": {}},
    ]
    assert prospective.common_success(records, ["a", "b"]) == records[:1]
