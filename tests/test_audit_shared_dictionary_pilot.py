"""Independent-audit checks using synthetic covers and fabricated score fixtures.

No model is loaded, and no experimental record or outcome is read. Synthetic
NLL evidence tests arithmetic/provenance checks, not the truth of model scores.
"""

import copy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import unicodedata
from unittest.mock import patch

import pytest

from synsteg_audit.fixed_codebook import freeze_codebook


ROOT = Path(__file__).resolve().parents[1]


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


pilot = import_script("audit_tests_shared_runner", "run_shared_dictionary_pilot.py")
auditor = import_script("audit_tests_shared_auditor", "audit_shared_dictionary_pilot.py")


class SyntheticAdapter:
    """Use the real codecs, but never represent synthetic costs as real inference."""

    def __init__(self):
        self.baseline_adapter = pilot.transfer.local.LocalAdapter()

    def generate_cover(self, sample, settings):
        word, count = ("small", 160) if sample["id"] == "s0" else ("fast", 40)
        return {"text": " ".join([word] * count), "metadata": {"synthetic_fixture": True}}

    def prepare_shared(self, cover, dictionary, protocol):
        analysis = pilot.codec.analyze_text(cover, dictionary, protocol)
        classes = {entry.class_id: entry.words for entry in dictionary.classes}
        costs, evidence, unique = {}, [], {}
        for slot in analysis.slots:
            words = classes[slot.class_id]
            for i, word in enumerate(words):
                cost = 0.0 if len(words) == 2 else 1.0 + i / 10.0
                costs[f"{slot.index}:{word}"] = cost
                item = {"slot": slot.index, "candidate": word, "cost": cost}
                if len(words) == 2:
                    item["kind"] = "singleton_groups_no_quality_choice_not_a_model_score"
                else:
                    text = cover[:slot.start] + word + cover[slot.end:]
                    sha = pilot.neural.text_hash(text)
                    score = {"ppl": math.exp(cost), "details": {
                        "mean_nll": cost, "sum_nll": cost, "input_tokens": 2,
                        "scored_input_tokens": 2, "predicted_tokens": 1,
                        "truncated": False, "ppl_policy": "sliding_window"},
                        "forward_passes": 1}
                    item.update(kind="original_cover_single_replacement_mean_nll", text_sha256=sha, score=score)
                    unique[sha] = score
                evidence.append(item)
        return {"analysis": asdict(analysis), "candidate_costs": costs,
                "candidate_score_evidence": evidence,
                "unique_full_text_scoring_calls": len(unique),
                "model_forward_passes": len(unique),
                "score_policy": "Synthetic unit-test costs; no language model was run."}

    def prepare_baseline(self, cover, sample, config):
        words = ["compact", "little", "small", "tiny"] if sample["id"] == "s0" else ["fast", "quick"]
        book = freeze_codebook(cover, [words] * len(cover.split()),
            seed="synthetic-baseline:" + sample["id"], max_bits_per_position=1,
            min_group_size=1, candidate_filter_id="synthetic-admission",
            scorer=lambda context: float(len(context.candidate)), scorer_id="synthetic-length-cost")
        return {"codebook": book.to_dict(), "codebook_sha256": book.sha256,
                "eligibility": book.eligibility, "capacity_bits": book.capacity_bits,
                "framing": "external_length", "external_length_bits": 32,
                "token_alignment": {"considered": 0, "admitted": 0, "rejected": 0},
                "semantic_equivalence_verified": False}

    def encode_decode(self, *args):
        return self.baseline_adapter.encode_decode(*args)


def add_synthetic_predictions(directory, config, records, evaluation, lock):
    """Write known perfect 0.2/0.8 score fixtures; no classifier is executed."""
    for framing in ("plain_length16", "aes_siv"):
        for bits in config["payload_bits"]:
            paired_arms = [f"{framing}__{selector}__{bits}" for selector in ("canonical", "quality")]
            successful = {alias: [row for row in rows if all(row["variants"][arm]["full_success"] for arm in paired_arms)]
                          for alias, rows in records.items()}
            matched = set.intersection(*({row["sample"]["id"] for row in rows} for rows in successful.values()))
            for alias, rows in successful.items():
                for cohort, selected in (("per_model_success", rows),
                        ("matched_all_models_success", [row for row in rows if row["sample"]["id"] in matched])):
                    cohort_id = f"{framing}__{bits}__{cohort}"
                    item = {"pairs": len(selected), "sample_ids": [row["sample"]["id"] for row in selected], "attacks": {}}
                    for selector in ("canonical", "quality"):
                        for feature in ("char", "word"):
                            name = f"{framing}__{selector}__{bits}__{feature}"
                            if not selected:
                                item["attacks"][name] = {"status": "no_common_success_pairs"}
                                continue
                            source_name = f"fixed_{selector}__{feature}"
                            source = lock["attacks"][source_name]
                            predictions = [{"sample_id": row["sample"]["id"], "label": label,
                                "score": 0.2 if label == 0 else 0.8, "prediction": label,
                                "prediction_at_source_dev_threshold": label}
                                for row in selected for label in (0, 1)]
                            path = directory / "predictions" / alias / cohort_id / (name + ".json")
                            pilot.atomic_json(path, predictions)
                            item["attacks"][name] = {"status": "complete", "source_attack": source_name,
                                "accuracy": 1.0, "auc": 1.0, "fpr_at_development_threshold": 0.0,
                                "tpr_at_development_threshold": 1.0, "source_development_threshold": source["threshold"],
                                "source_score_direction": source["score_direction"], "predictions_sha256": pilot.file_hash(path)}
                    evaluation["models"][alias]["cohorts"][cohort_id] = item


def synthetic_run(directory, interrupted=False):
    config = pilot.read_json(ROOT / "configs/shared_dictionary_pilot_v1.json")
    config.update(models=[{"alias": alias, "model_id": "synthetic/model", "revision": "a" * 40}
                          for alias in ("a", "b")],
                  samples=[{"id": f"s{i}", "topic_id": f"t{i}", "genre_id": "fixture",
                            "prompt": f"Synthetic fixture {i}", "generation_seed": i}
                           for i in range(2)],
                  external_resource_files=[], additional_identity_files=[])
    config["baseline_method_config"].update(candidate_filter_id="synthetic-admission", scorer_id="synthetic-length-cost")
    dictionary = pilot.codec.PublicDictionary.from_mapping({
        "size": ["compact", "little", "small", "tiny"], "pace": ["fast", "quick"]},
        version="synthetic-auditor-fixture-v1")
    attacks, attack_files = {}, {}
    for variant in ("fixed_canonical", "fixed_quality"):
        for feature in ("char", "word"):
            name = variant + "__" + feature
            path = directory / "source_attacks" / (name + ".joblib")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("UNFITTED SYNTHETIC AUDIT FIXTURE: " + name).encode())
            sha = pilot.file_hash(path)
            attack_files[str(path.relative_to(directory))] = sha
            attacks[name] = {"variant": variant, "features": feature,
                "score_direction": -1 if feature == "char" else 1,
                "decision_threshold": 0.5, "threshold": 0.6,
                "artifact": path.name, "artifact_sha256": sha}
    lock = {"attacks": attacks, "freeze_id": "synthetic-source-freeze"}
    lock["lock_id"] = pilot.digest(lock)
    pilot.atomic_json(directory / "source_attacks/attack_lock.json", lock)
    config["source_run"] = str(directory / "source_attacks")
    config["source_attack_lock_sha256"] = pilot.file_hash(directory / "source_attacks/attack_lock.json")
    attack_files["source_attacks/attack_lock.json"] = config["source_attack_lock_sha256"]
    pilot.atomic_json(directory / "protocol.json", config)
    pilot.atomic_json(directory / "samples.json", config["samples"])
    pilot.atomic_json(directory / "public_dictionary.json", dictionary.to_dict())
    freeze = {"schema": config["schema"], "dictionary_sha256": dictionary.sha256,
              "dictionary_file_bytes": (directory / "public_dictionary.json").stat().st_size,
              "dictionary_file_sha256": pilot.file_hash(directory / "public_dictionary.json"),
              "protocol_sha256": pilot.file_hash(directory / "protocol.json"),
              "samples_sha256": pilot.file_hash(directory / "samples.json"),
              "code_sha256": {"src/synsteg_audit/shared_dictionary.py": pilot.file_hash(ROOT / "src/synsteg_audit/shared_dictionary.py")},
              "resource_sha256": {},
              "source_lock_id": lock["lock_id"],
              "source_attack_files_sha256": attack_files,
              "scorer_job_files_sha256": {name: pilot.file_hash(path) for name, path in pilot.JOBS.items()},
              "environment": {"unicode_database_version": unicodedata.unidata_version},
              "fixture_key_is_public_not_secret": True}
    freeze["freeze_id"] = pilot.digest(freeze)
    pilot.atomic_json(directory / "freeze.json", freeze)
    if interrupted:
        pilot.atomic_json(directory / "models/a/attempt_started/s0.json", {"time_unix": 1.0, "freeze_id": freeze["freeze_id"]})
    with patch.object(pilot, "verify", return_value=(config, freeze, dictionary, lock)):
        for alias in ("a", "b"):
            pilot.generate(directory, alias, adapter=SyntheticAdapter())
    records = {alias: pilot.read_records(directory, alias, complete=True) for alias in ("a", "b")}
    bundles = {alias: pilot.read_json(directory / "models" / alias / "bundle.json") for alias in ("a", "b")}
    events = {alias: [json.loads(line) for line in (directory / "models" / alias / "events.jsonl").read_text().splitlines()]
              for alias in ("a", "b")}
    identity = {"freeze_id": freeze["freeze_id"], "source_lock_id": lock["lock_id"],
                "all_model_records_sha256": {alias: pilot.digest(rows) for alias, rows in records.items()},
                "all_model_bundle_sha256": {alias: pilot.file_hash(directory / "models" / alias / "bundle.json") for alias in records}}
    evaluation = {**identity, "schema": config["schema"], "status": "complete_shared_dictionary_feasibility",
                  "evaluation_resume_count": 0,
                  "models": {alias: {"planned_covers": len(rows), "generated_covers": sum(bool(row.get("cover_text")) for row in rows),
                              "reliability": {arm["id"]: pilot.reliability(rows, arm) for arm in pilot.arms(config)}, "cohorts": {}}
                             for alias, rows in records.items()}}
    add_synthetic_predictions(directory, config, records, evaluation, lock)
    pilot.atomic_json(directory / "evaluation.json", evaluation)
    opening_time = max(event["time_unix"] for ledger in events.values() for event in ledger) + 1
    pilot.atomic_json(directory / "evaluation_opened.json", {**identity, "opened_at_unix": opening_time})
    root_events = [{"event": "shared_design_dictionary_and_attacks_frozen_before_generation", "time_unix": 0.0, "freeze_id": freeze["freeze_id"]},
                   {"event": "shared_evaluation_opened_once", "time_unix": opening_time, **identity},
                   {"event": "shared_evaluation_completed", "time_unix": opening_time + 1,
                    "evaluation_sha256": pilot.file_hash(directory / "evaluation.json")}]
    (directory / "events.jsonl").write_bytes(b"".join(pilot.canonical(event) + b"\n" for event in root_events))
    return {"run": directory, "config": config, "freeze": freeze, "dictionary": dictionary,
            "records": records, "bundles": bundles, "model_events": events, "lock": lock}


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    return synthetic_run(tmp_path_factory.mktemp("shared-audit-fixture"))


@pytest.fixture
def evidence(fixture_run):
    return copy.deepcopy(fixture_run)


def check(evidence, *, fresh_receiver=False, model_events=True):
    return auditor.independent_checks(evidence["config"], evidence["freeze"], evidence["dictionary"],
        evidence["records"], evidence["bundles"],
        model_events=evidence["model_events"] if model_events else None,
        fresh_receiver=fresh_receiver)


def sync_prepared(row):
    row["prepared_shared_sha256"] = pilot.digest(row["prepared_shared"])


@pytest.mark.parametrize("missing", ["evaluation.json", "evaluation_opened.json", "events.jsonl"])
def test_incomplete_primary_gate_refuses_before_any_outcome_read(tmp_path, monkeypatch, missing):
    for name in ("evaluation.json", "evaluation_opened.json", "events.jsonl"):
        if name != missing:
            (tmp_path / name).write_text("{}\n")

    def forbidden_read(path):
        raise AssertionError(f"A closed gate must not read outcomes: {path}")

    monkeypatch.setattr(auditor, "read_json", forbidden_read)
    with pytest.raises(auditor.AuditError):
        auditor.load_completed(tmp_path)


def test_audit_cannot_bypass_completed_primary_gate(tmp_path, monkeypatch):
    def closed(run):
        raise auditor.AuditError("synthetic primary gate closed")

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent outcome checks ran before opening")

    monkeypatch.setattr(auditor, "load_completed", closed)
    monkeypatch.setattr(auditor, "independent_checks", forbidden)
    with pytest.raises(auditor.AuditError, match="gate closed"):
        auditor.audit(tmp_path)


def test_intact_synthetic_completion_gate_and_full_audit(fixture_run):
    config, frozen, dictionary, evaluation, records, bundles = auditor.load_completed(fixture_run["run"])
    assert frozen["freeze_id"] == fixture_run["freeze"]["freeze_id"]
    assert config["samples"] == fixture_run["config"]["samples"]
    assert dictionary.sha256 == fixture_run["dictionary"].sha256
    assert set(records) == set(bundles) == {"a", "b"}
    result = auditor.audit(fixture_run["run"], fresh_receiver=False)
    assert result["status"] == "passed"
    assert result["shared_attempts"] == 32


@pytest.mark.parametrize("corruption", ["evaluation_hash", "duplicate_opening"])
def test_corrupt_primary_control_gate_rejects_before_raw_records(fixture_run, tmp_path, monkeypatch, corruption):
    run = tmp_path / "corrupted"
    shutil.copytree(fixture_run["run"], run)
    if corruption == "evaluation_hash":
        path = run / "evaluation.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = run / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        duplicate = {**events[1], "time_unix": events[-1]["time_unix"] + 1}
        path.write_bytes(path.read_bytes() + pilot.canonical(duplicate) + b"\n")
    actual_read = auditor.read_json

    def no_raw_records(path):
        assert "/records/" not in str(path), "Raw outcome opened before primary integrity gate"
        return actual_read(path)

    monkeypatch.setattr(auditor, "read_json", no_raw_records)
    with pytest.raises(auditor.AuditError):
        auditor.load_completed(run)


@pytest.mark.parametrize("unfinished", ["incomplete_generation", "active_writer"])
def test_every_model_is_preflighted_before_any_raw_record_read(fixture_run, tmp_path, monkeypatch, unfinished):
    run = tmp_path / "unfinished-later-model"
    shutil.copytree(fixture_run["run"], run)
    if unfinished == "incomplete_generation":
        path = run / "models/b/generation.json"
        generation = pilot.read_json(path)
        generation["complete"] = False
        pilot.atomic_json(path, generation)
    else:
        (run / "models/b/.writer.lock").write_text("synthetic active writer")
    actual_read = auditor.read_json

    def no_raw_records(path):
        assert "/records/" not in str(path), "A finished earlier model leaked before all models passed the gate"
        return actual_read(path)

    monkeypatch.setattr(auditor, "read_json", no_raw_records)
    with pytest.raises(auditor.AuditError):
        auditor.load_completed(run)


def test_independent_complete_roundtrips_and_all_attempt_denominators(evidence):
    summary = check(evidence, fresh_receiver=True)
    assert summary["planned_covers"] == 4
    assert summary["shared_attempts"] == 32
    assert summary["baseline_attempts"] == 8
    assert summary["bundle_attempts"] == 4
    assert summary["shared_successes"] == 24
    assert summary["baseline_successes"] == 8
    assert summary["bundle_successes"] == 4
    assert summary["public_key_is_not_secret"] is True
    costs = summary["models"]["a"]["resource_accounting_by_arm"]
    assert costs["plain_length16__canonical__8"]["successful_frame_bits"] == 48
    assert costs["aes_siv__canonical__8"]["successful_frame_bits"] == 152
    assert costs["aes_siv__canonical__8"]["individual_codebook_json_bytes_across_successes"] == 0
    assert costs["fixed_quality__8"]["external_length_bits_per_attempt"] == 32
    assert costs["fixed_quality__8"]["individual_codebook_json_bytes_across_successes"] > 0


def test_independent_receiver_does_not_call_production_shared_decoder(evidence, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production shared decoder must not validate its own output")

    monkeypatch.setattr(auditor.codec, "decode", forbidden)
    assert check(evidence)["shared_successes"] == 24


@pytest.mark.parametrize("mutation", ["model", "record", "arm", "bundle_arm"])
def test_incomplete_declared_panel_or_arm_set_rejected(evidence, mutation):
    if mutation == "model":
        evidence["records"].pop("b")
    elif mutation == "record":
        evidence["records"]["a"].pop()
    elif mutation == "arm":
        evidence["records"]["a"][0]["variants"].pop("aes_siv__quality__8")
    else:
        evidence["bundles"]["a"]["variants"].pop("16")
    with pytest.raises(auditor.AuditError):
        check(evidence, model_events=False)


def test_prepared_hash_and_candidate_score_provenance_are_checked(evidence):
    row = evidence["records"]["a"][0]
    row["prepared_shared"]["candidate_score_evidence"][0]["text_sha256"] = "0" * 64
    with pytest.raises(auditor.AuditError, match="preparation checksum"):
        check(evidence, model_events=False)
    sync_prepared(row)
    with pytest.raises(auditor.AuditError, match="unchanged cover context"):
        check(evidence, model_events=False)


def test_inconsistent_nll_arithmetic_rejected_even_with_updated_preparation_hash(evidence):
    row = evidence["records"]["a"][0]
    row["prepared_shared"]["candidate_score_evidence"][0]["score"]["details"]["sum_nll"] += 1
    sync_prepared(row)
    with pytest.raises(auditor.AuditError, match="NLL/PPL arithmetic"):
        check(evidence, model_events=False)


def test_false_capacity_failure_does_not_hide_a_deliverable_arm(evidence):
    arm = evidence["records"]["a"][0]["variants"]["aes_siv__canonical__8"]
    arm.update(full_success=False, stego_text=None, error="CapacityError: fabricated")
    with pytest.raises(auditor.AuditError, match="False capacity failure"):
        check(evidence, model_events=False)


@pytest.mark.parametrize("corrupt_production_replay", [False, True])
def test_valid_decode_is_not_sufficient_for_claimed_quality_selection(evidence, monkeypatch, corrupt_production_replay):
    row, dictionary = evidence["records"]["a"][0], evidence["dictionary"]
    arm = row["variants"]["plain_length16__quality__8"]
    protocol = pilot.public_protocol(evidence["config"], "plain_length16")
    payload = bytes.fromhex(row["payloads_hex"]["8"])
    costs = row["prepared_shared"]["candidate_costs"]
    wrong = pilot.codec.encode(row["cover_text"], payload, pilot.fixture_key(evidence["config"]), dictionary,
        protocol, selector="quality", candidate_cost=lambda slot, candidate: -costs[f"{slot.index}:{candidate}"])
    assert wrong.stego_text != arm["stego_text"]
    assert pilot.codec.decode(wrong.stego_text, pilot.fixture_key(evidence["config"]), dictionary, protocol) == payload
    arm.update(stego_text=wrong.stego_text,
               encoding={key: value for key, value in asdict(wrong).items() if key != "stego_text"})
    if corrupt_production_replay:
        original_encode = auditor.codec.encode

        def corrupted_replay(cover, data, key, public_dictionary, public_protocol, *, selector="canonical", candidate_cost=None):
            if cover == row["cover_text"] and data == payload and public_protocol.framing == "plain_length16" and selector == "quality":
                return wrong
            return original_encode(cover, data, key, public_dictionary, public_protocol,
                                   selector=selector, candidate_cost=candidate_cost)

        monkeypatch.setattr(auditor.codec, "encode", corrupted_replay)
    message = "Independent required-group minimum selection differs" if corrupt_production_replay else "group selection differs"
    with pytest.raises(auditor.AuditError, match=message):
        check(evidence, model_events=False)


def test_receiver_contract_cannot_smuggle_cover_or_individual_codebook(evidence):
    receiver = evidence["records"]["a"][0]["variants"]["aes_siv__canonical__8"]["receiver"]
    receiver["input_keys"].append("cover_text")
    with pytest.raises(auditor.AuditError, match="receiver proof changed"):
        check(evidence, model_events=False)


def test_wrong_public_fixture_key_is_rejected(evidence):
    evidence["records"]["a"][0]["public_fixture_key_hex"] = bytes(32).hex()
    with pytest.raises(auditor.AuditError, match="Fixture key differs"):
        check(evidence, model_events=False)


def test_bundle_cannot_select_a_subset_of_declared_covers(evidence):
    bundle = evidence["bundles"]["a"]
    bundle["sample_ids"] = ["s0"]
    bundle["cover_text"] = evidence["records"]["a"][0]["cover_text"]
    with pytest.raises(auditor.AuditError, match="Bundle identity/order"):
        check(evidence, model_events=False)


def test_payload_preparation_event_must_precede_terminal_attempt(evidence):
    events = evidence["model_events"]["a"]
    prepared = next(event for event in events if event["event"] == "methods_prepared_before_payload")
    completed = next(event for event in events if event["event"] == "attempt_completed")
    events.remove(prepared)
    events.insert(events.index(completed) + 1, prepared)
    with pytest.raises(auditor.AuditError):
        check(evidence)


def test_interrupted_cover_keeps_all_attempts_and_failed_bundle(tmp_path):
    evidence = synthetic_run(tmp_path / "interrupted", interrupted=True)
    summary = check(evidence)
    assert summary["planned_covers"] == 4
    assert summary["shared_attempts"] == 32
    assert summary["baseline_attempts"] == 8
    assert summary["bundle_attempts"] == 4
    assert summary["bundle_successes"] == 2
    assert evidence["bundles"]["a"]["cover_text"] is None
    assert not any(arm["full_success"] for arm in evidence["bundles"]["a"]["variants"].values())
