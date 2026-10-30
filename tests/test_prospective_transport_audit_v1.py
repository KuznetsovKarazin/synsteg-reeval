"""Synthetic-only validation of the independent prospective transport auditor."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load("prospective_transport_audit_tested", ROOT / "scripts/audit_prospective_development_transport_v1.py")
CODEC_PATH = ROOT / "runtime/prospective-development-transport-v1/frozen/src/synsteg_audit/shared_dictionary.py"


@pytest.fixture(scope="module")
def codec():
    if not CODEC_PATH.exists():
        pytest.skip("strict-hash restored frozen codec is required for synthetic interoperability tests")
    assert audit.digest(CODEC_PATH.read_bytes()) == "21dc3dcfb2b0cd6396424280fc2a47695481eb815b29ab2381cfdcf7f2d47d2f"
    return load("prospective_transport_frozen_codec_test_only", CODEC_PATH)


def test_hkdf_rfc5869_vector():
    result = audit.hkdf_sha256(bytes.fromhex("0b" * 22), bytes.fromhex("000102030405060708090a0b0c"),
                               bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"), 42)
    assert result.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                            "34007208d5b887185865")


@pytest.mark.parametrize("framing", ["plain_length16", "aes_siv"])
@pytest.mark.parametrize("group_size,width", [(1, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("payload", [b"", b"\x00\xff", bytes(range(16)), bytes(range(64))])
def test_independent_oracle_synthetic_roundtrip(codec, framing, group_size, width, payload):
    dictionary = codec.PublicDictionary.from_mapping({"shape": ["azure", "blue", "cyan", "navy"]}, "synthetic-audit-v1")
    protocol = codec.Protocol(group_size, width, framing)
    cover = "Literal marker: " + " ".join(["blue", "Blue,", "BLUE!", "(cyan)"] * 180) + " tail."
    key = bytes(range(32))
    encoded = codec.encode(cover, payload, key, dictionary, protocol)
    before = audit.public_analysis(cover, dictionary.to_dict(), protocol.to_dict())
    after = audit.public_analysis(encoded.stego_text, dictionary.to_dict(), protocol.to_dict())
    assert before[1:] == after[1:]
    assert after[2] == encoded.capacity_bits
    assert audit.public_oracle(encoded.stego_text, key, dictionary.to_dict(), protocol.to_dict()) == payload


def test_lexer_rejects_compounds_digits_mixed_case_and_underscore(codec):
    dictionary = codec.PublicDictionary.from_mapping({"shape": ["azure", "blue", "cyan", "navy"]})
    protocol = codec.Protocol(1, 2, "plain_length16")
    text = "blue (Blue) CYAN! blue_navy blue-navy bluE 2blue blue2 \u20acblue blue'"
    own = audit.public_analysis(text, dictionary.to_dict(), protocol.to_dict())
    expected = codec.analyze_text(text, dictionary, protocol)
    assert [slot["surface"] for slot in own[0]] == ["blue", "Blue", "CYAN", "blue"]
    assert own[1:] == (expected.normalization_sha256, expected.capacity_bits)


@pytest.mark.parametrize("mutation", ["wrong_key", "literal_change", "truncated"])
def test_oracle_rejects_invalid_authenticated_frame(codec, mutation):
    dictionary = codec.PublicDictionary.from_mapping({"shape": ["azure", "blue", "cyan", "navy"]})
    protocol = codec.Protocol(1, 2, "aes_siv")
    key = bytes(range(32))
    text = codec.encode("blue " * 150 + "literal.", b"secret", key, dictionary, protocol).stego_text
    if mutation == "wrong_key":
        key = bytes(reversed(range(32)))
    elif mutation == "literal_change":
        text += "extra"
    else:
        text = " ".join(text.split()[:10])
    from cryptography.exceptions import InvalidTag
    with pytest.raises((ValueError, InvalidTag)):
        audit.public_oracle(text, key, dictionary.to_dict(), protocol.to_dict())


def test_oracle_rejects_dictionary_duplicates():
    raw = {"schema": "synsteg-public-dictionary-v1", "version": "fixture",
           "classes": [{"id": "a", "words": ["blue", "blue"]}]}
    with pytest.raises(ValueError, match="uniqueness"):
        audit.dictionary_canonical(raw)


def test_group_valid_noncanonical_synonym_is_rejected_by_sender_audit(codec):
    dictionary = codec.PublicDictionary.from_mapping({"shape": ["azure", "blue", "cyan", "navy", "red", "pink", "gold", "green"]})
    protocol = codec.Protocol(1, 2, "aes_siv")
    key, payload = bytes(range(32)), b"fixture"
    encoded = codec.encode("blue " * 200, payload, key, dictionary, protocol)
    audit.verify_canonical_members(encoded.stego_text, key, dictionary.to_dict(), protocol.to_dict(), encoded.consumed_slots)
    analysis = codec.analyze_text(encoded.stego_text, dictionary, protocol)
    slot = analysis.slots[0]
    _, mapping_key = codec._keys(key, dictionary.sha256, protocol)
    groups = codec._groups(slot, dictionary.classes[0].words, mapping_key, dictionary.sha256,
                            protocol.sha256, analysis.normalization_sha256)
    group = next(group for group in groups if slot.surface.lower() in group)
    alternative = next(word for word in group if word != slot.surface.lower())
    changed = encoded.stego_text[:slot.start] + alternative + encoded.stego_text[slot.end:]
    assert audit.public_oracle(changed, key, dictionary.to_dict(), protocol.to_dict()) == payload
    with pytest.raises(ValueError, match="noncanonical"):
        audit.verify_canonical_members(changed, key, dictionary.to_dict(), protocol.to_dict(), encoded.consumed_slots)


def test_oracle_has_no_production_imports():
    import ast
    tree = ast.parse((ROOT / "scripts/audit_prospective_development_transport_v1.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("synsteg_audit")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("synsteg_audit") for alias in node.names)


@pytest.fixture(scope="module")
def synthetic_grid():
    records, payloads, cells = [], [], []
    for order in range(1, 101):
        record_id = f"synthetic-source-{order:03d}"
        for parent in (128, 256, 512):
            bits = ("0000000100100011" * (parent // 16))
            payloads.append({"record_id": record_id, "source_order": order,
                "primary_payload_length": parent, "payload_bits": bits,
                "secondary16_bits": bits[:16], "payload_sha256": audit.digest(bits.encode()),
                "secondary16_sha256": audit.digest(bits[:16].encode())})
        for model in ("qwen05", "tinyllama11", "falcon3_1b"):
            status = "completed" if order == 1 and model == "tinyllama11" else "not_attempted"
            record = {"record_id": record_id, "model_alias": model, "source_order": order,
                      "identity": audit.digest(f"{model}:{record_id}".encode()), "cover_status": status}
            records.append(record)
            for parent in (128, 256, 512):
                bits = "0000000100100011" * (parent // 16)
                for grid, arms in (("primary", audit.PUBLIC_ARMS | audit.FIXED_ARMS | {"original_native"}),
                                   ("secondary16", audit.PUBLIC_ARMS | audit.FIXED_ARMS)):
                    expected = bits if grid == "primary" else bits[:16]
                    for arm in sorted(arms):
                        resource = "ready" if arm in audit.PUBLIC_ARMS else audit.BLOCKED
                        execution = ("not_run_cover_unavailable" if status != "completed" else
                                     "not_run_resource_blocked" if resource == audit.BLOCKED else
                                     "insufficient_capacity")
                        cells.append({**record, "row_id": audit.cell_identity(model, record_id, parent, grid, arm),
                            "parent_payload_length": parent, "grid": grid, "arm": arm,
                            "requested_payload_bits": len(expected), "payload_sha256": audit.digest(expected.encode()),
                            "resource_status": resource, "execution_status": execution,
                            "capacity_bits": 0 if execution == "insufficient_capacity" else None,
                            "delivered_payload_bits": 0 if execution == "insufficient_capacity" else None})
    return cells, records, {"rows": payloads}


def test_complete_synthetic_grid_and_null_zero_distinction(synthetic_grid):
    cells, records, payload_map = synthetic_grid
    result = audit.validate_cell_grid(cells, records, payload_map)
    assert result["cells"] == 15300
    assert result["primary_cells"] == 8100
    assert result["secondary16_cells"] == 7200
    assert result["resource_statuses"] == {"ready": 7200, audit.BLOCKED: 8100}
    assert result["execution_statuses"] == {
        "not_run_cover_unavailable": 15249, "not_run_resource_blocked": 27, "insufficient_capacity": 24}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "null_to_zero", "wrong_precedence", "wrong_payload", "source_order"])
def test_grid_rejects_integrity_errors(synthetic_grid, mutation):
    import copy
    cells, records, payload_map = synthetic_grid
    cells = copy.deepcopy(cells)
    if mutation == "missing":
        cells.pop()
    elif mutation == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif mutation == "null_to_zero":
        cells[0]["delivered_payload_bits"] = 0
    elif mutation == "wrong_precedence":
        row = next(row for row in cells if row["cover_status"] == "not_attempted" and row["resource_status"] == audit.BLOCKED)
        row["execution_status"] = "not_run_resource_blocked"
    elif mutation == "wrong_payload":
        cells[0]["payload_sha256"] = "0" * 64
    else:
        cells[0]["source_order"] = 2
    with pytest.raises(ValueError):
        audit.validate_cell_grid(cells, records, payload_map)


@pytest.fixture(scope="module")
def synthetic_milestone(tmp_path_factory):
    """Create a wholly synthetic full-format milestone, never reading real covers."""
    import copy
    import json
    import shutil
    import subprocess
    from collections import Counter
    runner = load("prospective_runner_for_independent_test", ROOT / "scripts/run_prospective_development_transport_v1.py")
    source_tests = load("prospective_runner_fixtures_for_independent_test", ROOT / "tests/test_prospective_development_transport_v1.py")
    frozen = ROOT / "runtime/prospective-development-transport-v1/frozen"
    dependency = runner.dependencies(ROOT, frozen)
    plan, config, payload_map, shared, dictionaries, resources, pins, _ = dependency
    records = source_tests.fixture_records(config, dictionaries["v2"])
    output = tmp_path_factory.mktemp("synthetic_transport_milestone")
    metadata = {r["record_id"]: r for r in config["selected_record_metadata"]}
    models = {r["alias"]: r for r in config["local_models"]}
    cp_records, hashes = [], {}
    for row in records:
        row["model"], row["revision"] = models[row["model_alias"]]["model_id"], models[row["model_alias"]]["revision"]
        row["whitespace_words"] = len(row["cover_text"].split()) if row["cover_text"] is not None else None
        if row["status"] == "completed":
            relative = f"runs/large-campaign-development-v2/models/{row['model_alias']}/attempts/{row['identity']}.json"
            attempt = {"status": "completed", "identity": row["identity"], "record": metadata[row["record_id"]],
                       "result": {"cover_text": row["cover_text"]}}
            path = output / "input_snapshot" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            runner.write_json(path, attempt)
            hashes[relative] = runner.sha(path.read_bytes())
            row["attempt_sha256"] = hashes[relative]
        cp_records.append({k: v for k, v in row.items() if k not in ("source_order", "cover_text", "attempt_sha256")})
    counts = dict(Counter(r["status"] for r in records))
    model_counts = {m: sum(r["status"] == "completed" and r["model_alias"] == m for r in records) for m in models}
    cp = {"status": "verified", "consistency_issues": [], "planned_model_record_pairs": 300,
          "completed_generations": 3, "counts": counts, "model_counts": model_counts, "records": cp_records,
          "run_files_sha256": hashes, "paid_calls": 0, "closed_test_opened": False}
    runner.write_json(output / "input_checkpoint.json", cp)
    shutil.copyfile(ROOT / plan["cover_cohort"]["plan"], output / "input_config.json")
    shutil.copyfile(ROOT / plan["payloads"]["map"], output / "payload-map.json")
    shutil.copyfile(ROOT / runner.PLAN, output / "feasibility-protocol.json")
    runtime = runner.runtime_identity()
    implementation = {relative: runner.sha((ROOT / relative).read_bytes()) for relative in runner.IMPLEMENTATION_FILES}
    activation = {"experiment_id": runner.VERSION, "feasibility_protocol_sha256": runner.PLAN_SHA,
        "runtime": runtime, "implementation_files_sha256": implementation,
        "observed_completed_covers_at_implementation_freeze": 0,
        "target": {"checkpoint": "synthetic-fixture/after-checkpoint.json", "cover_counts": counts,
                   "model_counts": model_counts, "completed_covers": 3}}
    runner.write_json(output / "activation.json", activation)
    receipt = {"remote_readback_verified": True, "implementation_freeze_commit": "a" * 40,
        "checkpoint_commit": "b" * 40, "observed_completed_covers_at_implementation_freeze": 0,
        "synthetic_fixture_not_remote_verification": True,
        "verified_files_sha256": {**implementation, runner.ACTIVATION: runner.sha((output / "activation.json").read_bytes()),
            runner.PLAN: runner.PLAN_SHA, plan["payloads"]["map"]: plan["payloads"]["map_sha256"],
            activation["target"]["checkpoint"]: runner.sha((output / "input_checkpoint.json").read_bytes())}}
    runner.write_json(output / "git-readback-receipt.json", receipt)
    runner.write_rows(output / "records.jsonl.gz", records)
    cells, jobs, expected = runner.evaluate(records, payload_map, plan, shared, dictionaries)
    runner.write_rows(output / "cells_before_receiver.jsonl.gz", cells)
    recipient = output / "recipient"
    recipient.mkdir()
    shutil.copyfile(frozen / "src/synsteg_audit/shared_dictionary.py", recipient / "shared_dictionary.py")
    for version in ("v1", "v2"):
        shutil.copyfile(frozen / f"data/shared_dictionary_recovery_{version}.json", recipient / f"dictionary_{version}.json")
    runner.write_rows(recipient / "receiver_jobs.jsonl.gz", jobs)
    process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_prospective_development_transport_v1.py"),
        "--receiver", str(recipient)], cwd=recipient, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    runner.write_json(output / "receiver_process.json", {"returncode": 0, "stdout": process.stdout, "stderr": process.stderr,
        "job_count": len(jobs), "payload_or_cover_argument": False, "receiver_cwd": "recipient",
        "receiver_resource_files": sorted(p.name for p in recipient.iterdir())})
    runner.finalize(cells, expected, runner.read_rows(recipient / "receiver_results.jsonl.gz"))
    runner.write_rows(output / "cells.jsonl.gz", cells)
    summary, resource_output = runner.summarize(cells, records, resources)
    runner.write_json(output / "summary.json", summary)
    runner.write_json(output / "resources.json", resource_output)
    n_calls = sum(row["encoder_called"] is True for row in cells)
    execution = {"experiment_id": runner.VERSION, "activation_sha256": runner.sha((output / "activation.json").read_bytes()),
        "feasibility_protocol_sha256": runner.PLAN_SHA, "payload_map_sha256": plan["payloads"]["map_sha256"],
        "config_sha256": plan["cover_cohort"]["plan_sha256"], "checkpoint_sha256": runner.sha((output / "input_checkpoint.json").read_bytes()),
        "source_checkpoint": activation["target"]["checkpoint"], "runtime": runtime, "implementation_freeze_commit": "a" * 40,
        "checkpoint_commit": "b" * 40, "observed_completed_covers_at_implementation_freeze": 0,
        "resource_files_sha256": pins, "model_calls": 0, "paid_calls": 0, "closed_test_opened": False,
        "encoder_calls": n_calls, "receiver_jobs": len(jobs), "wall_seconds": 0,
        "fresh_receiver_has_cover_or_payload": False, "semantics_or_security_inferred": False}
    runner.write_json(output / "execution.json", execution)
    runner.write_json(output / "status.json", {"status": "completed_pending_independent_audit", "experiment_id": runner.VERSION,
        "completed_covers": 3, "full_denominator": 300, "total_cells": 15300, "unexpected_errors": 0,
        "encoder_calls": n_calls, "receiver_jobs": len(jobs), "model_calls": 0, "paid_calls": 0, "closed_test_opened": False})
    runner.write_json(output / "output_manifest.json", {str(p.relative_to(output)): runner.sha(p.read_bytes())
        for p in sorted(output.rglob("*")) if p.is_file()})
    return output, frozen, runner


def test_complete_synthetic_milestone_independent_audit(synthetic_milestone):
    output, frozen, _ = synthetic_milestone
    result = audit.audit(output, frozen, ROOT / "experiments/prospective-development-transport-v1/protocol.json")
    assert result["status"] == "passed"
    assert result["cells"] == 15300 and result["input_records"] == 300
    assert result["completed_covers"] == 3 and result["independent_capacity_checks"] == 72
    assert result["independent_decodes"] == result["receiver_jobs"] > 0
    assert result["unexpected_errors_preserved"] == 0


@pytest.mark.parametrize("mutation", ["output_hash", "snapshot_hash", "recipient_leak", "summary", "null_as_zero"])
def test_complete_milestone_rejects_tampering(synthetic_milestone, tmp_path, mutation):
    import shutil
    output, frozen, runner = synthetic_milestone
    target = tmp_path / "tampered"
    shutil.copytree(output, target)
    if mutation == "output_hash":
        (target / "summary.json").write_text("{}")
    elif mutation == "snapshot_hash":
        next((target / "input_snapshot").rglob("*.json")).write_text("{}")
    elif mutation == "recipient_leak":
        jobs = runner.read_rows(target / "recipient/receiver_jobs.jsonl.gz")
        jobs[0]["expected_bits"] = "0101"
        runner.write_rows(target / "recipient/receiver_jobs.jsonl.gz", jobs)
    elif mutation == "summary":
        summary = runner.read_json(target / "summary.json")
        summary["groups"][0]["planned_cells"] -= 1
        runner.write_json(target / "summary.json", summary)
    else:
        cells = runner.read_rows(target / "cells.jsonl.gz")
        cells[0]["delivered_payload_bits"] = 0
        runner.write_rows(target / "cells.jsonl.gz", cells)
    if mutation != "output_hash":
        runner.write_json(target / "output_manifest.json", {str(p.relative_to(target)): runner.sha(p.read_bytes())
            for p in sorted(target.rglob("*")) if p.is_file() and p.name != "output_manifest.json"})
    with pytest.raises(ValueError):
        audit.audit(target, frozen)


def test_real_runner_gates_on_handcrafted_synthetic_batch15(tmp_path):
    """Exercise run() gates/receiver/audit using30 wholly synthetic covers."""
    from collections import Counter
    import shutil
    runner = load("prospective_runner_real_entry_synthetic", ROOT / "scripts/run_prospective_development_transport_v1.py")
    frozen = ROOT / "runtime/prospective-development-transport-v1/frozen"
    plan = runner.read_json(ROOT / runner.PLAN)
    config = runner.read_json(ROOT / plan["cover_cohort"]["plan"])["development_candidate"]
    root = tmp_path / "synthetic-root"
    for relative in (runner.PLAN, plan["cover_cohort"]["plan"], plan["payloads"]["map"], config["source_prompts"],
                     *runner.IMPLEMENTATION_FILES):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    checkpoint = "results/large-campaign-development-v2/batches/batch15/after-checkpoint.json"
    snapshot = (root / checkpoint).parent / "after"
    source_records, hashes = [], {}
    for model in config["local_models"]:
        for order, meta in enumerate(config["selected_record_metadata"], 1):
            complete = model["alias"] == "tinyllama11" and order <= 30
            identity = runner.sha(runner.canonical(["synthetic-independent-batch15", model["alias"], meta["record_id"]]))
            row = dict(meta, model_alias=model["alias"], model=model["model_id"], revision=model["revision"],
                identity=identity, status="completed" if complete else "not_attempted", cover_sha256=None, whitespace_words=None)
            if complete:
                text = ["dexterous " * 800, "zzzzzz " * 20, "dexterous " * 20][(order - 1) % 3]
                relative = f"runs/large-campaign-development-v2/models/{model['alias']}/attempts/{identity}.json"
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                runner.write_json(path, {"status": "completed", "identity": identity, "record": meta, "result": {"cover_text": text}})
                hashes[relative] = runner.sha(path.read_bytes())
                row.update(cover_sha256=runner.sha(text.encode()), whitespace_words=len(text.split()))
            source_records.append(row)
    counts = dict(Counter(r["status"] for r in source_records))
    model_counts = {model["alias"]: sum(r["status"] == "completed" and r["model_alias"] == model["alias"] for r in source_records)
                    for model in config["local_models"]}
    cp = {"status": "verified", "consistency_issues": [], "planned_model_record_pairs": 300,
          "completed_generations": 30, "counts": counts, "model_counts": model_counts, "records": source_records,
          "run_files_sha256": hashes, "paid_calls": 0, "closed_test_opened": False}
    runner.write_json(root / checkpoint, cp)
    implementation = {relative: runner.sha((root / relative).read_bytes()) for relative in runner.IMPLEMENTATION_FILES}
    activation = {"experiment_id": runner.VERSION, "feasibility_protocol_sha256": runner.PLAN_SHA,
        "runtime": runner.runtime_identity(), "implementation_files_sha256": implementation,
        "observed_completed_covers_at_implementation_freeze": 12,
        "target": {"checkpoint": checkpoint, "batch": "batch15", "completed_covers": 30,
                   "cover_counts": counts, "model_counts": model_counts, "output": "results/synthetic-transport-only"}}
    runner.write_json(root / runner.ACTIVATION, activation)
    receipt = {"remote_readback_verified": True, "implementation_freeze_commit": "a" * 40, "checkpoint_commit": "b" * 40,
        "observed_completed_covers_at_implementation_freeze": 12, "synthetic_fixture_not_remote_verification": True,
        "verified_files_sha256": {**implementation, runner.ACTIVATION: runner.sha((root / runner.ACTIVATION).read_bytes()),
            runner.PLAN: runner.PLAN_SHA, plan["payloads"]["map"]: plan["payloads"]["map_sha256"],
            checkpoint: runner.sha((root / checkpoint).read_bytes())}}
    runner.write_json(root / "synthetic-receipt.json", receipt)
    output = root / activation["target"]["output"]
    status = runner.run(root, frozen, output, root / "synthetic-receipt.json")
    assert status["completed_covers"] == 30 and status["total_cells"] == 15300
    result = audit.audit(output, frozen, root / runner.PLAN)
    assert result["completed_covers"] == 30 and result["independent_capacity_checks"] == 720
    assert result["encoder_calls_recorded"] == 720 and result["unexpected_errors_preserved"] == 0
    assert result["independent_decodes"] == result["receiver_jobs"] > 0
    assert all(r["result"]["cover_text"].split()[0] in ("dexterous", "zzzzzz")
               for r in [runner.read_json(p) for p in snapshot.rglob("*.json")])
