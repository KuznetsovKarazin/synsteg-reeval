#!/usr/bin/env python3
"""Independent, post-opening integrity/transport audit; no model inference.

The auditor does not import the experiment runner or trust its aggregate
success counts. It checks its completion gate before reading raw records,
replays public codec selection, and invokes a separate receiver process.
Neural scores are checked for identity/arithmetic consistency, not recomputed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from synsteg_audit import shared_dictionary as codec
from synsteg_audit.fixed_codebook import FrozenCodebook, encode_bits, decode_bits
import shared_dictionary_audit_core as independent
from shared_dictionary_prediction_audit import audit_predictions

SCHEMA = "synsteg-shared-dictionary-pilot-v1"
FRAMINGS = ("plain_length16", "aes_siv")
SELECTORS = ("canonical", "quality")
RECEIVER_KEYS = ["key_hex", "public_dictionary", "public_protocol", "stego_text"]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(path):
    values = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    require(all(isinstance(row.get("time_unix"), (int, float)) and math.isfinite(row["time_unix"]) for row in values),
            "Event timestamps must be finite")
    require(all(a["time_unix"] <= b["time_unix"] for a, b in zip(values, values[1:])), "Event timestamps moved backwards")
    return values


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "Missing or escaping evidence path: " + str(relative))
    return path


def protocol_for(config, framing, minimum=None):
    method = config["shared_method"]
    return codec.Protocol(min_group_size=method["min_group_size"] if minimum is None else minimum,
                          max_bits_per_slot=method["max_bits_per_slot"], framing=framing)


def expected_arms(config):
    result = []
    for bits in config["payload_bits"]:
        for framing in FRAMINGS:
            for selector in SELECTORS:
                result.append({"id": f"{framing}__{selector}__{bits}", "framing": framing,
                               "selector": selector, "payload_bits": bits, "method": "shared_dictionary"})
    if config["include_fixed_quality_baseline"]:
        result += [{"id": f"fixed_quality__{bits}", "framing": "external_length", "selector": "quality",
                    "payload_bits": bits, "method": "fixed_codebook"} for bits in config["payload_bits"]]
    return result


def fixture_key(config):
    return hashlib.sha256(("PUBLIC-NONPRODUCTION-SHARED-KEY:" + config["public_demo_key_seed"]).encode()).digest()


def fixture_payload(config, sample_id, bits):
    return hashlib.sha256(canonical([config["payload_seed"], sample_id])).digest()[:bits // 8]


def _one(rows, name):
    matches = [row for row in rows if row["event"] == name]
    require(len(matches) == 1, "Exactly one event required: " + name)
    return matches[0]


def load_completed(run):
    """Gate first; inspect raw records only after an intact completed opening."""
    run = Path(run).resolve()
    for name in ("evaluation.json", "evaluation_opened.json", "events.jsonl"):
        require((run / name).is_file(), "Completed evaluation required before raw record access")
    ledger = events(run / "events.jsonl")
    opening = _one(ledger, "shared_evaluation_opened_once")
    completion = _one(ledger, "shared_evaluation_completed")
    freezing = _one(ledger, "shared_design_dictionary_and_attacks_frozen_before_generation")
    require(ledger.index(freezing) < ledger.index(opening) < ledger.index(completion), "Root event order changed")
    require(completion["evaluation_sha256"] == file_hash(run / "evaluation.json"), "Evaluation checksum changed")
    evaluation = read_json(run / "evaluation.json")
    require(evaluation.get("status") == "complete_shared_dictionary_feasibility", "Evaluation is incomplete")
    frozen, config, marker = (read_json(run / name) for name in ("freeze.json", "protocol.json", "evaluation_opened.json"))
    require(config.get("schema") == SCHEMA and frozen.get("schema") == SCHEMA, "Unexpected experiment schema")
    require(digest({k: v for k, v in frozen.items() if k != "freeze_id"}) == frozen["freeze_id"], "Freeze identity changed")
    require(freezing["freeze_id"] == frozen["freeze_id"], "Freeze event identity changed")
    require(file_hash(run / "protocol.json") == frozen["protocol_sha256"] and
            file_hash(run / "samples.json") == frozen["samples_sha256"], "Frozen design checksum changed")
    require(read_json(run / "samples.json") == config["samples"], "Sample design changed")
    require(config["payload_bits"] == [8, 16] and config["framings"] == list(FRAMINGS)
            and config["selectors"] == list(SELECTORS), "Unexpected arm contract")
    require(config["shared_method"]["min_group_size"] == config["shared_method"]["max_bits_per_slot"] == 1,
            "This audit covers the registered one-bit transport protocol")
    require(frozen.get("fixture_key_is_public_not_secret") is True, "Public fixture-key caveat missing")
    for relative, sha in frozen["code_sha256"].items():
        require(file_hash(inside(ROOT, relative)) == sha, "Frozen source checksum changed: " + relative)
    for name, sha in frozen["resource_sha256"].items():
        require(file_hash(name) == sha, "Frozen external resource checksum changed")
    resources = {str(Path(item["path"]).resolve()): item["sha256"]
                 for item in config.get("external_resource_files", []) + config.get("additional_identity_files", [])}
    require(resources == frozen["resource_sha256"], "Resource declarations changed")
    jobs = {"gpt2": "configs/neural/gpt2-full.json", "qwen_base": "configs/neural/crossmodel-qwen-base-ppl.json"}
    require(set(frozen["scorer_job_files_sha256"]) == set(jobs), "Scorer job identities missing")
    for name, relative in jobs.items():
        require(file_hash(ROOT / relative) == frozen["scorer_job_files_sha256"][name], "Frozen scorer job changed")
    require(file_hash(run / "public_dictionary.json") == frozen["dictionary_file_sha256"], "Public dictionary file changed")
    dictionary = codec.PublicDictionary.from_dict(read_json(run / "public_dictionary.json"))
    require(dictionary.sha256 == frozen["dictionary_sha256"], "Public vocabulary identity changed")
    for relative, sha in frozen["source_attack_files_sha256"].items():
        require(file_hash(inside(run, relative)) == sha, "Transferred attack checksum changed")
    lock = read_json(run / "source_attacks/attack_lock.json")
    require(digest({k: v for k, v in lock.items() if k != "lock_id"}) == lock["lock_id"] == frozen["source_lock_id"],
            "Source attack lock identity changed")
    require(file_hash(run / "source_attacks/attack_lock.json") == config["source_attack_lock_sha256"], "Declared source lock changed")
    attack_names = {f"fixed_{selector}__{feature}" for selector in SELECTORS for feature in ("char", "word")}
    require(set(lock["attacks"]) == attack_names, "Four registered transferred attacks required")
    require(set(frozen["source_attack_files_sha256"]) == {"source_attacks/attack_lock.json"} |
            {"source_attacks/" + name + ".joblib" for name in attack_names}, "Transferred attack manifest incomplete")
    for name, attack in lock["attacks"].items():
        require(attack["score_direction"] in (-1, 1) and attack["decision_threshold"] == .5 and
                attack["artifact_sha256"] == file_hash(run / "source_attacks" / (name + ".joblib")), "Transferred attack identity changed")
    if "unicode_database_version" in frozen.get("environment", {}):
        require(frozen["environment"]["unicode_database_version"] == unicodedata.unidata_version,
                "Audit Unicode classification runtime differs from frozen protocol")
    for name in ("freeze_id", "source_lock_id", "all_model_records_sha256", "all_model_bundle_sha256"):
        require(evaluation[name] == marker[name] == opening[name], "Opening identity changed: " + name)
    require(evaluation["freeze_id"] == frozen["freeze_id"] and evaluation["source_lock_id"] == frozen["source_lock_id"],
            "Evaluation does not use frozen design/attacks")
    require(not (run / "evaluation_control/.writer.lock").exists() and not (run / ".writer.lock").exists(), "A writer is still active")
    aliases = [model["alias"] for model in config["models"]]
    require(len(set(aliases)) == len(aliases) and len({s["id"] for s in config["samples"]}) == len(config["samples"]),
            "Duplicate declared identity")
    require(set(evaluation["all_model_records_sha256"]) == set(aliases) == set(evaluation["all_model_bundle_sha256"]),
            "Evaluation model denominator differs from design")
    resumes = [row for row in ledger if row["event"] == "shared_evaluation_resumed"]
    for resume in resumes:
        require(ledger.index(opening) < ledger.index(resume) < ledger.index(completion), "Resume outside opening/completion")
        for field in ("freeze_id", "source_lock_id", "all_model_records_sha256", "all_model_bundle_sha256"):
            require(resume[field] == evaluation[field], "Resume identity changed")
    if "evaluation_resume_count" in evaluation:
        require(evaluation["evaluation_resume_count"] == len(resumes), "Evaluation resume count changed")
    # Verify EVERY model's terminal metadata before the first raw record read.
    generation_reports = {}
    for alias in aliases:
        directory = run / "models" / alias
        require(not (directory / ".writer.lock").exists(), "Model writer is active")
        generation = read_json(directory / "generation.json")
        require(generation["complete"] is True and generation["records"] == generation["expected"] == len(config["samples"]),
                "Incomplete generation denominator")
        require(generation["freeze_id"] == frozen["freeze_id"] and generation["model_alias"] == alias, "Generation identity changed")
        expected_names = {sample["id"] + ".json" for sample in config["samples"]}
        require({path.name for path in (directory / "records").glob("*.json")} == expected_names, "Missing or extra record file")
        generation_reports[alias] = generation
    # Everything above is a metadata/hash gate. Only now read raw outcomes.
    records, bundles = {}, {}
    for alias in aliases:
        directory = run / "models" / alias
        generation = generation_reports[alias]
        rows = [read_json(directory / "records" / (sample["id"] + ".json")) for sample in config["samples"]]
        require(digest(rows) == generation["records_sha256"] == evaluation["all_model_records_sha256"][alias], "Record checksum changed")
        bundle = read_json(directory / "bundle.json")
        require(digest(bundle) == generation["bundle_sha256"] and bundle["source_records_sha256"] == digest(rows), "Bundle source identity changed")
        require(file_hash(directory / "bundle.json") == evaluation["all_model_bundle_sha256"][alias], "Opened bundle checksum changed")
        model_events = events(directory / "events.jsonl")
        _check_model_events(rows, bundle, model_events)
        require(model_events[-1]["time_unix"] <= opening["time_unix"], "Generation continued after opening")
        require(model_events[0]["time_unix"] >= freezing["time_unix"], "Generation preceded freeze event")
        for row in rows:
            attempt = read_json(directory / "attempt_started" / (row["sample"]["id"] + ".json"))
            require(attempt["freeze_id"] == frozen["freeze_id"] and attempt["time_unix"] == row["started_at"], "Attempt marker identity changed")
            require(row["started_at"] <= row["finished_at"] <= opening["time_unix"], "Attempt timing violates closed panel")
        records[alias], bundles[alias] = rows, bundle
    return config, frozen, dictionary, evaluation, records, bundles


def _check_model_events(rows, bundle, ledger):
    require(all(event.get("sample_id") in {row["sample"]["id"] for row in rows} for event in ledger if "sample_id" in event),
            "Event ledger contains an undeclared sample")
    for row in rows:
        sid = row["sample"]["id"]
        matching = [event for event in ledger if event.get("sample_id") == sid]
        interrupted = row.get("failure") == "interrupted_attempt_not_retried"
        terminal = _one(matching, "attempt_interrupted_not_retried" if interrupted else "attempt_completed")
        require(sum(event["event"] in ("attempt_completed", "attempt_interrupted_not_retried") for event in matching) == 1,
                "More than one terminal event for an attempt")
        if not interrupted:
            require(terminal["status"] == row["status"] and terminal["failure"] == row.get("failure"), "Attempt terminal status changed")
        starts = [event for event in matching if event["event"] == "attempt_started"]
        require(len(starts) <= 1 if interrupted else len(starts) == 1, "Attempt retried or missing start event")
        if starts:
            require(ledger.index(starts[0]) < ledger.index(terminal), "Attempt terminal preceded its start")
        preparations = [event for event in matching if event["event"] == "methods_prepared_before_payload"]
        require(len(preparations) == int("prepared_shared" in row), "Preparation event count disagrees with saved state")
        if preparations:
            require(starts and ledger.index(starts[0]) < ledger.index(preparations[0]) < ledger.index(terminal), "Method preparation event order changed")
            require(preparations[0]["shared_sha256"] == row["prepared_shared_sha256"] and
                    preparations[0].get("baseline_sha256") == row.get("prepared_baseline_sha256"), "Preparation identity changed")
    bundled = _one(ledger, "fixed_all_cover_bundle_completed")
    completed = _one(ledger, "generation_completed")
    require(bundled["bundle_sha256"] == digest(bundle) and completed["records_sha256"] == digest(rows), "Terminal generation event hash changed")
    require(ledger.index(bundled) < ledger.index(completed), "Generation completed before bundle")
    require(all(ledger.index(event) < ledger.index(bundled) for event in ledger
                if event["event"] in ("attempt_completed", "attempt_interrupted_not_retried")), "Bundle preceded an attempt terminal")


def _score_valid(score):
    detail = score["details"]
    nll, total, ppl = detail["mean_nll"], detail["sum_nll"], score["ppl"]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in (nll, total, ppl)), "Nonfinite neural score evidence")
    n = detail["input_tokens"]
    require(type(n) is int and n >= 2 and detail["scored_input_tokens"] == n and detail["predicted_tokens"] == n - 1,
            "Incomplete full-text token accounting")
    require(nll >= 0 and ppl >= 1 and detail["truncated"] is False and detail["ppl_policy"] == "sliding_window", "Wrong full-text scoring policy")
    require(math.isclose(total / (n - 1), nll, rel_tol=1e-10, abs_tol=1e-12)
            and math.isclose(math.log(ppl), nll, rel_tol=1e-10, abs_tol=1e-12), "Inconsistent NLL/PPL arithmetic")
    expected_forwards = 1 + max(0, math.ceil((n - 1024) / 512))
    require(score["forward_passes"] == expected_forwards, "Scoring window count changed")


def _check_preparation(row, dictionary, protocol):
    prepared, cover = row["prepared_shared"], row["cover_text"]
    require(digest(prepared) == row["prepared_shared_sha256"], "Sender preparation checksum changed")
    analysis = codec.analyze_text(cover, dictionary, protocol)
    require(canonical(prepared["analysis"]) == canonical(asdict(analysis)), "Saved eligibility differs from public invariant rules")
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    evidence = {(item["slot"], item["candidate"]): item for item in prepared["candidate_score_evidence"]}
    require(len(evidence) == len(prepared["candidate_score_evidence"]), "Duplicate candidate score evidence")
    expected, unique = set(), {}
    for slot in analysis.slots:
        words = classes[slot.class_id]
        for word in words:
            candidate = word.upper() if slot.case_style == "upper" else word.capitalize() if slot.case_style == "title" else word
            identity = (slot.index, candidate)
            expected.add(identity)
            require(identity in evidence, "Missing pre-payload candidate cost")
            value = evidence[identity]
            cost = prepared["candidate_costs"].get(str(slot.index) + ":" + candidate)
            require(type(cost) in (float, int) and math.isfinite(cost) and value["cost"] == cost, "Candidate costs differ from score evidence")
            if len(words) == 2:
                require(cost == 0.0 and value["kind"] == "singleton_groups_no_quality_choice_not_a_model_score",
                        "Singleton bookkeeping falsely presented as a model score")
            else:
                require(value["kind"] == "original_cover_single_replacement_mean_nll", "Wrong candidate-scoring policy")
                text = cover[:slot.start] + candidate + cover[slot.end:]
                sha = text_hash(text)
                require(value["text_sha256"] == sha, "Candidate was not scored on unchanged cover context")
                _score_valid(value["score"])
                require(cost == value["score"]["details"]["mean_nll"], "Selector cost differs from saved NLL")
                if sha in unique:
                    require(unique[sha] == value["score"], "Duplicate candidate text has inconsistent cached scores")
                unique[sha] = value["score"]
    require(set(evidence) == expected and set(prepared["candidate_costs"]) == {str(i) + ":" + word for i, word in expected},
            "Candidate set changed or contains extras")
    require(prepared["unique_full_text_scoring_calls"] == len(unique) and
            prepared["model_forward_passes"] == sum(score["forward_passes"] for score in unique.values()), "Candidate inference accounting changed")
    return analysis


def receiver_main():
    """Receiver branch imports public code only; request has no audit/original state."""
    import socket
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    warm = AESSIV(bytes(64))
    warm.decrypt(warm.encrypt(b"", [b"audit-public-warmup"]), [b"audit-public-warmup"])
    def guard(event, args):
        if event in {"open", "os.listdir", "os.scandir", "os.system", "os.posix_spawn"} or event.startswith(("socket.", "subprocess.")):
            raise PermissionError("Receiver external I/O forbidden")
    sys.addaudithook(guard)
    for operation in (lambda: open("/audit-forbidden"), socket.socket, lambda: subprocess.Popen(["audit-forbidden"])):
        try:
            operation()
        except PermissionError:
            pass
        else:
            raise AuditError("Receiver I/O guard self-check failed")
    request = json.loads(sys.stdin.buffer.read())
    require(sorted(request) == RECEIVER_KEYS, "Receiver received extra state")
    result = independent.decode(request["stego_text"], bytes.fromhex(request["key_hex"]),
                                request["public_dictionary"], request["public_protocol"])
    print(json.dumps({"payload_hex": result["payload_hex"], "input_keys": RECEIVER_KEYS, "io_guard_self_checks": True}))


def fresh_decode(text, key, dictionary, protocol):
    request = {"key_hex": key.hex(), "stego_text": text,
               "public_dictionary": dictionary.to_dict(), "public_protocol": protocol.to_dict()}
    with tempfile.TemporaryDirectory(prefix="synsteg-independent-receiver-") as directory:
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "receiver"], input=canonical(request),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory,
                                env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"}, timeout=30)
    require(result.returncode == 0, "Independent fresh receiver rejected: " + result.stderr.decode(errors="replace")[-500:])
    response = json.loads(result.stdout)
    require(response["input_keys"] == RECEIVER_KEYS and response["io_guard_self_checks"] is True, "Receiver isolation evidence missing")
    return bytes.fromhex(response["payload_hex"])


def _check_success(output, encoded, payload, key, dictionary, protocol, fresh_receiver, dictionary_bytes=None,
                   *, original_text, selector="canonical", candidate_costs=None):
    require(output["stego_text"] == encoded.stego_text, "Exact canonical/quality group selection differs from pre-payload costs")
    require(output["decoded_payload_hex"] == payload.hex(), "Full payload evidence differs")
    require(output["encoding"] == {k: v for k, v in asdict(encoded).items() if k != "stego_text"}, "Frame/capacity accounting differs")
    require(output.get("external_document_metadata_bits") == output.get("individual_codebook_bytes") == 0 and
            output.get("shared_secret_key_bytes") == 32, "No-codebook wire accounting differs")
    if "protocol" in output:
        require(output["protocol"] == protocol.to_dict(), "Output public protocol differs")
    if "public_dictionary_sha256" in output:
        require(output["public_dictionary_sha256"] == dictionary.sha256, "Output public dictionary differs")
    if dictionary_bytes is not None and "public_dictionary_file_bytes" in output:
        require(output["public_dictionary_file_bytes"] == dictionary_bytes, "Shared dictionary byte cost differs")
    if "public_dictionary_canonical_wire_bytes" in output:
        require(output["public_dictionary_canonical_wire_bytes"] == len(canonical(dictionary.to_dict())), "Public dictionary wire byte cost differs")
    proof = output["receiver"]
    require(proof["input_keys"] == RECEIVER_KEYS and proof["status"] == "decoded" and
            proof["actual_returncode"] == 0 and proof["payload_hex"] == payload.hex(), "Original receiver proof changed")
    require(proof["io_guard_self_checks"] == {"file": True, "socket": True, "subprocess": True} and
            proof["public_dictionary_sha256"] == dictionary.sha256 and proof["public_protocol_sha256"] == protocol.sha256,
            "Receiver received different public resources or missing I/O guard")
    independent_result = independent.decode(output["stego_text"], key, dictionary.to_dict(), protocol.to_dict())
    independent_original = independent.analyze(original_text, dictionary.to_dict(), protocol.to_dict())
    require(independent_result["payload_hex"] == payload.hex() and
            independent_result["analysis"]["normalization_sha256"] == independent_original["normalization_sha256"] == encoded.normalization_sha256,
            "Independent decoder/invariant reconstruction disagrees")
    require(independent_result["analysis"]["capacity_bits"] == independent_original["capacity_bits"] == encoded.capacity_bits,
            "Independent capacity reconstruction disagrees")
    for field in ("frame_bits", "padding_bits", "consumed_slots"):
        require(independent_result[field] == getattr(encoded, field), "Independent frame accounting disagrees: " + field)
    for group in independent_result["used_groups"]:
        slot = independent_result["analysis"]["slots"][group["slot"]]
        candidates = [word.upper() if slot["case_style"] == "upper" else word.capitalize()
                      if slot["case_style"] == "title" else word for word in group["candidates"]]
        chosen = min(candidates) if selector == "canonical" else min(candidates,
            key=lambda word: (candidate_costs[str(slot["index"]) + ":" + word], word))
        require(slot["surface"] == chosen, "Independent required-group minimum selection differs")
    actual = fresh_decode(output["stego_text"], key, dictionary, protocol) if fresh_receiver else bytes.fromhex(independent_result["payload_hex"])
    require(actual == payload, "Independent decoder did not recover complete payload")
    require(codec.analyze_text(output["stego_text"], dictionary, protocol).normalization_sha256 == encoded.normalization_sha256,
            "Invariant normalization changed")


def _check_baseline(row, output, payload):
    state = row["prepared_baseline"]
    require(digest(state) == row["prepared_baseline_sha256"], "Baseline state changed after preparation")
    book = FrozenCodebook.from_dict(state["codebook"])
    require(book.cover == row["cover_text"] and book.sha256 == state["codebook_sha256"], "Baseline codebook identity changed")
    bits = "".join(f"{byte:08b}" for byte in payload)
    encoded = encode_bits(book, bits, selection="quality", framing="external_length")
    if output["full_success"]:
        require(output["stego_text"] == encoded.text and canonical(output["key"]) == canonical(book.to_dict()), "Baseline selection or document codebook changed")
        require(decode_bits(book, output["stego_text"], framing="external_length", expected_bits=len(bits)) == bits,
                "Baseline complete decode failed")
        require(output["decoded_payload_hex"] == payload.hex() and output["external_document_metadata_bits"] == 32,
                "Baseline full delivery/external length accounting differs")
        expected_metadata = asdict(encoded)
        expected_metadata["text"] = None
        require(all(output["metadata"].get(name) == value for name, value in expected_metadata.items()) and
                output["metadata"].get("external_length_bits") == 32, "Baseline frame/padding metadata differs")
        metrics = output["delivery_metrics"]
        require(metrics["requested_bits"] == metrics["used_bits"] == metrics["decoded_bits"] == len(bits)
                and metrics["full_message_success"] is True and metrics["bit_error_rate"] == 0,
                "Baseline full-delivery metrics changed")
        require(metrics["key_json_bytes"] == len(json.dumps(output["key"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()),
                "Baseline serialized individual codebook byte cost changed")
        serialized_book = json.dumps(book.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        require(metrics["key_zlib9_bytes"] == len(zlib.compress(serialized_book, level=9)), "Baseline compressed codebook byte cost changed")
    return encoded


def independent_checks(config, frozen, dictionary, records, bundles, model_events=None, fresh_receiver=True):
    """Audit supplied complete records. CLI always gates and uses fresh receivers."""
    aliases = [model["alias"] for model in config["models"]]
    require(set(records) == set(bundles) == set(aliases), "Missing or extra model denominator")
    require(frozen["dictionary_sha256"] == dictionary.sha256, "Dictionary identity differs from frozen state")
    arms, key = expected_arms(config), fixture_key(config)
    summary = {"planned_covers": len(aliases) * len(config["samples"]), "shared_attempts": 0, "baseline_attempts": 0,
               "bundle_attempts": 0, "shared_successes": 0, "baseline_successes": 0, "bundle_successes": 0,
               "public_key_is_not_secret": True, "models": {}, "unexpected_runtime_failures": [],
               "fresh_receivers_used": fresh_receiver, "inference_recomputed": False}
    for alias in aliases:
        rows, bundle = records[alias], bundles[alias]
        require(len(rows) == len(config["samples"]), "Missing planned sample denominator")
        if model_events is not None:
            _check_model_events(rows, bundle, model_events[alias])
        stats = {arm["id"]: {"attempted_samples": 0, "full_success_samples": 0, "requested_payload_bits": 0,
                 "delivered_payload_bits": 0, "capacity_failures": 0, "failures": []} for arm in arms}
        resource_costs = {arm["id"]: {"payload_bits_per_attempt": arm["payload_bits"],
            "embedded_overhead_bits_per_frame": 0 if arm["method"] == "fixed_codebook" else (144 if arm["framing"] == "aes_siv" else 16),
            "external_length_bits_per_attempt": 32 if arm["method"] == "fixed_codebook" else 0,
            "successful_frame_bits": 0, "successful_padding_bits": 0,
            "individual_codebook_json_bytes_across_successes": 0} for arm in arms}
        word_total, generated, failure_count = 0, 0, 0
        capacities = []
        for row, sample in zip(rows, config["samples"]):
            sid = sample["id"]
            require(row["sample"] == sample and row["model_alias"] == alias and row["freeze_id"] == frozen["freeze_id"]
                    and row["attempt"] == 1, "Record differs from frozen sample identity")
            require(row["status"] in ("completed", "failed") and set(row["variants"]) == {arm["id"] for arm in arms},
                    "Missing or extra arm denominator")
            cover = row.get("cover_text")
            if isinstance(cover, str) and cover.strip():
                generated += 1
                word_total += len(cover.split())
                analysis = codec.analyze_text(cover, dictionary, protocol_for(config, "plain_length16"))
                reconstructed = independent.analyze(cover, dictionary.to_dict(), protocol_for(config, "plain_length16").to_dict())
                require(reconstructed["capacity_bits"] == analysis.capacity_bits and reconstructed["normalization_sha256"] == analysis.normalization_sha256,
                        "Independent original-text eligibility reconstruction disagrees")
                capacities.append({"sample_id": sid, "capacity_bits": analysis.capacity_bits,
                    "min_group_size_2_capacity_bits": codec.analyze_text(cover, dictionary, protocol_for(config, "plain_length16", 2)).capacity_bits})
            else:
                analysis = None
            failure_count += int(row.get("failure") is not None)
            if "prepared_shared" in row:
                require(bool(cover), "Preparation exists without generated cover")
                _check_preparation(row, dictionary, protocol_for(config, "plain_length16"))
                expected_ablation = {str(minimum): {"capacity_bits": codec.analyze_text(cover, dictionary,
                    protocol_for(config, "plain_length16", minimum)).capacity_bits, "min_group_size": minimum,
                    "max_bits_per_slot": config["shared_method"]["max_bits_per_slot"]} for minimum in (1, 2)}
                require(row["capacity_ablations"] == expected_ablation, "Capacity ablation changed")
                require(row["public_fixture_key_hex"] == key.hex(), "Fixture key differs from declared derivation")
                expected_payloads = {str(bits): fixture_payload(config, sid, bits).hex() for bits in config["payload_bits"]}
                require(row["payloads_hex"] == expected_payloads, "Paired payload derivation changed")
            for arm in arms:
                output = row["variants"][arm["id"]]
                require(all(output[name] == value for name, value in arm.items()), "Arm identity changed")
                require(type(output["full_success"]) is bool, "Nonboolean success flag")
                successful = output["full_success"]
                require(not successful or (row["status"] == "completed" and row.get("failure") is None), "Success declared on failed sample generation/preparation")
                require((successful and output.get("error") is None and output.get("stego_text"))
                        or (not successful and output.get("stego_text") is None and output.get("error")), "Success/failure text ledger inconsistent")
                family = "shared" if arm["method"] == "shared_dictionary" else "baseline"
                summary[family + "_attempts"] += 1
                summary[family + "_successes"] += int(successful)
                if successful:
                    resource_costs[arm["id"]]["successful_frame_bits"] += arm["payload_bits"] + resource_costs[arm["id"]]["embedded_overhead_bits_per_frame"]
                    resource_costs[arm["id"]]["successful_padding_bits"] += output["encoding"]["padding_bits"] if family == "shared" else output["metadata"]["padding_bits"]
                    if family == "baseline":
                        resource_costs[arm["id"]]["individual_codebook_json_bytes_across_successes"] += output["delivery_metrics"]["key_json_bytes"]
                stat = stats[arm["id"]]
                stat["attempted_samples"] += 1
                stat["full_success_samples"] += int(successful)
                stat["requested_payload_bits"] += arm["payload_bits"]
                stat["delivered_payload_bits"] += arm["payload_bits"] * int(successful)
                if not successful:
                    stat["failures"].append({"sample_id": sid, "error": row.get("failure") or output["error"]})
                capacity_error = "CapacityError:" in (output.get("error") or "")
                stat["capacity_failures"] += int(capacity_error)
                payload = fixture_payload(config, sid, arm["payload_bits"])
                if "prepared_shared" not in row:
                    require(not successful and not capacity_error, "Outcome claims encoding without completed preparation")
                    continue
                if family == "shared":
                    protocol = protocol_for(config, arm["framing"])
                    needed = protocol.frame_overhead_bits + arm["payload_bits"]
                    if analysis.capacity_bits < needed:
                        require(not successful and capacity_error, "Insufficient complete-frame capacity was not retained as failure")
                        continue
                    require(not capacity_error, "False capacity failure despite complete frame fitting")
                    costs = row["prepared_shared"]["candidate_costs"]
                    candidate_cost = (lambda slot, word: costs[str(slot.index) + ":" + word]) if arm["selector"] == "quality" else None
                    encoded = codec.encode(cover, payload, key, dictionary, protocol, selector=arm["selector"], candidate_cost=candidate_cost)
                    if successful:
                        _check_success(output, encoded, payload, key, dictionary, protocol, fresh_receiver,
                                       frozen.get("dictionary_file_bytes"), original_text=cover,
                                       selector=arm["selector"], candidate_costs=costs)
                    else:
                        summary["unexpected_runtime_failures"].append({"model_alias": alias, "sample_id": sid, "arm": arm["id"], "error": output["error"]})
                elif "prepared_baseline" in row:
                    try:
                        _check_baseline(row, output, payload)
                    except Exception as error:
                        from synsteg_audit.fixed_codebook import CapacityError
                        if isinstance(error, CapacityError):
                            require(not successful and capacity_error, "Baseline insufficient capacity counted as success")
                        else:
                            raise
                    else:
                        require(not capacity_error, "False baseline capacity failure")
                else:
                    require(not successful and not capacity_error, "Baseline claimed delivery without its required document codebook")
        for stat in stats.values():
            stat["full_success_rate"] = stat["full_success_samples"] / stat["attempted_samples"]
            stat["goodput_bits_per_all_generated_cover_words"] = stat["delivered_payload_bits"] / word_total if word_total else None
        require(bundle["model_alias"] == alias and bundle["freeze_id"] == frozen["freeze_id"] and
                bundle["sample_ids"] == [sample["id"] for sample in config["samples"]] and
                bundle["source_records_sha256"] == digest(rows) and bundle["policy"] == config["bundle_policy"] and
                bundle["independent_container_count"] == 1, "Bundle identity/order differs from complete panel")
        ready = all(isinstance(row.get("cover_text"), str) and row["cover_text"].strip()
                    and "quarantined_duplicate_of:" not in (row.get("failure") or "") for row in rows)
        expected_cover = "\n\n".join(row["cover_text"] for row in rows) if ready else None
        require(bundle["cover_text"] == expected_cover, "Bundle dropped, selected, or rearranged declared covers")
        bundle_words = len(expected_cover.split()) if expected_cover else 0
        require(bundle["cover_words"] == bundle_words and bundle["public_fixture_key_hex"] == key.hex() and
                bundle["public_dictionary_sha256"] == dictionary.sha256, "Bundle resource accounting changed")
        if "public_dictionary_canonical_wire_bytes" in bundle:
            require(bundle["public_dictionary_canonical_wire_bytes"] == len(canonical(dictionary.to_dict())), "Bundle shared dictionary wire cost differs")
        if frozen.get("dictionary_file_bytes") is not None and "public_dictionary_file_bytes" in bundle:
            require(bundle["public_dictionary_file_bytes"] == frozen["dictionary_file_bytes"], "Bundle shared dictionary file cost differs")
        protocol = protocol_for(config, "aes_siv")
        require(bundle["protocol"] == protocol.to_dict() and set(bundle["variants"]) == {str(bits) for bits in config["payload_bits"]}, "Bundle arm denominator changed")
        bundle_stats = {}
        for bits in config["payload_bits"]:
            output = bundle["variants"][str(bits)]
            payload = fixture_payload(config, "all_covers_bundle", bits)
            require(output["payload_bits"] == bits and output["payload_hex"] == payload.hex(), "Bundle payload derivation changed")
            require(type(output["full_success"]) is bool, "Nonboolean bundle success flag")
            successful = output["full_success"]
            require((successful and output.get("error") is None and output.get("stego_text"))
                    or (not successful and output.get("stego_text") is None and output.get("error")), "Bundle success/failure text ledger inconsistent")
            summary["bundle_attempts"] += 1
            summary["bundle_successes"] += int(successful)
            require(output["goodput_bits_per_bundle_word"] == (bits * int(successful) / bundle_words if bundle_words else None), "Bundle goodput denominator changed")
            if ready:
                capacity = codec.analyze_text(expected_cover, dictionary, protocol).capacity_bits
                require(bundle["capacity_bits"] == capacity, "Bundle public capacity changed")
                if capacity < protocol.frame_overhead_bits + bits:
                    require(not successful and "CapacityError:" in (output.get("error") or ""), "Bundle insufficient capacity omitted")
                else:
                    require("CapacityError:" not in (output.get("error") or ""), "False bundle capacity failure")
                    encoded = codec.encode(expected_cover, payload, key, dictionary, protocol)
                    if successful:
                        require(output.get("error") is None, "Successful bundle retains error")
                        _check_success(output, encoded, payload, key, dictionary, protocol, fresh_receiver, original_text=expected_cover)
                    else:
                        summary["unexpected_runtime_failures"].append({"model_alias": alias, "sample_id": "all_covers_bundle", "arm": str(bits), "error": output.get("error")})
            else:
                require(not successful and output["stego_text"] is None and output["error"] == "all_declared_nonquarantined_covers_required",
                        "Missing-generation bundle failure was dropped or altered")
            bundle_stats[str(bits)] = {"full_success": successful, "requested_payload_bits": bits,
                                      "delivered_payload_bits": bits * int(successful), "goodput_bits_per_bundle_word": output["goodput_bits_per_bundle_word"]}
        summary["models"][alias] = {"planned_covers": len(rows), "generated_covers": generated,
            "terminal_generation_or_preparation_failures": failure_count, "all_generated_cover_words": word_total,
            "capacity_by_sample": capacities, "reliability": stats, "resource_accounting_by_arm": resource_costs,
            "bundle": bundle_stats}
    require(summary["shared_attempts"] == summary["planned_covers"] * len(config["payload_bits"]) * len(FRAMINGS) * len(SELECTORS), "Shared attempt denominator incomplete")
    require(summary["baseline_attempts"] == summary["planned_covers"] * len(config["payload_bits"]) * int(config["include_fixed_quality_baseline"]), "Baseline denominator incomplete")
    require(summary["bundle_attempts"] == len(aliases) * len(config["payload_bits"]), "Bundle denominator incomplete")
    return summary


def audit(run, *, fresh_receiver=True):
    run = Path(run).resolve()
    config, frozen, dictionary, evaluation, records, bundles = load_completed(run)
    evidence = dict(frozen, dictionary_file_bytes=(run / "public_dictionary.json").stat().st_size)
    result = independent_checks(config, evidence, dictionary, records, bundles,
                               model_events={alias: events(run / "models" / alias / "events.jsonl") for alias in records},
                               fresh_receiver=fresh_receiver)
    for alias, model in result["models"].items():
        saved = evaluation["models"][alias]
        require(saved["planned_covers"] == model["planned_covers"] and saved["generated_covers"] == model["generated_covers"], "Evaluation cover denominator changed")
        require(saved["reliability"] == model["reliability"], "Evaluation reliability or goodput differs from independent recomputation")
    result["prediction_audit"] = audit_predictions(run, config, evaluation, records, read_json(run / "source_attacks/attack_lock.json"))
    result.update(schema="synsteg-shared-dictionary-independent-audit-v1", status="passed",
                  freeze_id=frozen["freeze_id"], evaluation_sha256=file_hash(run / "evaluation.json"),
                  audit_script_sha256=file_hash(__file__), codec_sha256=file_hash(ROOT / "src/synsteg_audit/shared_dictionary.py"),
                  independent_decoder_sha256=file_hash(ROOT / "scripts/shared_dictionary_audit_core.py"),
                  prediction_auditor_sha256=file_hash(ROOT / "scripts/shared_dictionary_prediction_audit.py"),
                  dictionary_file_sha256=file_hash(run / "public_dictionary.json"), dictionary_sha256=dictionary.sha256,
                  dictionary_file_bytes=(run / "public_dictionary.json").stat().st_size,
                  public_dictionary_canonical_wire_bytes=len(canonical(dictionary.to_dict())),
                  audit_environment={"python": sys.version, "cryptography": importlib.metadata.version("cryptography"),
                                     "unicode_database_version": unicodedata.unidata_version},
                  all_model_records_sha256=evaluation["all_model_records_sha256"], all_model_bundle_sha256=evaluation["all_model_bundle_sha256"],
                  limitations=["Public fixture keys/payloads offer no confidentiality; this is transport feasibility.",
                    "Plain framing is unauthenticated; AES-SIV authenticates message/invariant, not every synonym spelling.",
                    "The public dictionary/protocol and 32-byte secret are shared resources; only individual document metadata is zero.",
                    "Scores were checked for identity and arithmetic, not recomputed with neural models.",
                    "Transferred detector metrics were recomputed from saved predictions; classifier scores were not regenerated.",
                    "Python receiver audit guards are inspected-program checks, not OS sandboxes.",
                    "Payload sizes and selectors share covers; bundle containers reuse every cover and are dependent observations."])
    return result


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "receiver":
        receiver_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite prior independent audit")
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "shared_attempts": result["shared_attempts"],
                      "baseline_attempts": result["baseline_attempts"], "bundle_attempts": result["bundle_attempts"],
                      "output_sha256": file_hash(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
