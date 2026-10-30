#!/usr/bin/env python3
"""Recompute saved v1 gate results without calling universal codec/evaluator.

The frozen base tokenizer is shared. Eligibility is derived directly from the
declared exact-class/lowercase/article rule. No encoding or generation occurs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import shared_dictionary as base

RESULT = ROOT / "results/universal-context-development-v1"
COUNTS = {"source_excerpts": 100, "historical_covers": 4, "grounded_covers": 20, "long_summary_covers": 20}
ARM_BITS = {"plain_length16__8": 24, "plain_length16__16": 32, "aes_siv__8": 152}


def read(path):
    return json.loads(path.read_bytes())


def jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core(token):
    match = re.fullmatch(r"([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)", token)
    if match is None:
        return None
    if any(c == "_" or not unicodedata.category(c).startswith("P") for c in match[1] + match[3]):
        return None
    return match[2].lower()


def oracle(text, dictionary, reference, radius=1, cap=4096):
    tokens = text.split()
    raw = base.analyze_text(text, dictionary, base.Protocol(1, 1, "plain_length16"))
    by_token = {s.token_index: s for s in raw.slots}
    classes = {c.class_id: c.words for c in dictionary.classes}
    ref_classes = {c.class_id: c.words for c in reference.classes}
    decisions = []
    for slot in raw.slots:
        index = slot.token_index
        product_size = 1
        for neighbor in range(max(0, index - max(radius, 1)), min(len(tokens), index + radius + 1)):
            if neighbor in by_token:
                product_size *= len(classes[by_token[neighbor].class_id])
        if index and index - 1 in by_token:
            previous_cores = set(classes[by_token[index - 1].class_id])
        else:
            previous_cores = {core(tokens[index - 1])} if index else set()
        if product_size > cap:
            reason = "orbit_cap_exceeded"
        elif ref_classes.get(slot.class_id) != classes[slot.class_id]:
            reason = "class_not_exact_reference"
        elif slot.case_style != "lower":
            reason = "case_not_lowercase"
        elif previous_cores & {"a", "an"}:
            reason = "adjacent_indefinite_article"
        else:
            reason = "accepted"
        decisions.append({"token_index": index, "class_id": slot.class_id, "orbit_size": product_size,
                          "accepted": reason == "accepted", "reason": reason})
    return raw.capacity_bits, decisions


def stats(values):
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values),
            "zero_capacity": sum(v == 0 for v in values)} if values else {"count": 0}


def execute():
    protocol = read(RESULT / "protocol.json")
    execution = read(RESULT / "audit/execution.json")
    for location, manifest in ((ROOT, protocol["source_files"]), (ROOT, protocol["code_files"]), (RESULT, protocol["frozen_files"])):
        for name, digest in manifest.items():
            assert sha(location / name) == digest, name
    for name, digest in execution["output_sha256"].items():
        assert sha(RESULT / "audit" / name) == digest, name
    inputs = jsonl(RESULT / "inputs.jsonl")
    records = jsonl(RESULT / "audit/records.jsonl")
    summary = read(RESULT / "audit/summary.json")
    assert len(inputs) == len(records) == 144
    assert Counter(r["cohort"] for r in inputs) == Counter(COUNTS)
    assert Counter(r["cohort"] for r in records) == Counter(COUNTS)
    assert all(r["split"] == "development" for r in inputs)
    assert len({r["audit_record_id"] for r in inputs}) == len({r["audit_record_id"] for r in records}) == 144
    dictionaries = {v: base.PublicDictionary.from_dict(read(ROOT / f"data/shared_dictionary_recovery_{v}.json")) for v in ("v1", "v2")}
    policies = {v: read(RESULT / f"policy_{v}.json") for v in dictionaries}
    assert all(p["radius"] == 1 and p["orbit_cap"] == 4096 for p in policies.values())
    assert all(base.PublicDictionary.from_dict(p["strict_dictionary"]).sha256 == dictionaries["v2"].sha256 for p in policies.values())
    capacities = {c: {v: [] for v in dictionaries} for c in COUNTS}
    baselines = {c: {v: [] for v in dictionaries} for c in COUNTS}
    reasons = {c: {v: Counter() for v in dictionaries} for c in COUNTS}
    outcomes, slot_checks, capacity_checks = Counter(), 0, 0
    individual, simultaneous = 0, 0
    for source, record in zip(inputs, records):
        for name in ("audit_record_id", "cohort", "source_group_id", "text_sha256", "availability"):
            assert source[name] == record[name], name
        assert set(record["methods"]) == {"universal_v1", "universal_v2"}
        for v, dictionary in dictionaries.items():
            method = record["methods"]["universal_" + v]
            assert set(method["arms"]) == set(ARM_BITS)
            if source["availability"] == "available":
                raw, decisions = oracle(source["text"], dictionary, dictionaries["v2"])
                expected = sum(d["accepted"] for d in decisions)
                assert raw == record["capacity_only_references"]["base_" + v]["capacity_bits"]
                assert expected == method["capacity_bits"]
                assert len(decisions) == len(method["analysis"]["slot_decisions"])
                assert list(d["token_index"] for d in decisions if d["accepted"]) == method["analysis"]["eligible_token_indices"]
                for decision, observed in zip(decisions, method["analysis"]["slot_decisions"]):
                    assert all(observed[k] == value for k, value in decision.items())
                    if observed["accepted"]:
                        assert observed["evaluated_variants"] == observed["orbit_size"]
                    elif observed["reason"] == "orbit_cap_exceeded":
                        assert observed["evaluated_variants"] == 0
                    else:
                        assert 1 <= observed["evaluated_variants"] <= observed["orbit_size"]
                    slot_checks += 1
                assert method["input_status"] == "available"
                alias = method["alias_audit"]
                assert len(alias["simultaneous_checks"]) == 2 and alias["all_equal"] is True
                assert all(c["equal"] and c["signature"] == alias["reference_signature"] for c in alias["individual_checks"] + alias["simultaneous_checks"])
                individual += len(alias["individual_checks"])
                simultaneous += len(alias["simultaneous_checks"])
                capacity_checks += 1
                capacities[source["cohort"]][v].append(expected)
                baselines[source["cohort"]][v].append(raw)
                reasons[source["cohort"]][v].update(d["reason"] for d in decisions)
                for arm_name, arm in method["arms"].items():
                    assert expected < ARM_BITS[arm_name], "This verifier's no-delivery expectation no longer fits saved results"
                    assert arm["status"] == "insufficient_capacity"
            else:
                assert source["availability"] == "source_gate_abstention"
                assert method["capacity_bits"] is None and method["analysis"] is None and method["alias_audit"] is None
                assert all(a["status"] == "source_gate_abstention" for a in method["arms"].values())
            assert all(not a["encoding_called"] and not a["mechanical_success"] for a in method["arms"].values())
            outcomes.update(a["status"] for a in method["arms"].values())
    independent = {}
    for cohort, count in COUNTS.items():
        selected = [r for r in inputs if r["cohort"] == cohort]
        actual = summary["cohorts"][cohort]
        groups = len({r["source_group_id"] for r in selected})
        assert groups == (2 if cohort == "historical_covers" else count)
        assert actual["planned_records"] == count and actual["distinct_source_groups"] == groups
        assert actual["availability_counts"] == dict(Counter(r["availability"] for r in selected))
        independent[cohort] = {"planned": count, "source_groups": groups, "capacity_bits": {}}
        for v in dictionaries:
            assert actual["capacity_only_references"]["base_" + v] == stats(baselines[cohort][v])
            method_summary = actual["methods"]["universal_" + v]
            assert method_summary["capacity_bits"] == stats(capacities[cohort][v])
            assert method_summary["slot_reasons"] == dict(reasons[cohort][v])
            for arm in method_summary["arms"].values():
                assert arm["planned_denominator"] == count and arm["available_denominator"] == len(capacities[cohort][v])
                assert arm["mechanical_successes"] == 0
            independent[cohort]["capacity_bits"][v] = stats(capacities[cohort][v])
    assert dict(outcomes) == summary["all_outcome_counts"] == {"insufficient_capacity": 834, "source_gate_abstention": 30}
    assert sum(outcomes.values()) == 864
    return {"schema": "synsteg-universal-independent-results-v1", "status": "matched",
            "records_checked": len(records), "source_excerpts_checked": 100, "source_gate_records": 5,
            "cohorts_independent": False, "capacity_pairs_checked": capacity_checks,
            "baseline_capacity_pairs_checked": capacity_checks, "slot_decisions_checked": slot_checks,
            "saved_individual_alias_evidence_checked": individual, "saved_simultaneous_alias_evidence_checked": simultaneous,
            "alias_checks_rerun_by_this_script": False, "outcomes": dict(outcomes),
            "independent_capacity_summary": independent, "frozen_commit": execution["frozen_commit"],
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [RESULT / "inputs.jsonl", RESULT / "audit/records.jsonl", RESULT / "audit/summary.json", RESULT / "protocol.json"]},
            "oracle_scope": "Direct declared-rule calculation sharing only frozen base tokenizer; does not call universal analyzer or evaluator",
            "new_model_calls": 0, "encoding_calls": 0, "closed_test_opened": False}


if __name__ == "__main__":
    print(json.dumps(execute(), sort_keys=True, indent=2))
