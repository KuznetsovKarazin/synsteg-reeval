#!/usr/bin/env python3
"""Prepare a disabled, offline candidate-format component; never generate text."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_format_v2", ROOT / "src/synsteg_audit/author_candidate_format_v2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    archive = ROOT.parent / "project_sources/01-synsteg_reproduction.zip"
    archive_sha = m.digest(archive.read_bytes())
    if archive_sha != "793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790":
        raise ValueError("author ZIP differs")
    prior = ROOT / "experiments/author-candidate-local-native-v1/requests.json"
    requests = m.build_requests(prior.read_bytes())
    base = "experiments/author-candidate-format-v2/"
    prompt_path = ROOT / (base + "system-prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("xb") as stream:
        stream.write((m.SYSTEM_PROMPT + "\n").encode())
    requests_sha = save(base + "requests.json", requests)
    profile_path = "experiments/framed-interval-public-v1/native-public-profile.json"
    profile_sha = m.digest((ROOT / profile_path).read_bytes())
    if profile_sha != "f32097750dc898fce418a29146587a4229eae1dcb76e3cbcd374b692846adc8a":
        raise ValueError("pinned native resource profile differs")
    protocol = {
        "schema": m.ID + "-protocol", "id": m.ID,
        "component_scope": "offline prompt, exact coverage/count gate and whole-prefix grammar only",
        "native_execution_enabled": False, "native_calls": 0,
        "future_native_variant_id": "author-candidate-local-native-v2",
        "future_native_variant_is_not_implemented_or_authorized_by_this_file": True,
        "not_original_provider_replication": True,
        "original_author_source_zip_sha256": archive_sha,
        "adaptations_from_v1": [
            "JSON-only unambiguous prompt with no fenced example or placeholder keys",
            "exact ordered input keys and exactly eight distinct alternatives or explicit empty abstention",
            "irreversibly invalid whole-JSON prefix stops after the current committed token",
            "strict source identity/count/duplicate checks; no semantic certification"
        ],
        "causal_prompt_only_comparison": False,
        "cohort": {"selection": "unchanged first2chunks of first fixed baseline cover",
                   "source_requests_path": str(prior.relative_to(ROOT)),
                   "source_requests_sha256": m.AUTHOR_REQUESTS_SHA256,
                   "cover_sha256": m.AUTHOR_COVER_SHA256,
                   "whitespace_words_in_full_cover": 543, "full_cover_chunks": 109,
                   "diagnostic_chunks": 2, "outside_diagnostic_chunks": 107,
                   "same_five_token_context_as_v1": True, "additional_context_supplied": False},
        "requests": {"path": base + "requests.json", "sha256": requests_sha},
        "fixed_request_sha256": [x["request_sha256"] for x in requests["requests"]],
        "bits_per_word": 3, "additional_bits": 0,
        "response_schema": {"root": "one object with only words field",
                            "rows": "exactly one ordered singleton object per source word",
                            "keys": "exact source strings including capitalization/punctuation",
                            "values": "[] abstention or exactly8 nonempty unpadded strings",
                            "uniqueness": "casefold after collapsing whitespace, excluding source surface",
                            "no_silent_repair_or_filter": True,
                            "reject_duplicate_JSON_keys": True,
                            "forbidden_candidate_and_source_char_categories": ["Cc", "Cf", "Zl", "Zp"],
                            "unicode_rule_scope": "Conservative surface-format restriction for this English cohort; not a general multilingual linguistic rule",
                            "case_POS_morphology_sense_punctuation_candidate_correctness_verified": False},
        "abstention": {"format_accepted": True, "candidate_strings": 0,
                       "semantic_retry_allowed": False,
                       "usable_candidate_rows": None, "capacity_bits": None,
                       "full_cover_complete_not_implied": True},
        "context_limitation": "Five tokens can underdetermine sense, reference and morphology; instruction compliance is not semantic proof.",
        "proposed_native_settings": {"profile_path": profile_path, "profile_sha256": profile_sha,
            "profile_scope": "resources/runtime/READY/resident identities only; inherited temperature0.8 is not proposed sampler",
            "model": "Qwen2.5-1.5B-Instruct Q4_K_M", "temperature": 0.7, "top_k": 50,
            "integer_total": 16777216, "token_cap_per_attempt": 512,
            "attempt_deadline_seconds": 600, "physical_attempts_per_chunk_max": 3,
            "diagnostic_physical_attempts_max": 6, "reset": "fresh owned native process per attempt",
            "EOS_policy": "same existing EOS/special mask; not vanilla EOS-terminated generation",
            "seed_policy": "HMAC/draw implementation must be separately frozen under new native ID, bound to requestSHA and fixed attempt0..2; no seed search",
            "all_settings_are_pending_native_integration": True},
        "prefix_policy": {"input": "exact bytes of entire committed response, not a sliced convenience file",
            "complete": "entire object plus only JSON whitespace; run strict response gate and stop",
            "invalid_prefix": "stop and record known format refusal; no scanning, fence stripping, repair or token truncation",
            "incomplete": "continue until complete/invalid/resource refusal or committed512token cap",
            "incomplete_at_cap": "known format refusal",
            "partial_UTF8": "wait conservatively without lossy decoding; invalid UTF8 is refused",
            "resource_refusal": "known parser-limit refusal; stop campaign without automatic retry",
            "max_raw_bytes": m.MAX_RAW_BYTES, "max_JSON_depth": m.MAX_JSON_DEPTH,
            "trailing_junk_within_final_token": "whole token retained, response refused",
            "known_refusal_next_attempt": "only next predeclared attempt, maximum3, with complete verified journal",
            "accepted_including_all_abstentions": "stop chunk; no semantic retry",
            "unknown_incomplete_journal_or_transport_failure": "stop campaign; no retry or resume"},
        "journal_gate": {"component": "candidate-journal-guard-v2",
            "required_before_format_acceptance": "verify_bundle on the persisted sealed attempt; trusted terminal commitment and resource/request/protocol bindings; use verified response bytes",
            "native_adapter_exists": False,
            "post_save_check": "independent exact Git readback and sealed-bundle verification; later tampering/truncation must be recorded, never filled in",
            "no_OS_immutability_claim": True},
        "next_native_requires": ["new integrated native runner and source closure",
            "independent source and fault-injection review", "native activation and immutable execution intent",
            "actual Git source/readback gate", "root exclusive NN lease"],
        "full_cover_policy": "All109 ordered chunks must be accounted for, including abstentions/refusals/unattempted. No original/fixed matched transport from a two-chunk diagnostic.",
        "all50_workload_scope": "metadata-only counts from pinned first50 records; not native authorization or a runtime prediction",
        "pay_api_allowed": False, "closed_test_allowed": False,
        "model_calls": 0, "tokenizer_calls": 0, "encoder_calls": 0, "decoder_calls": 0
    }
    protocol_sha = save(base + "protocol.json", protocol)
    records_path = "results/prospective-development-transport-first50-v1/milestones/tinyllama-first50/records.jsonl.gz"
    records_raw = (ROOT / records_path).read_bytes()
    if m.digest(records_raw) != "49fb7482feb12c03d31cb7262ca22ead1420d68cac5e02316debe9a95b119475":
        raise ValueError("immutable first50 records digest differs")
    rows = [json.loads(x) for x in gzip.decompress(records_raw).splitlines()]
    completed = [r for r in rows if r["status"] == "completed"]
    if len(rows) != 300 or len(completed) != 50:
        raise ValueError("expected immutable first50 snapshot")
    known = []
    for r in completed:
        if r["model_alias"] != "tinyllama11" or r["split"] != "development":
            raise ValueError("snapshot cohort differs")
        words = len(r["cover_text"].split())
        chunks = math.ceil(words / 5)
        known.append({"record_id": r["record_id"], "source_order": r["source_order"],
                      "cover_sha256": m.digest(r["cover_text"].encode()),
                      "words": words, "five_word_chunks": chunks,
                      "max_physical_attempts": chunks * 3,
                      "max_committed_output_tokens": chunks * 3 * 512})
    known.sort(key=lambda x: x["source_order"])
    if (known[0]["cover_sha256"] != m.AUTHOR_COVER_SHA256
            or known[0]["five_word_chunks"] != 109 or known[0]["words"] != 543):
        raise ValueError("first cover changed")
    chunks = sum(x["five_word_chunks"] for x in known)
    cost = {"schema": m.ID + "-metadata-workload", "records_path": records_path,
            "records_sha256": m.digest(records_raw), "full_model_record_pairs": 300,
            "available_covers": 50, "unavailable_covers": 250,
            "observed_words": sum(x["words"] for x in known),
            "known_five_word_chunks": chunks, "known_max_physical_attempts": chunks * 3,
            "known_max_committed_output_tokens": chunks * 3 * 512,
            "unknown_remaining_chunks": None, "runtime_prediction_seconds": None,
            "diagnostic": {"chunks": 2, "max_physical_attempts": 6, "max_output_tokens": 3072},
            "first_cover": {"chunks": 109, "max_physical_attempts": 327,
                            "max_output_tokens": 167424,
                            "sequential_attempt_deadline_sum_seconds": 196200,
                            "deadline_sum_is_not_runtime_prediction_and_excludes_overhead": True},
            "records": known, "model_calls": 0, "tokenizer_calls": 0,
            "encoding_calls": 0, "paid_calls": 0, "closed_test_opened": False,
            "native_execution_authorized_by_workload": False}
    cost_sha = save("results/author-candidate-format-v2-preparation/workload.json", cost)
    save("results/author-candidate-format-v2-preparation/status.json", {
        "schema": m.ID + "-preparation-status", "status": "offline_component_prepared_native_disabled",
        "protocol_sha256": protocol_sha, "requests_sha256": requests_sha,
        "workload_sha256": cost_sha, "source_zip_sha256": archive_sha,
        "new_native_attempts": 0, "prior_native_attempts_replayed": 0,
        "independent_review_required": True, "native_activation_enabled": False,
        "usable_candidate_rows": None, "capacity_bits": None,
        "full_cover_resource_complete": False})
    print(json.dumps({"status": "prepared_native_disabled", "diagnostic_chunks": 2,
                      "full_cover_chunks": 109, "known50cover_chunks": chunks,
                      "model_calls": 0, "encoder_decoder_calls": 0}))


if __name__ == "__main__":
    main()
