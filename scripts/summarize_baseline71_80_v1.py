#!/usr/bin/env python3
"""Summarize immutable baseline batches36–40 without model or codec execution.

All 80 attempts are read from the fixed batch40 after snapshot. The complete
100-record development input and 300-cell campaign denominator are checked.
Source-fidelity annotations and the separate mask-order study are not inferred
from these mechanical generation/length checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

SOURCE = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
SOURCE_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
RUN = "runs/large-campaign-development-v2/models/tinyllama11"
BATCH40 = "results/large-campaign-development-v2/batches/batch40"
BATCH35 = "results/large-campaign-development-v2/batches/batch35"
MODELS = ("tinyllama11", "qwen05", "falcon3_1b")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def summarize(root):
    inputs = {}

    def read(path):
        raw = (root / path).read_bytes()
        inputs[path] = digest(raw)
        return json.loads(raw)

    source_raw = (root / SOURCE).read_bytes()
    assert digest(source_raw) == SOURCE_SHA
    inputs[SOURCE] = SOURCE_SHA
    source = [json.loads(line) for line in source_raw.decode().splitlines()]
    assert len(source) == 100 and all(x["split"] == "development" for x in source)
    assert len({x["record_id"] for x in source}) == 100
    for src in source:
        assert digest(src["excerpt"].encode()) == src["excerpt_sha256"]
        assert digest(src["prompt"].encode()) == src["prompt_sha256"]

    audits = []
    for batch in range(36, 41):
        item = read(f"results/large-campaign-continuation-v3-audit/batch{batch}-postrun-audit.json")
        assert item["status"] == "passed" and item["completed"] == batch * 2
        assert item["prior_completed_entries_preserved"] == batch * 2 - 2
        assert item["prior_completed_attempt_files_byte_identical"] is True
        assert item["full_model_record_denominator"] == 300
        assert item["new_model_calls_by_auditor"] == item["new_tokenizer_calls_by_auditor"] == 0
        audits.append(item)
    rows = [dict(row) for audit in audits for row in audit["new_completed_rows"]]
    assert [x["source_order"] for x in rows] == list(range(71, 81))
    assert len({x["identity"] for x in rows}) == 10

    checkpoint = read(BATCH40 + "/after-checkpoint.json")
    prior70 = read(BATCH35 + "/after-checkpoint.json")
    assert checkpoint["status"] == prior70["status"] == "verified"
    assert checkpoint["completed_generations"] == checkpoint["attempted"] == 80
    assert prior70["completed_generations"] == prior70["attempted"] == 70
    assert checkpoint["planned_model_record_pairs"] == prior70["planned_model_record_pairs"] == 300
    all_records = checkpoint["records"]
    assert len(all_records) == 300
    assert Counter(x["model_alias"] for x in all_records) == {name: 100 for name in MODELS}
    assert Counter(x["status"] for x in all_records) == {"completed": 80, "not_attempted": 220}
    model_completed = {name: sum(x["status"] == "completed" for x in all_records if x["model_alias"] == name)
                       for name in MODELS}
    assert model_completed == {"tinyllama11": 80, "qwen05": 0, "falcon3_1b": 0}
    records = [x for x in all_records if x["model_alias"] == "tinyllama11"]
    assert len(records) == 100
    assert [x["status"] for x in records] == ["completed"] * 80 + ["not_attempted"] * 20
    attempts = {}
    for order, row in enumerate(records[:80], 1):
        relative = RUN + "/attempts/" + row["identity"] + ".json"
        snapshot_path = BATCH40 + "/after/" + relative
        raw = (root / snapshot_path).read_bytes()
        actual_sha = digest(raw)
        assert actual_sha == checkpoint["run_files_sha256"][relative]
        if order <= 70:
            assert actual_sha == prior70["run_files_sha256"][relative]
        inputs[snapshot_path] = actual_sha
        item = json.loads(raw)
        assert item["status"] == "completed" and item["automatic_retries"] == 0
        assert item["identity"] == row["identity"]
        assert item["record"]["record_id"] == row["record_id"] == source[order - 1]["record_id"]
        attempts[row["identity"]] = item
    for row in rows:
        src = source[row["source_order"] - 1]
        item = attempts[row["identity"]]
        relative = RUN + "/attempts/" + row["identity"] + ".json"
        assert inputs[BATCH40 + "/after/" + relative] == row["attempt_sha256"]
        assert src["record_id"] == row["record_id"] and src["title"] == row["source_title"]
        assert item["result"]["usage"] == row["usage"]
        assert item["result"]["stop_reason"] == row["stop_reason"]
        assert digest(item["result"]["cover_text"].encode()) == row["cover_sha256"]
        assert len(item["result"]["cover_text"].split()) == row["words"]
        row.update(source_excerpt_words=len(src["excerpt"].split()),
                   source_excerpt_sha256=src["excerpt_sha256"],
                   prompt_sha256=src["prompt_sha256"],
                   frozen_topic_label=src["topic"], genre=src["genre"])
    tokens = sum(x["result"]["usage"]["output_tokens"] for x in attempts.values())
    assert tokens == audits[-1]["cumulative_generated_output_tokens"]
    prior_tokens = sum(attempts[x["identity"]]["result"]["usage"]["output_tokens"] for x in records[:70])
    assert prior_tokens == 45643
    new_tokens = sum(x["usage"]["output_tokens"] for x in rows)
    assert tokens == prior_tokens + new_tokens
    # The earlier input issue remains a fixed panel limitation, not a new label.
    assert source[69]["title"] == "Oasis (band)" and source[69]["topic"] == "chemistry"
    return {
        "schema": "baseline71-80-saved-summary-v1",
        "source_snapshot": BATCH40 + "/after", "new_rows": rows,
        "new_completed": 10, "new_output_tokens": new_tokens,
        "new_whitespace_words": sum(x["words"] for x in rows),
        "new_source_excerpt_words": sum(x["source_excerpt_words"] for x in rows),
        "new_target_words": sum(x["target_words"] for x in rows),
        "new_stop_reasons": dict(Counter(x["stop_reason"] for x in rows)),
        "completed_generations": 80, "full_model_record_denominator": 300,
        "not_attempted": 220, "model_completed": model_completed,
        "cumulative_output_tokens": tokens,
        "cumulative_stop_reasons": dict(Counter(x["result"]["stop_reason"] for x in attempts.values())),
        "full_development_source_records_verified": 100,
        "full_development_excerpt_and_prompt_hashes_verified": 200,
        "saved_attempt_hashes_verified": 80,
        "prior70_attempt_hashes_unchanged": True,
        "prior70_output_tokens_unchanged": prior_tokens,
        "public_transport_evaluated_covers": 60,
        "covers_awaiting_public_transport": list(range(61, 81)),
        "all_new_outcomes_retained": True,
        "factual_or_steganographic_success_inferred_from_completion": False,
        "length_measure": "Whitespace-delimited words; target is an aim, not an invented pass/fail threshold.",
        "inherited_input_metadata_limitation": {"source_order": 70,
            "title": source[69]["title"], "frozen_topic_label": source[69]["topic"],
            "label_or_prompt_changed": False,
            "scope": "Historical Oasis (band) under chemistry retained; no post-hoc relabeling or exclusion. New input-only review71–80 is separate."},
        "inputs_sha256": inputs, "new_model_calls_by_this_summary": 0,
        "tokenizer_calls": 0, "encoder_calls": 0, "decoder_calls": 0, "paid_calls": 0,
        "closed_test_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    raw = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        print(raw, end="")
