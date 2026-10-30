#!/usr/bin/env python3
"""Summarize immutable baseline batches33–35 without model or codec execution.

All cumulative attempts come from the fixed batch35 after snapshot, so later
continuation of the live run cannot change this summary. Source fidelity labels
are deliberately outside this mechanical generation/length aggregate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
SOURCE_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
RUN = "runs/large-campaign-development-v2/models/tinyllama11"
BATCH35 = "results/large-campaign-development-v2/batches/batch35"


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
    audits = []
    for batch in (33, 34, 35):
        item = read(f"results/large-campaign-continuation-v3-audit/batch{batch}-postrun-audit.json")
        assert item["status"] == "passed" and item["completed"] == batch * 2
        assert item["prior_completed_entries_preserved"] == batch * 2 - 2
        assert item["prior_completed_attempt_files_byte_identical"] is True
        audits.append(item)
    rows = [dict(row) for audit in audits for row in audit["new_completed_rows"]]
    assert [x["source_order"] for x in rows] == list(range(65, 71))
    assert len({x["identity"] for x in rows}) == 6

    checkpoint = read(BATCH35 + "/after-checkpoint.json")
    assert checkpoint["status"] == "verified"
    assert checkpoint["completed_generations"] == checkpoint["attempted"] == 70
    assert checkpoint["planned_model_record_pairs"] == 300
    all_records = checkpoint["records"]
    assert len(all_records) == 300
    assert Counter(x["model_alias"] for x in all_records) == {"tinyllama11": 100, "qwen05": 100, "falcon3_1b": 100}
    assert Counter(x["status"] for x in all_records) == {"completed": 70, "not_attempted": 230}
    model_completed = {name: sum(x["status"] == "completed" for x in all_records if x["model_alias"] == name)
                       for name in ("tinyllama11", "qwen05", "falcon3_1b")}
    assert model_completed == {"tinyllama11": 70, "qwen05": 0, "falcon3_1b": 0}
    records = [x for x in checkpoint["records"] if x["model_alias"] == "tinyllama11"]
    assert len(records) == 100
    assert [x["status"] for x in records] == ["completed"] * 70 + ["not_attempted"] * 30
    attempts = {}
    for order, row in enumerate(records[:70], 1):
        relative = RUN + "/attempts/" + row["identity"] + ".json"
        raw = (root / BATCH35 / "after" / relative).read_bytes()
        assert digest(raw) == checkpoint["run_files_sha256"][relative]
        inputs[BATCH35 + "/after/" + relative] = digest(raw)
        item = json.loads(raw)
        assert item["status"] == "completed" and item["automatic_retries"] == 0
        assert item["identity"] == row["identity"]
        assert item["record"]["record_id"] == row["record_id"] == source[order - 1]["record_id"]
        attempts[row["identity"]] = item
    for row in rows:
        src = source[row["source_order"] - 1]
        item = attempts[row["identity"]]
        relative = RUN + "/attempts/" + row["identity"] + ".json"
        assert inputs[BATCH35 + "/after/" + relative] == row["attempt_sha256"]
        assert src["record_id"] == row["record_id"] and src["title"] == row["source_title"]
        assert digest(src["excerpt"].encode()) == src["excerpt_sha256"]
        assert digest(src["prompt"].encode()) == src["prompt_sha256"]
        assert item["result"]["usage"] == row["usage"]
        assert item["result"]["stop_reason"] == row["stop_reason"]
        assert digest(item["result"]["cover_text"].encode()) == row["cover_sha256"]
        assert len(item["result"]["cover_text"].split()) == row["words"]
        row.update(source_excerpt_words=len(src["excerpt"].split()),
                   source_excerpt_sha256=src["excerpt_sha256"],
                   prompt_sha256=src["prompt_sha256"],
                   frozen_topic_label=src["topic"], genre=src["genre"])
    tokens = sum(x["result"]["usage"]["output_tokens"] for x in attempts.values())
    assert tokens == audits[-1]["cumulative_generated_output_tokens"] == 45643
    new_tokens = sum(x["usage"]["output_tokens"] for x in rows)
    assert new_tokens == 2623
    return {
        "schema": "baseline65-70-saved-summary-v1",
        "source_snapshot": BATCH35 + "/after", "new_rows": rows,
        "new_completed": 6, "new_output_tokens": new_tokens,
        "new_whitespace_words": sum(x["words"] for x in rows),
        "new_stop_reasons": dict(Counter(x["stop_reason"] for x in rows)),
        "completed_generations": 70, "full_model_record_denominator": 300,
        "not_attempted": 230,
        "model_completed": model_completed,
        "cumulative_output_tokens": tokens,
        "cumulative_stop_reasons": dict(Counter(x["result"]["stop_reason"] for x in attempts.values())),
        "public_transport_evaluated_covers": 60,
        "covers_awaiting_public_transport": list(range(61, 71)),
        "all_new_outcomes_retained": True,
        "factual_or_steganographic_success_inferred_from_completion": False,
        "length_measure": "Whitespace-delimited words; target is an aim, not an invented pass/fail threshold.",
        "input_metadata_limitation": {"source_order": 70, "title": rows[-1]["source_title"],
            "frozen_topic_label": rows[-1]["frozen_topic_label"],
            "topic_label_matches_source_subject": False,
            "label_or_prompt_changed": False,
            "scope": "Oasis (band) is retained under the frozen chemistry label; no post-hoc relabeling or exclusion."},
        "inputs_sha256": inputs, "new_model_calls_by_this_summary": 0,
        "encoder_calls": 0, "decoder_calls": 0, "paid_calls": 0,
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
    else:
        print(raw, end="")
