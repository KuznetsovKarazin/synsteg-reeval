"""Read-only inventory of observed substitutions; never calls an encoder/model.

Fresh runs create outputs exclusively. --verify-existing compares reconstructed
bytes to existing artifacts without rewriting them. Frozen inputs stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ID = "prospective-development-transport-first40-delta-slot-review-v1"
MILESTONE = ROOT / "results/prospective-development-transport-first40-v1/milestones/tinyllama-first40"
OUT = ROOT / "results" / ID
SOURCE = ROOT / "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
WORD = re.compile(r"(?P<prefix>[^A-Za-z]*)(?P<word>[A-Za-z]+)(?P<suffix>[^A-Za-z]*)\Z")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def rel(path):
    return str(path.relative_to(ROOT))


def context(text, tokens, index):
    start = tokens[max(0, index - 10)].start()
    end = tokens[min(len(tokens) - 1, index + 10)].end()
    return {"start": start, "end": end, "quote": text[start:end]}


def whitespace_gaps(text, tokens):
    positions = [(0, tokens[0].start())]
    positions += [(a.end(), b.start()) for a, b in zip(tokens, tokens[1:])]
    positions += [(tokens[-1].end(), len(text))]
    return [text[a:b] for a, b in positions]


def build():
    sources = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    assert len(sources) == 100
    source_by_id = {row["record_id"]: row for row in sources}
    cells = [x for x in load_gz(MILESTONE / "cells.jsonl.gz") if x["model_alias"] == "tinyllama11" and 31 <= x["source_order"] <= 40]
    record_rows = load_gz(MILESTONE / "records.jsonl.gz")
    jobs = load_gz(MILESTONE / "recipient/receiver_jobs.jsonl.gz")
    results = load_gz(MILESTONE / "recipient/receiver_results.jsonl.gz")
    assert len(cells) == 510 and len(record_rows) == 300
    assert len({row["row_id"] for row in cells}) == len(cells)
    successes = {row["row_id"]: row for row in cells if row.get("full_exact") is True}
    jobs = [row for row in jobs if row["row_id"] in successes]
    results = [row for row in results if row["row_id"] in successes]
    jobs_by_id = {row["row_id"]: row for row in jobs}
    result_by_id = {row["row_id"]: row for row in results}
    assert len(successes) == len(jobs) == len(results) == 24
    assert set(successes) == set(jobs_by_id) == set(result_by_id)
    completed = [row for row in record_rows if row["status"] == "completed" and 31 <= row["source_order"] <= 40]
    assert len(completed) == 10
    completed.sort(key=lambda row: row["source_order"])
    assert [row["source_order"] for row in completed] == list(range(31, 41))
    covers = []
    frozen_paths = [MILESTONE / name for name in ["cells.jsonl.gz", "records.jsonl.gz", "recipient/receiver_jobs.jsonl.gz", "recipient/receiver_results.jsonl.gz", "input_config.json", "input_checkpoint.json", "status.json"]]
    frozen_paths.append(SOURCE)
    for row in completed:
        source = source_by_id[row["record_id"]]
        attempt_path = MILESTONE / "input_snapshot/runs/large-campaign-development-v2/models/tinyllama11/attempts" / (row["identity"] + ".json")
        attempt = json.loads(attempt_path.read_text())
        frozen_paths.append(attempt_path)
        cover = attempt["result"]["cover_text"]
        assert attempt["status"] == "completed" and cover == row["cover_text"]
        assert sha(attempt_path.read_bytes()) == row["attempt_sha256"]
        assert source["prompt_sha256"] == row["prompt_sha256"]
        assert sha(source["excerpt"].encode()) == source["excerpt_sha256"]
        covered_cells = [cell for cell in cells if cell["identity"] == row["identity"]]
        assert len(covered_cells) == 51
        covers.append({"source_order": row["source_order"], "record_id": row["record_id"], "title": source["title"], "cover_text": cover, "cover_sha256": sha(cover.encode()), "source_excerpt": source["excerpt"], "source_excerpt_sha256": source["excerpt_sha256"], "attempt_snapshot_path": rel(attempt_path), "attempt_sha256": row["attempt_sha256"], "all_cell_execution_status_counts": dict(sorted(Counter(cell["execution_status"] for cell in covered_cells).items())), "successful_row_ids": sorted(cell["row_id"] for cell in covered_cells if cell.get("full_exact") is True)})
    cover_by_id = {row["record_id"]: row for row in covers}
    unique = {}
    outcomes = []
    total_changes = 0
    for row_id, cell in sorted(successes.items(), key=lambda item: (item[1]["source_order"], item[1]["parent_payload_length"], item[0])):
        job = jobs_by_id[row_id]
        receiver = result_by_id[row_id]
        assert receiver["status"] == "decoded" and receiver["error"] is None
        assert cell["fresh_receiver_exact"] is True and cell["encoder_called"] is True
        decoded = receiver["decoded_bits"]
        assert set(decoded) <= {"0", "1"}
        assert len(decoded) == cell["decoded_bits"] == cell["requested_payload_bits"]
        assert sha(decoded.encode("ascii")) == cell["decoded_sha256"] == cell["payload_sha256"]
        original = cover_by_id[cell["record_id"]]
        cover = original["cover_text"]
        stego = job["stego_text"]
        assert sha(cover.encode()) == cell["cover_sha256"]
        assert sha(stego.encode()) == cell["stego_sha256"]
        assert original["attempt_sha256"] == cell["attempt_sha256"]
        left = list(re.finditer(r"\S+", cover))
        right = list(re.finditer(r"\S+", stego))
        assert len(left) == len(right) == cell["cover_words"] == cell["stego_words"]
        assert whitespace_gaps(cover, left) == whitespace_gaps(stego, right)
        changed_ids = []
        for index, (a, b) in enumerate(zip(left, right)):
            if a.group() == b.group():
                continue
            aw, bw = WORD.fullmatch(a.group()), WORD.fullmatch(b.group())
            assert aw is not None and bw is not None
            assert aw["prefix"] == bw["prefix"] and aw["suffix"] == bw["suffix"]
            key = [cell["record_id"], index, aw["word"], bw["word"]]
            slot_id = sha(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode())
            start, end = a.start() + aw.start("word"), a.start() + aw.end("word")
            stego_start, stego_end = b.start() + bw.start("word"), b.start() + bw.end("word")
            slot = unique.setdefault(slot_id, {"slot_id": slot_id, "deduplication_key": key, "source_order": cell["source_order"], "record_id": cell["record_id"], "token_index_zero_based": index, "original": aw["word"], "replacement": bw["word"], "original_token": a.group(), "replacement_token": b.group(), "cover_sha256": cell["cover_sha256"], "source_excerpt_sha256": original["source_excerpt_sha256"], "original_span": {"start": start, "end": end, "quote": cover[start:end]}, "original_context": context(cover, left, index), "occurrences": []})
            slot["occurrences"].append({"row_id": row_id, "parent_payload_length": cell["parent_payload_length"], "stego_sha256": cell["stego_sha256"], "stego_span": {"start": stego_start, "end": stego_end, "quote": stego[stego_start:stego_end]}, "stego_context": context(stego, right, index)})
            changed_ids.append(slot_id)
        assert len(changed_ids) == cell["changed_positions"]
        total_changes += len(changed_ids)
        outcomes.append({"row_id": row_id, "source_order": cell["source_order"], "record_id": cell["record_id"], "arm": cell["arm"], "grid": cell["grid"], "parent_payload_length": cell["parent_payload_length"], "requested_payload_bits": cell["requested_payload_bits"], "execution_status": cell["execution_status"], "receiver_status": receiver["status"], "cover_sha256": cell["cover_sha256"], "stego_sha256": cell["stego_sha256"], "stego_text": stego, "changed_positions": cell["changed_positions"], "changed_slot_ids": changed_ids, "cell": cell})
    slots = sorted(unique.values(), key=lambda row: (row["source_order"], row["token_index_zero_based"], row["original"], row["replacement"]))
    assert total_changes == sum(x["changed_positions"] for x in successes.values()) and len(slots) > 0
    data = {"covers.jsonl": b"".join(encoded(row) for row in covers), "outcomes.jsonl": b"".join(encoded(row) for row in outcomes), "unique_slots.jsonl": b"".join(encoded(row) for row in slots)}
    inventory = {"schema": "synsteg-posthoc-slot-review-inventory-v1", "id": ID, "milestone": rel(MILESTONE), "generation_calls": 0, "encoder_calls": 0, "receiver_calls": 0, "planned_model_record_rows": 300, "completed_covers": 10, "cumulative_completed_covers": 40, "source_orders_reviewed": list(range(31, 41)), "previous_full_exact_outcomes_not_reannotated": 75, "previous_encoder_calls_reused_not_reviewed_again": 720, "full_cumulative_planned_cells": 15300, "all_planned_cells": len(cells), "actual_encoder_attempts": sum(row.get("encoder_called") is True for row in cells), "execution_status_counts_all_cells": dict(sorted(Counter(row["execution_status"] for row in cells).items())), "full_exact_outcomes": len(successes), "insufficient_capacity": sum(row["execution_status"] == "insufficient_capacity" for row in cells), "changed_slot_occurrences": total_changes, "unique_changed_slots": len(slots), "successful_cover_orders": [row["source_order"] for row in covers if row["successful_row_ids"]], "cover_orders_without_successful_stego": [row["source_order"] for row in covers if not row["successful_row_ids"]], "deduplication_key": ["record_id", "zero-based whitespace token index", "case-preserving original ASCII word core", "case-preserving replacement ASCII word core"], "offset_convention": "zero-based, end-exclusive Unicode code-point offsets", "scope": "All observed substitutions in all full_exact outcomes; unsuccessful attempts and all ten new covers retained in denominators. This inventory is not a quality rating or new codec execution.", "frozen_inputs": [{"path": rel(path), "sha256": sha(path.read_bytes())} for path in frozen_paths], "artifact_sha256": {name: sha(content) for name, content in data.items()}}
    assert inventory["actual_encoder_attempts"] == 240 and inventory["insufficient_capacity"] == 216
    data["inventory.json"] = (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    data = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in data.items():
        path = OUT / name
        if args.verify_existing:
            assert path.read_bytes() == content, name
        else:
            with path.open("xb") as stream:
                stream.write(content)
    print(json.dumps({"mode": "verified" if args.verify_existing else "created", "outputs": {name: sha(content) for name, content in data.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
