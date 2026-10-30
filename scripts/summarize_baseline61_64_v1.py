#!/usr/bin/env python3
"""Aggregate saved baseline61-64 evidence; no inference or codec execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(root):
    audits = []
    inputs = {}
    for batch in (31, 32):
        path = root / f"results/large-campaign-continuation-v3-audit/batch{batch}-postrun-audit.json"
        raw = path.read_bytes()
        item = json.loads(raw)
        assert item["status"] == "passed" and item["completed"] == batch * 2
        assert item["prior_completed_entries_preserved"] == batch * 2 - 2
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        audits.append(item)
    rows = [row for audit in audits for row in audit["new_completed_rows"]]
    assert [x["source_order"] for x in rows] == [61, 62, 63, 64]
    assert len({x["identity"] for x in rows}) == 4
    run = root / "runs/large-campaign-development-v2/models/tinyllama11"
    attempts = []
    for path in sorted((run / "attempts").glob("*.json")):
        item = json.loads(path.read_bytes())
        assert item["status"] == "completed" and item["automatic_retries"] == 0
        attempts.append(item)
    assert len(attempts) == 64
    by_identity = {item["identity"]: item for item in attempts}
    for row in rows:
        path = run / "attempts" / (row["identity"] + ".json")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["attempt_sha256"]
        item = by_identity[row["identity"]]
        assert item["record"]["record_id"] == row["record_id"]
        assert item["result"]["usage"] == row["usage"]
        assert len(item["result"]["cover_text"].split()) == row["words"]
    tokens = sum(x["result"]["usage"]["output_tokens"] for x in attempts)
    assert tokens == audits[-1]["cumulative_generated_output_tokens"]
    return {
        "schema": "baseline61-64-saved-summary-v1",
        "new_rows": rows,
        "new_completed": 4,
        "new_output_tokens": sum(x["usage"]["output_tokens"] for x in rows),
        "new_stop_reasons": dict(Counter(x["stop_reason"] for x in rows)),
        "completed_generations": 64,
        "full_model_record_denominator": 300,
        "not_attempted": 236,
        "model_completed": {"tinyllama11": 64, "qwen05": 0, "falcon3_1b": 0},
        "cumulative_output_tokens": tokens,
        "cumulative_stop_reasons": dict(Counter(x["result"]["stop_reason"] for x in attempts)),
        "public_transport_evaluated_covers": 60,
        "new_covers_awaiting_public_transport": [61, 62, 63, 64],
        "all_new_outcomes_retained": True,
        "factual_or_steganographic_success_inferred_from_completion": False,
        "inputs_sha256": inputs,
        "new_model_calls_by_this_summary": 0,
        "encoder_calls": 0,
        "decoder_calls": 0,
        "paid_calls": 0,
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
