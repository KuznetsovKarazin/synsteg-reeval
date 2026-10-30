"""Read-only regression on the already saved v1 gap; never resumes inference.

The v1 journal format is not silently migrated into v2. This narrow diagnostic
checks its hash chain, acknowledged token prefix and result/raw agreement. It
does not re-audit logits, sampling, model execution or candidate semantics.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synsteg_audit.candidate_journal_guard_v2 import canonical, sha, strict_json, EvidenceError


RELATIVE = Path("results/author-candidate-local-native-v1/native-diagnostic-v1/chunk1-attempt0")
PINS = {
    "journal.jsonl": "9c388db691fbf7d77e901f39cc2151862fa4b6f9b76ee57a001eee56a9a557f0",
    "response.raw": "aa9a30697f260af24f23da22e1d9c4e87fc8b11891d3883134c31b5406234ec7",
    "result.json": "c12b5e9593c68ae15874000d6a7474cc386c6ccb9b4880684ef314e7f753fec0",
}


def check(root):
    directory = Path(root) / RELATIVE
    files = {name: (directory / name).read_bytes() for name in PINS}
    for name, raw in files.items():
        if sha(raw) != PINS[name]:
            raise EvidenceError("historical fixture differs: " + name)
    journal = files["journal.jsonl"]
    if not journal.endswith(b"\n"):
        raise EvidenceError("historical fixture has unexpected torn byte tail")
    previous = "0" * 64
    ids = []
    committed_raw = b""
    pending = None
    known = None
    rows = [strict_json(line) for line in journal.splitlines()]
    result = json.loads(files["result.json"])
    for index, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != "event_sha256"}
        if (type(row["sequence"]) is not int or row["sequence"] != index
                or row["previous_sha256"] != previous or sha(canonical(body)) != row["event_sha256"]):
            raise EvidenceError("historical fixture chain differs")
        previous = row["event_sha256"]
        kind = row["kind"]
        if kind == "step_prepared":
            if pending is not None or row["token_index"] != len(ids) or row["prefix_ids"] != ids:
                raise EvidenceError("historical prepared prefix differs")
            raw = bytes.fromhex(row["intended_raw_hex"])
            if sha(raw) != row["intended_raw_sha256"]:
                raise EvidenceError("historical rendered hash differs")
            pending = row
        elif kind == "step_committed":
            if (pending is None or row["prepared_sha256"] != pending["event_sha256"]
                    or row["token_id"] != pending["token_id"] or row["token_index"] != len(ids)
                    or row["raw_sha256"] != pending["intended_raw_sha256"]):
                raise EvidenceError("historical acknowledgement differs")
            ids.append(row["token_id"])
            committed_raw = bytes.fromhex(pending["intended_raw_hex"])
            pending = None
        elif kind == "known_response":
            known = row
    discrepancies = []
    if pending is not None:
        discrepancies.append("prepared_STEP_without_saved_acknowledgement")
    if known is None:
        discrepancies.append("no_saved_known_response_terminal")
    if result["committed_output_tokens"] != len(ids):
        discrepancies.append("result_counter_does_not_match_saved_commits")
    if result["generated_ids"] != ids:
        discrepancies.append("result_token_list_does_not_match_saved_commits")
    if files["response.raw"] != committed_raw:
        discrepancies.append("full_response_does_not_match_last_acknowledged_prefix")
    if not discrepancies:
        raise EvidenceError("negative-control fixture unexpectedly complete")
    return {
        "schema": "candidate-journal-guard-v2-legacy-negative-control",
        "status": "evidence_UNKNOWN",
        "status_scope": "New read-only evidence gate; original runtime outcome is not rewritten.",
        "source_path": str(RELATIVE), "source_sha256": PINS,
        "original_runtime_outcome": result["outcome"],
        "original_runtime_pending_step_unknown": result["pending_step_unknown"],
        "claimed_tokens": result["committed_output_tokens"], "saved_committed_tokens": len(ids),
        "saved_journal_rows": len(rows), "pending_token_index": pending["token_index"],
        "last_committed_response_bytes": len(committed_raw),
        "last_committed_response_sha256": sha(committed_raw),
        "saved_full_response_bytes": len(files["response.raw"]),
        "reason_codes": discrepancies, "accepted_as_complete": False,
        "missing_events_reconstructed": False, "runtime_outcome_reclassified": False,
        "source_files_changed": False, "model_tokenizer_encoder_decoder_calls": 0,
        "sampling_reaudited": False, "gap_origin": "unknown",
    }


if __name__ == "__main__":
    # Output is a new artifact chosen by the coordinator; existing files refused.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "claimed_tokens", "saved_committed_tokens", "accepted_as_complete")}))
