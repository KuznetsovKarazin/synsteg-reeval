#!/usr/bin/env python3
"""Posthoc arithmetic of saved bounded native evidence; no NN or decoder calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "results/framed-interval-bounded-v1/native-pilot-v1"
AUDIT = "results/framed-interval-bounded-v1-native-audit/independent-audit.json"
PROCESS_CHECK = "results/framed-interval-bounded-v1-native-audit/process-completion-check.json"
PROCESS = "results/framed-interval-bounded-v1-process/native-pilot-v1/process.json"
QUALITY = "results/framed-interval-bounded-v1-quality-review/native-pilot-v1.json"
PROTOCOL = "experiments/framed-interval-bounded-v1/public-protocol.json"
ACTIVATION = "experiments/framed-interval-bounded-v1/activation-v1.json"
ARTIFACT_INDEX = "results/framed-interval-bounded-v1-artifacts/index.json"
OUTPUT = "results/framed-interval-bounded-v1-native-metrics"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def record(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def stats(values):
    values = list(values)
    return {"n": len(values), "sum": sum(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "min": min(values), "max": max(values)}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
        "value": numerator/denominator if denominator else None}


def collect(root):
    root = Path(root).resolve()
    inputs, cache = {}, {}
    def read(path, *, json_value=True):
        if path not in cache:
            raw = (root/path).read_bytes()
            inputs[path] = record(raw)
            cache[path] = raw
        raw = cache[path]
        return json.loads(raw) if json_value else raw
    audit = read(AUDIT)
    if audit.get("status") != "passed" or audit.get("failures") != []:
        raise ValueError("failed/incomplete independent audit cannot authorize numerator")
    process_check = read(PROCESS_CHECK)
    if process_check.get("status") != "passed" or process_check.get("all24_worker_exit_zero") is not True:
        raise ValueError("complete24 worker process evidence required")
    protocol, activation = read(PROTOCOL), read(ACTIVATION)
    ledger, process = read(CAMPAIGN+"/campaign-status.json"), read(PROCESS)
    manifest = read(CAMPAIGN+"/campaign-manifest.json")
    index = read(ARTIFACT_INDEX)
    restore = read("results/framed-interval-bounded-v1-artifacts/fresh-restore.json")
    expected = tuple(run["run_id"] for run in protocol["runs"])
    if (len(expected) != 16 or tuple(r["run_id"] for r in ledger["cases"]) != expected
            or manifest["public_runs"] != protocol["runs"]
            or ledger["state"] != "finished_all_planned_outcomes"
            or ledger["authenticated_delivery_denominator"] != 8):
        raise ValueError("complete fixed16+8 denominator differs")
    audited = {(row["run_id"], row["role"]): row for row in audit["endpoints"]}
    if len(audited) != 24:
        raise ValueError("independent audit endpoint set differs")
    quality = read(QUALITY) if (root/QUALITY).exists() else None
    quality_cases = {row["run_id"]: row for row in quality["cases"]} if quality else {}
    if quality and (set(quality_cases) != set(expected) or quality["excluded_cases"] != []
            or quality["method"]["human_gold"] is not False
            or quality["campaign_status"]["sha256"] != inputs[CAMPAIGN+"/campaign-status.json"]["sha256"]):
        raise ValueError("quality review denominator/evidence differs")
    rows, keys = [], set()
    for run, case in zip(protocol["runs"], ledger["cases"], strict=True):
        rid = run["run_id"]
        base = CAMPAIGN+"/"+rid+"/sender/"
        worker, rec, request = read(base+"worker-result.json"), read(base+"reconciliation.json"), read(base+"request.json")
        envelope = read(base+"envelope.json")
        envelope_raw = read(base+"envelope.json", json_value=False)
        wire = read(base+"committed-wire.bin", json_value=False)
        core, verified = worker["core"], audited[(rid, "sender")]
        if (worker != case["sender"]["worker"] or worker["status"] != "returned"
                or case["sender"]["process"]["exit_code"] != 0
                or verified["wire_sha256"] != sha(wire)
                or core["wire_sha256"] != sha(wire)
                or core["wire_bytes"] != len(wire)
                or verified["committed_tokens"] != core["emitted_tokens"]
                or rec["known_committed_tokens"] != core["emitted_tokens"]
                or not verified["legal_public_stop"] or rec["unknown_native_step"]
                or core["consumed_bits"] != core["resolved_bits"] + core["pending_bits"] + 32):
            raise ValueError("saved sender identities/counters differ: "+rid)
        if envelope["protocol_sha256"] != inputs[ACTIVATION]["sha256"]:
            raise ValueError("envelope activation binding differs")
        keys.add(request["key_hex"])
        text = wire.decode("utf-8", "strict")
        payload_bytes, receiver_tokens, receiver_wall = None, 0, None
        success = None
        if run["arm"] == "stego":
            rbase = CAMPAIGN+"/"+rid+"/receiver/"
            received = read(rbase+"worker-result.json")
            rrec = read(rbase+"reconciliation.json")
            rv = audited[(rid, "receiver")]
            success = (case["receiver"]["authenticated_delivery"] is True
                and case["receiver"]["exact_expected_payload"] is True
                and rv["authenticated_delivery_verified"] is True
                and received["core"]["authenticated_delivery"] is True)
            if not success or received != case["receiver"]["worker"] or rrec["unknown_native_step"]:
                raise ValueError("saved audited delivery differs: "+rid)
            payload_bytes = len(bytes.fromhex(received["core"]["payload_hex"]))
            receiver_tokens = received["core"]["processed_tokens"]
            receiver_wall = case["receiver"]["process"]["wall_seconds"]
            if receiver_tokens != core["emitted_tokens"] or payload_bytes != 16:
                raise ValueError("saved payload/replay size differs")
        q = quality_cases.get(rid)
        if q and (q["wire_sha256"] != sha(wire) or q["wire_text"] != text):
            raise ValueError("quality wire identity differs: "+rid)
        labels = dict(Counter(obs["label"] for obs in q["source_support_observations"])) if q else None
        row = {"run_id": rid, "prompt_id": run["prompt_id"], "config_id": run["config_id"], "arm": run["arm"],
            "sender_worker_status": worker["status"], "sender_core_status": core["status"],
            "sender_tokens": core["emitted_tokens"], "sender_whitespace_words": len(text.split()),
            "wire_bytes": len(wire), "wire_sha256": sha(wire),
            "envelope_saved_json_bytes": len(envelope_raw),
            "envelope_canonical_json_bytes": len(canonical(envelope)),
            "wire_plus_canonical_envelope_bytes": len(wire)+len(canonical(envelope)),
            "resolved_bits": core["resolved_bits"], "consumed_bits": core["consumed_bits"],
            "pending_bits": core["pending_bits"], "lookahead_buffer_bits": 32,
            "first_threshold_token": core["first_threshold_token"],
            "frame_bits": envelope["frame_bits"], "token_cap": envelope["token_cap"],
            "delivered_payload_bytes": payload_bytes, "authenticated_exact_delivery": success,
            "receiver_token_steps": receiver_tokens,
            "sender_wall_seconds": case["sender"]["process"]["wall_seconds"],
            "receiver_wall_seconds": receiver_wall,
            "useful_bits_per_sender_token": None if payload_bytes is None else ratio(8*payload_bytes, core["emitted_tokens"]),
            "useful_bits_per_wire_byte": None if payload_bytes is None else ratio(8*payload_bytes, len(wire)),
            "quality_source_labels_descriptive": labels}
        rows.append(row)
    stego = [row for row in rows if row["arm"] == "stego"]
    ordinary = [row for row in rows if row["arm"] == "ordinary"]
    delivered = sum(row["delivered_payload_bytes"] for row in stego)
    if (delivered != 128 or audit["authenticated_deliveries_independently_verified"] != 8
            or audit["usable_delivery_numerator"] != 8):
        raise ValueError("complete useful-payload denominator differs")
    paired = []
    measures = ("sender_tokens", "sender_whitespace_words", "wire_bytes",
        "envelope_saved_json_bytes", "envelope_canonical_json_bytes",
        "wire_plus_canonical_envelope_bytes")
    for s, o in zip(stego, ordinary, strict=True):
        if (s["prompt_id"], s["config_id"]) != (o["prompt_id"], o["config_id"]):
            raise ValueError("paired fixed order differs")
        paired.append({"prompt_id": s["prompt_id"], "config_id": s["config_id"],
            "stego_run": s["run_id"], "ordinary_run": o["run_id"],
            **{name: {"stego": s[name], "ordinary": o[name], "stego_minus_ordinary": s[name]-o[name]} for name in measures}})
    descriptive = {arm: {name: stats(row[name] for row in group)
        for name in (*measures, "resolved_bits", "pending_bits", "sender_wall_seconds")}
        for arm, group in (("all16", rows), ("stego8", stego), ("ordinary8", ordinary))}
    delta = {name: stats(pair[name]["stego_minus_ordinary"] for pair in paired) for name in measures}
    all_tokens = sum(row["sender_tokens"] for row in rows)
    stego_tokens = sum(row["sender_tokens"] for row in stego)
    receiver_tokens = sum(row["receiver_token_steps"] for row in rows)
    wire_bytes = sum(row["wire_bytes"] for row in stego)
    canonical_bytes = sum(row["envelope_canonical_json_bytes"] for row in stego)
    if all_tokens != audit["sender_committed_tokens"] or receiver_tokens != audit["receiver_committed_tokens"]:
        raise ValueError("audited aggregate token counters differ")
    totals = {"planned_senders": 16, "planned_stego_receivers": 8,
        "authenticated_exact_payloads": 8, "delivery_denominator": 8,
        "payload_bytes_per_delivery": 16, "delivered_payload_bytes": delivered,
        "delivered_payload_bits": 8*delivered,
        "all_sender_tokens": all_tokens, "stego_sender_tokens": stego_tokens,
        "ordinary_sender_tokens": all_tokens-stego_tokens, "receiver_token_steps": receiver_tokens,
        "all_committed_token_STEP_requests": all_tokens+receiver_tokens,
        "PROMPT_requests": 24, "lower_level_graph_forward_invocations": None,
        "stego_wire_bytes": wire_bytes, "stego_saved_envelope_bytes": sum(r["envelope_saved_json_bytes"] for r in stego),
        "stego_canonical_envelope_bytes": canonical_bytes,
        "stego_wire_plus_canonical_envelope_bytes": wire_bytes+canonical_bytes,
        "useful_bits_per_stego_sender_token": ratio(8*delivered, stego_tokens),
        "useful_bits_per_stego_wire_byte": ratio(8*delivered, wire_bytes),
        "useful_payload_byte_fraction_of_stego_wire": ratio(delivered, wire_bytes),
        "useful_payload_byte_fraction_of_wire_plus_canonical_envelope": ratio(delivered, wire_bytes+canonical_bytes),
        "distinct_fixture_keys": len(keys), "bytes_per_shared_key": 32, "distinct_fixture_key_bytes": 32*len(keys),
        "nonce_bytes_per_run": 12, "nonce_bytes_all16": 16*12,
        "nonce_cost_note": "Nonce is already represented inside measured JSON envelope; do not add it twice. Shared key setup is separate.",
        "model_file_bytes": protocol["model"]["bytes"], "shared_runtime_and_other_resources_bytes": None,
        "process_exit_code": process["exit_code"], "campaign_wall_seconds": process["wall_seconds"],
        "worker_exit_zero_count": 24, "independent_audit_checks": audit["passed_checks"],
        "primary_files": index["primary_file_count"], "archive_bundles": len(index["bundles"]),
        "fresh_restore_exact_files": restore["verified_primary_files"]}
    # Every read primary is compared with the independent audit / packaged map.
    for path, value in inputs.items():
        if path in audit["evidence"] and value != audit["evidence"][path]:
            raise ValueError("independent audit input identity differs: "+path)
        if path in index["primary_files"] and value != index["primary_files"][path]:
            raise ValueError("packaged primary identity differs: "+path)
    summary = {"schema": "framed-interval-bounded-v1-native-posthoc-metrics-v1", "status": "passed",
        "method": "Deterministic arithmetic over saved audited JSON and exact UTF8 wires. No NN/tokenizer/decoder invocation.",
        "totals": totals, "descriptive_statistics": descriptive, "paired_stego_minus_ordinary": delta,
        "quality": {"status": "available_descriptive_agent_review" if quality else "pending",
            "human_gold": False, "validated_error_rate": None,
            "summary": quality["summary"] if quality else None},
        "deployment_binding_limitation": {
            "envelope_protocol_digest_is_activation_sha256": True,
            "activation_sha256": inputs[ACTIVATION]["sha256"],
            "activation_binds_sender_fixture_hash": activation["sender_fixtures"],
            "message_independent_public_binding": False,
            "arbitrary_message_deployment_established": False},
        "causal_comparison_to_fixed512_pilot_permitted": False,
        "causal_limit": "New keys/nonces, method domains, activation digest and stop rule;8/8 versus earlier1/2 is not an isolated causal improvement estimate.",
        "model_calls": 0, "tokenizer_calls": 0, "decoder_calls": 0}
    return summary, rows, paired, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    summary, rows, paired, inputs = collect(ROOT)
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"summary.json": summary, "rows.json": rows, "paired-costs.json": paired,
        "inputs.json": {"schema": "bounded-native-metrics-inputs-v1", "files": inputs}}
    for name, value in artifacts.items():
        with (output/name).open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
    status = {"schema": "bounded-native-metrics-status-v1", "status": "completed",
        "sender_rows": 16, "paired_rows": 8, "input_files": len(inputs),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "outputs": {name: record((output/name).read_bytes()) for name in artifacts},
        "model_calls": 0, "tokenizer_calls": 0, "decoder_calls": 0}
    (output/"status.json").write_text(json.dumps(status, indent=2, sort_keys=True)+"\n")
    print(json.dumps(summary["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
