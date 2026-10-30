#!/usr/bin/env python3
"""Independent audit of the frozen seeded LLM channel; no production imports.

Re-derives quantization, HMAC sampling and payload framing from saved endpoint
evidence. Does not run neural inference or claim an independent implementation
of llama.cpp, its tokenizer, or the full-vocabulary top-k operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import hmac
import json
import math
from pathlib import Path
import struct

VERSION = "seeded-pair-llm-v1"
TOTAL = 1 << 24


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_for(key, domain):
    assert isinstance(key, bytes) and len(key) == 32
    return hmac.digest(key, (VERSION + ":" + domain).encode(), "sha256")


def independent_weights(rows, temperature=.8):
    """Reference float32 decode and largest-remainder categorical law."""
    assert len(rows) == 50 and len({r["token_id"] for r in rows}) == 50
    values = []
    for r in rows:
        assert type(r["token_id"]) is int and r["token_id"] >= 0
        assert len(r["logit_hex"]) == 8
        value, = struct.unpack("!f", bytes.fromhex(r["logit_hex"]))
        assert math.isfinite(value)
        values.append((r["token_id"], r["logit_hex"], value))
    values.sort(key=lambda item: (-item[2], item[0]))
    maximum = values[0][2]
    unnormalized = [math.exp((v - maximum) / temperature) for _, _, v in values]
    total_mass = math.fsum(unnormalized)
    targets = [p / total_mass * TOTAL for p in unnormalized]
    counts = [int(math.floor(p)) for p in targets]
    residual = TOTAL - sum(counts)
    assert 0 <= residual <= len(values)
    ranked = sorted(range(len(values)),
                    key=lambda i: (-(targets[i] - counts[i]), values[i][0]))
    for index in ranked[:residual]:
        counts[index] += 1
    expected = [{"token_id": token, "logit_hex": raw, "weight": count}
                for (token, raw, _), count in zip(values, counts)]
    assert rows == expected, "quantization/CDF ordering mismatch"
    assert sum(counts) == TOTAL
    return expected


def independent_draw(weights, key, context, domain, branch):
    # TOTAL=2**24 divides 2**256, hence this protocol never rejects a draw.
    assert sum(row["weight"] for row in weights) == TOTAL
    word = hmac.digest(key_for(key, domain),
                       canonical([context, branch, 0]), "sha256")
    value = int.from_bytes(word, "big") % TOTAL
    cumulative = 0
    for row in weights:
        cumulative += row["weight"]
        if value < cumulative:
            return row["token_id"]
    raise AssertionError("CDF does not cover draw")


def context(envelope, step, prompt_ids, prefix):
    return [VERSION, envelope["protocol_sha256"], envelope["resources_sha256"],
            envelope["prompt_id"], envelope["nonce_hex"], step,
            sha(prompt_ids + prefix)]


def expected_frame(payload, key, envelope):
    body = (8 * len(payload)).to_bytes(2, "big") + payload
    mac = hmac.digest(key_for(key, "frame-auth"),
                      canonical(envelope) + b"\0" + body, "sha256")[:16]
    return "".join(format(byte, "08b") for byte in body + mac)


def independent_frame(bits, key, envelope):
    n = len(bits)
    if n < 16:
        return {"status": "insufficient_capacity", "decoded_bits": n}
    length = int(bits[:16], 2)
    if length % 8:
        return {"status": "invalid_frame", "decoded_bits": n}
    required = 16 + length + 128
    if n < required:
        return {"status": "insufficient_capacity", "decoded_bits": n,
                "required_frame_bits": required}
    frame = bytes(int(bits[i:i+8], 2) for i in range(0, required, 8))
    mac = hmac.digest(key_for(key, "frame-auth"),
                      canonical(envelope) + b"\0" + frame[:-16], "sha256")[:16]
    if not hmac.compare_digest(mac, frame[-16:]):
        return {"status": "authentication_failed", "decoded_bits": n}
    return {"status": "ok", "decoded_bits": n, "payload_hex": frame[2:-16].hex(),
            "required_frame_bits": required}


def padding(key, envelope, index):
    block = hmac.digest(key_for(key, "padding"),
                        canonical([envelope, index // 256]), "sha256")
    binary = "".join(format(x, "08b") for x in block)
    return int(binary[index % 256])


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_sender(trace, result, protocol):
    envelope = result["envelope"]
    key = bytes.fromhex(protocol["fixture_key_hex"])
    prompt_ids = result["prompt_token_ids"]
    prefix = []
    capacity = 0
    control_key = hashlib.sha256((VERSION + ":ordinary-control").encode()).digest()
    payload = hashlib.sha256((VERSION + ":payload:" + envelope["prompt_id"]).encode()).digest()[:16]
    frame = expected_frame(payload, key, envelope)
    assert len(frame) == protocol["frame_bits"] == 272
    raw = bytearray()
    for i, row in enumerate(trace):
        assert row["step"] == i
        weights = independent_weights(row["weights"], protocol["sampler"]["temperature"])
        assert row["weights_sha256"] == sha(weights)
        ctx = context(envelope, i, prompt_ids, prefix)
        assert row["prefix_sha256"] == ctx[-1]
        if envelope["arm"] == "stego":
            choices = [independent_draw(weights, key, ctx, "sample", branch)
                       for branch in (0, 1)]
            assert row["payload_cursor_before"] == capacity
            bit = None
            if choices[0] == choices[1]:
                selected = choices[0]
            else:
                bit = int(frame[capacity]) if capacity < len(frame) else padding(
                    key, envelope, capacity - len(frame))
                selected = choices[bit]
                capacity += 1
            assert row["capacity_bits_after"] == capacity
            assert row["carried_bit"] == bit
            q = 1 - sum(x["weight"] ** 2 for x in weights) / TOTAL**2
            assert math.isclose(row["raw_distinct_pair_probability"], q,
                                rel_tol=0, abs_tol=1e-15)
        else:
            choices = [independent_draw(weights, control_key, ctx, "ordinary-sample", 0)]
            selected = choices[0]
            assert row["carried_bit"] is None
        assert row["candidates"] == choices
        assert row["draw_counters"] == [1] * len(choices)
        assert row["selected_token"] == selected
        prefix.append(selected)
        raw.extend(bytes.fromhex(row["token_bytes_hex"]))
    assert result.get("generated_token_ids", prefix) == prefix
    if result.get("generated_tokens") is not None:
        assert result["generated_tokens"] == len(prefix)
    if result.get("raw_sha256") is not None:
        assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
        assert result["raw_bytes"] == len(raw)
    if result.get("capacity_bits") is not None:
        assert result["capacity_bits"] == capacity
        assert result["frame_completed"] == (capacity >= 272)
    return {"tokens": len(prefix), "capacity_bits": capacity,
            "raw_bytes": bytes(raw), "token_ids": prefix}


def audit_receiver(trace, result, envelope, protocol, key=None):
    key = bytes.fromhex(protocol["fixture_key_hex"]) if key is None else key
    prompt_ids = result["prompt_token_ids"]
    observed = result["retokenized_token_ids"]
    if len(observed) != envelope["units"]:
        assert not trace
        return {"status": "token_count_mismatch"}
    prefix, bits = [], []
    for i, row in enumerate(trace):
        assert row["step"] == i
        weights = independent_weights(row["weights"], protocol["sampler"]["temperature"])
        ctx = context(envelope, i, prompt_ids, prefix)
        assert row["weights_sha256"] == sha(weights)
        assert row["prefix_sha256"] == ctx[-1]
        pair = [independent_draw(weights, key, ctx, "sample", b) for b in (0, 1)]
        token = observed[i]
        if key == bytes.fromhex(protocol["fixture_key_hex"]):
            assert row["candidates"] == pair
            assert row["draw_counters"] == [1, 1]
            assert row["observed_token"] == token
        if token not in pair:
            return {"status": "unrecognized_token", "failure_step": i,
                    "decoded_bits": len(bits)}
        bit = None if pair[0] == pair[1] else int(token == pair[1])
        if bit is not None:
            bits.append(str(bit))
        if key == bytes.fromhex(protocol["fixture_key_hex"]):
            assert row["decoded_bit"] == bit
            assert row["decoded_bits_after"] == len(bits)
        prefix.append(token)
    assert len(trace) == len(observed), "receiver stopped without explicit refusal"
    return independent_frame("".join(bits), key, envelope)



def load_json(path):
    return json.loads(Path(path).read_text())


def audit_run(run_path, root):
    protocol_path = root / "experiments/seeded_pair_llm_v1/protocol.json"
    protocol = load_json(protocol_path)
    manifest = load_json(root / "experiments/seeded_pair_llm_v1/execution_manifest.json")
    execution = load_json(run_path / "execution.json")
    summary = load_json(run_path / "summary.json")
    assert protocol["experiment_id"] == VERSION
    assert protocol["output_tokens"] == 512 and protocol["payload_bytes"] == 16
    assert protocol["frame_bits"] == 272 and protocol["frame_overhead_bits"] == 144
    assert protocol["nonce_external_bits"] == 128
    assert protocol["prompt_ids"] == ["station", "library"]
    assert protocol["arms"] == ["stego", "ordinary"]
    assert protocol["fixture_key_hex"] == bytes(range(32)).hex()
    assert protocol["sampler"]["top_k"] == 50
    assert protocol["sampler"]["temperature"] == .8
    assert protocol["sampler"]["probability_integer_total"] == TOTAL
    assert protocol["model"]["sha256"] == "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
    assert protocol["runtime"]["archive_sha256"] == "ac75208179d00d86f378d5f10c65f9785028ce877cf57279f1a7b8dac54a25ff"
    assert execution["protocol_sha256"] == file_sha(protocol_path)
    assert execution["execution_manifest_sha256"] == file_sha(
        root / "experiments/seeded_pair_llm_v1/execution_manifest.json")
    assert execution["resources_sha256"] == manifest["resources_sha256"]
    assert sha(manifest["channel_resources"]) == manifest["resources_sha256"]
    assert manifest["channel_resources"]["runtime_parameters"] == protocol["runtime"]
    assert len(execution["freeze_commit"]) == 40
    for path, expected in manifest["frozen_files"].items():
        assert file_sha(root / path) == expected, ("frozen source changed", path)
    for path, info in manifest["channel_resources"]["files"].items():
        if path.endswith(protocol["model"]["filename"]):
            assert info["sha256"] == protocol["model"]["sha256"]
            assert info["bytes"] == protocol["model"]["bytes"]
    expected_plan = [{"prompt_id": p, "arm": arm, "status": "not_attempted"}
                     for p in protocol["prompt_ids"] for arm in protocol["arms"]]
    assert load_json(run_path / "plan.json") == expected_plan
    assert len(summary["ledger"]) == 4
    assert [(r["prompt_id"], r["arm"]) for r in summary["ledger"]] == [
        (r["prompt_id"], r["arm"]) for r in expected_plan]
    per_row = []
    sender_statuses, receiver_statuses = Counter(), Counter()
    totals = Counter()
    nonce_set = set()
    key = bytes.fromhex(protocol["fixture_key_hex"])
    for row in summary["ledger"]:
        name = row["prompt_id"] + "-" + row["arm"]
        assert load_json(run_path / (name + ".outcome.json")) == row
        request = load_json(run_path / (name + ".sender-input.json"))
        expected_keys = {"envelope", "key_hex"} | (
            {"payload_hex"} if row["arm"] == "stego" else set())
        assert set(request) == expected_keys
        assert request["key_hex"] == protocol["fixture_key_hex"]
        envelope = {
            "version": VERSION, "prompt_id": row["prompt_id"], "arm": row["arm"],
            "units": 512, "protocol_sha256": file_sha(protocol_path),
            "resources_sha256": manifest["resources_sha256"],
            "nonce_hex": hashlib.sha256(canonical(
                [VERSION, row["prompt_id"], row["arm"]])).digest()[:16].hex(),
        }
        assert request["envelope"] == envelope
        assert envelope["nonce_hex"] not in nonce_set
        nonce_set.add(envelope["nonce_hex"])
        payload = hashlib.sha256((VERSION + ":payload:" + row["prompt_id"]).encode()).digest()[:16]
        if row["arm"] == "stego":
            assert request["payload_hex"] == payload.hex()
        send_dir = run_path / name
        sent = load_json(send_dir / "result.json")
        sent_trace = read_jsonl(send_dir / "trace.jsonl")
        sender_statuses[sent["status"]] += 1
        totals["sender_trace_steps"] += len(sent_trace)
        assert len(sent_trace) <= 512
        assert row["generated_tokens"] == len(sent_trace)
        case = {"prompt_id": row["prompt_id"], "arm": row["arm"],
                "sender_status": sent["status"], "sender_trace_steps": len(sent_trace)}
        if "prompt_token_ids" in sent:
            checked = audit_sender(sent_trace, sent, protocol)
            assert checked["raw_bytes"] == (send_dir / "output.raw").read_bytes()
            totals["independently_checked_sender_steps"] += len(sent_trace)
            assert sent["prompt_tokens"] == len(sent["prompt_token_ids"])
            assert sent["envelope"] == envelope
            assert sent["sender_categorical_draws"] == len(sent_trace) * (
                2 if row["arm"] == "stego" else 1)
            if sent.get("stego_text") is not None:
                assert sent["stego_text"].encode("utf-8") == checked["raw_bytes"]
                assert (send_dir / "output.txt").read_text() == sent["stego_text"]
                assert sent["text_roundtrip_exact"] == (
                    sent["canonical_text_token_ids"] == sent["generated_token_ids"])
                totals["utf8_outputs"] += 1
                totals["canonical_outputs"] += int(sent["text_roundtrip_exact"])
            else:
                try:
                    checked["raw_bytes"].decode("utf-8", "strict")
                except UnicodeDecodeError:
                    assert sent["status"] == "invalid_utf8"
                else:
                    raise AssertionError("UTF8 output omitted without a decoding failure")
            case["capacity_bits"] = checked["capacity_bits"] if row["arm"] == "stego" else None
            if row["arm"] == "stego":
                assert row["capacity_bits"] == checked["capacity_bits"]
        else:
            assert sent["status"] in ("endpoint_error", "process_timeout", "process_error")
            case["partial_sender_crypto_audit_unavailable"] = True
        assert row["status"] == sent["status"]
        if len(sent_trace) == 512:
            totals["completed_512_token_texts"] += 1
        if row["arm"] == "stego":
            if row["receiver_process_attempted"]:
                totals["receiver_processes"] += 1
                recv_input = load_json(run_path / (name + ".receiver-input.json"))
                assert set(recv_input) == {"envelope", "key_hex", "stego_text"}
                assert recv_input == {"envelope": envelope, "key_hex": protocol["fixture_key_hex"],
                                      "stego_text": sent["stego_text"]}
                recv_dir = run_path / (name + "-receiver")
                received = load_json(recv_dir / "result.json")
                trace = read_jsonl(recv_dir / "trace.jsonl")
                receiver_statuses[received["status"]] += 1
                totals["receiver_trace_steps"] += len(trace)
                assert row["receiver_status"] == received["status"]
                assert row["receiver_trace_steps"] == len(trace)
                case["receiver_status"] = received["status"]
                if "prompt_token_ids" in received:
                    assert received["prompt_token_ids"] == sent["prompt_token_ids"]
                    assert received["retokenized_token_ids"] == sent["canonical_text_token_ids"]
                    independently = audit_receiver(trace, received, envelope, protocol)
                    for field, value in independently.items():
                        assert received[field] == value, (name, field)
                    totals["independently_checked_receiver_steps"] += len(trace)
                    exact = independently.get("status") == "ok" and independently.get("payload_hex") == payload.hex()
                    assert row["full_exact_delivery"] == exact
                    totals["exact_deliveries"] += exact
                    if trace:
                        negative = audit_receiver(trace, received, envelope, protocol, bytes([7]) * 32)
                        assert negative["status"] != "ok", "wrong-key control authenticated"
                        case["wrong_key_reference_status"] = negative["status"]
                    common = min(len(sent_trace), len(trace))
                    mismatches = [i for i in range(common) if (
                        sent_trace[i]["weights_sha256"] != trace[i]["weights_sha256"] or
                        sent_trace[i]["prefix_sha256"] != trace[i]["prefix_sha256"])]
                    totals["paired_trace_steps"] += common
                    totals["paired_trace_mismatches"] += len(mismatches)
                    comparison = row["trace_comparison"]
                    assert comparison["compared_steps"] == common
                    assert comparison["all_compared_weights_identical"] == (not mismatches)
                    assert comparison["first_mismatch"] == (mismatches[0] if mismatches else None)
                    case["paired_trace_steps"] = common
                    case["paired_trace_mismatches"] = len(mismatches)
                else:
                    assert received["status"] in ("endpoint_error", "process_timeout", "process_error")
                    assert not row["full_exact_delivery"]
                    case["partial_receiver_crypto_audit_unavailable"] = True
            else:
                assert sent.get("stego_text") is None
                assert row["receiver_status"] == "not_attempted_no_valid_UTF8_text"
                assert not row["full_exact_delivery"]
                receiver_statuses[row["receiver_status"]] += 1
        else:
            assert not row["receiver_process_attempted"] and not row["full_exact_delivery"]
        per_row.append(case)
    assert summary["planned_generated_texts"] == 4
    assert summary["attempted_generated_texts"] == 4
    assert summary["planned_stego_deliveries"] == 2
    assert summary["actual_generated_tokens"] == totals["sender_trace_steps"]
    assert summary["completed_512_token_texts"] == totals["completed_512_token_texts"]
    assert summary["full_exact_deliveries"] == totals["exact_deliveries"]
    assert summary["receiver_processes"] == totals["receiver_processes"]
    assert summary["receiver_trace_steps"] == totals["receiver_trace_steps"]
    assert summary["sender_statuses"] == dict(sender_statuses)
    assert summary["receiver_statuses"] == dict(receiver_statuses)
    assert summary["paid_calls"] == 0 and summary["closed_test_opened"] is False
    assert summary["semantic_or_security_validated"] is False
    outputs = load_json(run_path / "output_manifest.json")
    for name, expected in outputs.items():
        assert file_sha(run_path / name) == expected, ("output identity", name)
    return {
        "schema": "seeded-pair-llm-independent-audit-v1", "status": "passed",
        "planned_generated_texts": 4, "planned_stego_deliveries": 2,
        **dict(totals), "unique_public_nonces": len(nonce_set),
        "sender_statuses": dict(sender_statuses), "receiver_statuses": dict(receiver_statuses),
        "rows": per_row, "output_files_verified": len(outputs),
        "protocol_sha256": file_sha(protocol_path), "freeze_commit": execution["freeze_commit"],
        "audit_script_sha256": file_sha(Path(__file__)),
        "production_channel_imports": 0, "new_neural_inference": 0,
        "limitations": [
            "Independent arithmetic/framing audit uses saved native top50 logits; not an independent full-vocabulary LLM forward implementation.",
            "Tokenizer outputs and token-byte pieces are corroborated across endpoints, not a separate tokenizer implementation.",
            "Finite fixed-budget constrained-channel pilot with two synthetic public prompts does not validate factuality or steganographic security.",
            "Sender and receiver run in separate processes but are not filesystem-sandboxed from one another."
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve earlier audit")
    result = audit_run(args.run, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
