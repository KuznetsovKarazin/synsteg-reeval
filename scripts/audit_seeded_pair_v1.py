#!/usr/bin/env python3
"""Independent finite-channel ledger/receiver audit; no production imports.

This audit reconstructs sampler inputs and frame parsing from the frozen
protocol. It is an implementation cross-check, not a security proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import hmac
import itertools
import json
from pathlib import Path

VERSION = "seeded-pair-channel-v1"
KEY = bytes(range(32))
PROFILES = {
    "uniform4": {"symbols": "ABCD", "initial": [1, 1, 1, 1]},
    "skew4": {"symbols": "ABCD", "initial": [13, 1, 1, 1]},
    "constant1": {"symbols": "A", "initial": [1]},
    "context4": {"symbols": "ABCD", "initial": [4, 3, 2, 1],
                 "after": {"A": [1, 1, 1, 1], "B": [13, 1, 1, 1],
                           "C": [1, 1, 1, 1], "D": [13, 1, 1, 1]}},
}


def serialized(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("ascii")


def fingerprint(x):
    return hashlib.sha256(serialized(x)).hexdigest()


def mac(k, x):
    return hmac.new(k, x, hashlib.sha256).digest()


def independent_receive(e):
    """Public envelope and fixed key only: no encoder counters or payload."""
    assert set(e) == {"version", "profile", "nonce_hex", "units", "stego_text"}
    assert e["version"] == VERSION and len(bytes.fromhex(e["nonce_hex"])) == 16
    profile = PROFILES[e["profile"]]
    tokens = e["stego_text"].split(" ")
    assert len(tokens) == e["units"]
    samplekey = mac(KEY, (VERSION + ":sample").encode())
    prefix, bits = "", []
    for offset, symbol in enumerate(tokens):
        ws = profile.get("after", {}).get(prefix[-1:] or None, profile["initial"])
        total = sum(ws)
        threshold = (2**256 // total) * total
        alternatives = []
        for branch in range(2):
            counter = 0
            while True:
                domain = [VERSION, e["nonce_hex"], e["profile"], fingerprint(profile),
                          offset, prefix, branch, counter]
                n = int.from_bytes(mac(samplekey, serialized(domain)), "big")
                counter += 1
                if n < threshold:
                    position = n % total
                    cumulative = 0
                    for char, weight in zip(profile["symbols"], ws):
                        cumulative += weight
                        if position < cumulative:
                            alternatives.append(char)
                            break
                    break
        assert symbol in alternatives, (offset, symbol, alternatives)
        if alternatives[0] != alternatives[1]:
            bits.append(str(alternatives.index(symbol)))
        prefix += symbol
    capacity = len(bits)
    if capacity < 16:
        return {"status": "insufficient_capacity", "capacity": capacity}
    bit_length = int("".join(bits[:16]), 2)
    required = bit_length + 144
    assert bit_length % 8 == 0
    if capacity < required:
        return {"status": "insufficient_capacity", "capacity": capacity}
    frame = bytes(int("".join(bits[i:i + 8]), 2) for i in range(0, required, 8))
    metadata = {k: e[k] for k in ("version", "profile", "nonce_hex", "units")}
    metadata["profile_sha256"] = fingerprint(profile)
    expected = mac(mac(KEY, (VERSION + ":frame-auth").encode()),
                   serialized(metadata) + b"\0" + frame[:-16])[:16]
    assert hmac.compare_digest(expected, frame[-16:])
    return {"status": "ok", "capacity": capacity, "payload_hex": frame[2:-16].hex()}


def exact_one_step_checks():
    # Independent algebraic enumeration for both next bits and every distinct
    # profile weight vector. Collision outputs count in the original marginal.
    vectors = [(1,), (1, 1, 1, 1), (13, 1, 1, 1), (4, 3, 2, 1)]
    rows = []
    for ws in vectors:
        ps = [Fraction(w, sum(ws)) for w in ws]
        for bit in (0, 1):
            output = [Fraction(0) for _ in ws]
            for i, j in itertools.product(range(len(ws)), repeat=2):
                output[(i, j)[bit]] += ps[i] * ps[j]
            assert output == ps
        q = 1 - sum(p * p for p in ps)
        row = {"weights": list(ws), "distinct_pair_probability": str(q),
               "both_bit_output_marginals_equal_cover": True}
        if q:
            conditioned = [p * (1 - p) / q for p in ps]
            row["success_conditioned_one_unit_distribution"] = [str(p) for p in conditioned]
            row["success_conditioned_total_variation"] = str(sum(abs(a - b) for a, b in zip(ps, conditioned)) / 2)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    attempts = [json.loads(line) for line in (args.results / "attempts.jsonl").read_text().splitlines()]
    summary = json.loads((args.results / "summary.json").read_text())
    root = Path(__file__).resolve().parents[1]
    protocol_bytes = (root / "experiments/seeded_pair_channel_v1/protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["experiment_id"] == VERSION and protocol["expected_attempts"] == 1920
    assert protocol["profiles"] == list(PROFILES)
    assert protocol["fixed_output_units"] == [256, 512, 1024]
    assert protocol["payload_bytes"] == [0, 1, 4, 16, 32]
    assert protocol["replicates_per_cell"] == 32 and protocol["key_hex"] == KEY.hex()
    assert protocol["body_overhead_bits"] == 144
    assert protocol["transmitted_nonce_bits"] == 128
    assert protocol["minimum_total_overhead_bits"] == 272
    assert summary["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert summary["prereg_commit"] == "c3fd4fad402bfdfc0e23fde9931d8b9d706c73ff"
    for path, expected_sha in summary["code_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected_sha, path
    expected_cells = list(itertools.product(PROFILES, [256, 512, 1024], [0, 1, 4, 16, 32], range(32)))
    assert len(attempts) == len(expected_cells) == 1920
    cells, statuses, nonces = Counter(), Counter(), set()
    total_units = exact = completed = bits_recovered = zero_entropy = 0
    for i, (row, (profile, units, size, replicate)) in enumerate(zip(attempts, expected_cells)):
        assert row["attempt"] == i and row["replicate"] == replicate
        envelope = row["envelope"]
        assert (envelope["profile"], envelope["units"]) == (profile, units)
        nonce = hashlib.sha256(serialized([VERSION, profile, units, size, replicate])).digest()[:16].hex()
        assert nonce == envelope["nonce_hex"] and nonce not in nonces
        nonces.add(nonce)
        payload = hashlib.sha256(f"{VERSION}:payload:{replicate}".encode()).digest()[:size].hex()
        assert row["payload_hex"] == payload and row["payload_bits"] == size * 8
        actual = independent_receive(envelope)
        assert actual["status"] == row["receiver"]["status"]
        assert actual["capacity"] == row["capacity_bits"] == row["receiver"]["decoded_bits"]
        assert row["frame_bits"] == size * 8 + 144
        assert row["frame_completed"] == (actual["capacity"] >= row["frame_bits"])
        assert row["samples_drawn"] == 2 * units
        assert row["serialized_envelope_bytes"] == len(serialized(envelope))
        assert row["body_overhead_bits"] == 144 and row["nonce_bits"] == 128
        recovered = actual["status"] == "ok"
        assert recovered == row["authenticated_exact"] == row["frame_completed"]
        if recovered:
            assert actual["payload_hex"] == payload == row["receiver"]["payload_hex"]
        if profile == "constant1":
            assert actual["capacity"] == 0 and not recovered
            zero_entropy += 1
        cells[(profile, units, size)] += 1
        statuses[actual["status"]] += 1
        exact += int(recovered)
        completed += int(row["frame_completed"])
        bits_recovered += size * 8 * int(recovered)
        total_units += units
    assert len(cells) == 60 and set(cells.values()) == {32}
    assert dict(statuses) == summary["statuses"]
    assert exact == summary["authenticated_exact"] and completed == summary["frame_completed"]
    assert bits_recovered == summary["payload_bits_recovered"]
    assert total_units == summary["output_units"]
    assert 2 * total_units == summary["sender_categorical_sample_calls"] == summary["receiver_categorical_sample_calls"]
    assert zero_entropy == 480 and len(nonces) == 1920
    result = {
        "status": "passed", "independent_receiver_implementations": 1,
        "production_codec_imports": 0, "attempts_verified": len(attempts),
        "cells_verified": len(cells), "replicates_per_cell": 32,
        "authenticated_exact": exact, "statuses": dict(statuses),
        "nonce_duplicates": 0, "zero_entropy_refusals_retained": zero_entropy,
        "protocol_parameters_and_recorded_code_hashes_match": True,
        "one_step_exact_algebra": exact_one_step_checks(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [args.results / "attempts.jsonl", args.results / "summary.json"]},
        "limits": ["Finite fixed-length channel only; no LLM evidence or practical security proof.",
                   "HMAC payload authentication does not authenticate every unused padding spelling.",
                   "Success-only conditioning can change the token distribution; explicit example is recorded.",
                   "The nonce and profile envelope is public metadata, not claimed covert.",
                   "Independent decode reuses the frozen design and HMAC primitive, not production code."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "attempts_verified", "authenticated_exact", "statuses")}))


if __name__ == "__main__":
    main()
