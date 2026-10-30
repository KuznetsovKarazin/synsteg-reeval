#!/usr/bin/env python3
"""Execute the frozen finite-channel protocol; never sample an LLM."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.seeded_pair_channel import (  # noqa: E402
    DEMO_KEY, PROFILES, VERSION, canonical, decode, encode, exact_output_law,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg-commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "seeded-pair-channel-v1")
    args = parser.parse_args()
    if len(args.prereg_commit) != 40 or any(c not in "0123456789abcdef" for c in args.prereg_commit):
        raise ValueError("full prerequisite Git commit SHA required")
    protocol_path = ROOT / "experiments" / "seeded_pair_channel_v1" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["expected_attempts"] != 1920 or protocol["experiment_id"] != VERSION:
        raise ValueError("unexpected protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    attempts_path = args.output / "attempts.jsonl"
    if attempts_path.exists():
        raise FileExistsError("refusing to overwrite an existing run")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cells = defaultdict(lambda: Counter())
    statuses, seen_nonces = Counter(), set()
    total, exact, completed, total_payload_recovered = 0, 0, 0, 0
    total_units, total_sample_calls = 0, 0
    started = time.monotonic()
    with attempts_path.open("x") as out:
        for profile_id, units, payload_bytes, replicate in itertools.product(
                protocol["profiles"], protocol["fixed_output_units"],
                protocol["payload_bytes"], range(protocol["replicates_per_cell"])):
            nonce = hashlib.sha256(canonical([
                VERSION, profile_id, units, payload_bytes, replicate])).digest()[:16]
            if nonce in seen_nonces:
                raise RuntimeError("fixture nonce collision")
            seen_nonces.add(nonce)
            payload = hashlib.sha256(f"{VERSION}:payload:{replicate}".encode()).digest()[:payload_bytes]
            sent = encode(payload, DEMO_KEY, nonce, profile_id, units)
            receiver_input = {"envelope": sent["envelope"], "key_hex": DEMO_KEY.hex()}
            proc = subprocess.run([sys.executable, "-m", "synsteg_audit.seeded_pair_channel", "decode"],
                                  input=canonical(receiver_input), capture_output=True, env=env)
            if proc.returncode:
                received = {"status": "receiver_process_error", "returncode": proc.returncode,
                            "stderr": proc.stderr.decode(errors="replace")}
            else:
                try:
                    received = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    received = {"status": "invalid_receiver_json"}
            correct = received.get("status") == "ok" and received.get("payload_hex") == payload.hex()
            record = {"attempt": total, "replicate": replicate, "payload_hex": payload.hex(),
                      **sent, "receiver": received, "authenticated_exact": correct,
                      "serialized_envelope_bytes": len(canonical(sent["envelope"]))}
            out.write(json.dumps(record, sort_keys=True) + "\n")
            total += 1
            exact += int(correct)
            completed += int(sent["frame_completed"])
            total_payload_recovered += payload_bytes * 8 * int(correct)
            total_units += units
            total_sample_calls += sent["samples_drawn"]
            statuses[received["status"]] += 1
            cell = cells[(profile_id, units, payload_bytes)]
            cell.update(attempts=1, authenticated_exact=int(correct),
                        completed=int(sent["frame_completed"]),
                        capacity_bits=sent["capacity_bits"], output_units=units,
                        payload_bits_recovered=payload_bytes * 8 * int(correct))
            if total % 160 == 0:
                print(json.dumps({"attempts": total, "expected": 1920}), flush=True)
    laws = [{"profile": profile, "message_bits": "".join(bits), "units": 4,
             **exact_output_law(profile, "".join(bits), 4)}
            for profile in PROFILES for bits in itertools.product("01", repeat=4)]
    fixture = encode(b"control", DEMO_KEY, bytes(16), "uniform4", 512)
    changed = dict(fixture["envelope"])
    changed["stego_text"] = "Z" + changed["stego_text"][1:]
    controls = {
        "same_nonce_exact_repeat": fixture == encode(b"control", DEMO_KEY, bytes(16), "uniform4", 512),
        "different_nonce_different_text": fixture["envelope"]["stego_text"] != encode(
            b"control", DEMO_KEY, bytes([1]) * 16, "uniform4", 512)["envelope"]["stego_text"],
        "wrong_key_status": decode(fixture["envelope"], bytes([7]) * 32)["status"],
        "profile_drift_status": decode(fixture["envelope"], DEMO_KEY,
            {"symbols": "ABCD", "initial": [4, 3, 2, 1]})["status"],
        "text_mutation_status": decode(changed, DEMO_KEY)["status"],
        "exact_law_cases": len(laws),
        "exact_law_zero_tv": sum(x["total_variation"] == "0" for x in laws),
    }
    code_paths = [Path(__file__), ROOT / "src/synsteg_audit/seeded_pair_channel.py",
                  ROOT / "tests/test_seeded_pair_channel.py", protocol_path]
    summary = {
        "experiment_id": VERSION, "prereg_commit": args.prereg_commit,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "attempts": total, "expected_attempts": 1920,
        "frame_completed": completed, "authenticated_exact": exact,
        "receiver_fresh_processes": total, "statuses": dict(statuses),
        "output_units": total_units, "sender_categorical_sample_calls": total_sample_calls,
        "receiver_categorical_sample_calls": total_sample_calls,
        "payload_bits_recovered": total_payload_recovered,
        "sender_calls_per_recovered_payload_bit": total_sample_calls / total_payload_recovered,
        "body_overhead_bits": 144, "nonce_overhead_bits": 128,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "controls": controls,
        "cells": [{"profile": k[0], "units": k[1], "payload_bytes": k[2], **dict(v),
                   "capacity_bits_per_unit": v["capacity_bits"] / v["output_units"]}
                  for k, v in cells.items()],
        "limitations": ["Finite channel only; no LLM used.", "Demonstration key is public.",
                        "Authenticated frame is not encrypted.", "Public nonce envelope is not claimed covert.",
                        "Fixed-length output law concerns all texts, not only successful payloads.",
                        "Exact enumerations assume ideal independent sampler coins; no full HMAC system security theorem."],
    }
    for name, obj in [("summary.json", summary), ("exact_laws.json", laws)]:
        (args.output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("attempts", "authenticated_exact", "statuses", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
