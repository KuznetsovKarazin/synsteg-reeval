"""Executable synthetic profile for public-v1 interface/isolation checks.

This ASCII 8-symbol backend has no model and produces no natural language.
Native adapters use the same public core but validate their own real resources.
Receiver inputs are explicit public files plus wire and a 32-byte key file.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from . import framed_interval_public_v1 as core
from .canonical_channel_v2 import CanonicalGate, ContractError, PublicCodec
from .framed_interval_public_v1_nonce import reserve_nonce, reserve_random_nonce

PROFILE = {"schema": "public-v1-synthetic-ascii-eight-v1", "symbols": "abcdefgh",
           "weights": [1]*8, "prompt": "ignored-after-public-binding", "context_tokens": 2048}


def synthetic_endpoint():
    digest = lambda label: hashlib.sha256(core.canonical_json({"profile": PROFILE, "component": label})).hexdigest()
    def encode(text):
        if any(c not in "abcdefgh" for c in text): raise ValueError("outside synthetic alphabet")
        return tuple(ord(c)-97 for c in text)
    def render(ids):
        if any(type(t) is not int or not 0 <= t < 8 for t in ids): raise ValueError("outside synthetic alphabet")
        return bytes(t+97 for t in ids)
    codec = PublicCodec(encode, render, frozenset(), digest("codec"))
    return core.Endpoint(CanonicalGate(codec), resource_id="public-v1-synthetic-ascii-eight",
        resource_digest=digest("resources"), provider_digest=digest("integer-law"),
        distribution_callback=lambda ids: core.Distribution(tuple((t,1) for t in range(8))),
        context_tokens=2048)


def read_json(path):
    return core.load_public_json(Path(path).read_bytes())


def write(path, raw):
    with Path(path).open("xb") as output:
        output.write(raw); output.flush(); os.fsync(output.fileno())


def write_json(path, obj):
    write(path, core.canonical_json(obj) + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare = sub.add_parser("prepare-synthetic"); prepare.add_argument("--output", required=True)
    for role in ("send", "receive"):
        command = sub.add_parser(role)
        for field in ("spec", "context", "key-file", "output"):
            command.add_argument("--" + field, required=True)
        if role == "send":
            command.add_argument("--payload-file", required=True)
            command.add_argument("--nonce-ledger", required=True)
            command.add_argument("--nonce-hex", help="reproducible tests only; still reserved")
        else:
            command.add_argument("--wire", required=True); command.add_argument("--envelope", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)  # No endpoint retry into existing slot.
    if args.operation == "prepare-synthetic":
        write_json(destination / "public-spec.json", core.make_public_spec(synthetic_endpoint()))
        write_json(destination / "public-context.json", core.make_public_context((), context_id="synthetic-empty-v1"))
        return 0
    endpoint = synthetic_endpoint()
    spec, context, key = read_json(args.spec), read_json(args.context), Path(args.key_file).read_bytes()
    if type(key) is not bytes or len(key) != 32: raise ContractError("key file must contain32 raw bytes")
    if args.operation == "send":
        payload = Path(args.payload_file).read_bytes()
        if len(payload) > core.PAYLOAD_SLOT: raise ContractError("payload exceeds16 bytes")
        core.validate_public_spec(spec, endpoint); core.validate_public_context(context, spec)
        if args.nonce_hex is None:
            nonce, reservation = reserve_random_nonce(args.nonce_ledger, key)
        else:
            nonce = bytes.fromhex(args.nonce_hex)
            nonce, reservation = nonce, reserve_nonce(args.nonce_ledger, key, nonce)
        envelope = core.make_envelope(spec, context, nonce)
        write_json(destination / "nonce-reservation.json", reservation)
        write_json(destination / "envelope.json", envelope)
        result = core.generate(key, envelope, endpoint, public_spec=spec, public_context=context, payload=payload)
        write(destination / "wire.bin", result.wire)
        report = asdict(result); report.pop("wire"); report.pop("traces")
        report.update(execution_kind="synthetic_no_model", costs=core.metadata_costs(envelope, spec, context,
            wire_bytes=len(result.wire), payload_bytes=len(payload)))
    else:
        envelope, wire = read_json(args.envelope), Path(args.wire).read_bytes()
        result = core.receive(wire, key, envelope, endpoint, public_spec=spec, public_context=context)
        report = asdict(result); report.pop("payload"); report.pop("traces")
        report["execution_kind"] = "synthetic_no_model"
        if result.authenticated_delivery: write(destination / "payload.bin", result.payload)
    write_json(destination / "result.json", report)
    return 0 if (result.authenticated_delivery if args.operation == "receive" else result.status == "threshold_pending_receiver") else 2


if __name__ == "__main__":
    raise SystemExit(main())
