"""Finite-channel seeded-pair experiment, not a production stegosystem.

The public envelope contains an explicit nonce. Its visibility is not hidden by
the token-channel argument. HMAC authenticates the payload; it does not encrypt
it. A deterministic public channel is required at both endpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import hmac
import json
import sys

VERSION = "seeded-pair-channel-v1"
DEMO_KEY = bytes(range(32))
PROFILES = {
    "uniform4": {"symbols": "ABCD", "initial": [1, 1, 1, 1]},
    "skew4": {"symbols": "ABCD", "initial": [13, 1, 1, 1]},
    "constant1": {"symbols": "A", "initial": [1]},
    "context4": {"symbols": "ABCD", "initial": [4, 3, 2, 1],
                 "after": {"A": [1, 1, 1, 1], "B": [13, 1, 1, 1],
                           "C": [1, 1, 1, 1], "D": [13, 1, 1, 1]}},
}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def subkey(key, label):
    if len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    return hmac.new(key, (VERSION + ":" + label).encode(), hashlib.sha256).digest()


def weights(profile, prefix):
    if prefix and "after" in profile:
        return profile["after"][prefix[-1]]
    return profile["initial"]


def weighted_choice(profile, prefix, draw256):
    """Exact integer-weight law when draw256 supplies independent uniform ints."""
    ws = weights(profile, prefix)
    total = sum(ws)
    limit = (1 << 256) - (1 << 256) % total
    while True:
        value = draw256()
        if not 0 <= value < 1 << 256:
            raise ValueError("draw outside uint256")
        if value < limit:
            value %= total
            for symbol, weight in zip(profile["symbols"], ws):
                if value < weight:
                    return symbol
                value -= weight


def pair(key, nonce_hex, profile_id, prefix, profile=None):
    profile = PROFILES[profile_id] if profile is None else profile
    k = subkey(key, "sample")
    values = []
    for branch in (0, 1):
        counter = 0

        def draw():
            nonlocal counter
            # Canonical array makes boundaries and domains unambiguous.
            value = int.from_bytes(hmac.new(k, canonical([
                VERSION, nonce_hex, profile_id, digest(profile), len(prefix),
                prefix, branch, counter]), hashlib.sha256).digest(), "big")
            counter += 1
            return value

        values.append(weighted_choice(profile, prefix, draw))
    return tuple(values)


def _metadata(envelope, profile):
    return {**{k: envelope[k] for k in ("version", "profile", "nonce_hex", "units")},
            "profile_sha256": digest(profile)}


def _tag(key, envelope, profile, body):
    data = canonical(_metadata(envelope, profile)) + b"\x00" + body
    return hmac.new(subkey(key, "frame-auth"), data, hashlib.sha256).digest()[:16]


def _bits(raw):
    return "".join(f"{byte:08b}" for byte in raw)


def _padding_bit(key, nonce_hex, index):
    block = hmac.new(subkey(key, "padding"),
                     canonical([nonce_hex, index // 256]), hashlib.sha256).digest()
    return (block[(index % 256) // 8] >> (7 - index % 8)) & 1


def encode(payload, key, nonce, profile_id, units):
    if len(nonce) != 16 or not isinstance(units, int) or units < 0:
        raise ValueError("nonce must be 16 bytes and units nonnegative integer")
    if len(payload) * 8 > 65535:
        raise ValueError("payload exceeds uint16 bit-length field")
    profile = PROFILES[profile_id]
    envelope = {"version": VERSION, "profile": profile_id,
                "nonce_hex": nonce.hex(), "units": units}
    body = (len(payload) * 8).to_bytes(2, "big") + payload
    frame = _bits(body + _tag(key, envelope, profile, body))
    prefix, cursor = "", 0
    for _ in range(units):
        a, b = pair(key, nonce.hex(), profile_id, prefix)
        if a == b:
            prefix += a
            continue
        bit = int(frame[cursor]) if cursor < len(frame) else _padding_bit(
            key, nonce.hex(), cursor - len(frame))
        prefix += (a, b)[bit]
        cursor += 1
    envelope["stego_text"] = " ".join(prefix)
    return {"envelope": envelope, "capacity_bits": cursor,
            "frame_bits": len(frame), "frame_completed": cursor >= len(frame),
            "samples_drawn": 2 * units, "payload_bits": len(payload) * 8,
            "body_overhead_bits": 144, "nonce_bits": 128}


def decode(envelope, key, profile_override=None):
    """Only public envelope/key/profiles enter decoding; no encoder trace."""
    try:
        if envelope["version"] != VERSION:
            return {"status": "unsupported_version"}
        profile_id = envelope["profile"]
        profile = PROFILES[profile_id] if profile_override is None else profile_override
        nonce = bytes.fromhex(envelope["nonce_hex"])
        units = envelope["units"]
        text = envelope["stego_text"]
        if len(nonce) != 16 or not isinstance(units, int) or units < 0:
            raise ValueError("invalid public metadata")
        tokens = text.split(" ") if text else []
        if len(tokens) != units or any(len(t) != 1 or t not in profile["symbols"] for t in tokens):
            return {"status": "invalid_text"}
        prefix, extracted = "", []
        for token in tokens:
            a, b = pair(key, envelope["nonce_hex"], profile_id, prefix, profile)
            if token != a and token != b:
                return {"status": "unrecognized_symbol", "decoded_bits": len(extracted)}
            if a != b:
                extracted.append("0" if token == a else "1")
            prefix += token
        bits = "".join(extracted)
        if len(bits) < 16:
            return {"status": "insufficient_capacity", "decoded_bits": len(bits)}
        bit_length = int(bits[:16], 2)
        if bit_length % 8:
            return {"status": "invalid_frame", "decoded_bits": len(bits)}
        required = 16 + bit_length + 128
        if len(bits) < required:
            return {"status": "insufficient_capacity", "decoded_bits": len(bits),
                    "required_frame_bits": required}
        frame = int(bits[:required], 2).to_bytes(required // 8, "big")
        if not hmac.compare_digest(frame[-16:], _tag(key, envelope, profile, frame[:-16])):
            return {"status": "authentication_failed", "decoded_bits": len(bits)}
        return {"status": "ok", "payload_hex": frame[2:-16].hex(),
                "decoded_bits": len(bits), "required_frame_bits": required}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "invalid_input", "detail": str(exc)}


def exact_output_law(profile_id, message_bits, units):
    """Enumerate ideal independent pair coins, with exact rational arithmetic.

    This checks a small finite instance; it is not a cryptographic theorem.
    message_bits must cover every possible consumed position.
    """
    if not 0 <= units <= 6 or len(message_bits) < units:
        raise ValueError("bounded enumeration requires units<=6 and padded bits")
    profile = PROFILES[profile_id]
    states = {("", 0): Fraction(1)}
    for _ in range(units):
        following = defaultdict(Fraction)
        for (prefix, cursor), mass in states.items():
            ws = weights(profile, prefix)
            total = sum(ws)
            for i, a in enumerate(profile["symbols"]):
                for j, b in enumerate(profile["symbols"]):
                    token = (a, b)[int(message_bits[cursor])]
                    following[(prefix + token, cursor + (a != b))] += (
                        mass * Fraction(ws[i] * ws[j], total * total))
        states = following
    marginal = defaultdict(Fraction)
    for (prefix, _), mass in states.items():
        marginal[prefix] += mass
    cover = {"": Fraction(1)}
    for _ in range(units):
        following = {}
        for prefix, mass in cover.items():
            ws = weights(profile, prefix)
            for symbol, weight in zip(profile["symbols"], ws):
                following[prefix + symbol] = mass * Fraction(weight, sum(ws))
        cover = following
    tv = sum(abs(marginal[s] - cover[s]) for s in cover) / 2
    return {"total_variation": str(tv), "output_strings": len(cover),
            "probability_mass": str(sum(marginal.values())),
            "expected_capacity_bits": str(sum(m * cursor for (_, cursor), m in states.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["decode"])
    parser.parse_args()
    item = json.load(sys.stdin)
    result = decode(item["envelope"], bytes.fromhex(item["key_hex"]))
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
