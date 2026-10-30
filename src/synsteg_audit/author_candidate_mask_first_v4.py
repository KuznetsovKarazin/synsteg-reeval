"""Pure proposed v4 support construction; no native runner or model imports.

The caller must pin the frozen schema parser, all 50 genuine top-k float32
logits and their contextual byte pieces. This component cannot attest those
resources. Whitespace measurements are passive: they never change eligibility.
Neither a nonempty support nor a positive mass certifies useful candidates.
"""
from __future__ import annotations

import math
import re
import struct

ID = "author-candidate-mask-first-v4"
TOTAL = 1 << 24
TEMPERATURE = 0.7
MAX_BYTES = 262144
JSON_WS = frozenset(b" \t\r\n")


def structural_whitespace(raw: bytes, piece: bytes) -> bool:
    """True only for an entirely whitespace suffix outside a JSON string.

    This is a lexical observation for already parser-validated prefixes; it
    deliberately does not interpret content words as semantic progress.
    """
    if type(raw) is not bytes or type(piece) is not bytes or not piece:
        raise ValueError("exact prefix and nonempty suffix bytes required")
    inside = escaped = False
    for byte in raw:
        if inside:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                inside = False
        elif byte == 34:
            inside = True
    return not inside and all(byte in JSON_WS for byte in piece)


def whitespace_streak(raw: bytes, piece: bytes, previous: int) -> int:
    if type(previous) is not int or previous < 0:
        raise ValueError("nonnegative integer streak required")
    return previous + 1 if structural_whitespace(raw, piece) else 0


def _validated(rows):
    if type(rows) is not list or len(rows) != 50:
        raise ValueError("all 50 ordered token/logit/piece rows required")
    seen, decoded = set(), []
    for row in rows:
        if (type(row) is not dict or set(row) != {"token_id", "logit_hex", "piece_hex"}
                or type(row["token_id"]) is not int
                or not 0 <= row["token_id"] < (1 << 31)
                or row["token_id"] in seen
                or type(row["logit_hex"]) is not str
                or re.fullmatch(r"[0-9a-f]{8}", row["logit_hex"]) is None
                or type(row["piece_hex"]) is not str
                or not 0 < len(row["piece_hex"]) <= 2 * (MAX_BYTES + 65536)
                or re.fullmatch(r"(?:[0-9a-f]{2})+", row["piece_hex"]) is None):
            raise ValueError("malformed or duplicate complete top50 row")
        value, = struct.unpack("!f", bytes.fromhex(row["logit_hex"]))
        if not math.isfinite(value):
            raise ValueError("nonfinite logit")
        seen.add(row["token_id"])
        decoded.append((value, row["token_id"], row.copy()))
    ordered = sorted(decoded, key=lambda x: (-x[0], x[1]))
    if decoded != ordered:
        raise ValueError("top50 order must be descending logit, then token ID")
    return ordered


def filter_then_quantize(raw, words, rows, classify_prefix, *, max_bytes=MAX_BYTES):
    """Mask all 50 pieces, center on eligible logits, allocate 2^24 masses.

    Any parser resource refusal refuses the whole probe. Ineligible pieces and
    zero-mass eligible pieces remain in the returned evidence. No fallback,
    widening, repair, per-token mass floor, token selection or STEP occurs.
    """
    if type(raw) is not bytes or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("exact bytes and bounded integer byte budget required")
    ordered = _validated(rows)
    prefix = classify_prefix(raw, words, max_bytes=max_bytes)
    if prefix["state"] == "resource_refusal":
        return {"schema": ID, "state": "resource_refusal", "reason": "prefix byte budget"}
    if prefix["state"] != "incomplete":
        raise ValueError("probe requires a syntactically extendable prefix")
    observations, eligible = [], []
    for value, token, row in ordered:
        piece = bytes.fromhex(row["piece_hex"])
        parsed = classify_prefix(raw + piece, words, max_bytes=max_bytes)
        state = parsed["state"]
        if state not in {"incomplete", "complete", "invalid_prefix", "resource_refusal"}:
            raise ValueError("unknown parser state")
        observation = {**row, "prefix_state": state, "weight": 0,
                       "structural_whitespace": structural_whitespace(raw, piece)}
        observations.append(observation)
        if state in {"incomplete", "complete"}:
            eligible.append((value, token, observation))
    if any(row["prefix_state"] == "resource_refusal" for row in observations):
        return {"schema": ID, "state": "resource_refusal", "reason": "candidate byte budget",
                "rows": observations}
    if not eligible:
        return {"schema": ID, "state": "empty_schema_support", "rows": observations,
                "integer_total": 0, "positive_tokens": 0}
    exp = [math.exp((value - eligible[0][0]) / TEMPERATURE) for value, _, _ in eligible]
    normalizer = math.fsum(exp)
    ideal = [x / normalizer * TOTAL for x in exp]
    masses = [math.floor(x) for x in ideal]
    remainder = TOTAL - sum(masses)
    if not 0 <= remainder <= len(masses):
        raise ValueError("quantization residual outside contract")
    order = sorted(range(len(masses)), key=lambda i: (-(ideal[i] - masses[i]), eligible[i][1]))
    for i in order[:remainder]:
        masses[i] += 1
    if sum(masses) != TOTAL or max(masses) < TOTAL // len(eligible):
        raise AssertionError("mass conservation or nonempty-support guarantee failed")
    for (_, _, row), mass in zip(eligible, masses):
        row["weight"] = mass
    positive = [row for row in observations if row["weight"] > 0]
    ws_mass = sum(row["weight"] for row in positive if row["structural_whitespace"])
    return {"schema": ID, "state": "ready", "rows": observations,
            "integer_total": TOTAL, "temperature": TEMPERATURE,
            "eligible_tokens": len(eligible), "positive_tokens": len(positive),
            "positive_support_structural_whitespace_only": all(row["structural_whitespace"] for row in positive),
            "structural_whitespace_mass": {"numerator": ws_mass, "denominator": TOTAL},
            "exponential_underflow_ids": [row["token_id"] for (_, _, row), x in zip(eligible, exp) if x == 0],
            "semantic_capacity_bits": None}
