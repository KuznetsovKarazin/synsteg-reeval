"""Payload delivery and communication costs with explicit denominators.

These metrics inspect supplied values; they do not execute a decoder. Empty
requests remain visible but never count as successful nonempty deliveries.
"""
from __future__ import annotations

import json
import zlib
from typing import Any


def _bits(value: str, name: str) -> None:
    if not isinstance(value, str) or set(value) - {"0", "1"}:
        raise ValueError(f"{name} must be a binary string")


def serialized_key_cost(key: Any) -> dict[str, int]:
    """Compact UTF-8 JSON bytes, preserving order, and zlib level-9 bytes.

    This is a declared serialization convention, not a lower bound on the
    information in a key and not the cost of a shared, reusable key schedule.
    """
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return {"key_json_bytes": len(raw),
            "key_zlib9_bytes": len(zlib.compress(raw, level=9))}


def delivery_metrics(
    requested: str,
    decoded: str | None,
    *,
    used_bits: int,
    cover: str,
    stego: str,
    key: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Evaluate one attempt without counting a correct prefix as a delivery.

    Word counts use Python whitespace splitting. ``used_bits`` is the actual
    encoder count, which must not exceed the requested count. No metric silently
    substitutes requested bits for used or correctly recovered bits. BER is
    defined only for nonempty, equal-length bit strings; missing/extra bits are
    reported separately. ``None`` means unavailable or undefined, never zero.
    """
    _bits(requested, "requested")
    if decoded is not None:
        _bits(decoded, "decoded")
    if type(used_bits) is not int or not 0 <= used_bits <= len(requested):
        raise ValueError("used_bits must be an integer between 0 and requested length")
    if not isinstance(cover, str) or not isinstance(stego, str):
        raise TypeError("cover and stego must be strings")
    if error is not None and not isinstance(error, str):
        raise TypeError("error must be a string or None")
    n = len(requested)
    cw, sw = len(cover.split()), len(stego.split())
    nd = None if decoded is None else len(decoded)
    full = None if decoded is None else decoded == requested
    prefix = None if decoded is None else requested.startswith(decoded)
    correct_prefix = 0
    if decoded is not None:
        for a, b in zip(requested, decoded):
            if a != b:
                break
            correct_prefix += 1
    success = bool(n and full and used_bits == n and error is None)
    ratio = lambda x, den: None if x is None or den == 0 else x / den
    result = {
        "requested_bits": n, "used_bits": used_bits, "decoded_bits": nd,
        "payload_eligible": bool(n), "full_exact": full,
        "full_message_success": success, "prefix_exact": prefix,
        "nonempty_prefix_exact": bool(decoded and prefix),
        "correct_leading_bits": None if decoded is None else correct_prefix,
        "missing_bits": None if nd is None else max(0, n - nd),
        "extra_bits": None if nd is None else max(0, nd - n),
        "truncated": None if nd is None else nd < n,
        "bit_error_rate": (sum(a != b for a, b in zip(requested, decoded)) / n
                           if n and nd == n else None),
        "cover_words": cw, "stego_words": sw,
        "requested_bpw_cover": ratio(n, cw),
        "used_bpw_cover": ratio(used_bits, cw),
        "used_bpw_stego": ratio(used_bits, sw),
        "decoded_bpw_stego": ratio(nd, sw),
        "correct_leading_bpw_stego": ratio(None if decoded is None else correct_prefix, sw),
        "full_message_goodput_bpw_stego": ratio(n if success else 0, sw),
        "error": error,
    }
    if key is not None:
        result.update(serialized_key_cost(key))
    return result
