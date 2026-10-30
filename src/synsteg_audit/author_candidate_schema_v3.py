"""Pure schema-constrained token support; no model, tokenizer, or codec calls.

This new component conditions an explicitly supplied integer-mass distribution.
It cannot certify candidate meanings, restore lost probability mass, or run a
native experiment. An all-abstention response is a format success only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata

ID = "author-candidate-schema-v3"
MAX_RAW_BYTES = 262144
MAX_SUPPORT = 65536
MAX_MASS = 1 << 24
MAX_DRAW_BLOCKS = 128
FORBIDDEN = frozenset({"Cc", "Cf", "Zl", "Zp"})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def surface(value):
    return " ".join(value.split()).casefold()


def validate_words(words):
    if (type(words) not in (list, tuple) or not 1 <= len(words) <= 5
            or any(type(w) is not str or not w or any(c.isspace() for c in w)
                   or any(unicodedata.category(c) in FORBIDDEN for c in w)
                   for w in words)):
        raise ValueError("one to five exact nonempty source tokens required")
    for word in words:
        word.encode("utf-8")
    return tuple(words)


class _Incomplete(Exception):
    pass


class _Invalid(Exception):
    pass


def _partial_codepoints(tail):
    """All scalar codepoints consistent with a decoder-validated UTF-8 tail.

    Range construction does not replace decoder validation. The tail has one
    incomplete sequence, never a completed invalid byte or UTF-8 surrogate.
    """
    first = tail[0]
    width = 2 if first < 0xE0 else 3 if first < 0xF0 else 4
    prefix = first & ((1 << (7 - width)) - 1)
    for byte in tail[1:]:
        prefix = (prefix << 6) | (byte & 63)
    missing = width - len(tail)
    low = prefix << (6 * missing)
    high = ((prefix + 1) << (6 * missing)) - 1
    low = max(low, {2: 0x80, 3: 0x800, 4: 0x10000}[width])
    high = min(high, 0x10FFFF)
    for value in range(low, high + 1):
        if not 0xD800 <= value <= 0xDFFF:
            yield chr(value)


class _Schema:
    def __init__(self, text, tail, words):
        self.text = text
        self.tail = tail
        self.words = words
        self.i = 0

    def peek(self):
        if self.i == len(self.text):
            if self.tail:
                raise _Invalid("non-ASCII UTF-8 fragment in a structural or escape position")
            raise _Incomplete("extendable schema prefix")
        return self.text[self.i]

    def take(self, expected):
        if self.peek() != expected:
            raise _Invalid("unexpected schema delimiter")
        self.i += 1

    def ws(self):
        while self.i < len(self.text) and self.text[self.i] in " \t\r\n":
            self.i += 1

    def hex4(self, permitted):
        value = ""
        for _ in range(4):
            c = self.peek()
            if c not in "0123456789abcdefABCDEF":
                raise _Invalid("invalid Unicode escape")
            self.i += 1
            value += c
            low = int(value, 16) << (4 * (4 - len(value)))
            high = low + (1 << (4 * (4 - len(value))))
            if not any(permitted(point) for point in range(low, high)):
                raise _Invalid("Unicode escape prefix has no legal schema completion")
        return int(value, 16)

    def string(self, expected=None, seen=None):
        self.take('"')
        value = ""

        def admissible(character):
            if expected is not None:
                return expected.startswith(value + character)
            return (unicodedata.category(character) not in FORBIDDEN
                    and (bool(value) or not character.isspace()))

        while True:
            if self.i == len(self.text) and self.tail:
                if expected is not None:
                    remaining = expected[len(value):]
                    possible = bool(remaining) and remaining[0].encode("utf-8").startswith(self.tail)
                else:
                    possible = any(admissible(c) for c in _partial_codepoints(self.tail))
                if possible:
                    raise _Incomplete("partial UTF-8 sequence has a legal schema completion")
                raise _Invalid("partial UTF-8 sequence has no legal schema completion")
            c = self.peek()
            self.i += 1
            if c == '"':
                if expected is not None:
                    if value != expected:
                        raise _Invalid("source or words key differs")
                elif (not value or value.strip() != value or surface(value) in seen):
                    raise _Invalid("empty, padded, self, or duplicate candidate")
                return value
            if ord(c) < 32:
                raise _Invalid("unescaped control in JSON string")
            if c == "\\":
                if expected is not None and len(value) == len(expected):
                    raise _Invalid("key is already complete; only closing quote is allowed")
                c = self.peek()
                self.i += 1
                if c == "u":
                    target = ord(expected[len(value)]) if expected is not None else None
                    target_first = (0xD800 + ((target - 0x10000) >> 10)
                                    if target is not None and target > 0xFFFF else target)
                    def allowed_first(point):
                        if target is not None:
                            return point == target_first
                        if 0xD800 <= point <= 0xDBFF:
                            return True  # At least one legal scalar completion exists.
                        return not 0xDC00 <= point <= 0xDFFF and admissible(chr(point))
                    point = self.hex4(allowed_first)
                    if 0xD800 <= point <= 0xDBFF:
                        self.take("\\")
                        self.take("u")
                        def allowed_low(low):
                            if not 0xDC00 <= low <= 0xDFFF:
                                return False
                            scalar = 0x10000 + ((point - 0xD800) << 10) + low - 0xDC00
                            return admissible(chr(scalar))
                        low = self.hex4(allowed_low)
                        if not 0xDC00 <= low <= 0xDFFF:
                            raise _Invalid("high surrogate requires a low surrogate")
                        point = 0x10000 + ((point - 0xD800) << 10) + low - 0xDC00
                    elif 0xDC00 <= point <= 0xDFFF:
                        raise _Invalid("unpaired low surrogate")
                    c = chr(point)
                elif c in {'"', "\\", "/"}:
                    pass
                elif c in "bfnrt":
                    c = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}[c]
                else:
                    raise _Invalid("invalid JSON escape")
            if not admissible(c):
                raise _Invalid("wrong key prefix or forbidden candidate character")
            value += c

    def candidates(self, source):
        self.take("[")
        self.ws()
        if self.peek() == "]":
            self.i += 1
            return
        seen = {surface(source)}
        for index in range(8):
            if index:
                self.take(",")
                self.ws()
            value = self.string(seen=seen)
            seen.add(surface(value))
            self.ws()
        self.take("]")

    def parse(self):
        self.ws()
        self.take("{")
        self.ws()
        self.string(expected="words")
        self.ws()
        self.take(":")
        self.ws()
        self.take("[")
        self.ws()
        for index, word in enumerate(self.words):
            if index:
                self.take(",")
                self.ws()
            self.take("{")
            self.ws()
            self.string(expected=word)
            self.ws()
            self.take(":")
            self.ws()
            self.candidates(word)
            self.ws()
            self.take("}")
            self.ws()
        self.take("]")
        self.ws()
        self.take("}")
        self.ws()
        if self.i != len(self.text) or self.tail:
            raise _Invalid("trailing bytes after complete schema object")


def classify_prefix(raw, expected_words, *, max_bytes=MAX_RAW_BYTES):
    """Classify exact raw bytes against the entire ordered response schema.

    Resource refusal says nothing about JSON validity. An incomplete response
    may still run out of token/byte budget or of model token support later.
    """
    words = validate_words(expected_words)
    if type(raw) is not bytes:
        raise TypeError("exact bytes required")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_RAW_BYTES:
        raise ValueError("invalid byte budget")
    if len(raw) > max_bytes:
        return {"state": "resource_refusal", "reason": "raw byte limit", "byte_count": len(raw)}
    tail = b""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(raw):
            return {"state": "invalid_prefix", "reason": "invalid UTF-8", "byte_count": len(raw)}
        text = raw[:exc.start].decode("utf-8")
        tail = raw[exc.start:]
    parser = _Schema(text, tail, words)
    try:
        parser.parse()
    except _Incomplete as exc:
        state, reason = "incomplete", str(exc)
    except _Invalid as exc:
        state, reason = "invalid_prefix", str(exc)
    else:
        state, reason = "complete", "complete exact ordered schema object"
    return {"state": state, "reason": reason, "byte_count": len(raw)}


def _rows(rows, max_support):
    if type(max_support) is not int or not 1 <= max_support <= MAX_SUPPORT:
        raise ValueError("invalid support budget")
    if type(rows) is not list:
        raise ValueError("support must be an explicit list")
    if len(rows) > max_support:
        return None
    ids = set()
    for row in rows:
        if (type(row) is not dict or set(row) != {"token_id", "piece", "mass"}
                or type(row["token_id"]) is not int or not 0 <= row["token_id"] < (1 << 31)
                or row["token_id"] in ids or type(row["piece"]) is not bytes or not row["piece"]
                or type(row["mass"]) is not int or not 1 <= row["mass"] <= MAX_MASS):
            raise ValueError("malformed token id, nonempty piece, or positive integer mass")
        ids.add(row["token_id"])
    return sorted(rows, key=lambda row: row["token_id"])


def constrain_support(raw, expected_words, rows, *, max_bytes=MAX_RAW_BYTES, max_support=MAX_SUPPORT):
    """Condition a supplied discrete law on byte-prefix eligibility.

    Caller must exclude EOS/special tokens, bind genuine tokenizer pieces and
    a complete frozen base law. This pure component cannot verify those facts.
    Any resource-limited candidate refuses the whole step instead of secretly
    treating a parser limit as a grammatical rejection.
    """
    parsed = classify_prefix(raw, expected_words, max_bytes=max_bytes)
    ordered = _rows(rows, max_support)
    if ordered is None:
        return {"state": "resource_refusal", "reason": "support cardinality limit"}
    if parsed["state"] != "incomplete":
        return {"state": parsed["state"], "reason": parsed["reason"]}
    accepted, rejected = [], []
    for row in ordered:
        if len(row["piece"]) > max_bytes - len(raw):
            return {"state": "resource_refusal", "reason": "candidate exceeds parser byte budget",
                    "token_id": row["token_id"]}
        classification = classify_prefix(raw + row["piece"], expected_words, max_bytes=max_bytes)
        if classification["state"] == "resource_refusal":
            return {"state": "resource_refusal", "reason": "candidate exceeds parser byte budget",
                    "token_id": row["token_id"]}
        if classification["state"] in {"incomplete", "complete"}:
            accepted.append({"token_id": row["token_id"], "piece_hex": row["piece"].hex(),
                             "mass": row["mass"]})
        else:
            rejected.append({"token_id": row["token_id"], "reason": classification["reason"]})
    retained = sum(row["mass"] for row in accepted)
    input_mass = sum(row["mass"] for row in ordered)
    result = {"schema": ID + "-conditional-law", "state": "ready" if accepted else "empty_support",
              "prefix_sha256": sha(raw), "words_sha256": sha(canonical(list(expected_words))),
              "input_mass": input_mass, "retained_mass": retained,
              "retained_fraction": ({"numerator": retained, "denominator": input_mass}
                                    if input_mass else None),
              "accepted": accepted, "rejected": rejected,
              "base_law_sha256": sha(canonical([{**r, "piece": r["piece"].hex()} for r in ordered]))}
    for row in accepted:
        row["conditional_numerator"] = row["mass"]
        row["conditional_denominator"] = retained
    result["law_sha256"] = sha(canonical(result))
    return result


def select_token(support, key, request_sha256, attempt, step):
    """Bounded deterministic HMAC rejection draw; never search keys or seeds.

    Exact rational conditioning is retained: no second integer quantization.
    Uniformity is the ideal random-PRF interpretation, not a claim that a fixed
    key is random or that a schema-conditioned distribution is undetectable.
    """
    if (type(key) is not bytes or len(key) != 32 or type(request_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
            or type(attempt) is not int or not 0 <= attempt < 3
            or type(step) is not int or not 0 <= step < 512):
        raise ValueError("invalid fixed draw context")
    if type(support) is not dict or support.get("state") != "ready":
        raise ValueError("a ready conditional law is required")
    fields = {"schema", "state", "prefix_sha256", "words_sha256", "input_mass", "retained_mass",
              "retained_fraction", "accepted", "rejected", "base_law_sha256", "law_sha256"}
    if (set(support) != fields or support["schema"] != ID + "-conditional-law"
            or any(type(support[name]) is not str or re.fullmatch(r"[0-9a-f]{64}", support[name]) is None
                   for name in ("prefix_sha256", "words_sha256", "base_law_sha256", "law_sha256"))
            or type(support["input_mass"]) is not int
            or not 1 <= support["input_mass"] <= MAX_SUPPORT * MAX_MASS
            or type(support["retained_mass"]) is not int
            or not 1 <= support["retained_mass"] <= support["input_mass"]):
        raise ValueError("invalid conditional law metadata")
    fraction = support["retained_fraction"]
    if (type(fraction) is not dict or set(fraction) != {"numerator", "denominator"}
            or type(fraction["numerator"]) is not int or fraction["numerator"] != support["retained_mass"]
            or type(fraction["denominator"]) is not int or fraction["denominator"] != support["input_mass"]):
        raise ValueError("invalid retained fraction")
    claimed = support.get("law_sha256")
    body = {k: v for k, v in support.items() if k != "law_sha256"}
    if type(claimed) is not str or sha(canonical(body)) != claimed:
        raise ValueError("conditional law digest mismatch")
    accepted = support.get("accepted")
    rejected = support.get("rejected")
    total = support.get("retained_mass")
    if (type(accepted) is not list or not 1 <= len(accepted) <= MAX_SUPPORT
            or type(total) is not int or not 1 <= total <= MAX_SUPPORT * MAX_MASS
            or any(type(r) is not dict or set(r) != {"token_id", "piece_hex", "mass", "conditional_numerator", "conditional_denominator"}
                   or type(r["token_id"]) is not int or not 0 <= r["token_id"] < (1 << 31)
                   or type(r["mass"]) is not int or not 1 <= r["mass"] <= MAX_MASS
                   or type(r["conditional_numerator"]) is not int or r["conditional_numerator"] != r["mass"]
                   or type(r["conditional_denominator"]) is not int or r["conditional_denominator"] != total
                   or type(r["piece_hex"]) is not str or len(r["piece_hex"]) > MAX_RAW_BYTES * 2
                   or re.fullmatch(r"(?:[0-9a-f]{2})+", r["piece_hex"]) is None
                   for r in accepted)
            or sum(r["mass"] for r in accepted) != total
            or [r["token_id"] for r in accepted] != sorted({r["token_id"] for r in accepted})):
        raise ValueError("invalid conditional support structure")
    if (type(rejected) is not list or len(rejected) + len(accepted) > MAX_SUPPORT
            or any(type(r) is not dict or set(r) != {"token_id", "reason"}
                   or type(r["token_id"]) is not int or not 0 <= r["token_id"] < (1 << 31)
                   or type(r["reason"]) is not str or not r["reason"] for r in rejected)
            or [r["token_id"] for r in rejected] != sorted({r["token_id"] for r in rejected})
            or {r["token_id"] for r in rejected} & {r["token_id"] for r in accepted}
            or (not rejected and support["input_mass"] != total)
            or (rejected and not len(rejected) <= support["input_mass"] - total <= len(rejected) * MAX_MASS)):
        raise ValueError("invalid rejected support metadata")
    cutoff = ((1 << 256) // total) * total
    for counter in range(MAX_DRAW_BLOCKS):
        message = canonical({"domain": ID + "/draw-v1", "request_sha256": request_sha256,
                             "attempt": attempt, "step": step, "law_sha256": claimed,
                             "counter": counter})
        block = int.from_bytes(hmac.new(key, message, hashlib.sha256).digest(), "big")
        if block >= cutoff:
            continue
        ticket = block % total
        cumulative = 0
        for row in accepted:
            cumulative += row["mass"]
            if ticket < cumulative:
                return {"state": "selected", "token_id": row["token_id"], "piece_hex": row["piece_hex"],
                        "ticket": ticket, "counter": counter, "law_sha256": claimed}
        raise AssertionError("validated support has no selected interval")
    return {"state": "resource_refusal", "reason": "HMAC draw block limit", "law_sha256": claimed}
