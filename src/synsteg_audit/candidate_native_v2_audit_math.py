"""Independent offline replay of the frozen native-v2 format and sampling law.

Only standard-library imports. This is a second implementation, with an
iterative JSON recognizer and an arithmetic IEEE-754 binary32 decoder. It does
not invoke the production parser, quantizer, sampler, tokenizer or a model.
Replay verifies recorded support/law consistency, not the provenance or global
top-50 completeness of recorded logits. HMAC seeds here are public diagnostics,
not evidence of cryptographic secrecy. Format success certifies no semantics.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata

DOMAIN = "author-candidate-local-native-v2"
MAX_BYTES = 262144
MAX_DEPTH = 32
TOTAL = 16777216
REQUEST_HASHES = (
    "15f413e60449231fe505a45091d838f36155d15307bcdf16b27d1215ffa05104",
    "d9f3597b716fb4fbb888a83ceccd881f94635537132ac5dc00c0748ddd14b4f4",
)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


class _End(Exception):
    pass


class _Bad(Exception):
    pass


class _Limit(Exception):
    pass


class _Cursor:
    """Lexical scanner only; container grammar lives in an explicit work stack."""
    def __init__(self, text):
        self.text, self.pos = text, 0

    def space(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \r\n\t":
            self.pos += 1

    def char(self):
        if self.pos == len(self.text):
            raise _End
        return self.text[self.pos]

    def exact(self, character):
        if self.char() != character:
            raise _Bad
        self.pos += 1

    def string(self):
        self.exact('"')
        escaped = False
        hex_left = 0
        while True:
            c = self.char()
            self.pos += 1
            if hex_left:
                if c not in "0123456789abcdefABCDEF":
                    raise _Bad
                hex_left -= 1
            elif escaped:
                escaped = False
                if c == "u":
                    hex_left = 4
                elif c not in '"\\/bfnrt':
                    raise _Bad
            elif c == '"':
                return
            elif c == "\\":
                escaped = True
            elif ord(c) < 32:
                raise _Bad

    def number(self):
        # A transition table separates extendable incomplete states from
        # terminal numeric states; an unmatched suffix is left to the grammar.
        state = "start"
        while self.pos < len(self.text):
            c = self.text[self.pos]
            next_state = None
            if state == "start":
                next_state = "sign" if c == "-" else "zero" if c == "0" else "int" if c in "123456789" else None
            elif state == "sign":
                next_state = "zero" if c == "0" else "int" if c in "123456789" else None
            elif state in {"zero", "int", "fraction"}:
                if c in "0123456789" and state != "zero":
                    next_state = state
                elif c == "." and state in {"zero", "int"}:
                    next_state = "dot"
                elif c in "eE":
                    next_state = "exponent"
            elif state == "dot":
                next_state = "fraction" if c in "0123456789" else None
            elif state == "exponent":
                next_state = "exponent_sign" if c in "+-" else "exponent_digits" if c in "0123456789" else None
            elif state in {"exponent_sign", "exponent_digits"}:
                next_state = "exponent_digits" if c in "0123456789" else None
            if next_state is None:
                if state in {"zero", "int", "fraction", "exponent_digits"}:
                    return
                raise _Bad
            self.pos += 1
            state = next_state
        if state not in {"zero", "int", "fraction", "exponent_digits"}:
            raise _End


def prefix_state(raw):
    """Recognize the entire byte prefix under frozen size/depth/UTF-8 rules.

Duplicate keys and escaped surrogate syntax are classified as complete JSON;
they are refused separately by the shape gate. A trailing partial UTF-8 code
point is conservatively incomplete even if prior bytes contain obvious junk,
matching the frozen stopping policy. Resource refusal is not invalid syntax.
"""
    if type(raw) is not bytes:
        raise TypeError("exact bytes required")
    if len(raw) > MAX_BYTES:
        return "resource_refusal"
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return "incomplete" if exc.reason == "unexpected end of data" and exc.end == len(raw) else "invalid_prefix"
    cursor = _Cursor(text)
    stack = [("value", 0)]
    try:
        cursor.space()
        if cursor.char() != "{":
            raise _Bad
        while stack:
            action, depth = stack.pop()
            if action == "value":
                if depth > MAX_DEPTH:
                    raise _Limit
                cursor.space()
                c = cursor.char()
                if c == "{":
                    cursor.pos += 1
                    stack.append(("object_first", depth))
                elif c == "[":
                    cursor.pos += 1
                    stack.append(("array_first", depth))
                elif c == '"':
                    cursor.string()
                elif c in "-0123456789":
                    cursor.number()
                elif c in "tfn":
                    for expected in {"t": "true", "f": "false", "n": "null"}[c]:
                        cursor.exact(expected)
                else:
                    raise _Bad
            elif action in {"object_first", "object_key"}:
                cursor.space()
                if action == "object_first" and cursor.char() == "}":
                    cursor.pos += 1
                    continue
                cursor.string()
                cursor.space()
                cursor.exact(":")
                stack.extend((("object_next", depth), ("value", depth + 1)))
            elif action == "array_first":
                cursor.space()
                if cursor.char() == "]":
                    cursor.pos += 1
                else:
                    stack.extend((("array_next", depth), ("value", depth + 1)))
            else:
                cursor.space()
                close = "}" if action == "object_next" else "]"
                c = cursor.char()
                if c == close:
                    cursor.pos += 1
                elif c == ",":
                    cursor.pos += 1
                    if action == "object_next":
                        stack.append(("object_key", depth))
                    else:
                        stack.extend((("array_next", depth), ("value", depth + 1)))
                else:
                    raise _Bad
        cursor.space()
        return "complete" if cursor.pos == len(text) else "invalid_prefix"
    except _End:
        return "incomplete"
    except _Bad:
        return "invalid_prefix"
    except _Limit:
        return "resource_refusal"


def _visible(text):
    return not any(unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"} for c in text)


def _surface(text):
    return " ".join(text.split()).casefold()


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate key")
        result[name] = value
    return result


def _shape(raw, words):
    if type(words) not in (list, tuple) or len(words) != 5:
        raise ValueError("this auditor is scoped to exactly five fixed source tokens")
    for word in words:
        if type(word) is not str or not word or any(c.isspace() for c in word) or not _visible(word):
            raise ValueError("invalid source token")
        word.encode("utf-8", errors="strict")
    value = json.loads(raw, object_pairs_hook=_unique)
    if type(value) is not dict or list(value) != ["words"]:
        raise ValueError("wrong root")
    rows = value["words"]
    if type(rows) is not list or len(rows) != 5:
        raise ValueError("wrong row count")
    returned = 0
    for word, row in zip(words, rows):
        if type(row) is not dict or list(row) != [word]:
            raise ValueError("wrong source identity or order")
        values = row[word]
        if type(values) is not list or len(values) not in (0, 8):
            raise ValueError("wrong alternative count")
        identities = {_surface(word)}
        for text in values:
            if type(text) is not str or not text or text.strip() != text or not _visible(text):
                raise ValueError("invalid candidate surface")
            text.encode("utf-8", errors="strict")
            identity = _surface(text)
            if identity in identities:
                raise ValueError("self or duplicate candidate")
            identities.add(identity)
        returned += bool(values)
    return {"format_accepted": True, "raw_sha256": _hash(raw), "accepted_words": rows,
            "requested_word_rows": 5, "nonempty_candidate_rows": returned,
            "abstained_word_rows": 5 - returned, "candidate_strings_returned": returned * 8,
            "usable_candidate_rows": None, "capacity_bits": None,
            "semantic_equivalence_verified": False, "full_cover_resource_complete": False}


def assess_format(raw, expected_words):
    state = prefix_state(raw)
    outcome, result = "format_rejected", None
    if state == "resource_refusal":
        outcome = "parser_resource_refusal"
    elif state == "complete":
        try:
            result = _shape(raw, expected_words)
            outcome = "format_accepted"
        except (ValueError, TypeError, UnicodeError, RecursionError):
            pass
    return {"outcome": outcome, "prefix_state": state, "format_result": result}


def _float32(hexadecimal):
    if type(hexadecimal) is not str or re.fullmatch(r"[0-9a-fA-F]{8}", hexadecimal) is None:
        raise ValueError("exact binary32 hexadecimal required")
    bits = int(hexadecimal, 16)
    exponent, fraction = (bits >> 23) & 255, bits & 8388607
    if exponent == 255:
        raise ValueError("nonfinite binary32 logit")
    magnitude = math.ldexp(fraction, -149) if exponent == 0 else math.ldexp(8388608 + fraction, exponent - 150)
    return -magnitude if bits >> 31 else magnitude


def _law(rows):
    ranked = sorted(((row["token_id"], row["logit_hex"], _float32(row["logit_hex"]))
                     for row in rows), key=lambda entry: (-entry[2], entry[0]))
    maximum = ranked[0][2]
    mass = [math.exp((entry[2] - maximum) / 0.7) for entry in ranked]
    normalizer = math.fsum(mass)
    allocation = [x / normalizer * TOTAL for x in mass]
    counts = [math.floor(x) for x in allocation]
    residual = TOTAL - sum(counts)
    if residual < 0 or residual > len(rows):
        raise ValueError("floating allocation residual")
    priorities = sorted(range(len(ranked)), key=lambda index: (counts[index] - allocation[index], ranked[index][0]))
    for index in priorities[:residual]:
        counts[index] += 1
    return [{"token_id": entry[0], "logit_hex": entry[1], "weight": counts[index]}
            for index, entry in enumerate(ranked)]


def replay_sample(evidence, token_id, prefix_ids, request_sha, attempt_order, profile):
    """Fail closed on any recorded categorical-law/HMAC mismatch; return a summary.

The parent auditor independently verifies resource identity and journal ordering.
This helper checks support range/masking and replays only recorded logits.
"""
    if request_sha not in REQUEST_HASHES or type(request_sha) is not str:
        raise ValueError("unknown fixed request")
    if type(attempt_order) is not int or attempt_order not in (0, 1, 2):
        raise ValueError("unknown fixed attempt")
    if type(profile) is not dict or type(profile.get("ready")) is not dict:
        raise ValueError("missing profile")
    vocabulary = profile["ready"].get("vocabulary_size")
    special = profile.get("special_ids")
    if type(vocabulary) is not int or vocabulary < 50 or type(special) is not list or any(type(i) is not int or not 0 <= i < vocabulary for i in special):
        raise ValueError("invalid vocabulary or mask")
    def valid_id(value):
        return type(value) is int and 0 <= value < vocabulary and value not in special
    if type(prefix_ids) is not list or len(prefix_ids) >= 512 or not all(map(valid_id, prefix_ids)) or not valid_id(token_id):
        raise ValueError("invalid output IDs")
    keys = {"request_sha256", "seed_hex", "prefix_ids", "weights", "draw_uint24",
            "draw_context_sha256", "temperature", "integer_total"}
    if type(evidence) is not dict or set(evidence) != keys:
        raise ValueError("unexpected evidence schema")
    rows = evidence["weights"]
    if type(rows) is not list or len(rows) != 50:
        raise ValueError("exactly fifty recorded support rows required")
    for row in rows:
        if type(row) is not dict or set(row) != {"token_id", "logit_hex", "weight"} or not valid_id(row["token_id"]) or type(row["weight"]) is not int or row["weight"] < 0:
            raise ValueError("invalid support row")
    if len({row["token_id"] for row in rows}) != 50:
        raise ValueError("duplicate support ID")
    law = _law(rows)
    seed = _hash(_bytes([DOMAIN, "fixed-public-diagnostic-seed", request_sha, attempt_order]))
    context = _bytes([DOMAIN, request_sha, attempt_order, len(prefix_ids), _hash(_bytes(prefix_ids))])
    draw = int.from_bytes(hmac.digest(bytes.fromhex(seed), context, "sha256")[:3], "big")
    remainder, selected = draw, None
    for row in law:
        remainder -= row["weight"]
        if remainder < 0:
            selected = row["token_id"]
            break
    expected = {"request_sha256": request_sha, "seed_hex": seed, "prefix_ids": prefix_ids,
                "weights": law, "draw_uint24": draw, "draw_context_sha256": _hash(context),
                "temperature": 0.7, "integer_total": TOTAL}
    if _bytes(evidence) != _bytes(expected) or token_id != selected:
        raise ValueError("recorded quantization, seed, context, draw or selected token differs")
    return {"verified": True, "token_id": selected, "draw_uint24": draw,
            "draw_context_sha256": _hash(context), "seed_hex": seed,
            "support_size": 50, "integer_total": TOTAL}
