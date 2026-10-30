"""Offline, strict candidate-format adaptation. No provider/model/codec imports.

This new condition is not the author's GPT-4o API, nor a repair of v1 replies.
Empty candidate arrays are explicit abstentions, never eight safe alternatives.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

ID = "author-candidate-format-v2"
MAX_RAW_BYTES = 262144
MAX_JSON_DEPTH = 32
AUTHOR_REQUESTS_SHA256 = "e5256075993526101114f63ea37161d1d82369183e1758054ea924c74b522864"
AUTHOR_COVER_SHA256 = "a2a51f6f45afb664e8f06d5ebc160b189785b781e80de3c8d01939a9ec5f3a95"
AUTHOR_REQUEST_SHA256 = (
    "2a510a01e1b7ee830f393c97bce447889f28ff24e251f9c6ee82231259bc2247",
    "ec61d91995330841f842969e930c996004214e8822b6511c8f81383ae860d328",
)
FIXED_WORDS = (
    ("Popular", "Knowledge", "Article:", "Understanding", "Chinese"),
    ("Astronomy", "For", "a", "general", "audience,"),
)
SYSTEM_PROMPT = """Return one JSON object and nothing else. Its only top-level key must be \"words\".
The value of \"words\" must be a list with one entry for each input token, in exactly the input order.
Each entry must be an object with exactly one key: the corresponding input token copied exactly, including capitalization and punctuation.
For b=3 and a=0, each value must be either an array of exactly 8 distinct candidate strings or an empty array [].
The empty array is an explicit abstention: use it whenever you cannot supply all 8 defensible alternatives. Never fill a list with weak alternatives just to reach 8.
For a nonempty list, do not include the input token itself, duplicates, case-only duplicates, or whitespace-only variants of another candidate.
Use only substitutions that preserve the intended meaning, reference, part of speech, inflection, capitalization, punctuation and grammatical role in the supplied five-token context.
Preserve proper names, named entities and technical terms. Abstain if substitution would change them, or if the intended sense or morphology cannot be determined from the limited context.
Do not invent facts, names, explanations or extra keys. Do not return placeholders, Markdown fences, comments, introductory text or text after the JSON object.
Candidate strings must not contain control characters, invisible format characters, or line/paragraph separators.
An empty array is allowed for every token. Do not claim that a proposed alternative is semantically certified."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build_requests(source_raw):
    """Bind the exact already selected v1 cohort, without a new selection pass."""
    if type(source_raw) is not bytes or digest(source_raw) != AUTHOR_REQUESTS_SHA256:
        raise ValueError("fixed source request file differs")
    source = json.loads(source_raw)
    if (source["cover_sha256"] != AUTHOR_COVER_SHA256
            or source["full_cover_chunks"] != 109
            or source["diagnostic_chunks"] != 2):
        raise ValueError("fixed cohort denominator differs")
    requests = []
    for order, old in enumerate(source["requests"]):
        body = {k: v for k, v in old.items() if k != "request_sha256"}
        if (old["request_sha256"] != AUTHOR_REQUEST_SHA256[order]
                or digest(canonical(body)) != old["request_sha256"]
                or tuple(old["words"]) != FIXED_WORDS[order]
                or old["chunk_order"] != order or old["word_start"] != order * 5):
            raise ValueError("source request identity differs")
        user = canonical({"tokens": old["words"], "bits_per_word": 3,
                          "additional_bits": 0,
                          "context_scope": "Only these five tokens are supplied; abstain when ambiguous."}).decode()
        new = {"schema": ID + "-request", "chunk_order": order,
               "word_start": order * 5, "words": old["words"],
               "source_author_request_sha256": old["request_sha256"],
               "system": SYSTEM_PROMPT, "user": user,
               "bits_per_word": 3, "additional_bits": 0,
               "temperature_argument": 0.7, "json_shape_attempt_limit": 3}
        requests.append({**new, "request_sha256": digest(canonical(new))})
    if len(requests) != 2:
        raise ValueError("exactly two fixed requests required")
    return {"schema": ID + "-requests", "source_requests_sha256": digest(source_raw),
            "cover_sha256": AUTHOR_COVER_SHA256, "full_cover_chunks": 109,
            "diagnostic_chunks": 2, "outside_diagnostic_chunks": 107,
            "selection_rule": "unchanged first two chunks of first fixed baseline cover",
            "generation_enabled": False, "requests": requests}


class _Incomplete(Exception):
    pass


class _Invalid(Exception):
    pass


class _Resource(Exception):
    pass


class _Syntax:
    """A bounded JSON grammar recognizer; it does not normalize or repair input."""
    def __init__(self, text):
        self.text = text
        self.i = 0

    def ws(self):
        while self.i < len(self.text) and self.text[self.i] in " \t\r\n":
            self.i += 1

    def peek(self):
        if self.i == len(self.text):
            raise _Incomplete("end of an extendable JSON prefix")
        return self.text[self.i]

    def take(self, expected):
        if self.peek() != expected:
            raise _Invalid("unexpected character")
        self.i += 1

    def string(self):
        self.take('"')
        while True:
            c = self.peek()
            self.i += 1
            if c == '"':
                return
            if ord(c) < 32:
                raise _Invalid("unescaped control in JSON string")
            if c == "\\":
                c = self.peek()
                self.i += 1
                if c == "u":
                    for _ in range(4):
                        if self.peek() not in "0123456789abcdefABCDEF":
                            raise _Invalid("invalid Unicode escape")
                        self.i += 1
                elif c not in '"\\/bfnrt':
                    raise _Invalid("invalid JSON escape")

    def literal(self, word):
        for c in word:
            self.take(c)

    def number(self):
        if self.peek() == "-":
            self.i += 1
        c = self.peek()
        if c == "0":
            self.i += 1
        elif c in "123456789":
            self.i += 1
            while self.i < len(self.text) and self.text[self.i] in "0123456789":
                self.i += 1
        else:
            raise _Invalid("invalid JSON number")
        if self.i < len(self.text) and self.text[self.i] == ".":
            self.i += 1
            if self.peek() not in "0123456789":
                raise _Invalid("missing fractional digit")
            while self.i < len(self.text) and self.text[self.i] in "0123456789":
                self.i += 1
        if self.i < len(self.text) and self.text[self.i] in "eE":
            self.i += 1
            if self.peek() in "+-":
                self.i += 1
            if self.peek() not in "0123456789":
                raise _Invalid("missing exponent digit")
            while self.i < len(self.text) and self.text[self.i] in "0123456789":
                self.i += 1

    def value(self, depth=0):
        if depth > MAX_JSON_DEPTH:
            raise _Resource("JSON nesting limit")
        self.ws()
        c = self.peek()
        if c == '"':
            self.string()
        elif c == "{":
            self.i += 1
            self.ws()
            if self.peek() == "}":
                self.i += 1
                return
            while True:
                self.string()
                self.ws()
                self.take(":")
                self.value(depth + 1)
                self.ws()
                c = self.peek()
                self.i += 1
                if c == "}":
                    return
                if c != ",":
                    raise _Invalid("missing object separator")
                self.ws()
        elif c == "[":
            self.i += 1
            self.ws()
            if self.peek() == "]":
                self.i += 1
                return
            while True:
                self.value(depth + 1)
                self.ws()
                c = self.peek()
                self.i += 1
                if c == "]":
                    return
                if c != ",":
                    raise _Invalid("missing array separator")
        elif c in "-0123456789":
            self.number()
        elif c == "t":
            self.literal("true")
        elif c == "f":
            self.literal("false")
        elif c == "n":
            self.literal("null")
        else:
            raise _Invalid("invalid JSON value prefix")


def classify_prefix(raw):
    """Classify exact committed bytes; an invalid prefix cannot be repaired by appending.

    Resource refusals are not assertions of invalid JSON. A trailing partial
    UTF-8 code point stays incomplete conservatively. No sampling is performed.
    """
    if type(raw) is not bytes:
        raise TypeError("exact raw bytes required")
    if len(raw) > MAX_RAW_BYTES:
        return {"state": "resource_refusal", "reason": "raw byte limit"}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.end == len(raw):
            return {"state": "incomplete", "reason": "partial UTF-8 code point"}
        return {"state": "invalid_prefix", "reason": "invalid UTF-8"}
    parser = _Syntax(text)
    try:
        parser.ws()
        if parser.peek() != "{":
            raise _Invalid("top-level response must begin with an object")
        parser.value()
        parser.ws()
        if parser.i != len(text):
            raise _Invalid("trailing bytes after complete JSON object")
    except _Incomplete as exc:
        return {"state": "incomplete", "reason": str(exc)}
    except _Invalid as exc:
        return {"state": "invalid_prefix", "reason": str(exc)}
    except _Resource as exc:
        return {"state": "resource_refusal", "reason": str(exc)}
    return {"state": "complete", "reason": "complete entire JSON object"}


def _surface(s):
    return " ".join(s.split()).casefold()


def _forbidden_characters(s):
    return any(unicodedata.category(c) in {"Cc", "Cf", "Zl", "Zp"} for c in s)


def validate_response(raw, expected_words):
    """Strict format/coverage gate, not a semantic or usable-capacity verifier."""
    if (type(expected_words) not in (list, tuple) or not 1 <= len(expected_words) <= 5
            or any(type(w) is not str or not w or any(c.isspace() for c in w)
                   or _forbidden_characters(w) for w in expected_words)):
        raise ValueError("one to five exact nonempty source tokens required")
    for word in expected_words:
        word.encode("utf-8")  # Source identities must be representable as exact UTF-8 too.
    status = classify_prefix(raw)
    if status["state"] != "complete":
        raise ValueError(status["state"] + ": " + status["reason"])
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if type(value) is not dict or set(value) != {"words"}:
        raise ValueError("only words top-level key is allowed")
    rows = value["words"]
    if type(rows) is not list or len(rows) != len(expected_words):
        raise ValueError("one ordered row per source token required")
    nonempty = 0
    for expected, row in zip(expected_words, rows):
        if type(row) is not dict or list(row) != [expected]:
            raise ValueError("exact ordered source keys required")
        values = row[expected]
        if type(values) is not list or len(values) not in (0, 8):
            raise ValueError("each row must contain exactly eight alternatives or abstain with []")
        seen = {_surface(expected)}
        for candidate in values:
            if type(candidate) is not str or not candidate or candidate.strip() != candidate:
                raise ValueError("nonempty unpadded candidate strings required")
            candidate.encode("utf-8")  # Reject escaped unpaired surrogates.
            normalized = _surface(candidate)
            if normalized in seen:
                raise ValueError("self candidate or repeated surface form")
            if _forbidden_characters(candidate):
                raise ValueError("candidate Unicode Cc/Cf/Zl/Zp characters forbidden")
            seen.add(normalized)
        nonempty += bool(values)
    return {"format_accepted": True, "raw_sha256": digest(raw), "accepted_words": rows,
            "requested_word_rows": len(expected_words), "nonempty_candidate_rows": nonempty,
            "abstained_word_rows": len(expected_words) - nonempty,
            "candidate_strings_returned": nonempty * 8,
            "usable_candidate_rows": None, "capacity_bits": None,
            "semantic_equivalence_verified": False, "full_cover_resource_complete": False}
