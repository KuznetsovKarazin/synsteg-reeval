"""Independent stdlib JSON/schema oracle and adversarial fixtures for schema-v3.

Author: separate review agent; no production parsing/validation/normalization helpers
are used by this oracle. These are software checks, not neural generations or a
semantic quality benchmark.
"""
from __future__ import annotations
import hashlib
import hmac
from unittest import mock
import json
from pathlib import Path
import sys
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORDS = ("Popular", "Knowledge", "Article:", "Understanding", "Chinese")
DUPLICATE_WORDS = ("same", "same", "A:", "\u4e2d", "name")


def independent_schema_accepts(raw, words):
    """Full-document oracle built directly on strict stdlib JSON decoding."""
    if type(raw) is not bytes:
        return False
    try:
        source = raw.decode("utf-8", "strict")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate")
                result[key] = value
            return result
        document = json.loads(source, object_pairs_hook=pairs,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")))
        if type(document) is not dict or list(document) != ["words"]:
            return False
        rows = document["words"]
        if type(rows) is not list or len(rows) != len(words):
            return False
        for word, row in zip(words, rows):
            if type(row) is not dict or list(row) != [word]:
                return False
            candidates = row[word]
            if type(candidates) is not list or len(candidates) not in (0, 8):
                return False
            surfaces = {" ".join(word.split()).casefold()}
            for text in candidates:
                if type(text) is not str or not text or text.strip() != text:
                    return False
                text.encode("utf-8", "strict")
                if any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in text):
                    return False
                surface = " ".join(text.split()).casefold()
                if surface in surfaces:
                    return False
                surfaces.add(surface)
        return True
    except (UnicodeError, TypeError, ValueError):
        return False


def document(words=WORDS, candidate_rows=None, *, ascii_only=False, pretty=False):
    if candidate_rows is None:
        candidate_rows = [[] for _ in words]
    value = {"words": [{word: candidates} for word, candidates in zip(words, candidate_rows)]}
    return json.dumps(value, ensure_ascii=ascii_only,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")).encode("utf-8")


def valid_examples():
    cases = []
    for words in (WORDS, DUPLICATE_WORDS, ('say"', "back\\slash", "😀", "é", "a/b")):
        cases.append((words, document(words)))
        cases.append((words, document(words, pretty=True)))
        rows = [[f"candidate-{i}-{j}" for j in range(8)] for i in range(len(words))]
        rows[0] = ["\u00e9", "\u4e2d\u6587", "\U0001f600", "slash/", "back\\slash", 'a"b', "two words", "x\u00a0y"]
        cases.append((words, document(words, rows)))
        cases.append((words, document(words, rows, ascii_only=True)))
        cases.append((words, b" \t\r\n" + document(words, rows, pretty=True) + b"\n\r\t "))
        mix = [row if i % 2 else [] for i, row in enumerate(rows)]
        cases.append((words, document(words, mix)))
    return cases


def invalid_full_documents():
    good = json.loads(document())
    bad = []
    def add(value):
        bad.append(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode())
    add([]); add(None); add({}); add({"word": []}); add({"words": {}, "extra": 1})
    add({"words": []}); add({"words": good["words"][:-1]})
    add({"words": good["words"] + [{"extra": []}]})
    add({"words": list(reversed(good["words"]))})
    add({"words": [{"Popular": [], "Knowledge": []}] + good["words"][1:]})
    add({"words": [{"popular": []}] + good["words"][1:]})
    for candidates in [None, False, {}, "", ["only"], [str(i) for i in range(7)],
                       [str(i) for i in range(9)], list(range(8)),
                       ["same"] * 8, ["Popular"] + [str(i) for i in range(7)],
                       ["POPULAR"] + [str(i) for i in range(7)],
                       ["x y", "X  Y"] + [str(i) for i in range(6)],
                       ["x y", "x\u00a0y"] + [str(i) for i in range(6)]]:
        add({"words": [{"Popular": candidates}] + good["words"][1:]})
    for invalid in ["", " padded", "padded ", "\t", "x\n", "x\x00y", "x\u200by", "x\u2028y", "x\u2029y", "\ud800", "\udc00"]:
        add({"words": [{"Popular": [invalid] + [str(i) for i in range(7)]}] + good["words"][1:]})
    bad.extend([
        b'{"words":[],"words":[]}',
        b'{"words":[{"Popular":[],"Popular":[]},{"Knowledge":[]},{"Article:":[]},{"Understanding":[]},{"Chinese":[]}]}',
        document() + b"x", b"```" + document(), b"\xef\xbb\xbf" + document(),
        document().replace(b'"Popular"', b'"Popular\\x"'),
        document().replace(b'"Popular"', b'"\\uZZZZ"'),
        document().replace(b'"Popular"', b'"Pop\xffular"'),
    ])
    return bad


# Production API binding and unittest classes are added once the API is stable.
from synsteg_audit import author_candidate_schema_v3 as component


class IndependentSchemaTests(unittest.TestCase):
    def test_oracle_accepts_valid_examples(self):
        for words, raw in valid_examples():
            with self.subTest(raw_sha=hashlib.sha256(raw).hexdigest()):
                self.assertTrue(independent_schema_accepts(raw, words))
                self.assertEqual(component.classify_prefix(raw, words)["state"], "complete")

    def test_every_byte_prefix_of_valid_documents(self):
        for words, raw in valid_examples():
            for length in range(len(raw) + 1):
                with self.subTest(raw_sha=hashlib.sha256(raw).hexdigest(), length=length):
                    state = component.classify_prefix(raw[:length], words)["state"]
                    expected = "complete" if independent_schema_accepts(raw[:length], words) else "incomplete"
                    self.assertEqual(state, expected)

    def test_complete_invalid_documents_rejected(self):
        for raw in invalid_full_documents():
            with self.subTest(raw=repr(raw)):
                self.assertFalse(independent_schema_accepts(raw, WORDS))
                self.assertEqual(component.classify_prefix(raw, WORDS)["state"], "invalid_prefix")

    def test_irrecoverable_schema_prefixes(self):
        cases = [b"[", b"`", b'{"other"', b'{"words":{',
                 b'{"words":[{"WRONG"', b'{"words":[{"Popular":["one"]',
                 b'{"words":[{"Popular":["Popular"',
                 b'{"words":[{"Popular":["x","X"',
                 b'{"words":[{"Popular":[" padded',
                 b'{"words":[{"Popular":["\\n',
                 b'{"words":[{"Popular":["\\u200b',
                 b'{"words":[{"Popular":["\\udc00',
                 b'{"words":[{"Popular":["\\ud800x',
                 b'{"words":[{"Popular":[1',
                 b'{"\\u123', b'{"words":[{"Popular":["\\u000',
                 b'{"words":[{"Popular":["\\udc',
                 b'{"words":[{"Popular":[]},',
                 ]
        # A comma after the first row remains extendable: explicit negative control.
        cases.pop()
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(component.classify_prefix(raw, WORDS)["state"], "invalid_prefix")

    def test_bad_prefix_not_rescued_by_trailing_utf8_fragment(self):
        for raw in (b"[\xe4", b"`\xf0\x9f", b'{"other"\xe4', b'{"words":{\xe4'):
            with self.subTest(raw=raw):
                self.assertEqual(component.classify_prefix(raw, WORDS)["state"], "invalid_prefix")

    def test_incomplete_utf8_and_surrogate_sequences(self):
        start = b'{"words":[{"Popular":["'
        for suffix in (b"\xe4", b"\xe4\xb8", b"\xf0", b"\xf0\x9f", b"\xf0\x9f\x98", b"\\", b"\\u", b"\\ud", b"\\ud8", b"\\ud800", b"\\ud800\\", b"\\ud800\\u", b"\\ud800\\udc0"):
            with self.subTest(suffix=suffix):
                self.assertEqual(component.classify_prefix(start + suffix, WORDS)["state"], "incomplete")
        for suffix in (b"\xc0", b"\xff", b"\xed\xa0\x80", b"\xf4\x90", b"\\ud800\\u0041", b"\\udc00"):
            with self.subTest(suffix=suffix):
                self.assertEqual(component.classify_prefix(start + suffix, WORDS)["state"], "invalid_prefix")

    def test_resource_boundary_not_json_judgment(self):
        good = document()
        self.assertEqual(component.classify_prefix(good, WORDS, max_bytes=len(good))["state"], "complete")
        self.assertEqual(component.classify_prefix(good, WORDS, max_bytes=len(good) - 1)["state"], "resource_refusal")

    def test_malformed_expected_words(self):
        for words in ([], (), ["a"] * 6, [""], ["has space"], [True], ["x\n"], ["x\u200by"], ["\ud800"], "five"):
            with self.subTest(words=repr(words)):
                with self.assertRaises((ValueError, TypeError, UnicodeError)):
                    component.classify_prefix(b"", words)


class IndependentSupportTests(unittest.TestCase):
    def test_exact_conditional_mass(self):
        rows = [{"token_id": 1, "piece": b"{", "mass": 10},
                {"token_id": 2, "piece": b"[", "mass": 20},
                {"token_id": 3, "piece": b" \n", "mass": 30}]
        support = component.constrain_support(b"", WORDS, rows)
        self.assertEqual(support["state"], "ready")
        self.assertEqual(support["input_mass"], 60)
        self.assertEqual(support["retained_mass"], 40)
        self.assertEqual({row["token_id"] for row in support["accepted"]}, {1, 3})
        self.assertEqual({row["token_id"] for row in support["rejected"]}, {2})
        for row in support["accepted"]:
            self.assertEqual(row["conditional_numerator"], row["mass"])
            self.assertEqual(row["conditional_denominator"], 40)
        self.assertEqual(sum(row["mass"] for row in support["accepted"]), support["retained_mass"])

    def test_utf8_token_boundaries_and_syntax_spanning_tokens(self):
        prefix = b'{"words":[{"Popular":["\xe4'
        rows = [{"token_id": 9, "piece": b"\xb8\xad", "mass": 2},
                {"token_id": 7, "piece": b'\xb8\xad","a","b","c","d","e","f","g"]},{"Knowledge":[]}', "mass": 3},
                {"token_id": 8, "piece": b"\xff", "mass": 5}]
        support = component.constrain_support(prefix, WORDS, rows)
        self.assertEqual(support["state"], "ready")
        self.assertEqual({row["token_id"] for row in support["accepted"]}, {7, 9})
        self.assertEqual(support["retained_mass"], 5)

    def test_all_rejected_empty_support(self):
        support = component.constrain_support(b"", WORDS, [
            {"token_id": 1, "piece": b"[", "mass": 2},
            {"token_id": 2, "piece": b"```", "mass": 3},
        ])
        self.assertEqual(support["state"], "empty_support")
        self.assertEqual(support["retained_mass"], 0)
        self.assertEqual(support["accepted"], [])

    def test_input_empty_support(self):
        self.assertEqual(component.constrain_support(b"", WORDS, [])["state"], "empty_support")
        self.assertIsNone(component.constrain_support(b"", WORDS, [])["retained_fraction"])

    def test_complete_document_never_generates_extra_token(self):
        support = component.constrain_support(document(), WORDS, [
            {"token_id": 1, "piece": b" ", "mass": 1},
            {"token_id": 2, "piece": b"x", "mass": 2},
        ])
        self.assertEqual(support["state"], "complete")
        self.assertEqual(support.get("accepted", []), [])

    def test_duplicate_piece_distinct_ids_preserve_probability(self):
        support = component.constrain_support(b"", WORDS, [
            {"token_id": 2, "piece": b"{", "mass": 2},
            {"token_id": 1, "piece": b"{", "mass": 1},
        ])
        self.assertEqual(support["state"], "ready")
        self.assertEqual(len(support["accepted"]), 2)
        self.assertEqual(support["retained_mass"], 3)

    def test_input_support_malformed(self):
        valid = {"token_id": 1, "piece": b"{", "mass": 1}
        for changed in [dict(valid, token_id=True), dict(valid, token_id=-1),
                        dict(valid, token_id=1.0), dict(valid, mass=True),
                        dict(valid, mass=0), dict(valid, mass=-1), dict(valid, mass=1.0),
                        dict(valid, piece="{"), dict(valid, piece=b""),
                        {"token_id": 1, "piece": b"{"},
                        dict(valid, unexpected=True)]:
            with self.subTest(changed=repr(changed)):
                with self.assertRaises((ValueError, TypeError)):
                    component.constrain_support(b"", WORDS, [changed])
        with self.assertRaises((ValueError, TypeError)):
            component.constrain_support(b"", WORDS, [valid, valid])

    def test_support_resource_caps(self):
        rows = [{"token_id": i, "piece": b"{", "mass": i + 1} for i in range(3)]
        self.assertEqual(component.constrain_support(b"", WORDS, rows, max_support=2)["state"], "resource_refusal")
        good = document()
        self.assertEqual(component.constrain_support(good, WORDS, rows, max_bytes=1)["state"], "resource_refusal")



def independent_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def independent_draw(support, key, request_sha256, attempt, step):
    total = sum(row["mass"] for row in support["accepted"])
    bound = (2 ** 256 // total) * total
    for counter in range(128):
        text = {"domain": "author-candidate-schema-v3/draw-v1", "request_sha256": request_sha256,
                "attempt": attempt, "step": step, "law_sha256": support["law_sha256"], "counter": counter}
        random_integer = int.from_bytes(hmac.new(key, independent_canonical(text), hashlib.sha256).digest(), "big")
        if random_integer >= bound:
            continue
        ticket = random_integer % total
        remainder = ticket
        for row in sorted(support["accepted"], key=lambda r: r["token_id"]):
            if remainder < row["mass"]:
                return row["token_id"], ticket, counter
            remainder -= row["mass"]
    return None


class IndependentDrawTests(unittest.TestCase):
    @staticmethod
    def law():
        return component.constrain_support(b"", WORDS, [
            {"token_id": 33, "piece": b"{", "mass": 13},
            {"token_id": 4, "piece": b" ", "mass": 5},
            {"token_id": 17, "piece": b"\n", "mass": 2},
            {"token_id": 6, "piece": b"[", "mass": 7},
        ])

    def test_exact_reference_across_fixed_contexts(self):
        support = self.law()
        independent_hash = hashlib.sha256(independent_canonical({k: v for k, v in support.items() if k != "law_sha256"})).hexdigest()
        self.assertEqual(support["law_sha256"], independent_hash)
        for key in (bytes(32), bytes(range(32)), bytes([255]) * 32):
            for attempt in range(3):
                for step in (0, 1, 127, 511):
                    request_sha256 = hashlib.sha256(b"fixed-test-request").hexdigest()
                    expected = independent_draw(support, key, request_sha256, attempt, step)
                    actual = component.select_token(support, key, request_sha256, attempt, step)
                    self.assertEqual(actual["state"], "selected")
                    self.assertEqual((actual["token_id"], actual["ticket"], actual["counter"]), expected)

    def test_input_order_is_irrelevant(self):
        rows = [{"token_id": 1, "piece": b"{", "mass": 2},
                {"token_id": 3, "piece": b" ", "mass": 3},
                {"token_id": 2, "piece": b"[", "mass": 4}]
        one = component.constrain_support(b"", WORDS, rows)
        two = component.constrain_support(b"", WORDS, list(reversed(rows)))
        self.assertEqual(one, two)
        self.assertEqual(component.select_token(one, bytes(32), "0" * 64, 0, 0),
                         component.select_token(two, bytes(32), "0" * 64, 0, 0))

    def test_context_parameters_strict(self):
        valid = [self.law(), bytes(32), "a" * 64, 0, 0]
        for index, value in [(1, b""), (1, bytes(31)), (1, bytes(33)), (1, "x" * 32),
                             (2, "A" * 64), (2, "a" * 63), (2, "z" * 64),
                             (3, True), (3, -1), (3, 3), (3, 0.0),
                             (4, True), (4, -1), (4, 512), (4, 0.0)]:
            with self.subTest(index=index, value=repr(value)):
                args = valid.copy(); args[index] = value
                with self.assertRaises((ValueError, TypeError)):
                    component.select_token(*args)

    def test_tampered_categorical_fields_rejected_even_if_rehashed(self):
        for field, value in [("mass", 0), ("mass", True), ("conditional_numerator", 99),
                             ("conditional_denominator", 99), ("token_id", True),
                             ("piece_hex", ""), ("piece_hex", "FF"), ("piece_hex", "0")]:
            with self.subTest(field=field, value=value):
                support = json.loads(json.dumps(self.law()))
                support["accepted"][0][field] = value
                body = {k: v for k, v in support.items() if k != "law_sha256"}
                support["law_sha256"] = hashlib.sha256(independent_canonical(body)).hexdigest()
                with self.assertRaises((ValueError, TypeError)):
                    component.select_token(support, bytes(32), "a" * 64, 0, 0)

    def test_rehashed_malformed_metadata_rejected(self):
        for field, value in [("input_mass", 0), ("input_mass", False), ("input_mass", 19),
                             ("input_mass", 2 ** 40 + 1), ("schema", "wrong"),
                             ("prefix_sha256", "a" * 63), ("words_sha256", "A" * 64),
                             ("base_law_sha256", True), ("retained_fraction", None),
                             ("retained_fraction", {"numerator": True, "denominator": 27}),
                             ("retained_fraction", {"numerator": 20, "denominator": 28}),
                             ("rejected", [{"token_id": 4, "reason": "overlap"}]),
                             ("rejected", [{"token_id": 6, "reason": ""}]),
                             ("rejected", [{"token_id": 6, "reason": "x"}, {"token_id": 6, "reason": "x"}]),
                             ("rejected", []), ("unexpected", True)]:
            with self.subTest(field=field, value=repr(value)):
                support = json.loads(json.dumps(self.law())); support[field] = value
                body = {k: v for k, v in support.items() if k != "law_sha256"}
                support["law_sha256"] = hashlib.sha256(independent_canonical(body)).hexdigest()
                with self.assertRaises((ValueError, TypeError)):
                    component.select_token(support, bytes(32), "a" * 64, 0, 0)

    def test_digest_mismatch_rejected(self):
        support = self.law(); support["retained_mass"] += 1
        with self.assertRaises((ValueError, TypeError)):
            component.select_token(support, bytes(32), "a" * 64, 0, 0)

    def test_rejection_and_counter_exhaustion(self):
        support = component.constrain_support(b"", WORDS, [
            {"token_id": 1, "piece": b"{", "mass": 1},
            {"token_id": 2, "piece": b" ", "mass": 2},
        ])
        class FakeDigest:
            def __init__(self, raw): self.raw = raw
            def digest(self): return self.raw
        # For total 3, 2^256-1 is rejected and zero is accepted.
        with mock.patch.object(component.hmac, "new", side_effect=[FakeDigest(bytes([255]) * 32), FakeDigest(bytes(32))]) as call:
            result = component.select_token(support, bytes(32), "0" * 64, 0, 0)
            self.assertEqual((result["counter"], result["ticket"], result["token_id"]), (1, 0, 1))
            self.assertEqual(call.call_count, 2)
        with mock.patch.object(component.hmac, "new", return_value=FakeDigest(bytes([255]) * 32)) as call:
            result = component.select_token(support, bytes(32), "0" * 64, 0, 0)
            self.assertEqual(result["state"], "resource_refusal")
            self.assertEqual(call.call_count, 128)


if __name__ == "__main__":
    unittest.main()
