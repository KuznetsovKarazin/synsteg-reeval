"""Component tests. These are synthetic laws, never model observations."""
import hashlib
import json
import unittest
from unittest.mock import patch

from synsteg_audit import author_candidate_schema_v3 as m
from synsteg_audit.author_candidate_format_v2 import validate_response


class ComponentTests(unittest.TestCase):
    words = ["Popular", "Knowledge", "Article:", "Understanding", "Chinese"]

    def test_all_abstention_token_walk_stops_without_eos(self):
        pieces = [b'{"words":[', b'{"Popular":[]},{"Knowledge":[]},',
                  b'{"Article:":[]},{"Understanding":[]},{"Chinese":[]}]}']
        raw = b""
        for step, piece in enumerate(pieces):
            support = m.constrain_support(raw, self.words,
                                          [{"token_id": 8, "piece": piece, "mass": 7},
                                           {"token_id": 9, "piece": b"`", "mass": 3}])
            self.assertEqual("ready", support["state"])
            self.assertEqual(7, support["retained_mass"])
            chosen = m.select_token(support, bytes(range(32)), "a" * 64, 0, step)
            raw += bytes.fromhex(chosen["piece_hex"])
        self.assertEqual("complete", m.classify_prefix(raw, self.words)["state"])
        result = validate_response(raw, self.words)
        self.assertEqual(0, result["candidate_strings_returned"])
        self.assertIsNone(result["capacity_bits"])
        self.assertEqual("complete", m.constrain_support(raw, self.words, [
            {"token_id": 1, "piece": b" ", "mass": 1}])["state"])

    def test_crosscheck_nonempty_frozen_validator(self):
        values = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        raw = json.dumps({"words": [{w: values} for w in self.words]}, ensure_ascii=False).encode()
        self.assertTrue(validate_response(raw, self.words)["format_accepted"])
        self.assertEqual("complete", m.classify_prefix(raw, self.words)["state"])

    def test_source_binding_changes_law_and_draw_context(self):
        rows = [{"token_id": 0, "piece": b"{", "mass": 1}]
        first = m.constrain_support(b"", ["x"], rows)
        second = m.constrain_support(b"", ["y"], rows)
        self.assertNotEqual(first["law_sha256"], second["law_sha256"])
        self.assertNotEqual(first["words_sha256"], second["words_sha256"])

    def test_every_split_of_unicode_and_escaped_surrogate(self):
        raws = [b'{"words":[{"' + "😀".encode() + b'":[]}]}',
                b'{"words":[{"\\ud83d\\ude00":[]}]}']
        for raw in raws:
            for n in range(len(raw)):
                with self.subTest(raw=raw, n=n):
                    self.assertEqual("incomplete", m.classify_prefix(raw[:n], ["😀"])["state"])
            self.assertEqual("complete", m.classify_prefix(raw, ["😀"])["state"])

    def test_partial_unicode_escape_irreversibility(self):
        invalid = [b'{"\\u00f', b'{"words":[{"\\u005',
                   b'{"words":[{"x":["\\u000', b'{"words":[{"x":["\\udc',
                   b'{"words":[{"x":["\\u2028', b'{"words":[{"x":["\\u200b']
        for raw in invalid:
            self.assertEqual("invalid_prefix", m.classify_prefix(raw, ["x"])["state"], raw)

    def test_hmac_resource_cap_no_fallback_or_seed_change(self):
        support = m.constrain_support(b"", ["x"], [
            {"token_id": 0, "piece": b"{", "mass": 1},
            {"token_id": 1, "piece": b" ", "mass": 2}])
        class AllOnes:
            def digest(self):
                return b"\xff" * 32
        with patch.object(m.hmac, "new", return_value=AllOnes()) as calls:
            result = m.select_token(support, bytes(32), "1" * 64, 0, 0)
        self.assertEqual("resource_refusal", result["state"])
        self.assertEqual(128, calls.call_count)

    def test_hmac_rejection_then_second_counter(self):
        support = m.constrain_support(b"", ["x"], [
            {"token_id": 0, "piece": b"{", "mass": 1},
            {"token_id": 1, "piece": b" ", "mass": 2}])
        class Block:
            def __init__(self, raw): self.raw = raw
            def digest(self): return self.raw
        with patch.object(m.hmac, "new", side_effect=[Block(b"\xff" * 32), Block(bytes(32))]):
            result = m.select_token(support, bytes(32), "1" * 64, 0, 0)
        self.assertEqual(1, result["counter"])
        self.assertEqual(0, result["ticket"])
        self.assertEqual(0, result["token_id"])

    def test_byte_resource_limit_is_not_grammar_filter(self):
        support = m.constrain_support(b"", ["x"], [
            {"token_id": 0, "piece": b"{", "mass": 1},
            {"token_id": 1, "piece": b"invalid oversized", "mass": 2}], max_bytes=5)
        self.assertEqual("resource_refusal", support["state"])

    def test_empty_input_has_no_defined_retained_fraction(self):
        law = m.constrain_support(b"", ["x"], [])
        self.assertEqual("empty_support", law["state"])
        self.assertIsNone(law["retained_fraction"])

    def test_canonical_order_and_input_not_mutated(self):
        rows = [{"token_id": 8, "piece": b"{", "mass": 2},
                {"token_id": 3, "piece": b" ", "mass": 3}]
        a = m.constrain_support(b"", ["x"], rows)
        b = m.constrain_support(b"", ["x"], list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual([8, 3], [r["token_id"] for r in rows])


if __name__ == "__main__":
    unittest.main()
