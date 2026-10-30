"""Pure-format regression tests; never import native/model/tokenizer/codec code."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_format_v2", ROOT / "src/synsteg_audit/author_candidate_format_v2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def response(words, nonempty=0):
    return {"words": [{w: [f"Alternative{j}" for j in range(8)] if i < nonempty else []}
                      for i, w in enumerate(words)]}


class PrefixTests(unittest.TestCase):
    def test_all_byte_cuts_of_complete_json_objects_remain_possible(self):
        values = [{}, {"a": []}, {"a": [True, False, None]},
                  {"a": -1.25e30}, {"a": 0}, {"a": 1e-6},
                  {"a": "Привет 🌞 中文"}, {"a": "\\\"\n\t\r\b\f/"},
                  {"a": [{"nested": {"value": [1, 2, "x"]}}]},
                  response(m.FIXED_WORDS[0], 5)]
        for obj in values:
            for ascii_only in (False, True):
                raw = json.dumps(obj, ensure_ascii=ascii_only).encode()
                for cut in range(len(raw) + 1):
                    with self.subTest(obj=obj, ascii=ascii_only, cut=cut):
                        self.assertIn(m.classify_prefix(raw[:cut])["state"], ("incomplete", "complete"))
                self.assertEqual(m.classify_prefix(raw)["state"], "complete")

    def test_whitespace_permitted_without_rewriting(self):
        self.assertEqual(m.classify_prefix(b" \n\t\r{} \t\r\n")["state"], "complete")

    def test_empty_and_whitespace_incomplete(self):
        for raw in (b"", b" \t\n\r"):
            self.assertEqual(m.classify_prefix(raw)["state"], "incomplete")

    def test_fence_refused_at_first_byte(self):
        for raw in (b"`", b"```json\n{}\n```", b" \n```json"):
            self.assertEqual(m.classify_prefix(raw)["state"], "invalid_prefix")

    def test_nonobject_root_refused(self):
        for raw in (b"[", b"null", b"1", b'"x"', b"true"):
            self.assertEqual(m.classify_prefix(raw)["state"], "invalid_prefix")

    def test_closed_json_followed_by_junk_is_not_salvaged(self):
        for raw in (b"{}{}", b"{} prose", b"{}\n```", b"{}\0"):
            self.assertEqual(m.classify_prefix(raw)["state"], "invalid_prefix")

    def test_invalid_grammar(self):
        bad = [b'{"a":01}', b'{"a":+1}', b'{"a":.2}', b'{"a":1.}',
               b'{"a":1e}', b'{"a":1e+}', b'{"a":NaN}', b'{"a":Infinity}',
               b'{"a":True}', b'{"a":truX}', b'{"a" 1}', b'{"a":}',
               b'{"a":[],,}', b'{"a":[],}', b'{"a":[1,]}', b'{a:1}',
               b'{"a":"\\q"}', b'{"a":"\\u00X0"}', b'{"a":"\x01"}',
               b'\xef\xbb\xbf{}', b'\xc0\x80', b'{"a":"\xff"}']
        for raw in bad:
            with self.subTest(raw=raw):
                self.assertEqual(m.classify_prefix(raw)["state"], "invalid_prefix")

    def test_extendable_number_and_string_prefixes(self):
        for raw in (b'{"a":-', b'{"a":1.', b'{"a":1e', b'{"a":1e+',
                    b'{"a":"\\', b'{"a":"\\u', b'{"a":"\\u0',
                    b'{"a":"\\u00', b'{"a":"\\u000', b'{"a":tru'):
            self.assertEqual(m.classify_prefix(raw)["state"], "incomplete")

    def test_partial_utf8_is_not_replaced(self):
        self.assertEqual(m.classify_prefix(b'{"a":"\xf0\x9f')["state"], "incomplete")
        self.assertEqual(m.classify_prefix(b'{"a":"\xf0\x9f\x98\x80"}')["state"], "complete")

    def test_resource_limits_are_not_syntax_claims(self):
        self.assertEqual(m.classify_prefix(b" " * (m.MAX_RAW_BYTES + 1))["state"], "resource_refusal")
        self.assertEqual(m.classify_prefix(b'{"a":' + b'[' * 40)["state"], "resource_refusal")

    def test_requires_exact_bytes(self):
        for wrong in ("{}", bytearray(b"{}"), None, 0):
            with self.assertRaises(TypeError):
                m.classify_prefix(wrong)


class ResponseTests(unittest.TestCase):
    words = m.FIXED_WORDS[0]

    def validate(self, obj):
        return m.validate_response(m.canonical(obj), self.words)

    def test_all_abstention_is_format_only(self):
        out = self.validate(response(self.words))
        self.assertTrue(out["format_accepted"])
        self.assertEqual(out["abstained_word_rows"], 5)
        self.assertEqual(out["candidate_strings_returned"], 0)
        self.assertIsNone(out["capacity_bits"])
        self.assertIsNone(out["usable_candidate_rows"])
        self.assertFalse(out["semantic_equivalence_verified"])
        self.assertFalse(out["full_cover_resource_complete"])

    def test_nonempty_counts_are_not_usable_capacity(self):
        for n in range(6):
            out = self.validate(response(self.words, n))
            self.assertEqual(out["candidate_strings_returned"], n * 8)
            self.assertEqual(out["nonempty_candidate_rows"], n)
            self.assertEqual(out["abstained_word_rows"], 5 - n)
            self.assertIsNone(out["capacity_bits"])

    def test_raw_hash_is_exact_including_whitespace(self):
        raw = b"\n" + m.canonical(response(self.words)) + b" \t"
        out = m.validate_response(raw, self.words)
        self.assertEqual(out["raw_sha256"], m.digest(raw))

    def test_wrong_top_fields(self):
        for obj in ({"words": [], "note": "x"}, {"result": []}, {}, []):
            with self.assertRaises(ValueError):
                self.validate(obj)

    def test_all_source_rows_must_be_present(self):
        obj = response(self.words)
        obj["words"].pop()
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_extra_or_reordered_rows_refused(self):
        obj = response(self.words)
        obj["words"].reverse()
        with self.assertRaises(ValueError):
            self.validate(obj)
        obj = response(self.words)
        obj["words"].append({"extra": []})
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_placeholder_wrong_case_and_punctuation_keys_refused(self):
        for key in ("word_1", "popular", "Popular:"):
            obj = response(self.words)
            obj["words"][0] = {key: []}
            with self.assertRaises(ValueError):
                self.validate(obj)

    def test_row_extra_property_refused(self):
        obj = response(self.words)
        obj["words"][0]["note"] = []
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_duplicate_json_keys_not_silently_collapsed(self):
        for raw in (b'{"words":[],"words":[]}', b'{"words":[{"Popular":[],"Popular":[]}]}'):
            with self.assertRaises(ValueError):
                m.validate_response(raw, self.words)

    def test_candidate_count_is_zero_or_exactly_eight(self):
        for n in (1, 2, 4, 7, 9, 16):
            obj = response(self.words)
            obj["words"][0][self.words[0]] = [f"Alt{i}" for i in range(n)]
            with self.assertRaises(ValueError):
                self.validate(obj)

    def test_self_duplicate_case_and_whitespace_variants_refused(self):
        for replacement in ("Popular", "popular", "POPULAR", "Alternative1", "alternative1"):
            obj = response(self.words, 1)
            obj["words"][0][self.words[0]][0] = replacement
            with self.assertRaises(ValueError):
                self.validate(obj)
        obj = response(self.words, 1)
        obj["words"][0][self.words[0]][0:2] = ["two words", "two  words"]
        with self.assertRaises(ValueError):
            self.validate(obj)

    def test_padding_controls_nonstrings_and_surrogates_refused(self):
        for candidate in ("", " padded", "padded ", "line\nbreak", "a\tb", "a\x7fb",
                          "a\u0080b", "a\u009fb", "a\u200bb", "a\u200db", "a\u2028b", "a\u2029b",
                          1, True, None, "\ud800"):
            obj = response(self.words, 1)
            obj["words"][0][self.words[0]][0] = candidate
            raw = json.dumps(obj).encode()
            with self.assertRaises(ValueError):
                m.validate_response(raw, self.words)

    def test_repeated_source_tokens_have_separate_ordered_rows(self):
        raw = m.canonical({"words": [{"a": []}, {"a": []}]})
        self.assertEqual(m.validate_response(raw, ["a", "a"])["requested_word_rows"], 2)

    def test_invalid_expected_words_refused(self):
        for words in ([], ["x"] * 6, [""], [1], [" x"], ["two words"],
                      ["a\u0080b"], ["a\u200db"], ["a\nb"], ["\ud800"], "words"):
            with self.assertRaises(ValueError):
                m.validate_response(b"{}", words)

    def test_whole_fenced_or_trailing_response_not_repaired(self):
        raw = m.canonical(response(self.words))
        for prefix, suffix in ((b"```json\n", b"\n```"), (b"Here:", b""), (b"", b" explanation")):
            with self.assertRaises(ValueError):
                m.validate_response(prefix + raw + suffix, self.words)

    def test_linguistic_claim_not_mechanically_validated(self):
        out = self.validate(response(self.words, 5))
        self.assertTrue(out["format_accepted"])
        self.assertFalse(out["semantic_equivalence_verified"])
        self.assertIsNone(out["usable_candidate_rows"])


class BindingTests(unittest.TestCase):
    def test_exact_old_cohort_binds_new_prompt_without_reselection(self):
        raw = (ROOT / "experiments/author-candidate-local-native-v1/requests.json").read_bytes()
        new = m.build_requests(raw)
        self.assertFalse(new["generation_enabled"])
        self.assertEqual(new["full_cover_chunks"], 109)
        self.assertEqual(new["outside_diagnostic_chunks"], 107)
        for i, row in enumerate(new["requests"]):
            self.assertEqual(tuple(row["words"]), m.FIXED_WORDS[i])
            self.assertEqual(row["source_author_request_sha256"], m.AUTHOR_REQUEST_SHA256[i])
            self.assertNotEqual(row["request_sha256"], m.AUTHOR_REQUEST_SHA256[i])
            self.assertEqual(row["bits_per_word"], 3)
            self.assertEqual(row["additional_bits"], 0)
            self.assertNotIn("```", row["system"])
            self.assertNotIn("word_1", row["system"])

    def test_drifted_input_file_not_accepted(self):
        raw = (ROOT / "experiments/author-candidate-local-native-v1/requests.json").read_bytes()
        with self.assertRaises(ValueError):
            m.build_requests(raw + b" ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
