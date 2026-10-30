"""Pure synthetic independent-auditor tests; no production parser/sampler imports."""
import ast
import copy
from decimal import Decimal, localcontext
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import candidate_native_v2_audit_math as audit

WORDS = ["Popular", "Knowledge", "Article:", "Understanding", "Chinese"]
PROFILE = {"ready": {"vocabulary_size": 1000}, "special_ids": [0, 1, 2]}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def answer(words=WORDS, alternatives=None):
    return encode({"words": [{w: [] if alternatives is None else alternatives} for w in words]})


def reference_weights(values):
    """80-digit decimal reference independent of replay's binary exp/fsum law.

    Chosen non-boundary fixtures have identical integer allocations. Runtime
    binary arithmetic remains the specified replay law; this is a cross-check.
    """
    entries = sorted(((100 + i, struct.pack(">f", x).hex(), struct.unpack(">f", struct.pack(">f", x))[0])
                      for i, x in enumerate(values)), key=lambda item: (-item[2], item[0]))
    with localcontext() as ctx:
        ctx.prec = 80
        masses = [((Decimal.from_float(x) - Decimal.from_float(entries[0][2])) / Decimal.from_float(0.7)).exp()
                  for _, _, x in entries]
        masses = [x / sum(masses) * audit.TOTAL for x in masses]
        weights = [int(x) for x in masses]
        for i in sorted(range(50), key=lambda i: (-(masses[i] - weights[i]), entries[i][0]))[:audit.TOTAL - sum(weights)]:
            weights[i] += 1
    return [{"token_id": tid, "logit_hex": raw, "weight": weight}
            for (tid, raw, _), weight in zip(entries, weights)]


def evidence_fixture(weights=None, prefix=None, request=None, attempt=0):
    prefix = [] if prefix is None else prefix
    request = audit.REQUEST_HASHES[0] if request is None else request
    weights = [{"token_id": 100 + i, "logit_hex": "00000000", "weight": 335545 if i < 16 else 335544}
               for i in range(50)] if weights is None else weights
    seed_input = ["author-candidate-local-native-v2", "fixed-public-diagnostic-seed", request, attempt]
    seed = hashlib.sha256(encode(seed_input)).hexdigest()
    context = ["author-candidate-local-native-v2", request, attempt, len(prefix), hashlib.sha256(encode(prefix)).hexdigest()]
    context_bytes = encode(context)
    draw = int.from_bytes(hmac.new(bytes.fromhex(seed), context_bytes, hashlib.sha256).digest()[:3], "big")
    cumulative = 0
    selected = None
    for row in weights:
        cumulative += row["weight"]
        if selected is None and draw < cumulative:
            selected = row["token_id"]
    return ({"request_sha256": request, "seed_hex": seed, "prefix_ids": prefix, "weights": weights,
             "draw_uint24": draw, "draw_context_sha256": hashlib.sha256(context_bytes).hexdigest(),
             "temperature": 0.7, "integer_total": 16777216}, selected)


class PrefixTests(unittest.TestCase):
    def test_no_production_or_optional_imports(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        allowed = {"__future__", "hashlib", "hmac", "json", "math", "re", "unicodedata"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name in allowed for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_every_byte_prefix_of_independently_generated_valid_objects(self):
        rng = random.Random(740102)
        atoms = [None, True, False, 0, -1, 1.25, 1e20, "", "é", "汉字", "𝒜", 'a\\"b', "\n"]
        def node(depth):
            if depth == 0 or rng.randrange(3) == 0:
                return rng.choice(atoms)
            if rng.randrange(2):
                return [node(depth - 1) for _ in range(rng.randrange(4))]
            return {"key" + str(i): node(depth - 1) for i in range(rng.randrange(4))}
        documents = [{}, {"x": atoms}] + [{"tree": node(3)} for _ in range(100)]
        for obj in documents:
            for ascii_only in (False, True):
                raw = json.dumps(obj, ensure_ascii=ascii_only, separators=(",", ":")).encode()
                self.assertEqual(audit.prefix_state(raw), "complete")
                for end in range(len(raw)):
                    with self.subTest(raw=raw, end=end):
                        self.assertEqual(audit.prefix_state(raw[:end]), "incomplete")

    def test_complete_general_grammar_and_whitespace(self):
        for raw in [b"{}", b" \t\r\n{}\t", b'{"n":-1.2e+3}', b'{"t":true,"f":false,"n":null}',
                    b'{"a":[{},[],"",0]}', b'{"x":"\\ud800"}', b'{"x":1e999}', b'{"x":0,"x":1}']:
            self.assertEqual(audit.prefix_state(raw), "complete", raw)

    def test_incomplete_numbers_literals_strings_containers(self):
        for raw in [b"", b"   ", b"{", b'{"x"', b'{"x":', b'{"x":-', b'{"x":1.', b'{"x":1e',
                    b'{"x":1e+', b'{"x":1e-', b'{"x":tru', b'{"x":nul', b'{"x":fals',
                    b'{"x":"\\', b'{"x":"\\u', b'{"x":"\\u123', b'{"x":[', b'{"x":[],']:
            self.assertEqual(audit.prefix_state(raw), "incomplete", raw)

    def test_irreparable_json_sequences(self):
        for raw in [b"```json", b"Answer: {}", b"[]", b"null", b"\xef\xbb\xbf{}", b"{}x", b"{}{}",
                    b"{,", b"{'x':0}", b'{"x":01}', b'{"x":+1}', b'{"x":.1}', b'{"x":1.e0}',
                    b'{"x":1e+}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":undefined}', b'{"x":truE}',
                    b'{"x":"\\q"}', b'{"x":"\\u0g00"}', b'{"x":"\n"}', b'{"x":[1,]}', b'{"x":0,}',
                    b'{"x":true false}', b'{"x" 1}', b'{"x":[1 2]}', b'{"x":"a" "b"}']:
            self.assertEqual(audit.prefix_state(raw), "invalid_prefix", raw)

    def test_utf8_conservative_frozen_policy(self):
        for raw in [b'{"x":"\xc2', b'{"x":"\xe2\x82', b'{"x":"\xf0\x9d\x92', b'junk\xc2', b'{}\xe2']:
            self.assertEqual(audit.prefix_state(raw), "incomplete")
        for raw in [b'\xff', b'\xc0\x80', b'{"x":"\xed\xa0\x80"}', b'\xe2A', b'{}\x80']:
            self.assertEqual(audit.prefix_state(raw), "invalid_prefix")

    def test_size_limit_precedes_grammar(self):
        self.assertEqual(audit.prefix_state(b" " * 262144), "incomplete")
        self.assertEqual(audit.prefix_state(b" " * 262145), "resource_refusal")
        self.assertEqual(audit.prefix_state(b"x" * 262145), "resource_refusal")
        self.assertEqual(audit.prefix_state(b"{}" + b" " * 262142), "complete")

    def test_depth_limit_not_syntax_error(self):
        # Root depth0; first array depth1; scalar in 31 arrays depth32.
        self.assertEqual(audit.prefix_state(b'{"x":' + b'[' * 31 + b'0' + b']' * 31 + b'}'), "complete")
        self.assertEqual(audit.prefix_state(b'{"x":' + b'[' * 32 + b']' * 32 + b'}'), "complete")
        self.assertEqual(audit.prefix_state(b'{"x":' + b'[' * 32 + b'0' + b']' * 32 + b'}'), "resource_refusal")
        self.assertEqual(audit.prefix_state(b'{"x":' + b'[' * 32), "incomplete")
        self.assertEqual(audit.prefix_state(b'{"x":' + b'[' * 33), "resource_refusal")
        self.assertEqual(audit.prefix_state(b'{"x":' * 33), "resource_refusal")

    def test_type_required(self):
        for bad in ["{}", bytearray(b"{}"), None, 4]:
            with self.assertRaises(TypeError):
                audit.prefix_state(bad)


class FormatTests(unittest.TestCase):
    def accepted(self, raw, words=WORDS):
        result = audit.assess_format(raw, words)
        self.assertEqual(result["outcome"], "format_accepted")
        self.assertEqual(result["prefix_state"], "complete")
        return result["format_result"]

    def rejected(self, raw, words=WORDS):
        result = audit.assess_format(raw, words)
        self.assertEqual(result["outcome"], "format_rejected")
        self.assertIsNone(result["format_result"])

    def test_all_abstention_complete_format_only(self):
        raw = answer()
        result = self.accepted(raw)
        self.assertEqual(result["abstained_word_rows"], 5)
        self.assertEqual(result["candidate_strings_returned"], 0)
        self.assertEqual(result["raw_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertIsNone(result["usable_candidate_rows"])
        self.assertIsNone(result["capacity_bits"])
        self.assertIs(result["semantic_equivalence_verified"], False)
        self.assertIs(result["full_cover_resource_complete"], False)

    def test_mixed_rows_and_visible_unicode(self):
        obj = json.loads(answer())
        obj["words"][0][WORDS[0]] = ["é", "汉字", "😀", "beta", "gamma", "delta", "epsilon", "zeta"]
        result = self.accepted(encode(obj))
        self.assertEqual((result["abstained_word_rows"], result["nonempty_candidate_rows"], result["candidate_strings_returned"]), (4, 1, 8))

    def test_repeat_source_words_preserves_positions(self):
        words = ["the", "cat", "the", "runs", "the"]
        self.accepted(answer(words), words)

    def test_duplicate_keys_and_escaped_equivalent_keys_refused(self):
        raw = answer()
        self.rejected(raw.replace(b'"words":', b'"words":[],"words":'))
        self.rejected(raw.replace(b'"Popular":[]', b'"Popular":[],"Popular":[]'))
        self.rejected(raw.replace(b'"Popular":[]', b'"Popular":[],"P\\u006fpular":[]'))

    def test_shape_cardinality_identity_order(self):
        obj = json.loads(answer())
        variants = [{}, {"words": []}, {"words": obj["words"], "extra": 1}, {"words": {}},
                    {"words": obj["words"][:-1]}, {"words": obj["words"] + [obj["words"][0]]},
                    {"words": list(reversed(obj["words"]))}, {"words": [[w, []] for w in WORDS]}]
        for value in variants:
            self.rejected(encode(value))
        for count in [1, 7, 9]:
            self.rejected(answer(alternatives=[str(i) for i in range(count)]))

    def test_candidate_surface_refusals(self):
        for bad in ["", " alpha", "alpha ", None, 7, True, "Popular", "POPULAR", "x\x80y", "x\xadY",
                    "x\u2028y", "x\u2029y", "x\ny", "\ud800", "\udfff"]:
            obj = json.loads(answer())
            obj["words"][0][WORDS[0]] = [bad] + ["candidate" + str(i) for i in range(7)]
            self.rejected(json.dumps(obj).encode())
        for values in [["red", "RED"], ["red blue", "red  blue"], ["Straße", "STRASSE"]]:
            obj = json.loads(answer())
            obj["words"][0][WORDS[0]] = values + ["candidate" + str(i) for i in range(6)]
            self.rejected(encode(obj))

    def test_invalid_expected_tokens_cannot_be_hidden_by_matching_raw(self):
        for bad in ["", "a b", "a\tb", "x\x80y", "x\u200by", "x\u2028y", "\ud800"]:
            words = [bad] + WORDS[1:]
            self.rejected(json.dumps({"words": [{w: []} for w in words]}).encode(), words)
        for words in [WORDS[:4], WORDS + ["extra"], "abcde", [None] + WORDS[1:]]:
            self.rejected(answer(), words)

    def test_whole_committed_final_token_and_no_repairs(self):
        raw = answer()
        for malformed in [b"```json\n" + raw + b"\n```", raw + b" explanation", raw + raw, raw[:-1]]:
            self.rejected(malformed)
        self.accepted(raw + b"\n  ")
        self.assertEqual(audit.assess_format(b" " * 262145, WORDS)["outcome"], "parser_resource_refusal")


class SamplingTests(unittest.TestCase):
    def replay(self, evidence, token, prefix=None, request=None, attempt=0, profile=PROFILE):
        return audit.replay_sample(evidence, token, [] if prefix is None else prefix,
                                   audit.REQUEST_HASHES[0] if request is None else request, attempt, profile)

    def test_binary32_arithmetic_decoder_against_struct(self):
        rng = random.Random(2323)
        patterns = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff]
        patterns += [rng.randrange(1 << 32) for _ in range(2048)]
        for bits in patterns:
            hexadecimal = f"{bits:08x}"
            reference = struct.unpack(">f", bytes.fromhex(hexadecimal))[0]
            if math.isfinite(reference):
                result = audit._float32(hexadecimal)
                self.assertEqual(result, reference)
                if result == 0:
                    self.assertEqual(math.copysign(1, result), math.copysign(1, reference))
            else:
                with self.assertRaises(ValueError):
                    audit._float32(hexadecimal)

    def test_invalid_binary32_encodings(self):
        for raw in ["7f800000", "ff800000", "7fc00000", "ffffffff", "0000000", "000000000", "zzzzzzzz", "00 00 00", 0, None]:
            with self.assertRaises(ValueError):
                audit._float32(raw)

    def test_equal_logits_analytic_allocation_and_hmac_golden(self):
        evidence, token = evidence_fixture()
        result = self.replay(evidence, token)
        self.assertEqual(result["seed_hex"], "3ad523f14700d0485cbab33d69f1acf23f24b5260e0d164542b1bb4e009bac5d")
        self.assertEqual(result["draw_context_sha256"], "82ff40f6dcddb82936a4f3ee50ac72dfc845af3f7aa10223b7384eb768134f7f")
        self.assertEqual(result["draw_uint24"], 1404571)
        self.assertEqual(result["token_id"], 104)
        self.assertEqual(result["integer_total"], 16777216)

    def test_varied_distributions_against_high_precision_reference(self):
        rng = random.Random(683)
        # Keep construction explicit: multiple deterministic, non-boundary laws.
        fixtures = [[-i / 10 for i in range(50)], [0] * 25 + [-1] * 25]
        fixtures.extend([[rng.uniform(-20, 3) for _ in range(50)] for _ in range(12)])
        for values in fixtures:
            evidence, token = evidence_fixture(reference_weights(values))
            self.assertTrue(self.replay(evidence, token)["verified"])

    def test_underflow_zero_weights_and_negative_zero_ties(self):
        rows = [{"token_id": 100 + i, "logit_hex": "00000000" if i == 0 else "c47a0000",
                 "weight": 16777216 if i == 0 else 0} for i in range(50)]
        evidence, token = evidence_fixture(rows)
        self.assertEqual(self.replay(evidence, token)["token_id"], 100)
        evidence, token = evidence_fixture()
        evidence["weights"][1]["logit_hex"] = "80000000"
        self.assertTrue(self.replay(evidence, token)["verified"])

    def test_all_fixed_request_attempt_prefix_contexts(self):
        for request in audit.REQUEST_HASHES:
            for attempt in range(3):
                for prefix in [[], [104], [104, 110, 104], [104] * 511]:
                    evidence, token = evidence_fixture(prefix=prefix, request=request, attempt=attempt)
                    self.assertTrue(self.replay(evidence, token, prefix, request, attempt)["verified"])

    def test_each_evidence_identity_field_tamper_refused(self):
        evidence, token = evidence_fixture()
        mutations = {"seed_hex": "0" * 64, "request_sha256": audit.REQUEST_HASHES[1], "prefix_ids": [104],
                     "draw_uint24": evidence["draw_uint24"] + 1, "draw_context_sha256": "0" * 64,
                     "temperature": 0.8, "integer_total": 16777215}
        for key, value in mutations.items():
            altered = copy.deepcopy(evidence); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.replay(altered, token)
        for altered in [{**evidence, "extra": None}, {k: v for k, v in evidence.items() if k != "seed_hex"}]:
            with self.assertRaises(ValueError):
                self.replay(altered, token)
        with self.assertRaises(ValueError):
            self.replay(evidence, token + 1)

    def test_quantization_and_support_tamper_refused(self):
        original, token = evidence_fixture()
        mutations = []
        def changed(fn):
            value = copy.deepcopy(original); fn(value["weights"]); mutations.append(value)
        changed(lambda rows: rows[0].update(weight=rows[0]["weight"] + 1))
        changed(lambda rows: rows[0].update(weight=True))
        changed(lambda rows: rows[0].update(weight=-1))
        changed(lambda rows: rows[0].update(token_id=True))
        changed(lambda rows: rows[0].update(token_id=0))
        changed(lambda rows: rows[0].update(token_id=1000))
        changed(lambda rows: rows[0].update(token_id=101))
        changed(lambda rows: rows[0].update(logit_hex="7fc00000"))
        changed(lambda rows: rows[0].update(logit_hex="3f800000"))
        changed(lambda rows: rows.reverse())
        changed(lambda rows: rows.pop())
        changed(lambda rows: rows.append(rows[0]))
        changed(lambda rows: rows[0].update(extra=0))
        for altered in mutations:
            with self.assertRaises(ValueError):
                self.replay(altered, token)

    def test_context_and_profile_input_validation(self):
        evidence, token = evidence_fixture()
        for kwargs in [{"attempt": True}, {"attempt": -1}, {"attempt": 3}, {"request": "0" * 64},
                       {"prefix": [True]}, {"prefix": [0]}, {"prefix": [1000]}, {"prefix": [104] * 512},
                       {"profile": {}}, {"profile": {"ready": {"vocabulary_size": True}, "special_ids": []}},
                       {"profile": {"ready": {"vocabulary_size": 1000}, "special_ids": [True]}}]:
            with self.assertRaises(ValueError):
                self.replay(evidence, token, **kwargs)
        with self.assertRaises(ValueError):
            self.replay(evidence, True)


if __name__ == "__main__":
    unittest.main()
