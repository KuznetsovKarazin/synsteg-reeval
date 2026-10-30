"""Synthetic tests only. No saved real model trace is read by these tests."""
import importlib.util
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_candidate_mask_order_v1.py"
SPEC = importlib.util.spec_from_file_location("mask_order_analysis", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def row(token, value):
    return {"token_id": token, "logit_hex": struct.pack("!f", value).hex()}


def evidence(logits):
    weights = [{"token_id": r["token_id"], "logit_hex": r["logit_hex"], "weight": r["mass"]}
               for r in M.quantize(logits)["rows"]]
    pieces = [{"token_id": r["token_id"], "piece_hex": bytes([r["token_id"] + 32]).hex()}
              for r in weights]
    return weights, pieces


def classifier(eligible, resource=()):
    def classify(raw, words, max_bytes):
        if not raw:
            return {"state": "incomplete"}
        if raw[0] - 32 in resource:
            return {"state": "resource_refusal"}
        return {"state": "incomplete" if raw[0] - 32 in eligible else "invalid_prefix"}
    return classify


class QuantizerTests(unittest.TestCase):
    def test_singleton_all_mass(self):
        self.assertEqual(M.quantize([row(4, -1000)])["rows"][0]["mass"], M.TOTAL)

    def test_equal_logit_remainder_tie_by_id(self):
        result = M.quantize([row(3, 0), row(1, 0), row(2, 0)], total=2)
        self.assertEqual([(r["token_id"], r["mass"]) for r in result["rows"]], [(1, 1), (2, 1), (3, 0)])

    def test_input_permutation_invariant(self):
        rows = [row(7, -1), row(8, 2), row(9, 2)]
        self.assertEqual(M.quantize(rows), M.quantize(rows[::-1]))

    def test_extreme_spread_records_underflow(self):
        result = M.quantize([row(0, 3e38), row(1, -3e38)])
        self.assertEqual(result["exp_underflow_ids"], [1])
        self.assertEqual([r["mass"] for r in result["rows"]], [M.TOTAL, 0])

    def test_finite_shift_invariance_for_exact_inputs(self):
        masses = lambda x: [r["mass"] for r in M.quantize(x)["rows"]]
        self.assertEqual(masses([row(0, 1), row(1, 2)]), masses([row(0, 101), row(1, 102)]))

    def test_bool_and_duplicate_token_rejected(self):
        for rows in ([row(True, 0)], [row(1, 0), row(1, 1)]):
            with self.assertRaises(ValueError): M.quantize(rows)

    def test_nonfinite_logits_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError): M.quantize([row(0, value)])

    def test_closed_hex_schema(self):
        for entry in ({"token_id": 0, "logit_hex": "00"},
                      {"token_id": 0, "logit_hex": "3F800000"},
                      {"token_id": 0, "logit_hex": "00000000", "extra": 1}):
            with self.assertRaises(ValueError): M.quantize([entry])

    def test_invalid_total_temperature(self):
        for total in (True, 0, -1, M.TOTAL + 1, 2.0):
            with self.assertRaises(ValueError): M.quantize([row(0, 0)], total=total)
        for temp in (1, False, 0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError): M.quantize([row(0, 0)], temperature=temp)

    def test_cardinality_boundary(self):
        for rows in ([], [row(i, i) for i in range(51)]):
            with self.assertRaises(ValueError): M.quantize(rows)

    def test_masking_can_remove_individual_old_positive_mass(self):
        # Independent reviewer supplied this largest-remainder counterexample.
        # Nonempty total support is guaranteed, per-token monotonicity is not.
        logits = [{"token_id": i, "logit_hex": value} for i, value in enumerate(
            ["41328e02", "41328e02", "bf17d5f5", "00000000"])]
        masses = lambda values: {r["token_id"]: r["mass"] for r in M.quantize(values)["rows"]}
        self.assertEqual(masses(logits), {0: 8388607, 1: 8388607, 3: 1, 2: 1})
        self.assertEqual(masses(logits[:3]), {0: 8388608, 1: 8388608, 2: 0})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.weights, self.pieces = evidence([row(i, (50 - i) / 10) for i in range(50)])

    def compare(self, eligible, resource=()):
        return M.compare_probe(self.weights, self.pieces, b"", ["source"], classifier(eligible, resource))

    def test_all_legal_laws_identical(self):
        result = self.compare(set(range(50)))
        self.assertEqual(result["total_variation"]["numerator"], 0)
        self.assertEqual(result["old_retained_mass"], M.TOTAL)

    def test_all_illegal_empty_not_zero_success(self):
        result = self.compare(set())
        self.assertEqual(result["state"], "empty_schema_support")
        self.assertEqual(result["old_state"], "empty_support")
        self.assertIsNone(result["total_variation"])

    def test_zero_mass_legal_token_recovered(self):
        self.weights, self.pieces = evidence([row(i, 20 if i == 0 else -20 - i) for i in range(50)])
        result = self.compare({1})
        self.assertEqual(result["old_state"], "empty_support")
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["recovered_positive_ids"], [1])
        self.assertEqual(next(r for r in result["rows"] if r["token_id"] == 1)["alternative_mass"], M.TOTAL)
        self.assertIsNone(result["total_variation"])

    def test_underflow_before_mask_resolved_by_eligible_centering(self):
        self.weights, self.pieces = evidence([row(i, 3e38 if i == 0 else -3e38) for i in range(50)])
        result = self.compare({1, 2})
        self.assertEqual(result["recovered_positive_ids"], [1, 2])
        self.assertEqual(result["alternative_exp_underflow_ids"], [])
        self.assertEqual(result["new_positive_eligible_count"], 2)

    def test_resource_limit_in_zero_mass_row_refuses_whole_probe(self):
        self.weights, self.pieces = evidence([row(i, 20 if i == 0 else -20 - i) for i in range(50)])
        result = self.compare({0}, {49})
        self.assertEqual(result["state"], "resource_refusal")
        self.assertIsNone(result["total_variation"])

    def test_alternative_still_can_have_zero_mass(self):
        self.weights, self.pieces = evidence([row(i, 0 if i == 0 else -100 - i) for i in range(50)])
        result = self.compare({0, 1})
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["new_positive_eligible_count"], 1)

    def test_pinned_mass_tampering_rejected(self):
        self.weights[0]["weight"] += 1
        with self.assertRaises(ValueError): self.compare({0})

    def test_piece_reordering_rejected(self):
        self.pieces[0], self.pieces[1] = self.pieces[1], self.pieces[0]
        with self.assertRaises(ValueError): self.compare({0})

    def test_empty_oversized_malformed_piece_rejected(self):
        for value in ("", "ffg0", "00" * (262144 + 65536 + 1)):
            self.pieces[0]["piece_hex"] = value
            with self.assertRaises(ValueError): self.compare({0})

    def test_randomized_finite_nonempty_support_conserves_mass(self):
        # 100 nested deterministic synthetic cases, not independent model runs.
        rng = random.Random(20261026)
        for case in range(100):
            self.weights, self.pieces = evidence([row(i, rng.uniform(-100, 100)) for i in range(50)])
            eligible = set(rng.sample(range(50), rng.randint(1, 50)))
            result = self.compare(eligible)
            self.assertEqual(result["state"], "ready")
            self.assertEqual(sum(r["alternative_mass"] for r in result["rows"]), M.TOTAL)
            self.assertGreaterEqual(result["new_positive_eligible_count"], 1)
            self.assertTrue(all(r["alternative_mass"] == 0 for r in result["rows"] if r["token_id"] not in eligible))
            if result["total_variation"] is not None:
                x = result["total_variation"]
                self.assertTrue(0 <= x["numerator"] <= x["denominator"])


class InputTests(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError): M.strict_json(text)

    def test_nonregular_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            file = root / "file"; file.write_text("ok")
            link = root / "link"; link.symlink_to(file)
            with self.assertRaises(ValueError): M.regular_bytes(link)
            with self.assertRaises(ValueError): M.regular_bytes(root)
            self.assertEqual(M.regular_bytes(file), b"ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
