import json
import math
import unittest
import zlib

from synsteg_audit.metrics import delivery_metrics, serialized_key_cost


class DeliveryMetricsTests(unittest.TestCase):
    def metric(self, requested="1011", decoded="1011", used_bits=4, **kwargs):
        return delivery_metrics(requested, decoded, used_bits=used_bits,
                                cover="one two", stego="one two three four", **kwargs)

    def test_full_payload_and_actual_denominators(self):
        m = self.metric()
        self.assertTrue(m["full_message_success"])
        self.assertEqual(m["used_bpw_cover"], 2)
        self.assertEqual(m["used_bpw_stego"], 1)
        self.assertEqual(m["full_message_goodput_bpw_stego"], 1)
        self.assertEqual(m["bit_error_rate"], 0)

    def test_correct_prefix_is_not_full_delivery(self):
        m = self.metric(decoded="10", used_bits=2)
        self.assertTrue(m["prefix_exact"])
        self.assertTrue(m["truncated"])
        self.assertFalse(m["full_message_success"])
        self.assertEqual(m["used_bpw_stego"], .5)
        self.assertEqual(m["missing_bits"], 2)
        self.assertEqual(m["full_message_goodput_bpw_stego"], 0)
        self.assertIsNone(m["bit_error_rate"])

    def test_decoder_error_does_not_masquerade_as_empty_prefix_success(self):
        m = self.metric(decoded=None, used_bits=0, error="decoder failed")
        self.assertIsNone(m["full_exact"])
        self.assertIsNone(m["prefix_exact"])
        self.assertIsNone(m["decoded_bpw_stego"])
        self.assertFalse(m["full_message_success"])
        m2 = self.metric(decoded="", used_bits=0)
        self.assertTrue(m2["prefix_exact"])  # Mathematical empty prefix only.
        self.assertFalse(m2["nonempty_prefix_exact"])
        self.assertFalse(m2["full_message_success"])

    def test_empty_request_is_not_evidence_of_payload_reliability(self):
        m = self.metric(requested="", decoded="", used_bits=0)
        self.assertTrue(m["full_exact"])
        self.assertFalse(m["payload_eligible"])
        self.assertFalse(m["full_message_success"])
        self.assertIsNone(m["bit_error_rate"])

    def test_wrong_and_extra_bits_separately(self):
        m = self.metric(decoded="1001")
        self.assertEqual(m["bit_error_rate"], .25)
        self.assertEqual(m["correct_leading_bits"], 2)
        self.assertFalse(m["prefix_exact"])
        m = self.metric(decoded="10110")
        self.assertEqual(m["extra_bits"], 1)
        self.assertIsNone(m["bit_error_rate"])
        self.assertFalse(m["full_message_success"])

    def test_zero_word_denominator_stays_undefined_and_json_serializable(self):
        m = delivery_metrics("1", "1", used_bits=1, cover="", stego=" ")
        self.assertIsNone(m["used_bpw_stego"])
        self.assertIsNone(m["used_bpw_cover"])
        json.dumps(m, allow_nan=False)

    def test_error_or_encoder_shortfall_blocks_success_despite_matching_decoder(self):
        self.assertFalse(self.metric(error="untrusted decoder output")["full_message_success"])
        self.assertFalse(self.metric(used_bits=3)["full_message_success"])

    def test_invalid_data_fails_explicitly(self):
        for kwargs in [{"requested": "1021"}, {"decoded": "bits"},
                       {"used_bits": -1}, {"used_bits": 5}, {"used_bits": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.metric(**kwargs)

    def test_key_cost_counts_utf8_not_characters(self):
        key = {"слово": ["ліс", "forest"]}
        raw = '{"слово":["ліс","forest"]}'.encode("utf-8")
        result = serialized_key_cost(key)
        self.assertEqual(result["key_json_bytes"], len(raw))
        self.assertEqual(result["key_zlib9_bytes"], len(zlib.compress(raw, 9)))
        self.assertGreater(result["key_json_bytes"], len(raw.decode()))
        with self.assertRaises(ValueError):
            serialized_key_cost({"invalid": math.nan})


if __name__ == "__main__":
    unittest.main()
