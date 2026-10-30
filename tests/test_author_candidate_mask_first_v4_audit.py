"""Synthetic-only audit tests: no model, tokenizer, old prefix or request data."""
from __future__ import annotations

import copy
import hashlib
import hmac
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("SYNSTEG_SOURCE_ROOT", ROOT))
if "synsteg_audit" not in sys.modules:
    package = types.ModuleType("synsteg_audit")
    package.__path__ = [str(ROOT / "src/synsteg_audit"), str(FROZEN / "src/synsteg_audit")]
    sys.modules["synsteg_audit"] = package

from synsteg_audit import author_candidate_mask_first_v4_audit as audit
from synsteg_audit import candidate_journal_guard_v3 as guard
from synsteg_audit.author_candidate_schema_v3 import constrain_support, select_token


def rows(piece=b" ", logit=0.0):
    return [{"token_id": i, "logit_hex": struct.pack(">f", logit).hex(), "piece_hex": piece.hex()}
            for i in range(50)]


def context(*, vocabulary=100, special=None):
    request = {"chunk_order": 0, "words": ["alpha"], "fixture": "synthetic-audit-only"}
    request["request_sha256"] = guard.sha(guard.canonical(request))
    profile = {"ready": {"vocabulary_size": vocabulary}, "special_ids": special or []}
    binding = {"experiment_id": audit.ID, "attempt_id": "chunk0-attempt0",
               "request_sha256": request["request_sha256"],
               "resource_sha256": guard.sha(guard.canonical(profile)),
               "protocol_sha256": "2" * 64, "token_cap": 512}
    return binding, profile, request


def evidence(raw, candidates, request, ids=None):
    return audit.expected_evidence(raw, request["words"], candidates, request["request_sha256"], ids or [])


def sealed(path, binding, events, stop):
    """Create a genuine frozen guard bundle from synthetic evidence only.

    Each event is (evidence, optional committed raw). None means a refusal.
    Guard validates physical-style event ordering; arithmetic is audited later.
    """
    ids, raw = [], b""
    with guard.AttemptWriter(path, binding) as writer:
        prepared = writer.prepare_prompt([1])
        writer.commit_prompt(prepared, 1)
        for sample, committed in events:
            prepared = writer.prepare_distribution()
            writer.commit_distribution(prepared, 1 + len(ids), sample_evidence=sample)
            if committed is not None:
                token = sample["selection"]["token_id"]
                prepared = writer.prepare_step(token, committed, sample_evidence=sample)
                writer.commit_step(prepared, 2 + len(ids))
                ids.append(token)
                raw = committed
        return writer.seal(ids, raw, stop_label=stop)["terminal_sha256"]


class IndependentEvidenceTests(unittest.TestCase):
    def test_equal_logit_remainders_and_hmac_interval(self):
        binding, _, request = context()
        sample = evidence(b"", rows(), request)
        masses = [r["weight"] for r in sample["mask_first"]["rows"]]
        self.assertEqual(masses, [335545] * 16 + [335544] * 34)
        self.assertEqual(sum(masses), 1 << 24)
        self.assertTrue(sample["mask_first"]["positive_support_structural_whitespace_only"])
        draw = sample["selection"]
        draw_context = {"domain": "author-candidate-schema-v3/draw-v1",
                        "request_sha256": binding["request_sha256"], "attempt": 0, "step": 0,
                        "law_sha256": sample["conditional_law"]["law_sha256"], "counter": 0}
        block = hmac.new(bytes.fromhex(sample["key_hex"]), guard.canonical(draw_context), hashlib.sha256).digest()
        ticket = int.from_bytes(block, "big") % (1 << 24)
        self.assertEqual(draw["ticket"], ticket)
        self.assertLessEqual(sum(masses[:draw["token_id"]]), ticket)
        self.assertLess(ticket, sum(masses[:draw["token_id"] + 1]))

    def test_ineligible_dominant_logits_do_not_starve_legal_support(self):
        _, _, request = context()
        candidates = rows(b"x", 10000.0)
        for i in range(46, 50):
            candidates[i] = {"token_id": i, "logit_hex": struct.pack(">f", -20000.0 - i).hex(),
                             "piece_hex": "20"}
        result = evidence(b"", candidates, request)["mask_first"]
        self.assertEqual(result["eligible_tokens"], 4)
        self.assertEqual(sum(r["weight"] for r in result["rows"]), audit.TOTAL)
        self.assertTrue(all(r["weight"] == 0 for r in result["rows"][:46]))
        self.assertTrue(all(r["weight"] > 0 for r in result["rows"][46:]))

    def test_underflow_retained_as_zero_row(self):
        _, _, request = context()
        candidates = rows(b"x", -1000.0)
        candidates[0] = {"token_id": 0, "logit_hex": "00000000", "piece_hex": "20"}
        candidates[1]["piece_hex"] = "09"
        result = evidence(b"", candidates, request)["mask_first"]
        self.assertEqual(result["eligible_tokens"], 2)
        self.assertEqual(result["positive_tokens"], 1)
        self.assertEqual(result["exponential_underflow_ids"], [1])
        self.assertEqual(result["rows"][0]["weight"], audit.TOTAL)
        self.assertEqual(result["rows"][1]["weight"], 0)

    def test_empty_and_whole_probe_resource_refusal(self):
        _, _, request = context()
        empty = evidence(b"", rows(b"x"), request)
        self.assertEqual(empty["selection"], {"state": "empty_support", "reason": "empty_schema_support"})
        self.assertIsNone(empty["conditional_law"])
        candidates = rows()
        candidates[-1]["piece_hex"] = (b" " * (audit.MAX_BYTES + 1)).hex()
        refused = evidence(b"", candidates, request)
        self.assertEqual(refused["selection"], {"state": "resource_refusal", "reason": "candidate byte budget"})
        self.assertEqual(len(refused["mask_first"]["rows"]), 50)
        self.assertTrue(all(r["weight"] == 0 for r in refused["mask_first"]["rows"]))

    def test_closed_rows_finite_logits_order_and_exact_types(self):
        _, _, request = context()
        variants = []
        variants.append(rows()[:-1])
        duplicate = rows(); duplicate[-1]["token_id"] = 0; variants.append(duplicate)
        boolean = rows(); boolean[0]["token_id"] = False; variants.append(boolean)
        nonfinite = rows(); nonfinite[0]["logit_hex"] = "7fc00000"; variants.append(nonfinite)
        disorder = rows(); disorder[0], disorder[1] = disorder[1], disorder[0]; variants.append(disorder)
        empty = rows(); empty[0]["piece_hex"] = ""; variants.append(empty)
        extra = rows(); extra[0]["mass"] = 1; variants.append(extra)
        for candidates in variants:
            with self.subTest(candidates=candidates[0]):
                with self.assertRaises(ValueError):
                    evidence(b"", candidates, request)
        with self.assertRaises(ValueError):
            audit.expected_evidence(b"", request["words"], rows(), request["request_sha256"], [True])

    def test_passive_whitespace_outside_and_inside_strings(self):
        _, _, request = context()
        outside = evidence(b'{"words":[', rows(), request)["mask_first"]
        inside = evidence(b'{"words":[{"alpha":["foo', rows(), request)["mask_first"]
        escaped = evidence(b'{"words":[{"alpha":["foo\\"', rows(), request)["mask_first"]
        self.assertEqual(outside["structural_whitespace_mass"]["numerator"], audit.TOTAL)
        self.assertEqual(inside["structural_whitespace_mass"]["numerator"], 0)
        self.assertEqual(escaped["structural_whitespace_mass"]["numerator"], 0)
        self.assertEqual(inside["integer_total"], audit.TOTAL)

    def test_independent_law_and_draw_equal_frozen_contract(self):
        _, _, request = context()
        sample = evidence(b"", rows(), request)
        positive = [{"token_id": r["token_id"], "piece": bytes.fromhex(r["piece_hex"]), "mass": r["weight"]}
                    for r in sample["mask_first"]["rows"] if r["weight"]]
        frozen = constrain_support(b"", request["words"], positive)
        self.assertEqual(guard.canonical(sample["conditional_law"]), guard.canonical(frozen))
        selected = select_token(frozen, bytes.fromhex(sample["key_hex"]), request["request_sha256"], 0, 0)
        self.assertEqual(sample["selection"], selected)


class SealedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-v4-auditor-")
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "sealed"
        self.binding, self.profile, self.request = context()

    def audit(self, terminal):
        return audit.audit_attempt(self.bundle, self.binding, self.profile, self.request, terminal)

    def complete_sample(self):
        raw = b'{"words":[{"alpha":[]}]}'
        candidates = rows(b"x")
        candidates[0]["piece_hex"] = raw.hex()
        return raw, evidence(b"", candidates, self.request)

    def test_complete_shape_is_not_usable_dictionary(self):
        raw, sample = self.complete_sample()
        terminal = sealed(self.bundle, self.binding, [(sample, raw)], "complete")
        result = self.audit(terminal)
        self.assertEqual(result["outcome"], "format_accepted")
        self.assertEqual(result["committed_output_tokens"], 1)
        self.assertEqual(result["render_calls"], 50)
        self.assertEqual(result["format_result"]["abstained_word_rows"], 1)
        self.assertIsNone(result["capacity_bits"])
        self.assertIsNone(result["usable_candidate_rows"])
        self.assertFalse(result["semantic_equivalence_verified"])
        self.assertFalse(result["genuine_native_execution_verified"])

    def test_known_empty_refusal_keeps_zero_and_full_probe(self):
        sample = evidence(b"", rows(b"x"), self.request)
        terminal = sealed(self.bundle, self.binding, [(sample, None)], "empty_support")
        result = self.audit(terminal)
        self.assertEqual(result["outcome"], "empty_schema_support")
        self.assertEqual(result["committed_output_tokens"], 0)
        self.assertEqual(result["saved_distribution_rows"], 50)
        self.assertIsNone(result["format_result"])

    def test_false_mass_in_a_valid_hash_chain_rejected(self):
        sample = evidence(b"", rows(), self.request)
        sample["mask_first"]["rows"][0]["weight"] += 1
        terminal = sealed(self.bundle, self.binding, [(sample, b" ")], "token_cap")
        guard.verify_bundle(self.bundle, self.binding, expected_terminal_sha256=terminal)
        with self.assertRaisesRegex(ValueError, "saved sampling evidence"):
            self.audit(terminal)

    def test_wrong_intended_raw_in_valid_guard_bundle_rejected(self):
        raw, sample = self.complete_sample()
        terminal = sealed(self.bundle, self.binding, [(sample, raw + b" ")], "complete")
        with self.assertRaisesRegex(ValueError, "STEP token/raw"):
            self.audit(terminal)

    def test_probe_after_first_complete_rejected(self):
        raw, sample = self.complete_sample()
        refused = evidence(b"", rows(b"x"), self.request)
        terminal = sealed(self.bundle, self.binding, [(sample, raw), (refused, None)], "empty_support")
        with self.assertRaisesRegex(ValueError, "first complete"):
            self.audit(terminal)

    def test_early_token_cap_and_missing_external_pin_rejected(self):
        sample = evidence(b"", rows(), self.request)
        terminal = sealed(self.bundle, self.binding, [(sample, b" ")], "token_cap")
        with self.assertRaisesRegex(ValueError, "early or invalid token-cap"):
            self.audit(terminal)
        with self.assertRaises(ValueError):
            self.audit("0" * 64)

    def test_noncanonical_mass_type_cannot_pass_as_equal_integer(self):
        sample = evidence(b"", rows(), self.request)
        sample["mask_first"]["rows"][0]["weight"] = float(sample["mask_first"]["rows"][0]["weight"])
        terminal = sealed(self.bundle, self.binding, [(sample, b" ")], "token_cap")
        with self.assertRaisesRegex(ValueError, "saved sampling evidence"):
            self.audit(terminal)

    def test_special_token_rejected_even_with_zero_mass(self):
        self.binding, self.profile, self.request = context(special=[49])
        raw, sample = self.complete_sample()
        self.assertEqual(sample["mask_first"]["rows"][49]["weight"], 0)
        terminal = sealed(self.bundle, self.binding, [(sample, raw)], "complete")
        with self.assertRaisesRegex(ValueError, "special or out-of-vocabulary"):
            self.audit(terminal)

    def test_whitespace_run_resets_on_committed_nonwhitespace(self):
        raw, _ = self.complete_sample()
        first = evidence(b"", rows(), self.request)
        token = first["selection"]["token_id"]
        candidates = rows(b"x"); candidates[0]["piece_hex"] = raw.hex()
        second = evidence(b" ", candidates, self.request, [token])
        terminal = sealed(self.bundle, self.binding, [(first, b" "), (second, b" " + raw)], "complete")
        result = self.audit(terminal)
        self.assertEqual(result["whitespace"], {
            "structural_whitespace_steps": 1, "max_structural_whitespace_run": 1,
            "terminal_structural_whitespace_run": 0, "whitespace_only_positive_support_probes": 1})


if __name__ == "__main__":
    unittest.main()
