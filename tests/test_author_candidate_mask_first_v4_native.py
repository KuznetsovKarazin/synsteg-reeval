"""Synthetic bridge/process tests only: never invoke a model or tokenizer.

Forked test children execute handwritten fakes, including deliberate hangs.
These are software tests, not request/model experiments or old-prefix replays.
"""
from pathlib import Path
import hashlib
import importlib
import json
import os
import signal
import struct
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("SYNSTEG_SOURCE_ROOT", ROOT))
SUPPORT = Path(os.environ.get("SYNSTEG_V4_SUPPORT_ROOT", ROOT))
package = types.ModuleType("synsteg_audit")
package.__path__ = [str(ROOT / "src/synsteg_audit"), str(SUPPORT / "src/synsteg_audit"),
                    str(FROZEN / "src/synsteg_audit")]
sys.modules["synsteg_audit"] = package
m = importlib.import_module("synsteg_audit.author_candidate_mask_first_v4_native")
guard = m.guard
requests = json.loads((FROZEN / m.REQUESTS).read_text())["requests"]
profile = json.loads((FROZEN / m.PROFILE).read_text())


class FakeBridge:
    vocabulary_size = 152064
    context_tokens = 2048
    evaluated = 0

    def __init__(self, request, mode="complete"):
        self.request, self.mode = request, mode
        self.raw = b""
        self.process = types.SimpleNamespace(poll=lambda: 0)
        self.complete = m.canonical({"words": [{w: []} for w in request["words"]]})

    def tokenize(self, text, *flags):
        # Handwritten fixed IDs; this does not call any tokenizer.
        if self.mode == "tokenize_counter":
            self.evaluated = 1
        return [100, 101]

    def start(self, ids):
        self.evaluated = len(ids)

    def distribution(self):
        if self.mode == "hang_distribution":
            while True:
                signal.pause()
        return [(i, struct.pack("!f", 0.0).hex()) for i in range(50)]

    def render(self, ids):
        if self.mode == "hang_render":
            while True:
                signal.pause()
        piece = b"}" if self.mode == "empty" else self.complete
        if self.mode == "resource":
            piece = b"x" * 262145 if ids[-1] == 49 else b" "
        return self.raw + piece

    def step(self, token):
        self.evaluated += 2 if self.mode == "step_counter" else 1
        self.raw += self.complete

    def close(self):
        if self.mode == "hang_close":
            while True:
                signal.pause()
        if self.mode == "close_error":
            raise OSError("synthetic cleanup failure")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "chunk0-attempt0"
        m.mkdir(self.directory)
        self.commitment = self.root / "commitment.json"
        self.binding = m.binding_for(profile, requests[0], "a" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def worker(self, mode="complete", deadline=None):
        return m._worker(self.root, profile, requests[0], self.directory, self.commitment,
                         self.binding, deadline or time.monotonic() + 10,
                         bridge_factory=lambda *args: FakeBridge(requests[0], mode))

    def test_exact_prior_key_bytes_are_preserved(self):
        for request in requests:
            expected = hashlib.sha256(m.canonical(["author-candidate-local-native-v3",
                "fixed-public-diagnostic-key", request["request_sha256"]])).hexdigest()
            self.assertEqual(m.key_hex(request["request_sha256"]), expected)
        with self.assertRaises(ValueError):
            m.key_hex("0" * 64)

    def test_all50_mask_first_not_prequantized(self):
        rows = [{"token_id": i, "logit_hex": struct.pack("!f", -100.0 if i == 49 else 0.0).hex(),
                 "piece_hex": (b" " if i == 49 else b"}").hex()} for i in range(50)]
        evidence = m.sampling_evidence(b"", requests[0], rows, [])
        self.assertEqual(evidence["selection"]["token_id"], 49)
        self.assertEqual(evidence["mask_first"]["rows"][-1]["weight"], 1 << 24)
        self.assertEqual(len(evidence["top50_rows"]), 50)

    def test_complete_sealed_one_step_and_no_extra_probe(self):
        result = self.worker()
        self.assertEqual(result["status"], "sealed_pending_audit")
        self.assertEqual(result["invoked"]["step_invoked"], 1)
        self.assertEqual(result["invoked"]["render_invoked"], 50)
        checked = guard.verify_bundle(self.directory / "sealed", self.binding,
                                     expected_terminal_sha256=result["terminal_sha256"])
        self.assertEqual(checked["committed_tokens"], 1)
        self.assertEqual(json.loads(checked["response"])["words"], [{w: []} for w in requests[0]["words"]])

    def test_known_empty_support_seals_without_step(self):
        result = self.worker("empty")
        self.assertEqual(result["status"], "sealed_pending_audit")
        self.assertEqual(result["stop_reason"], "empty_support")
        self.assertEqual(result["invoked"]["step_invoked"], 0)
        self.assertEqual(result["invoked"]["render_invoked"], 50)

    def test_budget_refusal_keeps_all50_and_no_step(self):
        result = self.worker("resource")
        self.assertEqual(result["stop_reason"], "resource_refusal")
        self.assertEqual(result["invoked"]["render_invoked"], 50)
        self.assertEqual(result["invoked"]["step_invoked"], 0)

    def test_inflight_step_counter_mismatch_never_seals(self):
        result = self.worker("step_counter")
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(self.commitment.exists())
        trace = guard.verify_trace((self.directory / "sealed/journal.jsonl").read_bytes(),
                                   self.binding, require_terminal=False)
        self.assertEqual(trace["pending"][0], "step_prepared")
        self.assertEqual(trace["ids"], [])

    def test_tokenize_counter_error_never_starts_prompt(self):
        result = self.worker("tokenize_counter")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["invoked"]["prompt_invoked"], 0)

    def test_cleanup_failure_preserves_unknown(self):
        result = self.worker("close_error")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["committed_output_tokens"], 1)
        self.assertFalse(self.commitment.exists())

    def test_existing_attempt_cannot_be_replayed(self):
        self.worker("empty")
        with self.assertRaises(FileExistsError):
            self.worker("empty")

    def test_normal_supervisor_and_independent_audit(self):
        result = m._supervise(self.directory, lambda deadline: self.worker(deadline=deadline), seconds=3)
        self.assertEqual(result["status"], "completed_child")
        audited = m.audit_finished_attempt(self.directory, self.commitment, self.binding, profile, requests[0])
        self.assertEqual(audited["outcome"], "format_accepted")
        self.assertIsNone(audited["capacity_bits"])

    def test_watchdog_cuts_inflight_distribution_render_and_cleanup(self):
        for mode in ("hang_distribution", "hang_render", "hang_close"):
            with self.subTest(mode=mode):
                directory = self.root / mode; m.mkdir(directory)
                commitment = self.root / (mode + ".json")
                def run(deadline):
                    return m._worker(self.root, profile, requests[0], directory, commitment, self.binding,
                                     deadline, bridge_factory=lambda *args: FakeBridge(requests[0], mode))
                result = m._supervise(directory, run, seconds=0.2)
                self.assertEqual(result["status"], "unknown")
                self.assertTrue(result["deadline_expired"])
                self.assertLess(result["elapsed_seconds"], 2.5)
                self.assertFalse(commitment.exists())
                trace = guard.verify_trace((directory / "sealed/journal.jsonl").read_bytes(),
                                           self.binding, require_terminal=False)
                self.assertFalse(trace["terminal"])
                if mode != "hang_close":
                    self.assertEqual(trace["pending"][0], "distribution_prepared")
                else:
                    self.assertEqual(len(trace["ids"]), 1)

    def test_expired_deadline_before_factory_calls_nothing(self):
        result = self.worker(deadline=time.monotonic() - 1)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["invoked"]["factory_invoked"], 0)


class CampaignTests(unittest.TestCase):
    def test_dangling_ledger_links_are_not_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "reservations").symlink_to(output / "absent")
            with self.assertRaises(ValueError):
                m.audit_campaign_ledger(output, profile, requests, "a" * 64, {})
            (output / "reservations").unlink()
            (output / "chunk0-attempt0").symlink_to(output / "absent")
            result = m.audit_campaign_ledger(output, profile, requests, "a" * 64, {})
            self.assertEqual(result["chunks"][0]["status"], "unknown")

    def test_crash_before_reservations_preserves_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            result = m.audit_campaign_ledger(Path(temp), profile, requests, "a" * 64, {})
            self.assertEqual([r["status"] for r in result["chunks"]], ["not_attempted", "not_attempted"])
            self.assertEqual(result["requested_chunks"], 2)
            self.assertEqual(result["reservations"], 0)

    def test_orphan_commitment_cannot_be_not_attempted(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); (output / "commitments").mkdir()
            m.write_json(output / "commitments/chunk0-attempt0.json", {"terminal_sha256": "f" * 64})
            result = m.audit_campaign_ledger(output, profile, requests, "a" * 64, {})
            self.assertEqual(result["chunks"][0]["status"], "unknown")
            self.assertFalse(result["campaign_complete"])

    def test_fixed_denominator_and_stop_or_continue_policy(self):
        for first, count in (("resource_refusal", 1), ("unknown", 1),
                             ("empty_schema_support", 2), ("token_cap_refusal", 2), ("format_accepted", 2)):
            with self.subTest(first=first), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); (root / "results").mkdir()
                receipt = {"source_commit": "a" * 40, "activation_commit": "b" * 40}
                calls = []
                def fake_supervise(directory, worker):
                    # Do not call worker. Assert reservation already durable.
                    self.assertTrue((directory.parent / "reservations" / (directory.name + ".json")).is_file())
                    calls.append(directory.name)
                    return {"status": "synthetic_no_worker"}
                with mock.patch.object(m, "verify_gate", return_value=(requests, profile, receipt)), \
                     mock.patch.object(m, "sha_file", return_value="a" * 64), \
                     mock.patch.object(m, "_supervise", side_effect=fake_supervise), \
                     mock.patch.object(m, "audit_finished_attempt", side_effect=[{"outcome": first}, {"outcome": "format_accepted"}]):
                    result = m.run_campaign(root, root/"activation", root/"intent", root/"readback")
                    self.assertEqual(result["requested_chunks"], 2)
                    self.assertEqual(len(calls), count)
                    if count == 1:
                        self.assertEqual(result["chunks"][1]["status"], "not_attempted")
                    with self.assertRaises(FileExistsError):
                        m.run_campaign(root, root/"activation", root/"intent", root/"readback")
                    self.assertEqual(len(calls), count)

    def test_missing_gate_never_creates_campaign(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises((FileNotFoundError, ValueError)):
                m.run_campaign(root, root/"activation", root/"intent", root/"readback")
            self.assertFalse((root / m.OUTPUT).exists())

    def test_crash_ledger_reservation_is_unknown_without_external_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); (output / "reservations").mkdir(); (output / "commitments").mkdir()
            binding = m.binding_for(profile, requests[0], "a" * 64)
            m.write_json(output / "reservations/chunk0-attempt0.json", {"schema": m.ID + "-reservation",
                         "binding": binding, "no_retry": True, "outcome_known": False})
            result = m.audit_campaign_ledger(output, profile, requests, "a" * 64, {})
            self.assertEqual(result["requested_chunks"], 2)
            self.assertEqual([r["status"] for r in result["chunks"]], ["unknown", "not_attempted"])
            self.assertFalse(result["campaign_complete"])
            self.assertFalse(result["retry_allowed"])

    def test_crash_ledger_refuses_later_reserved_attempt_after_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); (output / "reservations").mkdir(); (output / "commitments").mkdir()
            for request in requests:
                binding = m.binding_for(profile, request, "a" * 64)
                m.write_json(output / "reservations" / (binding["attempt_id"] + ".json"),
                             {"schema": m.ID + "-reservation", "binding": binding,
                              "no_retry": True, "outcome_known": False})
            result = m.audit_campaign_ledger(output, profile, requests, "a" * 64, {})
            self.assertTrue(result["scheduling_violations"])
            self.assertEqual(result["reservations"], 2)


if __name__ == "__main__":
    unittest.main()
