"""Fault tests use synthetic byte streams only: no model/provider/tokenizer."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synsteg_audit import candidate_journal_guard_v2 as g


BINDING = {"experiment_id": "guard-mock-v2", "attempt_id": "chunk0-attempt0",
           "request_sha256": "1" * 64, "resource_sha256": "2" * 64,
           "protocol_sha256": "3" * 64, "token_cap": 3}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "attempt"
        self.writer = g.AttemptWriter(self.directory, BINDING)
        self.addCleanup(self.writer.close)

    def prompt(self):
        h = self.writer.prepare_prompt([10, 11])
        self.writer.commit_prompt(h, 2)

    def one(self, raw=b"{}"):
        self.prompt()
        h = self.writer.prepare_step(12, raw, sample_evidence={"fixture": "synthetic"})
        self.writer.commit_step(h, 3)

    def sealed(self, raw=b"{}"):
        self.one(raw)
        return self.writer.seal([12], raw, stop_label="complete_json")

    def verify(self, receipt):
        return g.verify_bundle(self.directory, BINDING,
                               expected_terminal_sha256=receipt["terminal_sha256"])

    def no_terminal(self):
        self.assertFalse((self.directory / "terminal.json").exists())

    def test_complete_transcript_and_external_pin(self):
        receipt = self.sealed()
        self.assertEqual(self.verify(receipt)["response"], b"{}")
        self.assertFalse(receipt["shape_or_semantics_accepted"])
        self.assertEqual(receipt["committed_tokens"], 1)

    def test_known_malformed_json_is_only_trace_complete(self):
        receipt = self.sealed(b"```json\n{}\n``` trailing prose")
        self.assertFalse(self.verify(receipt)["shape_or_semantics_accepted"])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(receipt["response"])

    def test_prompt_prepared_without_ack_is_unknown(self):
        self.writer.prepare_prompt([10])
        self.writer.close()
        with self.assertRaises(g.EvidenceError):
            g.verify_trace((self.directory / "journal.jsonl").read_bytes(), BINDING)
        self.no_terminal()

    def test_step_prepared_without_ack_is_unknown(self):
        self.prompt()
        self.writer.prepare_step(12, b"{}", sample_evidence={})
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_truncated_journal_before_seal_cannot_publish(self):
        self.one()
        p = self.directory / "journal.jsonl"
        p.write_bytes(p.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_same_size_rewrite_before_seal_cannot_publish(self):
        self.one()
        p = self.directory / "journal.jsonl"
        raw = p.read_bytes()
        altered = raw.replace(b"synthetic", b"synthEtic", 1)
        self.assertEqual(len(raw), len(altered))
        p.write_bytes(altered)
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_replaced_journal_inode_cannot_publish(self):
        self.one()
        p = self.directory / "journal.jsonl"
        p.rename(self.directory / "old-journal")
        p.write_bytes((self.directory / "old-journal").read_bytes())
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_replaced_lease_cannot_continue(self):
        p = self.directory / "writer.lock"
        p.rename(self.directory / "old-lock")
        p.write_bytes((self.directory / "old-lock").read_bytes())
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_prompt([10])
        self.no_terminal()

    def test_wrong_producer_response_rejected(self):
        self.one()
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"[]", stop_label="complete_json")
        self.no_terminal()

    def test_wrong_producer_token_list_rejected(self):
        self.one()
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([99], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_bool_token_id_rejected(self):
        self.prompt()
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_step(True, b"{}", sample_evidence={})

    def test_bool_output_list_cannot_equal_integer_list(self):
        self.prompt()
        h = self.writer.prepare_step(1, b"{}", sample_evidence={})
        self.writer.commit_step(h, 3)
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([True], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_wrong_prompt_evaluated_counter(self):
        h = self.writer.prepare_prompt([10, 11])
        with self.assertRaises(g.EvidenceError):
            self.writer.commit_prompt(h, 3)
        self.no_terminal()

    def test_wrong_step_evaluated_counter(self):
        self.prompt()
        h = self.writer.prepare_step(12, b"{}", sample_evidence={})
        with self.assertRaises(g.EvidenceError):
            self.writer.commit_step(h, 2)
        self.no_terminal()

    def test_cross_step_acknowledgement_rejected(self):
        self.prompt()
        self.writer.prepare_step(12, b"{}", sample_evidence={})
        with self.assertRaises(g.EvidenceError):
            self.writer.commit_step("9" * 64, 3)

    def test_second_prepare_while_ack_pending_rejected(self):
        self.prompt()
        self.writer.prepare_step(12, b"{", sample_evidence={})
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_step(13, b"{}", sample_evidence={})

    def test_cap_exceeded_rejected(self):
        self.prompt()
        for i in range(3):
            h = self.writer.prepare_step(12 + i, b"x" * (i + 1), sample_evidence={})
            self.writer.commit_step(h, 3 + i)
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_step(15, b"xxxx", sample_evidence={})

    def test_reader_rejects_active_writer(self):
        with self.assertRaisesRegex(g.EvidenceError, "active"):
            g.verify_bundle(self.directory, BINDING, expected_terminal_sha256="0" * 64)

    def test_no_reopen_or_automatic_resume(self):
        self.writer.close()
        with self.assertRaises(FileExistsError):
            g.AttemptWriter(self.directory, BINDING)

    def test_failed_writer_cannot_continue(self):
        with self.assertRaises(g.EvidenceError):
            self.writer.commit_prompt("0" * 64, 0)
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_prompt([10])

    def test_fsync_failure_preserves_unknown_and_forbids_retry(self):
        with patch.object(g.os, "fsync", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.writer.prepare_prompt([10])
        with self.assertRaises(g.EvidenceError):
            self.writer.prepare_prompt([10])
        self.no_terminal()
        self.assertTrue((self.directory / "journal.jsonl").exists())

    def test_partial_write_failure_keeps_torn_evidence(self):
        actual = os.write
        def partial_then_fail(fd, value):
            actual(fd, value[:7])
            raise OSError("injected partial write")
        with patch.object(g.os, "write", side_effect=partial_then_fail):
            with self.assertRaises(OSError):
                self.writer.prepare_prompt([10])
        with self.assertRaises(g.EvidenceError):
            g.verify_trace((self.directory / "journal.jsonl").read_bytes(), BINDING)
        self.no_terminal()

    def test_atomic_publish_failure_leaves_no_final_name(self):
        self.one()
        with patch.object(g.os, "link", side_effect=OSError("injected publication failure")):
            with self.assertRaises(OSError):
                self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()
        self.assertTrue((self.directory / "terminal.pending").exists())

    def test_failure_after_atomic_link_retains_fail_closed_staging(self):
        self.one()
        with patch.object(Path, "unlink", side_effect=OSError("injected after-link failure")):
            with self.assertRaises(OSError):
                self.writer.seal([12], b"{}", stop_label="complete_json")
        raw = (self.directory / "terminal.json").read_bytes()
        self.assertTrue((self.directory / "terminal.pending").exists())
        with self.assertRaises(g.EvidenceError):
            g.verify_bundle(self.directory, BINDING, expected_terminal_sha256=g.sha(raw))

    def test_fifo_lease_rejected_without_blocking(self):
        receipt = self.sealed()
        p = self.directory / "writer.lock"
        p.unlink(); os.mkfifo(p)
        program = ("import sys\nfrom pathlib import Path\n"
                   "sys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1] / "src")) + ")\n"
                   "from synsteg_audit import candidate_journal_guard_v2 as g\n"
                   "try:\n g.verify_bundle(" + repr(str(self.directory)) + ", " + repr(BINDING) + ", expected_terminal_sha256=" + repr(receipt["terminal_sha256"]) + ")\n"
                   "except g.EvidenceError:\n sys.exit(0)\n"
                   "sys.exit(7)\n")
        result = subprocess.run([sys.executable, "-B", "-c", program], timeout=2, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_terminal_is_never_overwritten(self):
        self.one()
        p = self.directory / "terminal.json"
        p.write_bytes(b"sentinel")
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.assertEqual(p.read_bytes(), b"sentinel")

    def test_unexpected_file_prevents_publication(self):
        self.one()
        (self.directory / "unexplained-partial-model").write_bytes(b"extra")
        with self.assertRaises(g.EvidenceError):
            self.writer.seal([12], b"{}", stop_label="complete_json")
        self.no_terminal()

    def test_postseal_journal_truncation_detected(self):
        r = self.sealed()
        p = self.directory / "journal.jsonl"
        p.write_bytes(p.read_bytes()[:-10])
        with self.assertRaises(g.EvidenceError):
            self.verify(r)

    def test_postseal_response_rewrite_detected(self):
        r = self.sealed()
        (self.directory / "response.raw").write_bytes(b"[]")
        with self.assertRaises(g.EvidenceError):
            self.verify(r)

    def test_postseal_result_counter_rewrite_detected(self):
        r = self.sealed()
        p = self.directory / "result.json"
        data = json.loads(p.read_bytes()); data["committed_output_tokens"] = 512
        p.write_bytes(g.canonical(data))
        with self.assertRaises(g.EvidenceError):
            self.verify(r)

    def test_recomputed_manifest_does_not_defeat_external_pin(self):
        r = self.sealed()
        p = self.directory / "terminal.json"
        data = json.loads(p.read_bytes()); data["journal_rows"] += 1
        p.write_bytes(g.canonical(data))
        with self.assertRaisesRegex(g.EvidenceError, "terminal digest"):
            self.verify(r)

    def test_malformed_but_pinned_bool_manifest_count_is_rejected(self):
        self.sealed(b"x")
        p = self.directory / "terminal.json"
        data = json.loads(p.read_bytes()); data["files"]["response.raw"]["bytes"] = True
        raw = g.canonical(data); p.write_bytes(raw)
        with self.assertRaises(g.EvidenceError):
            g.verify_bundle(self.directory, BINDING, expected_terminal_sha256=g.sha(raw))

    def test_malformed_but_pinned_float_manifest_count_is_rejected(self):
        self.sealed(b"x")
        p = self.directory / "terminal.json"
        data = json.loads(p.read_bytes()); data["files"]["response.raw"]["bytes"] = 1.0
        raw = g.canonical(data); p.write_bytes(raw)
        with self.assertRaises(g.EvidenceError):
            g.verify_bundle(self.directory, BINDING, expected_terminal_sha256=g.sha(raw))

    def test_cross_request_resource_protocol_or_attempt_binding(self):
        r = self.sealed()
        for field in ("request_sha256", "resource_sha256", "protocol_sha256", "attempt_id"):
            other = dict(BINDING); other[field] = "9" * 64 if field.endswith("sha256") else "other"
            with self.subTest(field=field), self.assertRaises(g.EvidenceError):
                g.verify_bundle(self.directory, other, expected_terminal_sha256=r["terminal_sha256"])

    def test_symlink_artifact_is_rejected(self):
        r = self.sealed()
        p = self.directory / "response.raw"
        original = Path(self.tmp.name) / "outside"
        p.rename(original); p.symlink_to(original)
        with self.assertRaises(OSError):
            self.verify(r)

    def test_symlink_attempt_directory_is_rejected(self):
        r = self.sealed()
        link = Path(self.tmp.name) / "link"; link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(g.EvidenceError):
            g.verify_bundle(link, BINDING, expected_terminal_sha256=r["terminal_sha256"])

    def test_postseal_extra_file_is_rejected(self):
        r = self.sealed()
        (self.directory / "extra").write_bytes(b"x")
        with self.assertRaises(g.EvidenceError):
            self.verify(r)

    def test_duplicate_keys_and_nonfinite_are_rejected(self):
        for raw in (b'{"a":1,"a":1}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(g.EvidenceError):
                g.strict_json(raw)

    def test_rehashed_wrong_order_cannot_pass(self):
        self.one()
        raw = (self.directory / "journal.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        rows[1], rows[2] = rows[2], rows[1]
        previous = g.ZERO
        for n, row in enumerate(rows):
            row["sequence"] = n; row["previous_sha256"] = previous
            body = {k: row[k] for k in ("sequence", "previous_sha256", "event")}
            previous = row["event_sha256"] = g.sha(g.canonical(body))
        with self.assertRaises(g.EvidenceError):
            g.verify_trace(b"\n".join(map(g.canonical, rows)) + b"\n", BINDING, require_terminal=False)

    def test_invalid_bindings(self):
        bads = []
        for value in (True, 0, -1, 2.0, 65537):
            b = dict(BINDING); b["token_cap"] = value; bads.append(b)
        b = dict(BINDING); b["extra"] = 1; bads.append(b)
        b = dict(BINDING); b["request_sha256"] = "A" * 64; bads.append(b)
        b = dict(BINDING); b["attempt_id"] = "../other"; bads.append(b)
        for b in bads:
            with self.subTest(binding=b), self.assertRaises(g.EvidenceError):
                g.validate_binding(b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
