"""Finite fake-worker evidence tests; no tokenizer, bridge, or model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from synsteg_audit.framed_interval_llm_v1_journal import TokenJournal, sha
from synsteg_audit.framed_interval_public_ending_v2 import VERSION, canonical_json
from synsteg_audit.framed_interval_public_ending_v2_collect import collect


class EndingCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / "endpoint"
        self.process = {"status": "exited", "exit_code": 0}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, obj):
        (self.directory / name).write_bytes(canonical_json(obj))

    def fake(self, suffixes=(b"word", b" tail", b"."), bits=(270, 2, 2), role="receive", **overrides):
        journal = TokenJournal(self.directory)
        traces, raw, ids, resolved, first, ending = [], b"", [], 0, None, None
        for index, (suffix, size) in enumerate(zip(suffixes, bits)):
            trace = {"step": index, "prefix_ids": list(ids), "prefix_sha256": sha(raw),
                     "observed_token": index, "emitted_bits": [0] * size}
            journal.prepare(index, suffix, trace)
            journal.commit()
            traces.append(trace)
            raw += suffix
            ids.append(index)
            resolved += size
            if first is None and resolved >= 272:
                first = index + 1
            if first is not None and ending is None and raw[-1:] in (b".", b"!", b"?"):
                ending = index + 1
        journal.close()
        core = {"status": "authenticated_delivery" if role == "receive" else "ending_pending_receiver",
                "authenticated_delivery": role == "receive", "resolved_bits": resolved,
                "first_threshold_token": first, "ending_token": ending,
                "tail_tokens": len(ids)-first if first is not None else None,
                "has_terminal_punctuation": raw[-1:] in (b".", b"!", b"?"),
                "trace_sha256": sha(canonical_json(traces)), "committed_trace_matches_returned_prefix": True,
                "noncommitted_trace_suffix": []}
        if role == "receive":
            core.update(processed_tokens=len(ids), wire_tokens=len(ids), payload_hex="0011")
            (self.directory / "received-payload.bin").write_bytes(b"\x00\x11")
        else:
            core.update(emitted_tokens=len(ids), wire_sha256=sha(raw), wire_bytes=len(raw), arm="stego")
        core.update(overrides)
        worker = {"schema": VERSION+"-native-worker-result", "role": role, "execution_kind": "native",
                  "status": "returned", "model_process_attempted": True, "core": core}
        self.save("worker-result.json", worker)
        self.save("traces.json", traces)
        return raw

    def run_collect(self, raw, role="receive", **kwargs):
        return collect(self.directory, self.process, role, raw, **kwargs)

    def rejected(self, raw, **kwargs):
        result = self.run_collect(raw, **kwargs)
        self.assertFalse(result["authenticated_delivery"])
        self.assertIn("delivery_evidence_error", result)

    def test_tail_delivery_and_exact_fixture(self):
        raw = self.fake()
        result = self.run_collect(raw, expected_payload=b"\x00\x11")
        self.assertTrue(result["authenticated_delivery"])
        self.assertTrue(result["exact_payload"])
        self.assertEqual(result["ending_evidence"]["first_threshold_token"], 2)
        self.assertEqual(result["ending_evidence"]["tail_tokens"], 1)
        self.assertEqual(result["ending_evidence"]["threshold_wire_bytes"], len(b"word tail"))
        self.assertEqual(result["ending_evidence"]["tail_wire_bytes"], 1)
        self.assertEqual(result["ending_evidence"]["end_wire_bytes"], len(raw))
        self.assertIsNone(result["ending_evidence"]["tail_seconds"])

    def test_same_token_threshold_and_ending(self):
        raw = self.fake((b"end.",), (272,))
        self.assertEqual(self.run_collect(raw)["ending_evidence"]["tail_tokens"], 0)

    def test_sender_and_ordinary_use_same_ending(self):
        raw = self.fake(role="generate", arm="ordinary", status="ordinary_ending_generated")
        result = self.run_collect(raw, role="generate")
        self.assertIn("ending_evidence", result)
        self.assertFalse(result["authenticated_delivery"])

    def test_earlier_punctuation_before_threshold_is_allowed(self):
        raw = self.fake((b"Intro.", b" more", b"!"), (270, 2, 2))
        self.assertTrue(self.run_collect(raw)["authenticated_delivery"])

    def test_late_output_after_first_ending_rejected(self):
        self.rejected(self.fake((b"word", b".", b"extra."), (270, 2, 2)))

    def test_missing_ending_rejected(self):
        self.rejected(self.fake((b"word", b" more"), (270, 2)))

    def test_trailing_whitespace_not_trimmed(self):
        self.rejected(self.fake((b"word", b". "), (270, 2)))

    def test_wrong_first_threshold_rejected(self):
        self.rejected(self.fake(first_threshold_token=3))

    def test_wrong_ending_rejected(self):
        self.rejected(self.fake(ending_token=2))

    def test_wrong_tail_count_rejected(self):
        self.rejected(self.fake(tail_tokens=0))

    def test_tail96_inclusive(self):
        raw = self.fake((b"word",)+tuple([b" a"]*95)+(b".",), (272,)+tuple([0]*96))
        self.assertEqual(self.run_collect(raw)["ending_evidence"]["tail_tokens"], 96)

    def test_tail97_rejected(self):
        raw = self.fake((b"word",)+tuple([b" a"]*96)+(b".",), (272,)+tuple([0]*97))
        self.rejected(raw)

    def test_total769_rejected(self):
        raw = self.fake(tuple([b"a"]*768)+(b".",), tuple([0]*767)+(272, 0))
        self.rejected(raw)

    def test_supplied_extra_or_missing_wire_rejected(self):
        raw = self.fake()
        self.rejected(raw+b"extra")

    def test_modified_tail_rejected(self):
        raw = self.fake()
        self.rejected(raw[:-1]+b"!")

    def test_mismatched_payload_rejected(self):
        self.rejected(self.fake(), expected_payload=b"different")

    def test_missing_payload_artifact_rejected(self):
        raw = self.fake()
        (self.directory / "received-payload.bin").unlink()
        self.rejected(raw)

    def test_saved_trace_change_rejected(self):
        raw = self.fake()
        rows = json.loads((self.directory / "traces.json").read_bytes())
        rows[-1]["emitted_bits"] = [1, 1]
        self.save("traces.json", rows)
        self.rejected(raw)

    def test_journal_corruption_rejected(self):
        raw = self.fake()
        path = self.directory / "token-journal.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"suffix_hex":"2e"', b'"suffix_hex":"21"'))
        self.rejected(raw)

    def test_torn_journal_rejected(self):
        raw = self.fake()
        path = self.directory / "token-journal.jsonl"
        path.write_bytes(path.read_bytes()+b'{"torn"')
        self.rejected(raw)

    def test_old_worker_schema_rejected(self):
        raw = self.fake()
        worker = json.loads((self.directory / "worker-result.json").read_bytes())
        worker["schema"] = "framed-interval-public-v1-native-worker-result"
        self.save("worker-result.json", worker)
        self.rejected(raw)

    def test_nonzero_process_rejected(self):
        raw = self.fake()
        self.process["exit_code"] = 1
        self.rejected(raw)

    def test_missing_ending_is_retained_as_failure_without_payload(self):
        raw = self.fake((b"word", b"more"), (270, 2), status="ending_not_reached",
                        authenticated_delivery=False, payload_hex=None)
        result = self.run_collect(raw)
        self.assertFalse(result["authenticated_delivery"])
        self.assertNotIn("delivery_evidence_error", result)

    def test_decimal_and_abbreviation_are_only_byte_predicate(self):
        raw = self.fake((b"Dr.",), (272,))
        result = self.run_collect(raw)
        self.assertTrue(result["authenticated_delivery"])
        self.assertTrue(result["ending_evidence"]["terminal_punctuation_only_not_semantic_completion"])


if __name__ == "__main__":
    unittest.main()
