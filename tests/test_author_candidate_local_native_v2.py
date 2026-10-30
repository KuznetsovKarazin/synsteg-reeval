"""Integrated native-v2 mock faults only. No actual native process/tokenizer."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import author_candidate_local_native_v2 as n

REQUESTS = json.loads((ROOT / n.REQUESTS).read_bytes())["requests"]
PROFILE = json.loads((ROOT / n.PROFILE).read_bytes())


def abstain(request):
    return json.dumps({"words": [{w: []} for w in request["words"]]}, separators=(",", ":")).encode()


class MockBridge:
    def __init__(self, responses, directory=None, fault=None):
        self.responses = responses; self.directory = directory; self.fault = fault
        self.evaluated = 0; self.vocabulary_size = PROFILE["ready"]["vocabulary_size"]
        self.context_tokens = 4096; self.process = None
        self.calls = {k: 0 for k in ("tokenize", "start", "distribution", "render", "step", "close")}

    def tokenize(self, *_):
        self.calls["tokenize"] += 1
        if self.fault == "tokenize_mutates": self.evaluated += 1
        if self.fault == "oversized_prompt": return [50] * 4096
        return [50, 51]

    def start(self, ids):
        self.calls["start"] += 1
        if self.fault == "prompt_error": raise RuntimeError("mock prompt error")
        self.evaluated = len(ids) + int(self.fault == "prompt_bad_counter")

    def distribution(self):
        self.calls["distribution"] += 1
        if self.fault == "dist_mutates": self.evaluated += 1
        rows = [(100 + i, struct.pack(">f", -float(i) / 10).hex()) for i in range(50)]
        if self.fault == "duplicate_law": rows[-1] = rows[0]
        if self.fault == "nan_law": rows[0] = (100, "7fc00000")
        if self.fault == "masked_law": rows[0] = (PROFILE["special_ids"][0], "00000000")
        return rows

    def render(self, ids):
        self.calls["render"] += 1
        if self.fault == "render_mutates": self.evaluated += 1
        return self.responses[min(len(ids) - 1, len(self.responses) - 1)]

    def step(self, token):
        self.calls["step"] += 1
        if self.fault == "step_error": raise RuntimeError("mock step error")
        self.evaluated += 2 if self.fault == "step_bad_counter" else 1

    def close(self):
        self.calls["close"] += 1
        if self.fault == "cleanup_error": raise RuntimeError("mock cleanup error")
        if self.fault == "truncate_before_seal":
            p = self.directory.parent / "sealed/journal.jsonl"
            p.write_bytes(p.read_bytes().splitlines(keepends=True)[0])


class NativeV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name); self.instances = []

    def factory(self, responses, fault=None):
        def create(root, profile, directory, deadline):
            bridge = MockBridge(responses, directory, fault)
            self.instances.append(bridge)
            return bridge
        return create

    def attempt(self, responses=None, fault=None, request=None, order=0, name="attempt"):
        request = REQUESTS[0] if request is None else request
        responses = [abstain(request)] if responses is None else responses
        directory = self.tmp / name; commitment = self.tmp / (name + ".commitment.json")
        result = n.run_attempt(ROOT, PROFILE, request, order, directory, commitment,
            protocol_sha256="9" * 64, bridge_factory=self.factory(responses, fault))
        return result, directory, commitment

    def verify(self, result, directory, commitment, request=None, order=0):
        return n.verify_attempt_outcome(PROFILE, request or REQUESTS[0], order,
                                       directory, commitment, "9" * 64, result)

    def fixture(self):
        root = self.tmp / "root"; root.mkdir()
        for relative in n.REQUIRED_CLOSURE:
            target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        closure = {"schema": n.ID + "-source-closure", "files_sha256": {
            p: n.sha_file(root / p) for p in n.REQUIRED_CLOSURE}}
        n.write_json(root / n.CLOSURE, closure)
        intent = root / n.BASE / "test-intent.json"; n.write_json(intent, n.execution_intent(root))
        activation = root / n.BASE / "test-activation.json"; n.write_json(activation, n.enabled_activation(root, intent))
        files = {**closure["files_sha256"], n.CLOSURE: n.sha_file(root / n.CLOSURE),
                 intent.relative_to(root).as_posix(): n.sha_file(intent),
                 activation.relative_to(root).as_posix(): n.sha_file(activation)}
        receipt = root / "test-readback.json"
        n.write_json(receipt, {"schema": n.ID + "-source-readback", "source_commit": "a" * 40,
            "activation_commit": "b" * 40, "all_files_exact": True, "files_sha256": files})
        (root / "results").mkdir(exist_ok=True)
        return root, root / n.OUTPUT, activation, intent, receipt

    def campaign_runner(self, choose_response):
        def run(root, profile, request, order, directory, commitment, **kwargs):
            responses, fault = choose_response(request, order)
            return n.run_attempt(root, profile, request, order, directory, commitment,
                **kwargs, bridge_factory=self.factory(responses, fault))
        return run

    def test_all_abstentions_pass_format_not_usability(self):
        result, directory, commitment = self.attempt()
        self.assertEqual(result["outcome"], "format_accepted")
        self.assertEqual(result["format_result"]["abstained_word_rows"], 5)
        self.assertEqual(result["format_result"]["candidate_strings_returned"], 0)
        self.assertIsNone(result["capacity_bits"])
        self.assertEqual(self.verify(result, directory, commitment), result)

    def test_nonempty_strings_still_not_semantic_certification(self):
        rows = [{w: []} for w in REQUESTS[0]["words"]]
        rows[0][REQUESTS[0]["words"][0]] = ["alternative" + str(i) for i in range(8)]
        result, directory, commitment = self.attempt([json.dumps({"words": rows}).encode()])
        self.assertEqual(result["outcome"], "format_accepted")
        self.assertEqual(result["format_result"]["candidate_strings_returned"], 8)
        self.assertFalse(result["semantic_equivalence_verified"])
        self.verify(result, directory, commitment)

    def test_fence_stops_after_first_whole_token(self):
        raw = b"```json"; result, directory, commitment = self.attempt([raw, abstain(REQUESTS[0])])
        self.assertEqual(result["outcome"], "format_rejected")
        self.assertEqual(result["stop_reason"], "invalid_prefix")
        self.assertEqual(self.instances[0].calls["step"], 1)
        self.assertEqual((directory / "sealed/response.raw").read_bytes(), raw)
        self.verify(result, directory, commitment)

    def test_trailing_junk_in_final_token_is_preserved(self):
        raw = abstain(REQUESTS[0]) + b" trailing"
        result, directory, commitment = self.attempt([raw])
        self.assertEqual(result["outcome"], "format_rejected")
        self.assertEqual((directory / "observed-response.raw").read_bytes(), raw)
        self.verify(result, directory, commitment)

    def test_partial_utf8_is_not_lossily_repaired(self):
        result, directory, commitment = self.attempt([b'{"x":"\xe2', '{"x":"€"}'.encode()])
        self.assertEqual(result["committed_output_tokens"], 2)
        self.assertEqual(result["outcome"], "format_rejected")
        self.verify(result, directory, commitment)

    def test_wrong_order_source_rows_are_rejected(self):
        rows = [{w: []} for w in reversed(REQUESTS[0]["words"])]
        result, directory, commitment = self.attempt([json.dumps({"words": rows}).encode()])
        self.assertEqual(result["outcome"], "format_rejected")
        self.verify(result, directory, commitment)

    def test_parser_resource_refusal_is_known_and_bounded(self):
        result, directory, commitment = self.attempt([b'{"x":' + b"[" * 40])
        self.assertEqual(result["outcome"], "parser_resource_refusal")
        self.assertEqual(result["committed_output_tokens"], 1)
        self.verify(result, directory, commitment)

    def test_incomplete_at_cap_is_known_refusal(self):
        result, directory, commitment = self.attempt([b"{"])
        self.assertEqual(result["committed_output_tokens"], 512)
        self.assertEqual(result["outcome"], "format_rejected")
        self.assertEqual(result["stop_reason"], "token_cap")
        self.verify(result, directory, commitment)

    def test_native_faults_never_reach_format_acceptance(self):
        faults = ("tokenize_mutates", "oversized_prompt", "prompt_error", "prompt_bad_counter",
                  "dist_mutates", "duplicate_law", "nan_law", "masked_law", "render_mutates",
                  "step_error", "step_bad_counter", "cleanup_error", "truncate_before_seal")
        for index, fault in enumerate(faults):
            with self.subTest(fault=fault):
                result, directory, commitment = self.attempt(fault=fault, name="fault" + str(index))
                self.assertEqual(result["outcome"], "unknown")
                self.assertIsNone(result["format_result"])
                self.assertFalse(commitment.exists())
                self.assertLessEqual(self.instances[-1].calls["step"], 1)
                self.assertEqual(self.instances[-1].calls["close"], 1)

    def test_missing_step_ack_keeps_pending_evidence(self):
        result, directory, _ = self.attempt(fault="step_error")
        self.assertTrue(result["pending_step_unknown"])
        self.assertEqual(result["step_calls_invoked"], 1)
        self.assertEqual(result["committed_output_tokens"], 0)
        self.assertFalse((directory / "sealed/terminal.json").exists())

    def test_postseal_truncation_rejected_by_coordinator(self):
        result, directory, commitment = self.attempt()
        p = directory / "sealed/journal.jsonl"; p.write_bytes(p.read_bytes()[:-7])
        with self.assertRaises(ValueError):
            self.verify(result, directory, commitment)

    def test_commitment_change_rejected(self):
        result, directory, commitment = self.attempt()
        data = json.loads(commitment.read_bytes()); data["terminal_sha256"] = "0" * 64
        commitment.write_bytes(n.canonical(data))
        with self.assertRaises(ValueError):
            self.verify(result, directory, commitment)

    def test_claimed_acceptance_of_rejected_raw_rejected(self):
        result, directory, commitment = self.attempt([b"```"])
        result["outcome"] = "format_accepted"
        (directory / "attempt-result.json").write_bytes(n.canonical(result))
        with self.assertRaises(ValueError):
            self.verify(result, directory, commitment)

    def test_returned_result_differs_from_persisted_rejected(self):
        result, directory, commitment = self.attempt()
        result["committed_output_tokens"] += 1
        with self.assertRaises(ValueError):
            self.verify(result, directory, commitment)

    def test_wrong_stop_claim_rejected(self):
        result, directory, commitment = self.attempt()
        result["stop_reason"] = "token_cap"
        (directory / "attempt-result.json").write_bytes(n.canonical(result))
        with self.assertRaises(ValueError):
            self.verify(result, directory, commitment)

    def test_request_mutation_rejected_before_backend(self):
        request = copy.deepcopy(REQUESTS[0]); request["user"] += "changed"
        with self.assertRaises(ValueError):
            self.attempt(request=request)
        self.assertFalse(self.instances)

    def test_seed_domain_and_attempt_binding(self):
        from synsteg_audit.author_candidate_local_native_v1 import seed_hex as old_seed
        seeds = [n.seed_hex(h, a) for h in n.REQUEST_HASHES for a in range(3)]
        self.assertEqual(len(set(seeds)), 6)
        self.assertNotEqual(seeds[0], old_seed(n.REQUEST_HASHES[0], 0))
        for order in (-1, 3, True, 1.0):
            with self.assertRaises(ValueError): n.seed_hex(n.REQUEST_HASHES[0], order)

    def test_actual_repository_default_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "disabled"):
            n.verify_gate(ROOT, ROOT / n.DEFAULT_ACTIVATION, self.tmp / "absent-intent", self.tmp / "absent-receipt")

    def test_mock_source_gate_accepts_exact_closure(self):
        root, output, activation, intent, receipt = self.fixture()
        self.assertEqual(len(n.verify_gate(root, activation, intent, receipt)[1]), 2)

    def test_source_gate_rejects_mutation_after_mock_readback(self):
        root, output, activation, intent, receipt = self.fixture()
        p = root / "src/synsteg_audit/author_candidate_local_native_v2.py"; p.write_bytes(p.read_bytes() + b"\n")
        with self.assertRaises(ValueError): n.verify_gate(root, activation, intent, receipt)

    def test_source_gate_rejects_fake_missing_file_set(self):
        root, output, activation, intent, receipt = self.fixture()
        data = json.loads(receipt.read_bytes()); data["files_sha256"].pop(n.PROTOCOL)
        receipt.write_bytes(n.canonical(data))
        with self.assertRaises(ValueError): n.verify_gate(root, activation, intent, receipt)

    def test_activation_numeric_boolean_rejected(self):
        root, output, activation, intent, receipt = self.fixture()
        data = json.loads(activation.read_bytes()); data["enabled"] = 1
        activation.write_bytes(n.canonical(data))
        with self.assertRaises(ValueError): n.verify_gate(root, activation, intent, receipt)

    def test_duplicate_activation_key_rejected(self):
        root, output, activation, intent, receipt = self.fixture()
        raw = activation.read_bytes()
        activation.write_bytes(b'{"enabled":false,' + raw[1:])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            n.verify_gate(root, activation, intent, receipt)

    def test_nonfinite_metadata_rejected(self):
        p = self.tmp / "nonfinite.json"; p.write_bytes(b'{"value":NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            n.read_json(p)

    def test_fifo_outer_commitment_rejected_without_blocking(self):
        result, directory, commitment = self.attempt()
        commitment.unlink(); os.mkfifo(commitment)
        code = ("import sys\nsys.path.insert(0," + repr(str(ROOT / "src")) + ")\n"
            "from synsteg_audit import author_candidate_local_native_v2 as n\n"
            "try:\n n.read_external_commitment(" + repr(str(commitment)) + "," + repr(result["binding"]) + ")\n"
            "except ValueError:\n sys.exit(0)\n"
            "sys.exit(7)\n")
        checked = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, timeout=2)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_all_abstention_ends_each_chunk_without_semantic_retry(self):
        args = self.fixture()
        result = n.run_campaign(*args, attempt_runner=self.campaign_runner(lambda r, a: ([abstain(r)], None)))
        self.assertEqual(result["attempts_executed"], 2)
        self.assertEqual(result["format_accepted_chunks_observed"], 2)
        self.assertIsNone(result["usable_candidate_rows"])
        self.assertFalse(result["post_save_Git_evidence_verified"])
        self.assertEqual(len(list((args[1] / "reservations").iterdir())), 2)

    def test_six_known_refusals_are_retained(self):
        result = n.run_campaign(*self.fixture(), attempt_runner=self.campaign_runner(lambda r, a: ([b"```"], None)))
        self.assertEqual(result["attempts_executed"], 6)
        self.assertEqual(result["format_accepted_chunks_observed"], 0)
        self.assertEqual([s["status"] for s in result["chunks"]], ["format_attempts_exhausted"] * 2)

    def test_known_refusal_then_declared_next_attempt(self):
        result = n.run_campaign(*self.fixture(), attempt_runner=self.campaign_runner(
            lambda r, a: ([b"```"] if a == 0 else [abstain(r)], None)))
        self.assertEqual(result["attempts_executed"], 4)
        self.assertEqual(result["format_accepted_chunks_observed"], 2)

    def test_unknown_stops_campaign_and_other_chunk_unattempted(self):
        result = n.run_campaign(*self.fixture(), attempt_runner=self.campaign_runner(lambda r, a: ([abstain(r)], "step_error")))
        self.assertEqual(result["status"], "unknown_terminal")
        self.assertEqual(result["attempts_executed"], 1)
        self.assertEqual(result["chunks"][1]["status"], "not_attempted")
        self.assertIsNone(result["complete_campaign_format_accepted_chunks"])

    def test_resource_refusal_stops_campaign_without_retry(self):
        result = n.run_campaign(*self.fixture(), attempt_runner=self.campaign_runner(lambda r, a: ([b'{"x":' + b"[" * 40], None)))
        self.assertEqual(result["status"], "parser_resource_refusal_terminal")
        self.assertEqual(result["attempts_executed"], 1)
        self.assertEqual(result["chunks"][1]["status"], "not_attempted")

    def test_invalid_return_contract_is_retained_unknown(self):
        result = n.run_campaign(*self.fixture(), attempt_runner=lambda *a, **kw: {"forged": "accepted"})
        self.assertEqual(result["attempts_executed"], 1)
        self.assertEqual(result["chunks"][0]["attempts"][0]["observed_returned_result"], {"forged": "accepted"})

    def test_fixed_output_no_resume_or_alternate_directory(self):
        args = self.fixture()
        n.run_campaign(*args, attempt_runner=self.campaign_runner(lambda r, a: ([abstain(r)], None)))
        with self.assertRaises(FileExistsError):
            n.run_campaign(*args, attempt_runner=lambda *a, **kw: self.fail("must not rerun"))
        root, output, activation, intent, receipt = args
        with self.assertRaises(ValueError):
            n.run_campaign(root, root / "elsewhere", activation, intent, receipt,
                           attempt_runner=lambda *a, **kw: self.fail("must not rerun"))

    def test_exclusive_project_lease_prevents_attempt(self):
        args = self.fixture(); lease = (args[0] / "results/project-nn.lock").open("w")
        self.addCleanup(lease.close); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(BlockingIOError):
            n.run_campaign(*args, attempt_runner=lambda *a, **kw: self.fail("must not start"))
        self.assertFalse(args[1].exists())

    def test_directory_and_reservation_fsync_precedes_backend(self):
        args = self.fixture(); root, output = args[:2]
        synced = []; backend_snapshots = []; original_fsync = os.fsync
        def sync(fd):
            original_fsync(fd)
            synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        def run(root, profile, request, order, directory, commitment, **kwargs):
            def factory(root, profile, native_dir, deadline):
                backend_snapshots.append(list(synced))
                return MockBridge([abstain(request)], native_dir)
            return n.run_attempt(root, profile, request, order, directory, commitment,
                                 **kwargs, bridge_factory=factory)
        with patch.object(n.os, "fsync", side_effect=sync):
            result = n.run_campaign(*args, attempt_runner=run)
        self.assertEqual(result["attempts_executed"], 2)
        for chunk, snapshot in enumerate(backend_snapshots):
            attempt = output / f"chunk{chunk}-attempt0"
            for path in (root / "results", output.parent, output,
                         output / "reservations",
                         output / "reservations" / f"chunk{chunk}-attempt0.json",
                         attempt, attempt / "launch.json"):
                self.assertIn(path, snapshot)
            self.assertLess(snapshot.index(root / "results"), snapshot.index(output.parent))
            self.assertGreaterEqual(snapshot.count(output), 3)

    def test_parent_fsync_failure_prevents_backend_and_leaves_no_resume(self):
        def failed_sync(path):
            raise OSError("mock directory durability failure")
        with patch.object(n, "fsync_directory", side_effect=failed_sync):
            with self.assertRaises(OSError):
                self.attempt()
        self.assertEqual(self.instances, [])
        self.assertTrue((self.tmp / "attempt").is_dir())
        with self.assertRaises(FileExistsError):
            self.attempt()

    def test_durable_directory_existing_symlink_is_refused(self):
        target = self.tmp / "target"; target.mkdir()
        link = self.tmp / "link"; link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            n.mkdir_durable(link, exist_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
