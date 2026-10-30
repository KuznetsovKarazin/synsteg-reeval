"""Pure synthetic tests for saved-evidence auditor, no actual generation outputs.

Native metadata below is deliberately artificial to test evidence classification.
No native process or tokenizer is executed, and these are not pilot deliveries.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from synsteg_audit import framed_interval_public_ending_v2 as core
from synsteg_audit import framed_interval_public_ending_v2_orchestration as worker
from synsteg_audit import integer_interval_channel_v1 as interval
from synsteg_audit.framed_interval_public_ending_v2_native import resource_descriptor
from synsteg_audit.canonical_channel_v2 import CanonicalGate, PublicCodec

ROOT = Path(os.environ.get("ENDING_V2_AUDIT_TEST_CHECKOUT", Path(__file__).resolve().parents[1]))
AUDIT_PATH = Path(os.environ.get("ENDING_V2_AUDIT_SCRIPT", ROOT / "scripts/audit_public_ending_v2_saved_v1.py"))
module_spec = importlib.util.spec_from_file_location("saved_audit_under_test", AUDIT_PATH)
audit = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(audit)
PROFILE = json.loads((ROOT / "experiments/framed-interval-public-v1/native-public-profile.json").read_bytes())
SPEC = json.loads((ROOT / (audit.EXP + "/native-public-spec.json")).read_bytes())
CONTEXT = core.make_public_context([], context_id="synthetic-audit-context")
FIXTURE = {"case_id":"o013-stego", "arm":"stego", "context_order":0, "key_hex":bytes(range(32)).hex(),
           "nonce_hex":bytes(range(12)).hex(), "payload_hex":bytes(range(16)).hex(),
           "ordinary_seed_hex":core.DEFAULT_ORDINARY_SEED.hex()}
ENV = core.make_envelope(SPEC, CONTEXT, bytes.fromhex(FIXTURE["nonce_hex"]))
BASE = audit.OUT + "/generation/o013-stego"


class FiniteService:
    kind = "finite_test_fixture_no_native_calls"
    instances = []
    def __init__(self, *args):
        self.process_attempted = False
        self.prepare_attempted = False
        self.calls = {"TOKENIZE":0,"TEXT":0,"PROMPT":0,"DIST":0,"STEP":0}
        self.instances.append(self)
    def prepare(self):
        self.prepare_attempted = True
        pieces = (b"a", b"b", b".")
        descriptor = resource_descriptor(PROFILE)
        def encode(s):
            self.calls["TOKENIZE"] += 1
            return tuple(pieces.index(ch.encode()) for ch in s)
        def render(ids):
            self.calls["TEXT"] += 1
            return b"".join(pieces[t] for t in ids)
        def distribution(ids):
            self.calls["DIST"] += 1
            return core.Distribution(((0,1),(1,1)) if len(ids)<278 else ((2,1),), {"synthetic_audit_test":True})
        def start(ids): self.calls["PROMPT"] += 1
        def step(token): self.calls["STEP"] += 1
        codec = PublicCodec(encode, render, frozenset(), descriptor["codec_sha256"])
        # The native adapter constructs the old bound gate, then its new gate.
        # These are pure fake-codec callbacks, not native protocol commands.
        CanonicalGate(codec)
        return core.Endpoint(CanonicalGate(codec), resource_id=descriptor["resource_id"],
            resource_digest=descriptor["resource_sha256"], provider_digest=descriptor["provider_sha256"],
            distribution_callback=distribution,start_callback=start,observe_callback=step)
    def close(self):
        pass


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.canonical(value))


def synthesize(root, role, wire=None):
    slot = "sender" if role == "generate" else "receiver"
    base = root / BASE
    (root / "runtime" / audit.ID).mkdir(parents=True, exist_ok=True)
    request = {"role":role, "key_hex":FIXTURE["key_hex"]}
    if role == "generate":
        request.update(payload_hex=FIXTURE["payload_hex"], nonce_ledger=str(root / "runtime" / audit.ID / "sender-nonces"))
    else:
        request["wire_hex"] = wire.hex()
    result = worker.execute_endpoint(request, root, PROFILE, SPEC, CONTEXT, ENV, base / slot,
                                     service_factory=FiniteService)
    # Artificial declarations exclusively exercise the auditor's classification.
    result.update(execution_kind="native", model_process_attempted=True)
    put(base / slot / "worker-result.json", result)
    put(base / (slot + "-process") / "reservation.json", {"synthetic":True})
    put(base / (slot + "-process") / "supervisor-result.json", {"status":"completed_child", "child_exit_code":0,
        "session_verified_before_worker":True,"group_shutdown_before_reap":True,
        "owned_group_present_after_cleanup":False,"shutdown_error":None})
    inputs = base / (slot + "-inputs"); inputs.mkdir()
    (inputs / "key.bin").write_bytes(bytes.fromhex(FIXTURE["key_hex"]))
    if wire is not None:
        (inputs / "wire.bin").write_bytes(wire)
    return result


def rechain(rows):
    previous, output = "0" * 64, []
    for i, row in enumerate(rows):
        row = {k:v for k,v in row.items() if k != "event_sha256"}
        row.update(sequence=i, previous_sha256=previous)
        if row["kind"] == "committed":
            row["prepared_sha256"] = previous
        previous = audit.digest(audit.canonical(row))
        output.append(audit.canonical({**row, "event_sha256":previous}) + b"\n")
    return b"".join(output)


def empty_campaign(root):
    """Future eight-case panel metadata only; all output endpoints absent."""
    cases = []
    for i,name in enumerate(audit.CASE_IDS):
        row = dict(FIXTURE, case_id=name, arm=name.rsplit("-",1)[1], context_order=i//2,
                   nonce_hex=(i+1).to_bytes(12,"big").hex())
        if row["arm"] == "ordinary": row["payload_hex"] = None
        cases.append(row)
    put(root / audit.EXP / "fixtures.json", {"cases":cases})
    put(root / audit.EXP / "native-public-spec.json", SPEC)
    contexts = [dict(CONTEXT, context_id=f"synthetic-c{i}") for i in range(4)]
    put(root / audit.OUT / "tokenization/contexts.json", {"contexts":contexts})
    put(root / audit.EXP / "selected-context-texts.json", {"rows":[{"context_id":x["context_id"]} for x in contexts]})
    for i,context in enumerate(contexts):
        directory = root / audit.OUT / f"tokenization/c{i:02d}"
        put(directory / "tokenization.json", {"status":"completed", "tokenizer_calls":1,"prompt_calls":0,"step_calls":0,
            "prompt_ids":context["prompt_ids"],"context_sha256":core.public_digest(context)})
        put(directory / "supervisor-result.json", {"status":"completed_child","child_exit_code":0,
            "session_verified_before_worker":True,"group_shutdown_before_reap":True,
            "owned_group_present_after_cleanup":False,"shutdown_error":None})
    return cases


class SavedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        synthesize(cls.root, "generate")
        cls.wire = (cls.root / BASE / "sender/committed-wire.bin").read_bytes()
        synthesize(cls.root, "receive", cls.wire)
        cls.services = FiniteService.instances[:2]
        cls.journal = (cls.root / BASE / "receiver/token-journal.jsonl").read_bytes()
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_synthetic_saved_evidence_is_verified(self):
        result = audit.endpoint(audit.Inputs(self.root), BASE, "receive", FIXTURE, CONTEXT, SPEC, core, interval)
        self.assertEqual(result["classification"], "verified_saved_evidence_success")
        self.assertTrue(result["authenticated_delivery_counted"])
        self.assertTrue(result["matching_sender_wire_and_full_trace_verified"])
        self.assertEqual(result["evidence"]["tail_tokens"], 7)
        self.assertFalse(result["evidence"]["token_to_bytes_mapping_independently_recomputed"])

    def test_source_count_formula_matches_finite_codec_callback_counts(self):
        for slot,role,service in zip(("sender","receiver"),("generate","receive"),self.services):
            with self.subTest(role=role):
                saved=audit.recover((self.root / BASE / slot / "token-journal.jsonl").read_bytes())
                returned=json.loads((self.root / BASE / slot / "traces.json").read_bytes())
                code=json.loads((self.root / BASE / slot / "worker-result.json").read_bytes())["core"]
                counts=audit.command_count_scope(saved,returned,code,role,True)
                self.assertEqual(counts["TOKENIZE_source_derived_total"],service.calls["TOKENIZE"])
                self.assertEqual(counts["TEXT_source_derived_total"],service.calls["TEXT"])
                self.assertEqual(counts["PROMPT_source_derived"],service.calls["PROMPT"])
                self.assertEqual(counts["STEP_journal_committed"],service.calls["STEP"])
                self.assertEqual(counts["DIST_recorded_lower_bound"],service.calls["DIST"])

    def test_rehashed_false_arithmetic_trace_is_detected(self):
        rows = [json.loads(line) for line in self.journal.splitlines()]
        rows[0]["trace"]["emitted_bits"][0] ^= 1
        saved = audit.recover(rechain(rows))
        with self.assertRaisesRegex(ValueError, "interval transition differs"):
            audit.replay_saved(saved, core, interval, SPEC, ENV, FIXTURE)

    def test_rehashed_wrong_key_order_is_detected(self):
        rows = [json.loads(line) for line in self.journal.splitlines()]
        rows[0]["trace"]["ordered_integer_law"].reverse()
        saved = audit.recover(rechain(rows))
        with self.assertRaisesRegex(ValueError, "keyed law order differs"):
            audit.replay_saved(saved, core, interval, SPEC, ENV, FIXTURE)

    def test_pending_and_torn_prefixes_remain_unknown(self):
        lines = self.journal.splitlines(keepends=True)
        saved = audit.recover(b"".join(lines[:543]))
        self.assertTrue(saved["unknown"])
        self.assertEqual(len(saved["ids"]), 271)
        ev = audit.replay_saved(saved, core, interval, SPEC, ENV, FIXTURE)
        self.assertIsNone(ev["first_threshold_token"])
        torn = audit.recover(b"".join(lines[:542]) + b'{"incomplete"')
        self.assertTrue(torn["unknown"])
        self.assertTrue(torn["torn"])

    def test_missing_readback_coverage_is_not_accepted(self):
        inputs = audit.Inputs(self.root, {"schema":"ending-v2-audit-git-readback-v1", "commit":"1"*40, "files_sha256":{}})
        with self.assertRaisesRegex(ValueError, "readback hash"):
            inputs.raw(BASE + "/receiver/token-journal.jsonl")

    def test_cleanup_failure_preserves_unknown_without_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "copy"; shutil.copytree(self.root, root)
            path = root / BASE / "receiver/worker-result.json"
            worker_result = json.loads(path.read_bytes())
            worker_result.update(status="worker_exception", service_close_error="injected cleanup failure")
            put(path, worker_result)
            result = audit.endpoint(audit.Inputs(root), BASE, "receive", FIXTURE, CONTEXT, SPEC, core, interval)
            self.assertEqual(result["classification"], "unknown")
            self.assertFalse(result["authenticated_delivery_counted"])
            self.assertEqual(result["errors"], [])

    def test_missing_nonce_burn_fails_attempted_sender(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "copy"; shutil.copytree(self.root, root)
            (root / BASE / "sender/nonce-reservation.json").unlink()
            result = audit.endpoint(audit.Inputs(root), BASE, "generate", FIXTURE, CONTEXT, SPEC, core, interval)
            self.assertEqual(result["classification"], "evidence_error")
            self.assertIn("nonce burn", result["errors"][0])

    def test_refused_modified_tail_is_not_reported_as_delivery_or_forged_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            synthesize(root, "generate")
            changed = self.wire[:274] + (b"a" if self.wire[274:275] == b"b" else b"b") + self.wire[275:]
            synthesize(root, "receive", changed)
            result = audit.endpoint(audit.Inputs(root), BASE, "receive", FIXTURE, CONTEXT, SPEC, core, interval)
            self.assertEqual(result["classification"], "known_refusal")
            self.assertFalse(result["authenticated_delivery_counted"])
            self.assertFalse(result["evidence"]["saved_law_full_prefix_replay"])
            self.assertEqual(result["errors"], [])

    def test_absent_outputs_preserve_all_twelve_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); empty_campaign(root)
            result=audit.run(audit.Inputs(root),core,interval)
            self.assertEqual(result["status"],"partial_unknown")
            self.assertEqual(len(result["endpoints"]),12)
            self.assertEqual((result["quality_denominator"],result["payload_delivery_denominator"]),(8,4))
            self.assertEqual(result["exact_deliveries_from_saved_evidence"],0)
            self.assertEqual({r["classification"] for r in result["endpoints"]},{"not_attempted"})

    def test_success_metadata_cannot_hide_missing_receiver_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cases=empty_campaign(root)
            put(root / audit.OUT / "generation/result.json", {"cases":cases,"planned_endpoints":12,
                "planned_senders":8,"planned_fresh_receivers":4,"quality_denominator":8,
                "payload_delivery_denominator":4,"exact_deliveries":4})
            def fake_endpoint(inputs,base,role,fixture,*args):
                return {"case_id":fixture["case_id"],"role":role,"classification":"verified_saved_evidence_success",
                        "authenticated_delivery_counted":role=="receive","model_process_attempted":True,"errors":[]}
            with patch.object(audit,"endpoint",fake_endpoint):
                result=audit.run(audit.Inputs(root),core,interval)
            self.assertEqual(result["status"],"inconsistency")
            self.assertIn("installation inventory",result["errors"][0])

    def test_failed_tokenization_cleanup_is_not_completed_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); empty_campaign(root)
            put(root / audit.OUT / "tokenization/c00/supervisor-result.json",
                {"status":"unknown","child_exit_code":0,"shutdown_error":"injected"})
            with self.assertRaisesRegex(ValueError,"tokenization supervisor/cleanup"):
                audit.run(audit.Inputs(root),core,interval)


if __name__ == "__main__":
    unittest.main(verbosity=2)
