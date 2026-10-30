"""Deny and malformed-input boundaries; no launch/backend/tokenizer imports."""
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("guard_under_test", ROOT / "scripts/continuation_replay_guard_v1.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
GUARD_PATH = ROOT / "results/current-git-status-audit-20261028-v1/no-replay-guard.json"
GUARD_BYTES = GUARD_PATH.read_bytes()
RECEIPT = json.loads(GUARD_BYTES)


def request(**overrides):
    value = {"schema": guard.REQUEST_SCHEMA, "experiment_id": "baseline-new-recovery-v2",
             "model_alias": "tinyllama11", "source_order": 92, "intent_identity": None,
             "key_sha256": None, "nonces": []}
    value.update(overrides)
    return value


def evaluate(value, receipt=GUARD_BYTES):
    return guard.evaluate_bytes(receipt, json.dumps(value).encode())


class ContinuationReplayGuardTests(unittest.TestCase):
    def test_exact_guard_pin_and_conservative_status(self):
        self.assertEqual(hashlib.sha256(GUARD_BYTES).hexdigest(), guard.GUARD_SHA256)
        self.assertFalse(guard.validate_guard(RECEIPT)["proof_of_all_nonce_use"])

    def test_unreserved92_is_not_denied_but_never_launch_authorized(self):
        out = evaluate(request())
        self.assertEqual(out["status"], "not_denied_by_this_receipt")
        self.assertIs(out["launch_authorized"], False)
        self.assertIs(out["frozen_runners_automatically_intercepted"], False)
        self.assertIn("canonical_nn_lease", out["required_other_gates"])

    def test_old_experiment_denied_without_any_nonce(self):
        out = evaluate(request(experiment_id="framed-interval-public-ending-v2", model_alias=None, source_order=None))
        self.assertEqual([r["kind"] for r in out["reasons"]], ["denied_experiment_id"])

    def test_all_eight_burned_pairs_denied_under_new_experiment_id(self):
        for pair in RECEIPT["conservatively_burned_nonce_pairs"]:
            with self.subTest(case=pair["case_id"]):
                out = evaluate(request(experiment_id="different-generative-v3", key_sha256=pair["key_sha256"], nonces=[pair["nonce_hex"]]))
                self.assertEqual(out["status"], "denied")
                self.assertEqual(out["reasons"], [{"kind": "conservatively_burned_key_nonce", "case_id": pair["case_id"]}])

    def test_all_eight_pairs_retained_in_multi_nonce_request(self):
        rows = RECEIPT["conservatively_burned_nonce_pairs"]
        out = evaluate(request(key_sha256=rows[0]["key_sha256"], nonces=[r["nonce_hex"] for r in rows]))
        self.assertEqual(len(out["reasons"]), 8)

    def test_different_key_same_nonce_not_denied_does_not_authorize(self):
        pair = RECEIPT["conservatively_burned_nonce_pairs"][0]
        out = evaluate(request(key_sha256="f" * 64, nonces=[pair["nonce_hex"]]))
        self.assertEqual(out["status"], "not_denied_by_this_receipt")
        self.assertFalse(out["launch_authorized"])

    def test_unknown82_blocked_independent_of_experiment_label(self):
        out = evaluate(request(experiment_id="new-name-cannot-clear-history", source_order=82))
        self.assertEqual(out["reasons"][0]["kind"], "historical_unknown_order")

    def test_reserved91_blocked_even_without_explicit_identity(self):
        out = evaluate(request(source_order=91))
        self.assertEqual(out["reasons"][0]["kind"], "blocked_reserved_order")

    def test_explicit_identity_blocked_even_when_order_model_are_changed(self):
        identity = RECEIPT["blocked_baseline_intents"][0]["identity"]
        out = evaluate(request(model_alias="qwen05", source_order=1, intent_identity=identity))
        self.assertEqual(out["reasons"][0]["kind"], "blocked_explicit_intent_identity")

    def test_all_matching_reasons_retained(self):
        pair = RECEIPT["conservatively_burned_nonce_pairs"][0]
        identity = RECEIPT["blocked_baseline_intents"][0]["identity"]
        out = evaluate(request(experiment_id="framed-interval-public-ending-v2", source_order=91,
                               intent_identity=identity, key_sha256=pair["key_sha256"], nonces=[pair["nonce_hex"]]))
        self.assertEqual(len(out["reasons"]), 4)

    def test_bool_float_and_out_of_range_source_orders_rejected(self):
        for value in [True, False, 92.0, "92", 0, -1, 101]:
            with self.subTest(value=value):
                with self.assertRaises(guard.GuardError):
                    evaluate(request(source_order=value))

    def test_missing_unknown_and_wrong_schema_request_fields_rejected(self):
        for variant in [dict(request(), unrecognized=True), {k:v for k,v in request().items() if k != "intent_identity"}, dict(request(), schema="unknown")]:
            with self.subTest(variant=variant):
                with self.assertRaises(guard.GuardError):
                    evaluate(variant)

    def test_model_order_and_key_nonce_pairs_must_be_complete(self):
        for changes in [{"model_alias": None}, {"source_order": None}, {"key_sha256": "a"*64}, {"nonces": ["a"*24]}]:
            with self.subTest(changes=changes):
                with self.assertRaises(guard.GuardError):
                    evaluate(request(**changes))

    def test_hex_lengths_case_and_nonce_duplicates_rejected(self):
        for changes in [{"intent_identity": "A"*64}, {"intent_identity": "a"*63}, {"key_sha256":"a"*64, "nonces":["a"*23]}, {"key_sha256":"a"*64, "nonces":["a"*24,"a"*24]}, {"key_sha256":"a"*64, "nonces":"a"*24}]:
            with self.subTest(changes=changes):
                with self.assertRaises(guard.GuardError):
                    evaluate(request(**changes))

    def test_duplicate_json_keys_nan_utf8_and_nonobjects_fail_closed(self):
        for raw in [b'{"schema":"one","schema":"two"}', b'{"x":NaN}', b'\xff', b'[]', b'null']:
            with self.subTest(raw=raw):
                with self.assertRaises(guard.GuardError):
                    guard.evaluate_bytes(GUARD_BYTES, raw)

    def test_missing_changed_truncated_or_oversized_guard_fails_closed(self):
        for raw in [b'', b'{}', GUARD_BYTES[:-1], GUARD_BYTES+b' ', b' '*65537]:
            with self.subTest(size=len(raw)):
                with self.assertRaises(guard.GuardError):
                    evaluate(request(), receipt=raw)

    def test_guard_validation_catches_incomplete_denial_records(self):
        variants = []
        a=copy.deepcopy(RECEIPT);a['conservatively_burned_nonce_pairs'].pop();variants.append(a)
        a=copy.deepcopy(RECEIPT);a['blocked_baseline_intents'][0]['source_order']=True;variants.append(a)
        a=copy.deepcopy(RECEIPT);a['proof_of_all_nonce_use']=True;variants.append(a)
        a=copy.deepcopy(RECEIPT);a['conservatively_burned_nonce_pairs'][1]=copy.deepcopy(a['conservatively_burned_nonce_pairs'][0]);variants.append(a)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(guard.GuardError):
                    guard.validate_guard(variant)

    def test_cli_denied_invalid_and_not_denied_codes_without_file_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);g=root/'guard.json';q=root/'request.json';g.write_bytes(GUARD_BYTES)
            for order,code,status in [(91,3,'denied'),(True,2,'invalid_or_missing_evidence'),(92,0,'not_denied_by_this_receipt')]:
                q.write_text(json.dumps(request(source_order=order)));before={p.name:p.read_bytes() for p in root.iterdir()}
                stdout=io.StringIO()
                with redirect_stdout(stdout): actual=guard.main(['--guard',str(g),'--request',str(q)])
                self.assertEqual(actual,code);out=json.loads(stdout.getvalue());self.assertEqual(out['status'],status);self.assertFalse(out['launch_authorized'])
                self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_cli_missing_guard_file_and_symlink_guard_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);q=root/'request.json';q.write_text(json.dumps(request()))
            paths=[root/'missing'];link=root/'link';link.symlink_to(GUARD_PATH);paths.append(link)
            for path in paths:
                stdout=io.StringIO()
                with redirect_stdout(stdout): code=guard.main(['--guard',str(path),'--request',str(q)])
                self.assertEqual(code,2);self.assertFalse(json.loads(stdout.getvalue())['launch_authorized'])

    def test_request_oversize_and_nonascii_labels_fail_closed(self):
        with self.assertRaises(guard.GuardError):
            guard.evaluate_bytes(GUARD_BYTES,b' '*65537)
        with self.assertRaises(guard.GuardError):
            evaluate(request(experiment_id='new\u2013id'))


if __name__ == '__main__':
    unittest.main()
