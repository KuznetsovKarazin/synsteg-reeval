"""Independent worker boundary tests using a finite character codec only."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synsteg_audit import framed_interval_public_ending_v2 as core
from synsteg_audit import framed_interval_public_ending_v2_orchestration as worker
from synsteg_audit.framed_interval_public_ending_v2_collect import collect
from synsteg_audit.framed_interval_public_ending_v2_native import resource_descriptor
from synsteg_audit.canonical_channel_v2 import CanonicalGate, PublicCodec, ContractError

ROOT = Path(os.environ.get('ENDING_V2_TEST_CHECKOUT', Path(__file__).resolve().parents[1]))
PROFILE = json.loads((ROOT/'experiments/framed-interval-public-v1/native-public-profile.json').read_bytes())
SPEC = json.loads((ROOT/'experiments/framed-interval-public-ending-v2/native-public-spec.json').read_bytes())
CONTEXT = core.make_public_context([], context_id='independent-worker-review')
KEY, PAYLOAD, NONCE = b'w'*32, b'finite review 16', b'v'*12


def fake_service(*, prepare_error=False, cleanup_error=False):
    class Service:
        kind = 'independent_synthetic_no_native_calls'
        instances = []
        def __init__(self, root, profile, spec, context, directory, deadline):
            self.process_attempted = False
            self.prepare_attempted = False
            self.closed = False
            self.instances.append(self)
        def prepare(self):
            self.prepare_attempted = True
            if prepare_error: raise RuntimeError('finite prepare failure')
            pieces = (b'a', b'b', b'.')
            descriptor = resource_descriptor(PROFILE)
            codec = PublicCodec(lambda s: tuple(pieces.index(c.encode()) for c in s),
                lambda ids: b''.join(pieces[t] for t in ids), frozenset(), descriptor['codec_sha256'])
            return core.Endpoint(CanonicalGate(codec), resource_id=descriptor['resource_id'],
                resource_digest=descriptor['resource_sha256'], provider_digest=descriptor['provider_sha256'],
                distribution_callback=lambda ids: core.Distribution(((0,1),(1,1)) if len(ids)<278 else ((2,1),),
                    {'synthetic':'independent-worker'}))
        def close(self):
            self.closed = True
            if cleanup_error: raise RuntimeError('finite cleanup failure')
    return Service


class IndependentWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.env = core.make_envelope(SPEC, CONTEXT, NONCE)
    def tearDown(self): self.tmp.cleanup()
    def sender(self, name='sender', service=None):
        return worker.execute_endpoint({'role':'generate','key_hex':KEY.hex(),'payload_hex':PAYLOAD.hex(),
            'nonce_ledger':str(self.path/'ledger')}, self.path, PROFILE, SPEC, CONTEXT, self.env,
            self.path/name, service_factory=service or fake_service())
    def receiver(self, wire, name='receiver', service=None):
        return worker.execute_endpoint({'role':'receive','key_hex':KEY.hex(),'wire_hex':wire.hex()},
            self.path, PROFILE, SPEC, CONTEXT, self.env, self.path/name,
            service_factory=service or fake_service())

    def test_failed_prepare_burns_nonce_and_second_attempt_never_prepares(self):
        first_service, second_service = fake_service(prepare_error=True), fake_service()
        first = self.sender(service=first_service)
        self.assertEqual(first['status'], 'worker_exception')
        self.assertTrue((self.path/'sender/nonce-reservation.json').is_file())
        self.assertTrue(first_service.instances[0].closed)
        second = self.sender('second', second_service)
        self.assertEqual(second['error_type'], 'NonceAlreadyReserved')
        self.assertFalse(second_service.instances[0].prepare_attempted)
        self.assertFalse(second['model_process_attempted'])

    def test_authenticated_frame_but_changed_tail_does_not_write_payload(self):
        self.sender()
        wire = (self.path/'sender/committed-wire.bin').read_bytes()
        changed = wire[:274] + (b'a' if wire[274:275] == b'b' else b'b') + wire[275:]
        opened = []
        original = core.open_frame
        def observe_aead(*args):
            plaintext = original(*args)
            self.assertFalse((self.path/'receiver/received-payload.bin').exists())
            opened.append(plaintext)
            return plaintext
        with patch.object(core, 'open_frame', observe_aead): result = self.receiver(changed)
        self.assertEqual(opened, [PAYLOAD])
        self.assertEqual(result['core']['reason'], 'noncanonical_fullwire_embedding')
        self.assertIsNone(result['core']['payload_hex'])
        self.assertFalse((self.path/'receiver/received-payload.bin').exists())

    def test_cleanup_failure_remains_exception_even_after_verified_payload(self):
        self.sender()
        wire = (self.path/'sender/committed-wire.bin').read_bytes()
        result = self.receiver(wire, service=fake_service(cleanup_error=True))
        self.assertTrue(result['core']['authenticated_delivery'])
        self.assertEqual(result['status'], 'worker_exception')
        self.assertIn('service_close_error', result)
        self.assertEqual((self.path/'receiver/received-payload.bin').read_bytes(), PAYLOAD)
        self.assertEqual(worker.reconcile(self.path/'receiver')['core_process_outcome_preserved'], 'worker_exception')
        # A finite evidence fixture declares native identity solely to exercise
        # the collector's independent cleanup classification. No model ran.
        result.update(execution_kind='native', model_process_attempted=True)
        (self.path/'receiver/worker-result.json').write_text(json.dumps(result))
        collected = collect(self.path/'receiver', {'status':'exited','exit_code':0}, 'receive', wire)
        self.assertFalse(collected['authenticated_delivery'])
        self.assertIn('clean terminal process evidence', collected['delivery_evidence_error'])

    def test_collector_accepts_complete_tail_evidence_from_finite_worker(self):
        self.sender()
        wire = (self.path/'sender/committed-wire.bin').read_bytes()
        result = self.receiver(wire)
        # As in the collector's own fixture tests, this is artificial evidence
        # with native metadata, not evidence of native inference or delivery.
        result.update(execution_kind='native', model_process_attempted=True)
        (self.path/'receiver/worker-result.json').write_text(json.dumps(result))
        collected = collect(self.path/'receiver', {'status':'exited','exit_code':0}, 'receive', wire,
            expected_payload=PAYLOAD)
        self.assertTrue(collected['authenticated_delivery'])
        self.assertEqual(collected['ending_evidence']['first_threshold_token'],272)
        self.assertEqual(collected['ending_evidence']['tail_tokens'],7)
        self.assertEqual(collected['ending_evidence']['threshold_wire_bytes'],272)
        self.assertEqual(collected['ending_evidence']['tail_wire_bytes'],7)
        self.assertEqual(collected['ending_evidence']['end_wire_bytes'],279)
        self.assertIsNone(collected['ending_evidence']['tail_seconds'])
        self.assertIn('not_measured', collected['ending_evidence']['tail_seconds_scope'])

    def test_receiver_rejects_experimental_payload_or_ledger_fields_before_service(self):
        for forbidden, value in [('payload_hex',PAYLOAD.hex()),('nonce_ledger',str(self.path/'ledger'))]:
            factory = fake_service()
            request = {'role':'receive','key_hex':KEY.hex(),'wire_hex':'',forbidden:value}
            with self.assertRaises(ContractError):
                worker.execute_endpoint(request, self.path, PROFILE, SPEC, CONTEXT, self.env,
                    self.path/'forbidden', service_factory=factory)
            self.assertEqual(factory.instances, [])
            self.assertFalse((self.path/'forbidden').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
