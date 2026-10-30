"""Pure one-attempt worker integration: toy character codec, zero native calls."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_framed_interval_public_ending_v2 import endpoint, KEY, PAYLOAD
from synsteg_audit import framed_interval_public_ending_v2 as core
from synsteg_audit import framed_interval_public_ending_v2_orchestration as worker
from synsteg_audit.framed_interval_public_ending_v2_native import resource_descriptor
from synsteg_audit.canonical_channel_v2 import CanonicalGate, ContractError

ROOT=Path(__file__).resolve().parents[1]
PROFILE=json.loads((ROOT/'experiments/framed-interval-public-v1/native-public-profile.json').read_bytes())
SPEC=json.loads((ROOT/'experiments/framed-interval-public-ending-v2/native-public-spec.json').read_bytes())
CONTEXT=core.make_public_context([],context_id='toy-worker-public-context')


def service_factory(*, no_ending=False, fail_step=None):
    class ToyService:
        kind='synthetic_no_model'
        calls=[]
        def __init__(self,root,profile,spec,context,directory,deadline):
            self.process_attempted=False;self.prepare_attempted=False
            self.__class__.calls.append(self)
        def prepare(self):
            if self.prepare_attempted:raise AssertionError('second prepare')
            self.prepare_attempted=True
            ep=endpoint(binary_end=272,ending=None if no_ending else b'.')
            d=resource_descriptor(PROFILE)
            ep.gate=CanonicalGate(replace(ep.gate.codec,resource_digest=d['codec_sha256']))
            ep.resource_id=d['resource_id'];ep.resource_digest=d['resource_sha256'];ep.provider_digest=d['provider_sha256']
            if fail_step is not None:
                original=ep._observe
                def observe(t):
                    if len(ep.fixture_state['observed'])==fail_step:raise RuntimeError('simulated unknown backend step')
                    original(t)
                ep._observe=observe
            return ep
        def close(self):pass
    return ToyService


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.env=core.make_envelope(SPEC,CONTEXT,b'\x83'*12)
    def tearDown(self):self.tmp.cleanup()
    def run_sender(self,*,factory=None,arm='stego',directory='sender'):
        env=core.make_envelope(SPEC,CONTEXT,b'\x83'*12,arm=arm)
        return worker.execute_endpoint(dict(role='generate',key_hex=KEY.hex(),payload_hex=PAYLOAD.hex() if arm=='stego' else None,
            nonce_ledger=str(self.root/'ledger')),self.root,PROFILE,SPEC,CONTEXT,env,self.root/directory,
            service_factory=factory or service_factory())
    def run_receiver(self,wire,*,factory=None,directory='receiver'):
        return worker.execute_endpoint(dict(role='receive',key_hex=KEY.hex(),wire_hex=wire.hex()),self.root,PROFILE,SPEC,CONTEXT,
            self.env,self.root/directory,service_factory=factory or service_factory())
    def test_tail_journal_and_fresh_receiver_public_input_contract(self):
        sent=self.run_sender();wire=(self.root/'sender/committed-wire.bin').read_bytes()
        got=self.run_receiver(wire)
        self.assertEqual(sent['core']['first_threshold_token'],272)
        self.assertEqual(sent['core']['ending_token'],273)
        self.assertEqual(got['core']['ending_token'],273)
        self.assertTrue(got['core']['authenticated_delivery'])
        self.assertEqual((self.root/'receiver/received-payload.bin').read_bytes(),PAYLOAD)
        for output in [sent,got]:
            self.assertFalse(output['model_process_attempted']);self.assertEqual(output['execution_kind'],'synthetic_no_model')
            self.assertTrue(output['core']['committed_trace_matches_returned_prefix'])
        self.assertFalse((self.root/'receiver/nonce-reservation.json').exists())
        self.assertEqual(worker.reconcile(self.root/'receiver')['known_committed_tokens'],273)
    def test_no_ending_keeps_complete_refusal_and_no_payload_file(self):
        sent=self.run_sender(factory=service_factory(no_ending=True))
        got=self.run_receiver((self.root/'sender/committed-wire.bin').read_bytes(),factory=service_factory(no_ending=True))
        self.assertEqual(sent['core']['status'],'ending_not_reached')
        self.assertEqual(got['core']['status'],'ending_not_reached')
        self.assertEqual(sent['core']['tail_tokens'],96)
        self.assertFalse((self.root/'receiver/received-payload.bin').exists())
        self.assertEqual(worker.reconcile(self.root/'sender')['known_committed_tokens'],368)
    def test_nonce_burns_before_prepare_and_cannot_retry_in_new_directory(self):
        self.run_sender();factory=service_factory()
        second=self.run_sender(factory=factory,directory='second')
        self.assertEqual(second['status'],'worker_exception')
        self.assertEqual(second['error_type'],'NonceAlreadyReserved')
        self.assertFalse(factory.calls[0].prepare_attempted)
        self.assertFalse(second['model_process_attempted'])
        self.assertTrue((self.root/'second/worker-result.json').exists())
    def test_existing_attempt_directory_fails_before_second_service(self):
        self.run_sender();factory=service_factory()
        with self.assertRaises(FileExistsError):self.run_sender(factory=factory)
        self.assertEqual(factory.calls,[])
    def test_unknown_step_after_first_threshold_keeps_prefix_without_delivery(self):
        sent=self.run_sender(factory=service_factory(fail_step=272))
        self.assertEqual(sent['core']['status'],'backend_error')
        evidence=json.loads((self.root/'sender/reconciliation.json').read_bytes())
        self.assertTrue(evidence['unknown_native_step']);self.assertEqual(evidence['known_committed_tokens'],272)
        self.assertFalse(sent['core']['authenticated_delivery'])
        self.assertTrue((self.root/'sender/nonce-reservation.json').exists())
    def test_extra_tail_never_creates_payload_file(self):
        self.run_sender();wire=(self.root/'sender/committed-wire.bin').read_bytes()+b'a'
        got=self.run_receiver(wire)
        self.assertEqual(got['core']['status'],'invalid_wire')
        self.assertFalse((self.root/'receiver/received-payload.bin').exists())
    def test_ordinary_arm_uses_same_ending_without_payload(self):
        sent=self.run_sender(arm='ordinary')
        self.assertEqual(sent['core']['status'],'ordinary_ending_generated')
        self.assertEqual((sent['core']['first_threshold_token'],sent['core']['ending_token']),(272,273))
        self.assertFalse(sent['core']['authenticated_delivery'])
    def test_timeout_and_role_input_bounds_fail_before_service_or_directory(self):
        factory=service_factory()
        req=dict(role='receive',key_hex=KEY.hex(),wire_hex='',payload_hex=PAYLOAD.hex())
        with self.assertRaises(ContractError):worker.execute_endpoint(req,self.root,PROFILE,SPEC,CONTEXT,self.env,self.root/'bad',service_factory=factory)
        req.pop('payload_hex')
        with self.assertRaises(ContractError):worker.execute_endpoint(req,self.root,PROFILE,SPEC,CONTEXT,self.env,self.root/'bad',timeout=301,service_factory=factory)
        self.assertEqual(factory.calls,[]);self.assertFalse((self.root/'bad').exists())

if __name__=='__main__':unittest.main()
