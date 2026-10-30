import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synsteg_audit import framed_interval_public_ending_v2 as core
from synsteg_audit import framed_interval_public_v1 as old
from synsteg_audit.canonical_channel_v2 import CanonicalGate, PublicCodec, ContractError
from synsteg_audit.framed_interval_public_ending_v2_journal import DurableEndpoint

KEY = bytes(range(32))
PAYLOAD = bytes(range(16))
NONCE = bytes(range(12))


def endpoint(*, binary_end=274, ending=b'.', warmup=0, threshold_punctuation=False):
    pieces = (b'a', b'b', b'.', b'!', b'?', b'c', b' ')
    # Character codec is synthetic, not a model tokenizer.
    codec = PublicCodec(lambda s: tuple(pieces.index(ch.encode()) for ch in s),
                        lambda ids: b''.join(pieces[i] for i in ids), frozenset(), 'a'*64)
    state = {'prompt': (), 'observed': []}
    def start(ids): state['prompt'] = tuple(ids)
    def observe(token): state['observed'].append(token)
    def distribution(ids):
        n = len(ids)-len(state['prompt'])
        if n < warmup: weights = ((5,1),)
        elif n < binary_end:
            weights = ((2,1),(3,1)) if threshold_punctuation and n==binary_end-1 else ((0,1),(1,1))
        else: weights = ((pieces.index(ending),1),) if ending is not None else ((0,1),)
        return core.Distribution(weights, {'synthetic': True})
    ep = core.Endpoint(CanonicalGate(codec), resource_id='finite-ending-fixture-v2',resource_digest='b'*64,
        provider_digest='c'*64, distribution_callback=distribution,start_callback=start,observe_callback=observe)
    ep.fixture_state = state
    return ep


def setup(**kwargs):
    ep = endpoint(**kwargs)
    spec = core.make_public_spec(ep)
    ctx = core.make_public_context([],context_id='finite-public-fixture')
    env = core.make_envelope(spec,ctx,NONCE)
    return ep,spec,ctx,env


def run_sender(**kwargs):
    ep,spec,ctx,env=setup(**kwargs)
    result=core.generate(KEY,env,ep,public_spec=spec,public_context=ctx,payload=PAYLOAD)
    return result,spec,ctx,env


class EndingTests(unittest.TestCase):
    def test_terminal_predicate_is_exact_bytes_not_semantics(self):
        for raw in [b'.',b'word!',b'word?',b'Dr.',b'3.',b'...']:
            self.assertTrue(core.terminal_punctuation(raw))
        for raw in [b'',b'word',b'Dr. ',b'word."',b'word.\n']:
            self.assertFalse(core.terminal_punctuation(raw))
        with self.assertRaises(ContractError): core.terminal_punctuation('word.')

    def test_roundtrip_and_first_threshold_survives_zero_bit_tail(self):
        r,s,c,e=run_sender(binary_end=272)
        self.assertEqual((r.first_threshold_token,r.ending_token,r.tail_tokens),(272,273,1))
        self.assertEqual(r.status,'ending_pending_receiver')
        receiver=endpoint(binary_end=272)
        v=core.receive(r.wire,KEY,e,receiver,public_spec=s,public_context=c)
        self.assertTrue(v.authenticated_delivery);self.assertEqual(v.payload,PAYLOAD)
        self.assertEqual((v.first_threshold_token,v.ending_token),(272,273))
        self.assertEqual(r.traces,v.traces)

    def test_same_token_threshold_and_ending(self):
        r,s,c,e=run_sender(binary_end=272,threshold_punctuation=True)
        self.assertEqual((r.first_threshold_token,r.ending_token,r.tail_tokens),(272,272,0))
        self.assertEqual(core.receive(r.wire,KEY,e,endpoint(binary_end=272,threshold_punctuation=True),public_spec=s,public_context=c).payload,PAYLOAD)

    def test_exact_tail96_accepts_without_97th_step(self):
        # 272 binary tokens;95 singleton letters; token368 is a punctuation singleton.
        ep,s,c,e=setup(binary_end=272,ending=None)
        original=ep._distribution
        ep._distribution=lambda ids: core.Distribution(((2,1),),{'synthetic':True}) if len(ids)==367 else original(ids)
        r=core.generate(KEY,e,ep,public_spec=s,public_context=c,payload=PAYLOAD)
        self.assertEqual((r.first_threshold_token,r.ending_token,r.tail_tokens),(272,368,96))
        self.assertEqual(len(ep.fixture_state['observed']),368)

    def test_tail96_refusal_no_extra_distribution(self):
        ep,s,c,e=setup(binary_end=272,ending=None)
        r=core.generate(KEY,e,ep,public_spec=s,public_context=c,payload=PAYLOAD)
        self.assertEqual(r.status,'ending_not_reached');self.assertEqual(r.tail_tokens,96)
        self.assertEqual(r.first_threshold_token,272);self.assertEqual(len(ep.fixture_state['observed']),368)
        v=core.receive(r.wire,KEY,e,endpoint(binary_end=272,ending=None),public_spec=s,public_context=c)
        self.assertEqual(v.status,'ending_not_reached');self.assertIsNone(v.payload)

    def test_threshold_at_total_cap_without_ending(self):
        r,s,c,e=run_sender(binary_end=768,warmup=496)
        self.assertEqual((r.status,r.first_threshold_token,r.tail_tokens),('ending_not_reached',768,0))
        v=core.receive(r.wire,KEY,e,endpoint(binary_end=768,warmup=496),public_spec=s,public_context=c)
        self.assertEqual(v.status,'ending_not_reached');self.assertIsNone(v.payload)

    def test_threshold_and_ending_at_total_cap(self):
        r,s,c,e=run_sender(binary_end=768,warmup=496,threshold_punctuation=True)
        self.assertEqual((r.status,r.first_threshold_token,r.ending_token),('ending_pending_receiver',768,768))
        self.assertEqual(core.receive(r.wire,KEY,e,endpoint(binary_end=768,warmup=496,threshold_punctuation=True),public_spec=s,public_context=c).payload,PAYLOAD)

    def test_incomplete_no_threshold_never_delivers(self):
        r,s,c,e=run_sender(binary_end=0,ending=b'a')
        self.assertEqual(r.status,'insufficient_capacity');self.assertIsNone(r.first_threshold_token)
        v=core.receive(r.wire,KEY,e,endpoint(binary_end=0,ending=b'a'),public_spec=s,public_context=c)
        self.assertEqual(v.status,'insufficient_capacity');self.assertIsNone(v.payload)

    def test_deleted_tail_or_early_ending_never_delivers(self):
        r,s,c,e=run_sender()
        for bad in [r.wire[:-1],r.wire[:272],r.wire[:273]+b'.']:
            v=core.receive(bad,KEY,e,endpoint(),public_spec=s,public_context=c)
            self.assertFalse(v.authenticated_delivery);self.assertIsNone(v.payload)

    def test_appended_tail_rejected_at_first_ending(self):
        r,s,c,e=run_sender()
        v=core.receive(r.wire+b'a',KEY,e,endpoint(),public_spec=s,public_context=c)
        self.assertEqual(v.reason,'trailing_tokens_after_first_ending');self.assertIsNone(v.payload)

    def test_valid_aead_does_not_release_modified_tail(self):
        r,s,c,e=run_sender()
        bad=r.wire[:272]+(b'b' if r.wire[272:273]==b'a' else b'a')+r.wire[273:]
        original=core.open_frame;seen=[]
        def spy(*args):
            payload=original(*args);seen.append(payload);return payload
        with patch.object(core,'open_frame',spy):
            v=core.receive(bad,KEY,e,endpoint(),public_spec=s,public_context=c)
        self.assertEqual(seen,[PAYLOAD]);self.assertEqual(v.reason,'noncanonical_fullwire_embedding')
        self.assertIsNone(v.payload);self.assertFalse(v.authenticated_delivery)

    def test_wrong_key_context_and_spec_fail_closed(self):
        r,s,c,e=run_sender()
        v=core.receive(r.wire,b'z'*32,e,endpoint(),public_spec=s,public_context=c)
        self.assertFalse(v.authenticated_delivery);self.assertIsNone(v.payload)
        c2=copy.deepcopy(c);c2['context_id']='other'
        with self.assertRaises(ContractError):core.receive(r.wire,KEY,e,endpoint(),public_spec=s,public_context=c2)
        s2=copy.deepcopy(s);s2['limits']['max_tail_tokens']=97
        with self.assertRaises(ContractError):core.receive(r.wire,KEY,e,endpoint(),public_spec=s2,public_context=c)

    def test_new_domain_rejects_old_spec_without_mutating_it(self):
        ep=endpoint();s=old.make_public_spec(ep);before=old.canonical_json(s)
        with self.assertRaises(ContractError):core.validate_public_spec(s)
        self.assertEqual(old.canonical_json(s),before)
        self.assertNotEqual(core.DOMAIN,old.DOMAIN)

    def test_byte_cap_after_threshold_does_not_create_success(self):
        ep,s,c,e=setup();s['limits']['max_wire_bytes']=273;e=core.make_envelope(s,c,NONCE)
        r=core.generate(KEY,e,ep,public_spec=s,public_context=c,payload=PAYLOAD)
        self.assertEqual((r.status,r.reason),('ending_not_reached','max_wire_bytes'))
        self.assertEqual(len(r.wire),273);self.assertFalse(r.authenticated_delivery)

    def test_ordinary_uses_same_ending_and_complete_replay(self):
        ep,s,c,e=setup();e=core.make_envelope(s,c,NONCE,arm='ordinary')
        r=core.generate(KEY,e,ep,public_spec=s,public_context=c)
        self.assertEqual(r.status,'ordinary_ending_generated')
        v=core.receive(r.wire,KEY,e,endpoint(),public_spec=s,public_context=c)
        self.assertEqual(v.status,'ordinary_ending_replay_verified');self.assertIsNone(v.payload)

    def test_durable_observer_allows_tail_and_stops_after_ending(self):
        with tempfile.TemporaryDirectory() as d:
            ep,s,c,e=setup(binary_end=272)
            durable=DurableEndpoint(ep,e,KEY,d,public_spec=s,public_context=c)
            try:
                r=core.generate(KEY,e,durable,public_spec=s,public_context=c,payload=PAYLOAD)
                self.assertEqual(r.status,'ending_pending_receiver');self.assertEqual(durable.first_threshold_token,272)
                self.assertEqual(len(durable.journal.ids),273)
                from synsteg_audit.canonical_channel_v2 import Prefix
                with self.assertRaises(ContractError):durable.distribution(Prefix(durable.journal.ids,durable.journal.raw))
            finally:durable.close()

if __name__=='__main__':unittest.main(verbosity=2)
