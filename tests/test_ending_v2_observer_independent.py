"""Independent finite observer tests; no real model, tokenizer, or subprocess."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synsteg_audit import framed_interval_public_ending_v2 as new
from synsteg_audit import framed_interval_public_ending_v2_journal as journal
from synsteg_audit.canonical_channel_v2 import CanonicalGate, PublicCodec, Prefix, ContractError


KEY = bytes(reversed(range(32)))
PAYLOAD = b'observer review!'
NONCE = bytes(range(12))


class ObserverFixture:
    def __init__(self, directory, *, ending_at=278, failure_at=None, warmup=0):
        pieces = (b'x', b'y', b'z', b'.', b'!', b'?')
        codec = PublicCodec(lambda text: tuple(pieces.index(char.encode()) for char in text),
            lambda ids: b''.join(pieces[token] for token in ids), frozenset(), 'd'*64)
        self.distributions = []
        self.before_native = []
        self.crossings_before_native = []
        self.ending_at, self.failure_at, self.warmup = ending_at, failure_at, warmup
        self.base = new.Endpoint(CanonicalGate(codec), resource_id='review-observer-synthetic',
            resource_digest='e'*64, provider_digest='f'*64,
            distribution_callback=self.distribution, observe_callback=self.observe)
        self.spec = new.make_public_spec(self.base)
        self.context = new.make_public_context([], context_id='independent-observer-review')
        self.envelope = new.make_envelope(self.spec, self.context, NONCE)
        self.wrapper = journal.DurableEndpoint(self.base, self.envelope, KEY, directory,
            public_spec=self.spec, public_context=self.context)

    def distribution(self, ids):
        self.distributions.append(len(ids)+1)
        if self.ending_at == len(ids)+1:
            weights = ((3, 1), (4, 1)) if len(ids)-self.warmup < 272 else ((3, 1),)
        elif self.warmup <= len(ids) < self.warmup+272:
            weights = ((0, 1), (1, 1))
        else:
            weights = ((2, 1),)
        return new.Distribution(weights, {'synthetic': 'independent-observer'})

    def observe(self, token):
        # Inspect the fsynced last record using an independent reader. Full
        # hash-chain recovery is performed once after each run, not per token.
        with self.wrapper.journal.path.open('rb') as stream:
            stream.seek(0, 2)
            length = stream.tell()
            block = min(length, 8192)
            while True:
                stream.seek(length-block)
                lines = stream.read().splitlines()
                if block == length or len(lines) > 1:
                    last = json.loads(lines[-1]); break
                block = min(length, block*2)
        self.before_native.append((last['token_index'],
            last['kind'] == 'prepared', self.wrapper.journal.pending is not None))
        self.crossings_before_native.append(self.wrapper.first_threshold_token)
        if self.failure_at == len(self.before_native):
            raise RuntimeError('injected native STEP failure')

    def run(self):
        return new.generate(KEY, self.envelope, self.wrapper, public_spec=self.spec,
            public_context=self.context, payload=PAYLOAD)

    def prefix(self):
        return Prefix(self.wrapper.journal.ids, self.wrapper.journal.raw)


class IndependentObserverTests(unittest.TestCase):
    def test_prepare_is_durable_before_every_step_and_crossing_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory)
            try:
                result = case.run()
                self.assertEqual(result.status, 'ending_pending_receiver')
                self.assertEqual(case.before_native, [(i, True, True) for i in range(278)])
                self.assertEqual(case.crossings_before_native[:272], [None]*272)
                self.assertEqual(case.crossings_before_native[272:], [272]*6)
                self.assertEqual(case.wrapper.first_threshold_token, 272)
                recovery = journal.recover_journal(case.wrapper.journal.path)
                self.assertFalse(recovery['unknown_native_step'])
                self.assertEqual(recovery['committed_wire'], result.wire)
                self.assertEqual(recovery['committed_traces'], list(result.traces))
                before = list(case.distributions)
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
                self.assertEqual(case.distributions, before)
            finally: case.wrapper.close()

    def test_first_threshold_native_failure_is_unknown_without_false_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory, failure_at=272)
            try:
                result = case.run()
                self.assertEqual(result.status, 'backend_error')
                self.assertIsNone(case.wrapper.first_threshold_token)
                recovery = journal.recover_journal(case.wrapper.journal.path)
                self.assertEqual(len(recovery['committed_token_ids']), 271)
                self.assertTrue(recovery['unknown_native_step'])
                self.assertFalse(recovery['retry_permitted'])
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
                with self.assertRaises(ContractError): case.wrapper.observe(0)
                self.assertEqual(len(case.before_native), 272)
            finally: case.wrapper.close()

    def test_tail_step_failure_keeps_crossing_and_unknown_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory, failure_at=275)
            try:
                result = case.run()
                self.assertEqual(result.status, 'backend_error')
                self.assertEqual(case.wrapper.first_threshold_token, 272)
                recovery = journal.recover_journal(case.wrapper.journal.path)
                self.assertEqual(len(recovery['committed_token_ids']), 274)
                self.assertTrue(recovery['unknown_native_step'])
                self.assertEqual(len(result.wire), 274)
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
            finally: case.wrapper.close()

    def test_exact_tail_cap_blocks_97th_distribution_without_punctuation(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory, ending_at=None)
            try:
                result = case.run()
                self.assertEqual((result.status, result.first_threshold_token, result.tail_tokens),
                    ('ending_not_reached', 272, 96))
                self.assertEqual(case.distributions, list(range(1,369)))
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
                self.assertEqual(len(case.before_native), 368)
            finally: case.wrapper.close()

    def test_exact_tail_cap_punctuation_is_committed_then_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory, ending_at=368)
            try:
                result = case.run()
                self.assertEqual((result.status, result.ending_token, result.tail_tokens),
                    ('ending_pending_receiver', 368, 96))
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
                self.assertEqual(len(case.before_native), 368)
            finally: case.wrapper.close()

    def test_total_cap_blocks_without_shifting_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            case = ObserverFixture(directory, ending_at=None, warmup=496)
            try:
                result = case.run()
                self.assertEqual((result.status, result.first_threshold_token, result.tail_tokens),
                    ('ending_not_reached', 768, 0))
                self.assertEqual(case.wrapper.first_threshold_token, 768)
                with self.assertRaises(ContractError): case.wrapper.distribution(case.prefix())
                self.assertEqual(len(case.before_native), 768)
            finally: case.wrapper.close()


if __name__ == '__main__': unittest.main(verbosity=2)
