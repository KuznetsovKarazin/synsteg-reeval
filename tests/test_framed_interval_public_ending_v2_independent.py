"""Independent finite adversarial tests; no model or native tokenizer execution.

The codec is a deliberately small prefix-free Unicode piece table and the law
is a public, position-indexed table. Cases target behavior at the wire boundary,
not internal implementations of arithmetic coding or cryptography.
"""
import copy
import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from synsteg_audit import framed_interval_public_ending_v2 as core
from synsteg_audit.canonical_channel_v2 import CanonicalGate, ContractError, PublicCodec

KEY = bytes(reversed(range(32)))
PAYLOAD = b"independent-v2!!"
PIECES = ("a", "b", ".", "!", "x.y", "x. ", 'x."', "é", "à", "🙂", "z", "<eos>")
BINARY = ((0, 1), (1, 1))
PUNCT = ((2, 1), (3, 1))
NONCES = itertools.count(1001)


class Fixture:
    def __init__(self, tail=(), *, data=BINARY, threshold_at=272, final=PUNCT, plan=None):
        self.plan = list(plan) if plan is not None else (
            [((10, 1),)] * (threshold_at - 272) + [data] * 272
            + [((index, 1),) for index in tail] + [final]
        )
        self.plan += [data] * (768 - len(self.plan))
        self.descriptor = json.dumps({"pieces": PIECES, "plan": self.plan}, sort_keys=True).encode()
        self.nonce = next(NONCES).to_bytes(12, "big")

    def endpoint(self):
        state = {"starts": 0, "distributions": 0, "observed": [], "prompt": ()}
        def encode(text):
            ids, offset = [], 0
            while offset < len(text):
                matches = [(len(piece), index) for index, piece in enumerate(PIECES)
                           if text.startswith(piece, offset)]
                if not matches:
                    raise ContractError("fixture text outside public piece table")
                size, index = max(matches)
                ids.append(index)
                offset += size
            return tuple(ids)
        codec = PublicCodec(encode, lambda ids: "".join(PIECES[i] for i in ids).encode(),
                            frozenset({11}), hashlib.sha256(repr(PIECES).encode()).hexdigest())
        def start(ids):
            state["starts"] += 1
            state["prompt"] = tuple(ids)
        def observe(token):
            state["observed"].append(token)
        def distribution(ids):
            state["distributions"] += 1
            index = len(ids) - len(state["prompt"])
            return core.Distribution(tuple(self.plan[index]), {"fixture": "independent-v2", "index": index})
        endpoint = core.Endpoint(CanonicalGate(codec), resource_id="independent-piece-table-v2",
            resource_digest=hashlib.sha256(b"independent finite resources").hexdigest(),
            provider_digest=hashlib.sha256(self.descriptor).hexdigest(),
            distribution_callback=distribution, start_callback=start, observe_callback=observe)
        endpoint.fixture = state
        return endpoint

    def setup(self, *, max_wire_bytes=65536, config=core.DEFAULT_CONFIG):
        sender = self.endpoint()
        spec = core.make_public_spec(sender, max_wire_bytes=max_wire_bytes, config=config)
        context = core.make_public_context([], context_id="independent-ending-boundaries")
        envelope = core.make_envelope(spec, context, self.nonce)
        return sender, spec, context, envelope

    def run(self, *, payload=PAYLOAD, **kwargs):
        sender, spec, context, envelope = self.setup(**kwargs)
        result = core.generate(KEY, envelope, sender, public_spec=spec, public_context=context, payload=payload)
        return result, sender, spec, context, envelope

    def receive(self, wire, spec, context, envelope):
        endpoint = self.endpoint()
        result = core.receive(wire, KEY, envelope, endpoint, public_spec=spec, public_context=context)
        return result, endpoint


class IndependentEndingTests(unittest.TestCase):
    def assert_not_delivered(self, result):
        self.assertFalse(result.authenticated_delivery)
        self.assertIsNone(result.payload)

    def test_bounded_payload_lengths_and_unicode_nonending_tail(self):
        for payload in (b"", b"x", bytes(range(16))):
            with self.subTest(payload_bytes=len(payload)):
                f = Fixture(tail=(4, 5, 6, 7, 9))
                sent, _, spec, ctx, env = f.run(payload=payload)
                self.assertEqual((sent.first_threshold_token, sent.ending_token, sent.tail_tokens), (272, 278, 6))
                self.assertIn('x.yx. x."é🙂'.encode(), sent.wire)
                received, ep = f.receive(sent.wire, spec, ctx, env)
                self.assertEqual(received.payload, payload)
                self.assertTrue(received.authenticated_delivery)
                self.assertEqual(len(ep.fixture["observed"]), 278)

    def test_punctuation_before_threshold_does_not_stop(self):
        f = Fixture(data=PUNCT)
        sent, ep, spec, ctx, env = f.run()
        self.assertEqual((sent.first_threshold_token, sent.ending_token, sent.tail_tokens), (272, 272, 0))
        self.assertEqual(len(ep.fixture["observed"]), 272)
        received, _ = f.receive(sent.wire, spec, ctx, env)
        self.assertEqual(received.payload, PAYLOAD)

    def test_every_entropy_tail_token_mutation_has_valid_aead_but_no_release(self):
        f = Fixture(plan=[BINARY] * 279 + [PUNCT])
        sent, _, spec, ctx, env = f.run()
        self.assertEqual((sent.first_threshold_token, sent.ending_token), (272, 280))
        open_frame = core.open_frame
        for offset in range(272, 280):
            with self.subTest(mutated_token=offset + 1):
                replacement = {ord("a"): ord("b"), ord("b"): ord("a"), ord("."): ord("!"), ord("!"): ord(".")}[sent.wire[offset]]
                bad = sent.wire[:offset] + bytes([replacement]) + sent.wire[offset + 1:]
                opened = []
                def spy(*args):
                    payload = open_frame(*args)
                    opened.append(payload)
                    return payload
                with patch.object(core, "open_frame", spy):
                    received, _ = f.receive(bad, spec, ctx, env)
                self.assertEqual(opened, [PAYLOAD])
                self.assertEqual(received.reason, "noncanonical_fullwire_embedding")
                self.assert_not_delivered(received)

    def test_every_truncated_tail_is_rejected_before_aead(self):
        f = Fixture(plan=[BINARY] * 279 + [PUNCT])
        sent, _, spec, ctx, env = f.run()
        for length in range(272, 280):
            with self.subTest(tokens=length):
                with patch.object(core, "open_frame", side_effect=AssertionError("early payload access")):
                    received, _ = f.receive(sent.wire[:length], spec, ctx, env)
                self.assertEqual(received.status, "incomplete_wire")
                self.assert_not_delivered(received)

    def test_appended_wire_does_not_step_past_first_ending(self):
        f = Fixture(plan=[BINARY] * 279 + [PUNCT])
        sent, _, spec, ctx, env = f.run()
        received, ep = f.receive(sent.wire + b"ab", spec, ctx, env)
        self.assertEqual(received.reason, "trailing_tokens_after_first_ending")
        self.assertEqual((ep.fixture["distributions"], len(ep.fixture["observed"])), (280, 280))
        self.assert_not_delivered(received)

    def test_entropy_tail_at96_is_received_without97th_step(self):
        f = Fixture(plan=[BINARY] * 367 + [PUNCT])
        sent, _, spec, ctx, env = f.run()
        received, ep = f.receive(sent.wire, spec, ctx, env)
        self.assertEqual((received.first_threshold_token, received.ending_token, received.tail_tokens), (272, 368, 96))
        self.assertEqual(received.payload, PAYLOAD)
        self.assertEqual((ep.fixture["distributions"], len(ep.fixture["observed"])), (368, 368))

    def test_ending_at97_is_rejected_before_observing_it(self):
        f = Fixture(plan=[BINARY] * 368 + [PUNCT])
        sent, _, spec, ctx, env = f.run()
        self.assertEqual(sent.status, "ending_not_reached")
        received, ep = f.receive(sent.wire + b".", spec, ctx, env)
        self.assertEqual(received.reason, "tail96_without_terminal_punctuation")
        self.assertEqual((ep.fixture["distributions"], len(ep.fixture["observed"])), (368, 368))
        self.assert_not_delivered(received)

    def test_utf8_budget_counts_bytes_and_retains_exact_threshold(self):
        f = Fixture(data=((7, 1), (8, 1)))
        sent, _, spec, ctx, env = f.run(max_wire_bytes=545)
        self.assertEqual(len(sent.wire), 545)
        received, _ = f.receive(sent.wire, spec, ctx, env)
        self.assertEqual(received.payload, PAYLOAD)
        f2 = Fixture(data=((7, 1), (8, 1)))
        refused, ep, spec, ctx, env = f2.run(max_wire_bytes=544)
        self.assertEqual((refused.status, refused.reason, refused.first_threshold_token), ("ending_not_reached", "max_wire_bytes", 272))
        self.assertEqual(len(refused.wire), 544)
        self.assertEqual((ep.fixture["distributions"], len(ep.fixture["observed"])), (273, 272))
        received, _ = f2.receive(refused.wire, spec, ctx, env)
        self.assert_not_delivered(received)

    def test_invalid_utf8_terminal_bytes_fail_before_backend_start(self):
        f = Fixture()
        _, spec, ctx, env = f.setup()
        received, ep = f.receive(b"\xff.", spec, ctx, env)
        self.assertEqual((received.status, received.reason), ("invalid_wire", "invalid_utf8"))
        self.assertEqual(ep.fixture["starts"], 0)
        self.assert_not_delivered(received)

    def test_special_eos_after_threshold_is_refusal_not_ending(self):
        f = Fixture(final=((11, 1),))
        sent, ep, spec, ctx, env = f.run()
        self.assertEqual(sent.status, "support_refusal")
        self.assertEqual(sent.first_threshold_token, 272)
        self.assertNotIn(11, ep.fixture["observed"])
        received, rep = f.receive(sent.wire + b"<eos>", spec, ctx, env)
        self.assertEqual(received.reason, "ineligible_token")
        self.assertNotIn(11, rep.fixture["observed"])
        self.assert_not_delivered(received)

    def test_bool_limit_not_accepted_as_integer_before_start(self):
        f = Fixture()
        sender, spec, ctx, env = f.setup()
        bad = copy.deepcopy(spec)
        bad["limits"]["max_tail_tokens"] = True
        with self.assertRaises(ContractError):
            core.generate(KEY, env, sender, public_spec=bad, public_context=ctx, payload=PAYLOAD)
        self.assertEqual(sender.fixture["starts"], 0)

    def test_postthreshold_stream_exhaustion_preserves_refusal(self):
        f = Fixture(plan=[BINARY] * 279 + [PUNCT])
        config = core.Config(state_bits=32, max_symbols=768, max_pending_bits=4096, max_stream_bits=304)
        sent, ep, _, _, _ = f.run(config=config)
        self.assertEqual(sent.status, "capacity_limit")
        self.assertEqual(sent.first_threshold_token, 272)
        self.assertEqual(len(ep.fixture["observed"]), 272)
        self.assertFalse(sent.authenticated_delivery)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentEndingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps({"schema": "ending-v2-independent-finite-tests-v1",
            "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "successful": result.wasSuccessful(), "model_calls": 0, "native_tokenizer_calls": 0,
            "scope": "pure synthetic prefix-free codec; not native decode or linguistic validation",
            "adversarial_tail_mutation_cases": 8, "truncated_tail_cases": 8,
            "bounded_payload_length_cases": [0, 1, 16]}, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
