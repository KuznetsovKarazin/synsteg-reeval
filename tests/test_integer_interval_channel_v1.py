"""Synthetic checks only; no neural model, tokenizer, network or secret key."""
from itertools import product
from fractions import Fraction

import pytest

from synsteg_audit.integer_interval_channel_v1 import (
    CapacityLimit, Config, ContractError, Law, Receiver, Sender, State,
    advance, bits_from_bytes, exact_bytes, integer_bins,
)


def compositions(total):
    if total == 0:
        yield ()
    for first in range(1, total + 1):
        for rest in compositions(total - first):
            yield (first,) + rest


def law(weights):
    return Law(tuple((10 + i * 3, value) for i, value in enumerate(weights)))


def test_uniform_four_symbols_deliver_two_bits_each():
    bits = bits_from_bytes(b"SynSteg\x00\xff")
    config = Config(state_bits=4)
    sender, receiver = Sender(bits, config), Receiver(config)
    support = law((1, 1, 1, 1))
    for index in range(len(bits) // 2):
        step = sender.take(support)
        observed = receiver.take(support, step.symbol)
        assert observed == step
        assert len(receiver.recovered_bits) == 2 * (index + 1)
    assert sender.input_complete
    assert receiver.recovered_bits == bits
    assert exact_bytes(receiver.recovered_bits) == b"SynSteg\x00\xff"
    assert sender.consumed_stream_bits == len(bits) + 4
    assert sender.padding_bits_read == 4


def test_lookahead_never_counts_as_delivery():
    sender, receiver = Sender((1, 0, 1), Config(8)), Receiver(Config(8))
    assert sender.consumed_stream_bits == 8
    assert sender.padding_bits_read == 5
    assert sender.delivered_input_bits == 0
    assert receiver.recovered_bits == ()
    singleton = law((3,))
    for _ in range(20):
        receiver.take(singleton, sender.take(singleton).symbol)
    assert sender.consumed_stream_bits == 8
    assert sender.delivered_input_bits == 0
    assert not sender.input_complete
    assert receiver.recovered_bits == ()


def test_middle_underflow_counterexample_no_finite_delivery_claim():
    config = Config(4, max_symbols=100)
    sender, receiver = Sender((1, 0, 0, 0), config), Receiver(config)
    support = law((1, 2, 1))
    for _ in range(100):
        step = sender.take(support)
        assert step.symbol == 13
        assert step.renormalizations == ("middle_half",)
        receiver.take(support, step.symbol)
    assert sender.code == 8
    assert sender.state == State(0, 15, 100, 100, 100)
    assert sender.consumed_stream_bits == 104
    assert sender.delivered_input_bits == 0
    assert receiver.recovered_bits == ()
    with pytest.raises(CapacityLimit, match="max_symbols"):
        sender.take(support)


@pytest.mark.parametrize("config,reason", [
    (Config(4, max_symbols=1), "max_symbols"),
    (Config(4, max_pending_bits=1), "max_pending_bits"),
    (Config(4, max_stream_bits=5), "max_stream_bits"),
])
def test_limits_refuse_atomically_at_both_endpoints(config, reason):
    sender, receiver = Sender((1, 0, 0, 0), config), Receiver(config)
    support = law((1, 2, 1))
    receiver.take(support, sender.take(support).symbol)
    before_sender = (sender.state, sender.code, sender.consumed_stream_bits, sender.resolved_bits)
    before_receiver = (receiver.state, receiver.recovered_bits)
    with pytest.raises(CapacityLimit, match=reason):
        sender.take(support)
    with pytest.raises(CapacityLimit, match=reason):
        receiver.take(support, 13)
    assert (sender.state, sender.code, sender.consumed_stream_bits, sender.resolved_bits) == before_sender
    assert (receiver.state, receiver.recovered_bits) == before_receiver


def test_empty_payload_is_complete_without_a_symbol():
    sender = Sender((), Config(4, max_symbols=0))
    assert sender.input_complete
    assert sender.delivered_input_bits == 0
    assert sender.state.symbols == 0
    with pytest.raises(CapacityLimit):
        sender.take(law((1,)))


def test_all_normalized_small_state_bins_partition_and_quantization():
    config = Config(4)
    half, quarter = 8, 4
    states = [State(lo, hi) for lo in range(half) for hi in range(half, 16)
              if not (lo >= quarter and hi < 3 * quarter)]
    for state in states:
        width = state.high - state.low + 1
        assert width >= quarter + 2
        for total in range(1, 5):
            for weights in compositions(total):
                support = law(weights)
                bins = integer_bins(state, support, config)
                assert [point for _, lo, hi in bins for point in range(lo, hi + 1)] == list(range(state.low, state.high + 1))
                for (_, lo, hi), weight in zip(bins, weights):
                    assert hi >= lo
                    assert abs(Fraction(hi - lo + 1, width) - Fraction(weight, total)) < Fraction(1, width)


def test_exhaustive_small_bit_strings_static_and_prefix_adaptive_laws():
    config = Config(4)
    static = [law(weights) for total in range(1, 5) for weights in compositions(total)]
    for length in range(8):
        for bits in product((0, 1), repeat=length):
            for fixed in static:
                sender, receiver = Sender(bits, config), Receiver(config)
                history = []
                for turn in range(12):
                    # Every other step uses a fixed, publicly prefix-dependent law.
                    support = fixed if turn % 2 == 0 else static[(sum(history) + turn) % len(static)]
                    step = sender.take(support)
                    history.append(step.symbol)
                    observed = receiver.take(support, step.symbol)
                    assert observed == step
                    recovered = receiver.recovered_bits
                    assert recovered == (bits + (0,) * len(recovered))[:len(recovered)]
                    assert recovered == sender.resolved_bits
                    assert sender.consumed_stream_bits == config.state_bits + len(recovered) + sender.state.pending
                    assert sender.state.shift_count == len(recovered) + sender.state.pending
                    assert sender.delivered_input_bits == min(len(bits), len(recovered))


def test_precision_extremes_and_nonmonotonic_symbol_order():
    for precision in (4, 8, 32, 64, 256):
        config = Config(precision)
        support = Law(((999, 1), (0, 1)))
        sender, receiver = Sender((1, 0, 1, 1), config), Receiver(config)
        seen = []
        for _ in range(4):
            symbol = sender.take(support).symbol
            seen.append(symbol)
            receiver.take(support, symbol)
        assert seen == [0, 999, 0, 0]
        assert receiver.recovered_bits == (1, 0, 1, 1)


@pytest.mark.parametrize("rows", [(), [], ((0, 0),), ((0, -1),), ((0, True),), ((True, 1),),
                                        ((-1, 1),), ((0, 1), (0, 2)), ((0, 1.5),), ((0,),), ([0, 1],)])
def test_invalid_laws(rows):
    with pytest.raises(ContractError):
        Law(rows)


@pytest.mark.parametrize("kwargs", [dict(state_bits=True), dict(state_bits=3), dict(state_bits=257),
                                          dict(max_symbols=-1), dict(max_pending_bits=True), dict(max_stream_bits=31)])
def test_invalid_configs(kwargs):
    with pytest.raises(ContractError):
        Config(**kwargs)


@pytest.mark.parametrize("bits", [(True,), (2,), (0.0,), None, "01"])
def test_invalid_bits(bits):
    with pytest.raises(ContractError):
        Sender(bits)


def test_invalid_input_does_not_modify_receiver():
    receiver = Receiver(Config(4))
    initial = receiver.state
    for support, symbol in ((law((5,)), 10), (law((1,)), 4), (law((1,)), True)):
        with pytest.raises(ContractError):
            receiver.take(support, symbol)
        assert receiver.state == initial
        assert receiver.recovered_bits == ()


def test_direct_function_contracts():
    for config in (None, 4, True):
        with pytest.raises(ContractError):
            integer_bins(State(0, 15), law((1,)), config)
    for state in (State(-1, 15), State(0, 16), State(0, 7), State(4, 11), State(0, 15, 1),
                  State(0, 15, 0, -1), State(0, 15, 0, 0, -1)):
        with pytest.raises(ContractError):
            advance(state, law((1,)), 10, Config(4))


def test_byte_helpers_reject_partial_bytes_and_boolean_bits():
    assert bits_from_bytes(b"\x00\xa5\xff") == (0,) * 8 + (1, 0, 1, 0, 0, 1, 0, 1) + (1,) * 8
    assert exact_bytes(bits_from_bytes(b"\x00\xa5\xff")) == b"\x00\xa5\xff"
    for values in ((1,), (True,) * 8, (2,) * 8, None):
        with pytest.raises(ContractError):
            exact_bytes(values)
    with pytest.raises(ContractError):
        bits_from_bytes(bytearray(b"a"))
