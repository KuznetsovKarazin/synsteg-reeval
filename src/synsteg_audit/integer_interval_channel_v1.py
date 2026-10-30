"""Finite-precision inverse arithmetic channel, without framing or encryption.

The sender decodes a finite bit string extended by zeros into public symbols.
The receiver encodes observed symbols and releases only resolved prefix bits.
There is deliberately no final flush: lookahead and pending underflow bits do
not establish delivered capacity. No model/tokenizer or random source is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

VERSION = "integer-interval-channel-v1"


class ContractError(ValueError):
    """Invalid state, symbol support or bit input."""


class CapacityLimit(RuntimeError):
    """Public resource bound reached; no transition or input read was committed.

    If the required frame is still incomplete, the application must retain an
    insufficient-capacity result. This exception never authorizes a flush,
    adaptive key search, or an automatic new generation attempt.
    """
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Config:
    state_bits: int = 32
    max_symbols: int = 65536
    max_pending_bits: int = 4096
    max_stream_bits: int = 1048576

    def __post_init__(self) -> None:
        if type(self.state_bits) is not int or not 4 <= self.state_bits <= 256:
            raise ContractError("state_bits must be an integer in [4, 256]")
        if any(type(value) is not int or value < 0 for value in
               (self.max_symbols, self.max_pending_bits, self.max_stream_bits)):
            raise ContractError("public limits must be nonnegative integers")
        if self.max_stream_bits < self.state_bits:
            raise ContractError("stream limit must accommodate initial lookahead")

    @property
    def full(self) -> int:
        return 1 << self.state_bits

    @property
    def maximum_total(self) -> int:
        return self.full // 4


@dataclass(frozen=True)
class Law:
    """Public order is part of the law; zero weights must be removed upstream.

    Symbol IDs may name canonical tokens or already validated token packets.
    This layer neither checks wire canonicality nor authenticates the law.
    """
    rows: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or not self.rows:
            raise ContractError("support must be a nonempty immutable tuple")
        if any(type(row) is not tuple or len(row) != 2
               or type(row[0]) is not int or row[0] < 0
               or type(row[1]) is not int or row[1] <= 0 for row in self.rows):
            raise ContractError("require nonnegative integer IDs and positive integer weights")
        if len({symbol for symbol, _ in self.rows}) != len(self.rows):
            raise ContractError("duplicate symbol IDs")

    @property
    def total(self) -> int:
        return sum(weight for _, weight in self.rows)

    def bounds(self, symbol: int) -> tuple[int, int]:
        if type(symbol) is not int:
            raise ContractError("symbol must be an integer")
        start = 0
        for candidate, weight in self.rows:
            end = start + weight
            if candidate == symbol:
                return start, end
            start = end
        raise ContractError("symbol outside the public support")


@dataclass(frozen=True)
class State:
    low: int
    high: int
    pending: int = 0
    symbols: int = 0
    shift_count: int = 0


@dataclass(frozen=True)
class Step:
    symbol: int
    before: State
    after: State
    emitted_bits: tuple[int, ...]
    renormalizations: tuple[str, ...]
    selected_integer_width: int
    parent_integer_width: int
    selected_weight: int
    weight_total: int


def _check_state(state: State, config: Config) -> None:
    if type(config) is not Config:
        raise ContractError("require Config")
    if (type(state) is not State
            or any(type(v) is not int for v in (state.low, state.high, state.pending,
                                               state.symbols, state.shift_count))
            or not 0 <= state.low <= state.high < config.full
            or not 0 <= state.pending <= min(state.shift_count, config.max_pending_bits)
            or not 0 <= state.symbols <= config.max_symbols
            or not 0 <= state.shift_count <= config.max_stream_bits - config.state_bits):
        raise ContractError("invalid interval state")
    # Symbol boundaries must be fully normalized; this implies span > quarter.
    half, quarter = config.full // 2, config.full // 4
    if (state.high < half or state.low >= half
            or (state.low >= quarter and state.high < 3 * quarter)):
        raise ContractError("state is not a normalized symbol boundary")


def _check_law(law: Law, config: Config) -> None:
    if type(law) is not Law or law.total > config.maximum_total:
        raise ContractError("law total exceeds the precision contract")


def integer_bins(state: State, law: Law, config: Config) -> tuple[tuple[int, int, int], ...]:
    """Disjoint inclusive bins; not a claim about the neural model's law."""
    _check_state(state, config)
    _check_law(law, config)
    width = state.high - state.low + 1
    total = law.total
    cumulative = 0
    rows = []
    for symbol, weight in law.rows:
        left = state.low + width * cumulative // total
        cumulative += weight
        right = state.low + width * cumulative // total - 1
        if left > right:
            raise AssertionError("positive weights must have nonempty bins")
        rows.append((symbol, left, right))
    return tuple(rows)


def advance(state: State, law: Law, symbol: int, config: Config) -> Step:
    """Public receiver transition, including only already determined bits."""
    _check_state(state, config)
    _check_law(law, config)
    if state.symbols >= config.max_symbols:
        raise CapacityLimit("max_symbols")
    lower, upper = law.bounds(symbol)
    width = state.high - state.low + 1
    total = law.total
    low = state.low + width * lower // total
    high = state.low + width * upper // total - 1
    selected_width = high - low + 1
    pending = state.pending
    emitted = []
    events = []
    half, quarter = config.full // 2, config.full // 4
    while True:
        if high < half:
            emitted.extend((0,) + (1,) * pending)
            pending = 0
            events.append("lower_half")
        elif low >= half:
            emitted.extend((1,) + (0,) * pending)
            pending = 0
            low -= half
            high -= half
            events.append("upper_half")
        elif low >= quarter and high < 3 * quarter:
            pending += 1
            if pending > config.max_pending_bits:
                raise CapacityLimit("max_pending_bits")
            low -= quarter
            high -= quarter
            events.append("middle_half")
        else:
            break
        if config.state_bits + state.shift_count + len(events) > config.max_stream_bits:
            raise CapacityLimit("max_stream_bits")
        low *= 2
        high = 2 * high + 1
    after = State(low, high, pending, state.symbols + 1, state.shift_count + len(events))
    _check_state(after, config)
    return Step(symbol, state, after, tuple(emitted), tuple(events),
                selected_width, width, upper - lower, total)


class Receiver:
    """Only needs symbols, the public law at each prefix, and public precision."""
    def __init__(self, config: Config = Config()):
        if type(config) is not Config:
            raise ContractError("require Config")
        self.config = config
        self.state = State(0, config.full - 1)
        self._bits: list[int] = []

    @property
    def recovered_bits(self) -> tuple[int, ...]:
        return tuple(self._bits)

    def take(self, law: Law, symbol: int) -> Step:
        result = advance(self.state, law, symbol, self.config)
        self.state = result.after
        self._bits.extend(result.emitted_bits)
        return result


class Sender:
    """Finite bit input plus a documented zero extension, never an API RNG.

    consumed_stream_bits counts buffered/read bits, including lookahead and
    zero padding. It is NOT capacity. resolved_bits mirrors the receiver and
    can be used to decide whether a separately framed message is recovered.
    """
    def __init__(self, bits: Iterable[int], config: Config = Config()):
        if type(config) is not Config:
            raise ContractError("require Config")
        try:
            bits = tuple(bits)
        except TypeError as exc:
            raise ContractError("bits must be iterable") from exc
        if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
            raise ContractError("bits must be integer zero or one")
        self.config = config
        self.input_bits = bits
        self.state = State(0, config.full - 1)
        self.consumed_stream_bits = 0
        self._resolved: list[int] = []
        self.code = 0
        for _ in range(config.state_bits):
            self.code = 2 * self.code + self._read()

    def _read(self) -> int:
        index = self.consumed_stream_bits
        self.consumed_stream_bits += 1
        return self.input_bits[index] if index < len(self.input_bits) else 0

    @property
    def padding_bits_read(self) -> int:
        return max(0, self.consumed_stream_bits - len(self.input_bits))

    @property
    def resolved_bits(self) -> tuple[int, ...]:
        return tuple(self._resolved)

    @property
    def delivered_input_bits(self) -> int:
        return min(len(self.input_bits), len(self._resolved))

    @property
    def input_complete(self) -> bool:
        return len(self._resolved) >= len(self.input_bits)

    def take(self, law: Law) -> Step:
        bins = integer_bins(self.state, law, self.config)
        if not self.state.low <= self.code <= self.state.high:
            raise ContractError("sender code is outside the interval")
        symbol = next(symbol for symbol, lo, hi in bins if lo <= self.code <= hi)
        result = advance(self.state, law, symbol, self.config)
        code = self.code
        for event in result.renormalizations:
            if event == "upper_half":
                code -= self.config.full // 2
            elif event == "middle_half":
                code -= self.config.full // 4
            code = 2 * code + self._read()
        # No termination flush: the receiver cannot infer still-pending bits.
        for bit in result.emitted_bits:
            index = len(self._resolved)
            expected = self.input_bits[index] if index < len(self.input_bits) else 0
            if bit != expected:
                raise AssertionError("resolved receiver prefix differs from source stream")
            self._resolved.append(bit)
        self.code = code
        self.state = result.after
        if not self.state.low <= code <= self.state.high:
            raise AssertionError("code left the selected renormalized interval")
        return result


def bits_from_bytes(raw: bytes) -> tuple[int, ...]:
    if type(raw) is not bytes:
        raise ContractError("require bytes")
    return tuple((value >> shift) & 1 for value in raw for shift in range(7, -1, -1))


def exact_bytes(bits: Iterable[int]) -> bytes:
    try:
        values = tuple(bits)
    except TypeError as exc:
        raise ContractError("bits must be iterable") from exc
    if len(values) % 8 or any(type(bit) is not int or bit not in (0, 1) for bit in values):
        raise ContractError("require an integer bit string with a whole number of bytes")
    return bytes(sum(bit << (7 - j) for j, bit in enumerate(values[i:i + 8]))
                 for i in range(0, len(values), 8))
