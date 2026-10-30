"""Append-only, prefix-canonical text transport gate, not a complete stegosystem.

The public codec renders the ENTIRE generated output prefix. Model prompt IDs
are separate. Admission conditions a supplied integer candidate law; it does
not preserve the unconstrained language-model law or guarantee useful capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Iterable, Sequence

VERSION = "canonical-channel-v2"
MAX_WEIGHT_TOTAL = 1 << 256


class ContractError(ValueError):
    """Invalid public codec/provider input, distinct from candidate rejection."""


def _ids(values: Iterable[int], label: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ContractError(label + ": token IDs must be iterable") from exc
    if any(type(token) is not int or token < 0 for token in result):
        raise ContractError(label + ": token IDs must be nonnegative integers")
    return result


@dataclass(frozen=True)
class PublicCodec:
    """Callbacks must disable automatic specials, cleanup and implicit context.

    render_ids returns exact bytes for a FULL OUTPUT prefix, including invalid
    partial UTF-8 when the backend exposes it. A renderer that already replaces
    invalid bytes can still be used, but encode(render(ids)) must recover IDs.
    The digest identifies the publicly pinned tokenizer/config/adapter bundle;
    this class cannot independently prove that callbacks implement that bundle.
    """
    encode_text: Callable[[str], Sequence[int]]
    render_ids: Callable[[tuple[int, ...]], bytes]
    special_token_ids: frozenset[int]
    resource_digest: str

    def encode(self, raw: bytes) -> tuple[int, ...]:
        text = raw.decode("utf-8", errors="strict")
        try:
            return _ids(self.encode_text(text), "encode")
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("encode backend failed: " + type(exc).__name__) from exc

    def render(self, ids: tuple[int, ...]) -> bytes:
        try:
            raw = self.render_ids(ids)
        except Exception as exc:
            raise ContractError("render backend failed: " + type(exc).__name__) from exc
        if type(raw) is not bytes:
            raise ContractError("renderer must return exact bytes")
        return raw


@dataclass(frozen=True)
class Prefix:
    ids: tuple[int, ...] = ()
    raw: bytes = b""


@dataclass(frozen=True)
class Verdict:
    token_id: int
    weight: int
    reason: str
    suffix: bytes = b""


@dataclass(frozen=True)
class Admission:
    codec_digest: str
    prefix: Prefix
    verdicts: tuple[Verdict, ...]

    def __post_init__(self):
        reasons = {"accepted", "zero_weight", "special_token", "rewrites_prior_bytes",
                   "empty_extension", "invalid_utf8", "retokenization_mismatch"}
        if type(self.verdicts) is not tuple or any(not isinstance(row, Verdict) for row in self.verdicts):
            raise ContractError("admission verdicts must be an immutable tuple")
        for row in self.verdicts:
            if (type(row.token_id) is not int or row.token_id < 0
                    or type(row.weight) is not int or row.weight < 0
                    or row.reason not in reasons or type(row.suffix) is not bytes):
                raise ContractError("invalid admission verdict")
            if row.reason == "accepted" and (row.weight <= 0 or not row.suffix):
                raise ContractError("accepted support requires positive weight and byte extension")
            if row.reason != "accepted" and row.suffix:
                raise ContractError("rejected candidate cannot supply an extension")
            if (row.reason == "zero_weight") != (row.weight == 0):
                raise ContractError("zero weight reason mismatch")
        if (len({row.token_id for row in self.verdicts}) != len(self.verdicts)
                or tuple(row.token_id for row in self.verdicts) != tuple(sorted(row.token_id for row in self.verdicts))
                or sum(row.weight for row in self.verdicts) > MAX_WEIGHT_TOTAL):
            raise ContractError("admission candidate order/total is invalid")

    @property
    def input_total(self) -> int:
        return sum(row.weight for row in self.verdicts)

    @property
    def weights(self) -> tuple[tuple[int, int], ...]:
        return tuple((row.token_id, row.weight) for row in self.verdicts if row.reason == "accepted")

    @property
    def retained_total(self) -> int:
        return sum(weight for _, weight in self.weights)

    @property
    def status(self) -> str:
        if not self.input_total:
            return "empty_input_distribution"
        return "ok" if self.retained_total else "empty_support"

    def choose(self, rank: int) -> int:
        """Map a supplied uniform integer in [0,S) to the conditional law.

        This performs no randomness or cryptography. Downstream keyed samplers
        must use S, not the pre-filter total W, and bind public resources.
        """
        if type(rank) is not int or not 0 <= rank < self.retained_total:
            raise ContractError("rank must be an integer below retained weight total")
        for token, weight in self.weights:
            if rank < weight:
                return token
            rank -= weight
        raise AssertionError("unreachable cumulative weight")

    def summary(self) -> dict:
        total, retained = self.input_total, self.retained_total
        return {
            "schema": VERSION + "-admission", "status": self.status,
            "codec_digest": self.codec_digest,
            "prefix_ids": list(self.prefix.ids),
            "prefix_bytes_sha256": hashlib.sha256(self.prefix.raw).hexdigest(),
            "input_candidates": len(self.verdicts),
            "input_positive_candidates": sum(row.weight > 0 for row in self.verdicts),
            "accepted_positive_candidates": len(self.weights),
            "input_weight_total": total, "retained_weight_total": retained,
            "removed_weight_total": total - retained,
            "removed_mass": None if not total else {"numerator": total - retained, "denominator": total},
            "conditioned_weights": [list(pair) for pair in self.weights],
            "verdicts": [{"token_id": row.token_id, "weight": row.weight,
                          "reason": row.reason, "appended_bytes": len(row.suffix)}
                         for row in self.verdicts],
        }


class CanonicalGate:
    def __init__(self, codec: PublicCodec):
        if (not isinstance(codec.resource_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", codec.resource_digest) is None):
            raise ContractError("require the public resource SHA256")
        if type(codec.special_token_ids) is not frozenset:
            raise ContractError("special token IDs must be a frozenset")
        _ids(codec.special_token_ids, "special tokens")
        self.codec = codec
        self.validate(Prefix())

    def validate(self, prefix: Prefix) -> None:
        if not isinstance(prefix, Prefix) or type(prefix.ids) is not tuple or type(prefix.raw) is not bytes:
            raise ContractError("invalid prefix state")
        ids = _ids(prefix.ids, "prefix")
        if any(token in self.codec.special_token_ids for token in ids):
            raise ContractError("output prefix contains a special token")
        try:
            encoded = self.codec.encode(prefix.raw)
        except UnicodeDecodeError as exc:
            raise ContractError("prefix is not strict UTF-8") from exc
        if self.codec.render(ids) != prefix.raw or encoded != ids:
            raise ContractError("prefix is not byte-exact and canonical")

    def _inspect(self, prefix: Prefix, token: int, weight: int) -> Verdict:
        if not weight:
            return Verdict(token, weight, "zero_weight")
        if token in self.codec.special_token_ids:
            return Verdict(token, weight, "special_token")
        expected = prefix.ids + (token,)
        raw = self.codec.render(expected)
        if not raw.startswith(prefix.raw):
            return Verdict(token, weight, "rewrites_prior_bytes")
        if raw == prefix.raw:
            return Verdict(token, weight, "empty_extension")
        try:
            encoded = self.codec.encode(raw)
        except UnicodeDecodeError:
            return Verdict(token, weight, "invalid_utf8")
        if encoded != expected:
            return Verdict(token, weight, "retokenization_mismatch")
        return Verdict(token, weight, "accepted", raw[len(prefix.raw):])

    def admit(self, prefix: Prefix, candidates: Iterable[tuple[int, int]]) -> Admission:
        self.validate(prefix)
        try:
            rows = tuple(tuple(row) for row in candidates)
        except TypeError as exc:
            raise ContractError("candidate rows must be (token_id, integer_weight)") from exc
        if any(len(row) != 2 or type(row[0]) is not int or row[0] < 0
               or type(row[1]) is not int or row[1] < 0 for row in rows):
            raise ContractError("candidate rows must be (nonnegative ID, nonnegative integer weight)")
        if len({row[0] for row in rows}) != len(rows):
            raise ContractError("duplicate candidate token ID")
        if sum(weight for _, weight in rows) > MAX_WEIGHT_TOTAL:
            raise ContractError("candidate total exceeds uint256 sampling contract")
        verdicts = tuple(self._inspect(prefix, token, weight) for token, weight in sorted(rows))
        return Admission(self.codec.resource_digest, prefix, verdicts)

    def take(self, prefix: Prefix, decision: Admission, token: int) -> Prefix:
        """Advance only an accepted positive-weight token at this exact prefix."""
        self.validate(prefix)
        if (not isinstance(decision, Admission) or type(token) is not int
                or decision.codec_digest != self.codec.resource_digest or decision.prefix != prefix):
            raise ContractError("decision does not belong to this prefix/resource")
        selected = [row for row in decision.verdicts if row.token_id == token and row.reason == "accepted"]
        if len(selected) != 1:
            raise ContractError("token is not in admitted positive support")
        # Recheck the backend rather than trusting a stale mutable external codec.
        fresh = self._inspect(prefix, token, selected[0].weight)
        if fresh != selected[0]:
            raise ContractError("codec changed after admission")
        return Prefix(prefix.ids + (token,), prefix.raw + fresh.suffix)


def recover_path(gate: CanonicalGate, received: bytes,
                 provider: Callable[[tuple[int, ...]], Iterable[tuple[int, int]]],
                 *, prompt_ids: Sequence[int] = (), expected_output_tokens: int | None = None) -> dict:
    """Receiver-only replay from transmitted bytes and public resources.

    provider sees EXACT prompt IDs + reconstructed output-prefix IDs. No sender
    IDs, candidate tables, selected bits or trace enter this function. Its result
    certifies canonical eligibility only, not authenticity or a decoded payload.
    """
    if type(received) is not bytes:
        raise ContractError("wire value must be bytes")
    prompt = _ids(prompt_ids, "model prompt")
    if expected_output_tokens is not None and (type(expected_output_tokens) is not int or expected_output_tokens < 0):
        raise ContractError("expected output token count must be a nonnegative integer")
    try:
        ids = gate.codec.encode(received)
    except UnicodeDecodeError:
        return {"status": "invalid_utf8", "steps": []}
    if gate.codec.render(ids) != received:
        return {"status": "noncanonical_wire_bytes", "steps": []}
    if expected_output_tokens is not None and len(ids) != expected_output_tokens:
        return {"status": "token_count_mismatch", "recovered_token_count": len(ids), "steps": []}
    prefix, traces = Prefix(), []
    for index, token in enumerate(ids):
        decision = gate.admit(prefix, provider(prompt + prefix.ids))
        trace = {"step": index, **decision.summary(), "observed_token": token}
        traces.append(trace)
        if token not in dict(decision.weights):
            return {"status": "ineligible_token", "failure_step": index, "steps": traces}
        prefix = gate.take(prefix, decision, token)
        if not received.startswith(prefix.raw):
            return {"status": "prefix_bytes_mismatch", "failure_step": index, "steps": traces}
    if prefix.raw != received:
        return {"status": "final_bytes_mismatch", "steps": traces}
    return {"status": "canonical_path_verified", "token_ids": list(ids), "steps": traces,
            "output_tokens": len(ids), "wire_bytes_sha256": hashlib.sha256(received).hexdigest(),
            "payload_authenticated": False}
