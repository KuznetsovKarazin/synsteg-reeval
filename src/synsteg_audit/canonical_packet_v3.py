"""Bounded prefix-free token packets with canonical text at packet boundaries.

This is a transport gate and a finite synthetic candidate builder, not a keyed
payload codec, language-model sampler, or cryptographic security construction.
The frozen v2 public codec and boundary-state validation are reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Iterable, Sequence

from .canonical_channel_v2 import (
    CanonicalGate, ContractError, MAX_WEIGHT_TOTAL, Prefix, PublicCodec,
)

VERSION = "canonical-packet-v3"
REASONS = frozenset({"accepted", "zero_weight", "special_token",
    "rewrites_prior_bytes", "empty_extension", "invalid_utf8", "retokenization_mismatch"})


def _ids(values, label):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ContractError(label + ": IDs must be iterable") from exc
    if any(type(t) is not int or t < 0 for t in result):
        raise ContractError(label + ": IDs must be nonnegative integers")
    return result


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("public resource/policy SHA256 required")


def _limit(value, label):
    if type(value) is not int or value <= 0:
        raise ContractError(label + " must be a positive integer")


def _packet(value, max_tokens):
    packet = _ids(value, "packet")
    if not 1 <= len(packet) <= max_tokens:
        raise ContractError("packet length must be between one and the public bound")
    return packet


def _prefix_free(packets):
    ordered = sorted(packets)
    for left, right in zip(ordered, ordered[1:]):
        if right[:len(left)] == left:
            raise ContractError("packet candidates must be unique and prefix-free")


class ConstructionRefusal(ValueError):
    """A declared construction budget was exceeded, not a backend failure."""


@dataclass(frozen=True)
class PacketVerdict:
    packet: tuple[int, ...]
    weight: int
    reason: str
    suffix: bytes = b""


@dataclass(frozen=True)
class PacketAdmission:
    codec_digest: str
    policy_digest: str
    max_packet_tokens: int
    prefix: Prefix
    verdicts: tuple[PacketVerdict, ...]

    def __post_init__(self):
        _digest(self.codec_digest)
        _digest(self.policy_digest)
        _limit(self.max_packet_tokens, "max_packet_tokens")
        if type(self.verdicts) is not tuple:
            raise ContractError("verdicts must be an immutable tuple")
        for row in self.verdicts:
            if not isinstance(row, PacketVerdict) or type(row.packet) is not tuple:
                raise ContractError("invalid packet verdict")
            _packet(row.packet, self.max_packet_tokens)
            if (type(row.weight) is not int or row.weight < 0 or row.reason not in REASONS
                    or type(row.suffix) is not bytes):
                raise ContractError("invalid packet weight/reason/bytes")
            if row.reason == "accepted" and (row.weight <= 0 or not row.suffix):
                raise ContractError("accepted packet requires positive weight and extension")
            if row.reason != "accepted" and row.suffix:
                raise ContractError("rejected packet cannot append bytes")
            if (row.reason == "zero_weight") != (row.weight == 0):
                raise ContractError("zero weight reason mismatch")
        packets = tuple(row.packet for row in self.verdicts)
        if packets != tuple(sorted(packets)):
            raise ContractError("packet verdicts must be lexicographically ordered")
        _prefix_free(packets)
        if self.input_total > MAX_WEIGHT_TOTAL:
            raise ContractError("candidate total exceeds uint256 sampling contract")

    @property
    def input_total(self):
        return sum(row.weight for row in self.verdicts)

    @property
    def weights(self):
        return tuple((row.packet, row.weight) for row in self.verdicts if row.reason == "accepted")

    @property
    def retained_total(self):
        return sum(weight for _, weight in self.weights)

    @property
    def status(self):
        if not self.input_total:
            return "empty_input_distribution"
        return "ok" if self.retained_total else "empty_support"

    def choose(self, rank):
        """Exact rank mapping only. A downstream sampler must draw below S."""
        if type(rank) is not int or not 0 <= rank < self.retained_total:
            raise ContractError("rank must be an integer below retained weight total")
        for packet, weight in self.weights:
            if rank < weight:
                return packet
            rank -= weight
        raise AssertionError("unreachable rank")

    def summary(self):
        total, retained = self.input_total, self.retained_total
        return {"schema": VERSION + "-admission", "status": self.status,
            "codec_digest": self.codec_digest, "policy_digest": self.policy_digest,
            "max_packet_tokens": self.max_packet_tokens, "prefix_ids": list(self.prefix.ids),
            "prefix_bytes_sha256": hashlib.sha256(self.prefix.raw).hexdigest(),
            "input_packets": len(self.verdicts), "input_weight_total": total,
            "accepted_packets": len(self.weights), "retained_weight_total": retained,
            "removed_weight_total": total - retained,
            "removed_mass": None if not total else {"numerator": total - retained, "denominator": total},
            "conditioned_weights": [[list(packet), weight] for packet, weight in self.weights],
            "verdicts": [{"packet": list(row.packet), "weight": row.weight,
                "reason": row.reason, "appended_bytes": len(row.suffix)} for row in self.verdicts]}


class CanonicalPacketGate:
    def __init__(self, codec: PublicCodec, *, max_packet_tokens: int, policy_digest: str):
        _limit(max_packet_tokens, "max_packet_tokens")
        _digest(policy_digest)
        self._boundary_gate = CanonicalGate(codec)
        self.codec = codec
        self.max_packet_tokens = max_packet_tokens
        self.policy_digest = policy_digest

    def validate(self, prefix: Prefix):
        self._boundary_gate.validate(prefix)

    def _inspect(self, prefix, packet, weight):
        if not weight:
            return PacketVerdict(packet, weight, "zero_weight")
        if any(token in self.codec.special_token_ids for token in packet):
            return PacketVerdict(packet, weight, "special_token")
        expected = prefix.ids + packet
        raw = self.codec.render(expected)
        if not raw.startswith(prefix.raw):
            return PacketVerdict(packet, weight, "rewrites_prior_bytes")
        if raw == prefix.raw:
            return PacketVerdict(packet, weight, "empty_extension")
        try:
            encoded = self.codec.encode(raw)
        except UnicodeDecodeError:
            return PacketVerdict(packet, weight, "invalid_utf8")
        if encoded != expected:
            return PacketVerdict(packet, weight, "retokenization_mismatch")
        return PacketVerdict(packet, weight, "accepted", raw[len(prefix.raw):])

    def admit(self, prefix: Prefix, candidates: Iterable[tuple[Sequence[int], int]]):
        self.validate(prefix)
        try:
            raw_rows = tuple(tuple(row) for row in candidates)
        except TypeError as exc:
            raise ContractError("candidate rows must be (packet, integer_weight)") from exc
        if any(len(row) != 2 or type(row[1]) is not int or row[1] < 0 for row in raw_rows):
            raise ContractError("candidate rows require nonnegative integer weights")
        rows = tuple((_packet(packet, self.max_packet_tokens), weight) for packet, weight in raw_rows)
        _prefix_free(packet for packet, _ in rows)
        if sum(weight for _, weight in rows) > MAX_WEIGHT_TOTAL:
            raise ContractError("candidate total exceeds uint256 sampling contract")
        verdicts = tuple(self._inspect(prefix, packet, weight) for packet, weight in sorted(rows))
        return PacketAdmission(self.codec.resource_digest, self.policy_digest,
                               self.max_packet_tokens, prefix, verdicts)

    def take(self, prefix: Prefix, decision: PacketAdmission, packet: Sequence[int]):
        self.validate(prefix)
        packet = _packet(packet, self.max_packet_tokens)
        if (not isinstance(decision, PacketAdmission) or decision.prefix != prefix
                or decision.codec_digest != self.codec.resource_digest
                or decision.policy_digest != self.policy_digest
                or decision.max_packet_tokens != self.max_packet_tokens):
            raise ContractError("decision does not belong to this boundary/resource/policy")
        selected = [row for row in decision.verdicts if row.packet == packet and row.reason == "accepted"]
        if len(selected) != 1:
            raise ContractError("packet is not in admitted positive support")
        fresh = self._inspect(prefix, packet, selected[0].weight)
        if fresh != selected[0]:
            raise ContractError("codec changed after admission")
        return Prefix(prefix.ids + packet, prefix.raw + fresh.suffix)


@dataclass(frozen=True)
class Frontier:
    status: str
    public_paths: int
    possible_trie_nodes: int
    node_budget: int
    inspections: tuple[PacketVerdict, ...]
    leaves: tuple[tuple[int, ...], ...]

    def candidates(self):
        if self.status != "ok":
            raise ConstructionRefusal(self.status)
        return tuple((packet, 1) for packet in self.leaves)

    def summary(self):
        return {"status": self.status, "public_paths": self.public_paths,
            "possible_trie_nodes": self.possible_trie_nodes, "node_budget": self.node_budget,
            "examined_nodes": len(self.inspections), "frontier_packets": len(self.leaves),
            "weight_law": "uniform over this public first-canonical finite frontier; not LM probabilities",
            "mass_outside_public_trie": None,
            "inspections": [{"packet": list(row.packet), "reason": row.reason,
                             "appended_bytes": len(row.suffix)} for row in self.inspections]}


def first_canonical_frontier(gate: CanonicalPacketGate, prefix: Prefix,
                             public_paths: Iterable[Sequence[int]], *, node_budget: int):
    """Finite public trie: first accepted node, special-containing node, or leaf.

    Noncanonical intermediate render may be extended. A strict upper bound on
    all possible nodes is checked before any inspection; overflow refuses the
    entire construction and never returns a successful search-order subset.
    Uniform *frontier* weights are diagnostic only, not token-path probability.
    """
    gate.validate(prefix)
    _limit(node_budget, "node_budget")
    paths = tuple(_packet(path, gate.max_packet_tokens) for path in public_paths)
    if len(set(paths)) != len(paths):
        raise ContractError("duplicate public trie path")
    children = {(): set()}
    for path in paths:
        for length in range(1, len(path) + 1):
            node, parent = path[:length], path[:length - 1]
            children.setdefault(parent, set()).add(node)
            children.setdefault(node, set())
    count = len(children) - 1
    if count > node_budget:
        return Frontier("construction_node_budget_exceeded", len(paths), count, node_budget, (), ())
    inspections, leaves = [], []
    stack = list(reversed(sorted(children[()])))
    while stack:
        packet = stack.pop()
        verdict = gate._inspect(prefix, packet, 1)
        inspections.append(verdict)
        if verdict.reason in {"accepted", "special_token"} or not children[packet]:
            leaves.append(packet)
        else:
            stack.extend(reversed(sorted(children[packet])))
    leaves = tuple(sorted(leaves))
    _prefix_free(leaves)
    return Frontier("ok", len(paths), count, node_budget, tuple(inspections), leaves)


def recover_packets(gate: CanonicalPacketGate, received: bytes,
                    provider: Callable[[tuple[int, ...]], Iterable[tuple[Sequence[int], int]]],
                    *, prompt_ids: Sequence[int] = (), expected_output_tokens: int | None = None,
                    expected_packets: int | None = None):
    """Reconstruct packet boundaries from bytes and public prefix-free choices.

    Without a public expected count a wire truncated exactly at a boundary can
    still verify. Neither count is an authenticator or a payload length frame.
    """
    if type(received) is not bytes:
        raise ContractError("wire must be exact bytes")
    prompt = _ids(prompt_ids, "model prompt")
    for count in (expected_output_tokens, expected_packets):
        if count is not None and (type(count) is not int or count < 0):
            raise ContractError("expected counts must be nonnegative integers")
    try:
        ids = gate.codec.encode(received)
    except UnicodeDecodeError:
        return {"status": "invalid_utf8", "steps": []}
    if gate.codec.render(ids) != received:
        return {"status": "noncanonical_wire_bytes", "steps": []}
    if expected_output_tokens is not None and len(ids) != expected_output_tokens:
        return {"status": "token_count_mismatch", "recovered_token_count": len(ids), "steps": []}
    prefix, traces, packets = Prefix(), [], []
    while len(prefix.ids) < len(ids):
        try:
            decision = gate.admit(prefix, provider(prompt + prefix.ids))
        except ConstructionRefusal as exc:
            return {"status": "construction_refusal", "reason": str(exc), "steps": traces}
        remaining = ids[len(prefix.ids):]
        matches = [row for row in decision.verdicts if row.weight > 0
                   and remaining[:len(row.packet)] == row.packet]
        trace = {"packet_index": len(packets), "token_offset": len(prefix.ids),
                 **decision.summary(), "observed_packet": None}
        traces.append(trace)
        if not matches:
            partial = any(row.weight > 0 and len(remaining) < len(row.packet)
                          and row.packet[:len(remaining)] == remaining for row in decision.verdicts)
            return {"status": "truncated_packet" if partial else "ineligible_packet", "steps": traces}
        if len(matches) != 1:
            raise AssertionError("prefix-free contract violated")
        row = matches[0]
        trace["observed_packet"] = list(row.packet)
        if row.reason != "accepted":
            return {"status": "rejected_packet", "reason": row.reason, "steps": traces}
        prefix = gate.take(prefix, decision, row.packet)
        packets.append(list(row.packet))
        if not received.startswith(prefix.raw):
            return {"status": "prefix_bytes_mismatch", "steps": traces}
    if prefix.raw != received:
        return {"status": "final_bytes_mismatch", "steps": traces}
    if expected_packets is not None and len(packets) != expected_packets:
        return {"status": "packet_count_mismatch", "recovered_packets": len(packets), "steps": traces}
    return {"status": "canonical_packet_path_verified", "token_ids": list(ids), "packets": packets,
        "output_tokens": len(ids), "packet_count": len(packets), "steps": traces,
        "wire_bytes_sha256": hashlib.sha256(received).hexdigest(),
        "public_count_checked": expected_output_tokens is not None or expected_packets is not None,
        "payload_authenticated": False, "payload_delivered": False}
