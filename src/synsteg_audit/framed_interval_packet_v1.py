"""Synthetic composition of packet transport, arithmetic coding and library AEAD.

No LLM, tokenizer, network, secret loading, entropy collection or API calls.
The caller supplies a 32-byte key and a nonce never reused with that key.
Only full authenticated frames with a deterministic exact-wire replay succeed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import re
from typing import Callable, Iterable, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .canonical_channel_v2 import ContractError, Prefix
from .canonical_packet_v3 import CanonicalPacketGate, ConstructionRefusal
from .integer_interval_channel_v1 import (
    CapacityLimit, Config, Law, Receiver, Sender, bits_from_bytes, exact_bytes,
)

VERSION = "framed-interval-packet-v1"
DOMAIN = VERSION.encode("ascii") + b"\x00"
HEADER_BYTES = 2
TAG_BYTES = 16
KEY_BYTES = 32
NONCE_BYTES = 12


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("resource identity must be SHA256 hex")


@dataclass(frozen=True)
class PublicBundle:
    gate: CanonicalPacketGate
    provider: Callable[[tuple[int, ...]], Iterable[tuple[Sequence[int], int]]]
    resource_id: str
    resource_digest: str
    provider_digest: str

    def __post_init__(self):
        if (not isinstance(self.gate, CanonicalPacketGate) or not callable(self.provider)
                or type(self.resource_id) is not str or not self.resource_id):
            raise ContractError("invalid public resource bundle")
        _digest(self.resource_digest)
        _digest(self.provider_digest)


def make_envelope(bundle: PublicBundle, nonce: bytes, *, config: Config = Config(32, 1024, 1024, 65536),
                  prompt_ids: tuple[int, ...] = (), max_payload_bytes: int = 64,
                  max_wire_bytes: int = 65536) -> dict:
    if type(nonce) is not bytes or len(nonce) != NONCE_BYTES:
        raise ContractError("nonce must contain exactly 12 bytes")
    envelope = {
        "schema": VERSION + "-public-envelope", "resource_id": bundle.resource_id,
        "resource_sha256": bundle.resource_digest, "provider_sha256": bundle.provider_digest,
        "codec_sha256": bundle.gate.codec.resource_digest, "policy_sha256": bundle.gate.policy_digest,
        "max_packet_tokens": bundle.gate.max_packet_tokens,
        "nonce_hex": nonce.hex(), "prompt_ids": list(prompt_ids),
        "interval_config": asdict(config), "max_payload_bytes": max_payload_bytes,
        "max_wire_bytes": max_wire_bytes, "length_bytes": HEADER_BYTES,
        "tag_bytes": TAG_BYTES, "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
        "frame_layout": "AEAD(encrypted-u16-length || payload || zero-pad-to-public-maximum)",
        "order": "HKDF-separated-HMAC-SHA256-sort-v1",
        "stop": "first packet resolving the complete frame; no flush",
    }
    validate_envelope(envelope, bundle)
    return envelope


def validate_envelope(envelope: dict, bundle: PublicBundle) -> Config:
    keys = {"schema", "resource_id", "resource_sha256", "provider_sha256", "codec_sha256",
            "policy_sha256", "max_packet_tokens", "nonce_hex", "prompt_ids", "interval_config",
            "max_payload_bytes", "max_wire_bytes", "length_bytes", "tag_bytes", "tail", "order", "stop"}
    keys.add("frame_layout")
    if type(envelope) is not dict or set(envelope) != keys:
        raise ContractError("public envelope fields differ from the protocol")
    expected = {
        "schema": VERSION + "-public-envelope", "resource_id": bundle.resource_id,
        "resource_sha256": bundle.resource_digest, "provider_sha256": bundle.provider_digest,
        "codec_sha256": bundle.gate.codec.resource_digest, "policy_sha256": bundle.gate.policy_digest,
        "max_packet_tokens": bundle.gate.max_packet_tokens, "length_bytes": HEADER_BYTES,
        "tag_bytes": TAG_BYTES, "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
        "frame_layout": "AEAD(encrypted-u16-length || payload || zero-pad-to-public-maximum)",
        "order": "HKDF-separated-HMAC-SHA256-sort-v1",
        "stop": "first packet resolving the complete frame; no flush",
    }
    if any(envelope[name] != value or type(envelope[name]) is not type(value)
           for name, value in expected.items()):
        raise ContractError("public resource/algorithm binding mismatch")
    if (type(envelope["nonce_hex"]) is not str
            or re.fullmatch(r"[0-9a-f]{24}", envelope["nonce_hex"]) is None):
        raise ContractError("invalid public nonce")
    if (type(envelope["prompt_ids"]) is not list
            or any(type(v) is not int or v < 0 for v in envelope["prompt_ids"])):
        raise ContractError("prompt IDs must be nonnegative integers")
    for name, minimum, maximum in (("max_payload_bytes", 0, 65535),
                                   ("max_wire_bytes", 1, 1048576)):
        if type(envelope[name]) is not int or not minimum <= envelope[name] <= maximum:
            raise ContractError("invalid " + name)
    if type(envelope["interval_config"]) is not dict or set(envelope["interval_config"]) != {
            "state_bits", "max_symbols", "max_pending_bits", "max_stream_bits"}:
        raise ContractError("invalid interval configuration")
    config = Config(**envelope["interval_config"])
    # This composition has finite operational bounds, independent of a caller's machine.
    if (config.max_symbols > 1024 or config.max_pending_bits > 65536
            or config.max_stream_bits > 1048576 or bundle.gate.max_packet_tokens > 16):
        raise ContractError("composition resource bound exceeded")
    canonical_json(envelope)
    return config


def derive_keys(key: bytes, envelope: dict) -> tuple[bytes, bytes, bytes]:
    if type(key) is not bytes or len(key) != KEY_BYTES:
        raise ContractError("shared key must contain exactly 32 bytes")
    context = hashlib.sha256(canonical_json(envelope)).digest()
    material = HKDF(algorithm=hashes.SHA256(), length=96,
                    salt=bytes.fromhex(envelope["nonce_hex"]),
                    info=DOMAIN + b"subkeys\x00" + context).derive(key)
    return material[:32], material[32:64], material[64:]


def _aad(envelope: dict) -> bytes:
    descriptor = canonical_json(envelope)
    return DOMAIN + b"aead\x00" + len(descriptor).to_bytes(4, "big") + descriptor


def seal_frame(payload: bytes, key: bytes, envelope: dict) -> bytes:
    """Standard library AEAD; caller enforces per-key nonce uniqueness."""
    if type(payload) is not bytes or len(payload) > envelope["max_payload_bytes"]:
        raise ContractError("payload exceeds its public maximum or is not bytes")
    encryption_key, _, _ = derive_keys(key, envelope)
    plaintext = (len(payload).to_bytes(HEADER_BYTES, "big") + payload
                 + bytes(envelope["max_payload_bytes"] - len(payload)))
    ciphertext = ChaCha20Poly1305(encryption_key).encrypt(bytes.fromhex(envelope["nonce_hex"]),
                                                       plaintext, _aad(envelope))
    return ciphertext


def open_frame(frame: bytes, key: bytes, envelope: dict) -> bytes:
    if type(frame) is not bytes or len(frame) != HEADER_BYTES + envelope["max_payload_bytes"] + TAG_BYTES:
        raise ContractError("ciphertext does not fill the public frame slot")
    encryption_key, _, _ = derive_keys(key, envelope)
    plaintext = ChaCha20Poly1305(encryption_key).decrypt(bytes.fromhex(envelope["nonce_hex"]),
                                                       frame, _aad(envelope))
    length = int.from_bytes(plaintext[:HEADER_BYTES], "big")
    if length > envelope["max_payload_bytes"]:
        raise ContractError("decrypted length exceeds the public maximum")
    if any(plaintext[HEADER_BYTES + length:]):
        raise ContractError("decrypted fixed-slot padding is nonzero")
    return plaintext[HEADER_BYTES:HEADER_BYTES + length]


def _input_bits(frame: bytes, tail_key: bytes, envelope: dict, config: Config) -> tuple[int, ...]:
    bits = bits_from_bytes(frame)
    remaining = max(0, config.max_stream_bits - len(bits))
    context = hashlib.sha256(canonical_json(envelope)).digest()
    tail = bytearray()
    counter = 0
    while len(tail) * 8 < remaining:
        tail.extend(hmac.digest(tail_key, DOMAIN + b"tail\x00" + context + counter.to_bytes(8, "big"), "sha256"))
        counter += 1
    return bits + bits_from_bytes(bytes(tail))[:remaining]


def _law(decision, ordering_key: bytes, envelope: dict, prefix: Prefix):
    # Eligibility and packet weights are public. Only their arithmetic-bin order is keyed.
    indexed = tuple(enumerate(decision.weights))
    context = hashlib.sha256(canonical_json(envelope)).digest()
    prefix_bytes = canonical_json(list(prefix.ids))
    def score(item):
        index, (packet, _) = item
        descriptor = canonical_json(list(packet))
        message = (DOMAIN + b"order\x00" + context + len(prefix_bytes).to_bytes(4, "big")
                   + prefix_bytes + len(descriptor).to_bytes(4, "big") + descriptor)
        return hmac.digest(ordering_key, message, "sha256"), packet
    ordered = sorted(indexed, key=score)
    support = Law(tuple((index, weight) for index, (_, weight) in ordered))
    packets = tuple(packet for packet, _ in decision.weights)
    return support, packets


@dataclass(frozen=True)
class EncodeResult:
    status: str
    reason: str | None
    wire: bytes
    required_frame_bits: int
    resolved_bits: int
    consumed_bits: int
    pending_bits: int
    packet_count: int
    frame_sha256: str
    traces: tuple[dict, ...]


@dataclass(frozen=True)
class DecodeResult:
    status: str
    reason: str | None
    payload: bytes | None
    required_frame_bits: int | None
    resolved_bits: int
    packet_count: int
    traces: tuple[dict, ...]


def _trace(prefix, decision, support, packet, step):
    return {"prefix_ids": list(prefix.ids), "prefix_bytes_sha256": _sha(prefix.raw),
            "admission": decision.summary(), "ordered_integer_law": [list(row) for row in support.rows],
            "observed_packet": list(packet), "symbol_id": step.symbol,
            "emitted_bits": list(step.emitted_bits), "interval_before": asdict(step.before),
            "interval_after": asdict(step.after), "selected_integer_width": step.selected_integer_width,
            "parent_integer_width": step.parent_integer_width,
            "selected_weight": step.selected_weight, "weight_total": step.weight_total}


def encode(payload: bytes, key: bytes, envelope: dict, bundle: PublicBundle) -> EncodeResult:
    config = validate_envelope(envelope, bundle)
    frame = seal_frame(payload, key, envelope)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    sender = Sender(_input_bits(frame, tail_key, envelope, config), config)
    prefix, traces = Prefix(), []
    required = len(frame) * 8
    def finish(status, reason=None):
        return EncodeResult(status, reason, prefix.raw, required, len(sender.resolved_bits),
                            sender.consumed_stream_bits, sender.state.pending, sender.state.symbols,
                            _sha(frame), tuple(traces))
    while len(sender.resolved_bits) < required:
        if sender.state.symbols >= config.max_symbols:
            return finish("insufficient_capacity", "max_symbols")
        try:
            decision = bundle.gate.admit(prefix, bundle.provider(tuple(envelope["prompt_ids"]) + prefix.ids))
        except ConstructionRefusal as exc:
            return finish("construction_refusal", str(exc))
        if decision.status != "ok":
            traces.append({"prefix_ids": list(prefix.ids), "admission": decision.summary()})
            return finish("insufficient_capacity", decision.status)
        support, packets = _law(decision, ordering_key, envelope, prefix)
        before_stats = (len(sender.resolved_bits), sender.consumed_stream_bits,
                        sender.state.pending, sender.state.symbols)
        try:
            step = sender.take(support)
        except CapacityLimit as exc:
            return finish("insufficient_capacity", exc.reason)
        packet = packets[step.symbol]
        advanced = bundle.gate.take(prefix, decision, packet)
        if len(advanced.raw) > envelope["max_wire_bytes"]:
            # The tentative symbol is not transmitted and cannot count as delivered bits.
            traces.append({"refused_packet": list(packet), "reason": "max_wire_bytes",
                           "transmitted": False})
            return EncodeResult("wire_limit_exceeded", "max_wire_bytes", prefix.raw,
                                required, *before_stats, _sha(frame), tuple(traces))
        traces.append(_trace(prefix, decision, support, packet, step))
        prefix = advanced
    return finish("frame_resolved_pending_receiver")


def decode(wire: bytes, key: bytes, envelope: dict, bundle: PublicBundle) -> DecodeResult:
    config = validate_envelope(envelope, bundle)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    if type(wire) is not bytes:
        raise ContractError("wire must be exact bytes")
    receiver = Receiver(config)
    prefix, traces, replay = Prefix(), [], []
    required = 8 * (HEADER_BYTES + envelope["max_payload_bytes"] + TAG_BYTES)
    def finish(status, reason=None, payload=None):
        return DecodeResult(status, reason, payload, required, len(receiver.recovered_bits),
                            receiver.state.symbols, tuple(traces))
    if len(wire) > envelope["max_wire_bytes"]:
        return finish("wire_limit_exceeded", "max_wire_bytes")
    try:
        ids = bundle.gate.codec.encode(wire)
    except (UnicodeDecodeError, ContractError):
        return finish("invalid_wire", "encode_failed")
    if bundle.gate.codec.render(ids) != wire:
        return finish("invalid_wire", "noncanonical_bytes")
    if len(ids) > config.max_symbols * bundle.gate.max_packet_tokens:
        return finish("wire_limit_exceeded", "output_token_bound")
    while len(prefix.ids) < len(ids):
        try:
            decision = bundle.gate.admit(prefix, bundle.provider(tuple(envelope["prompt_ids"]) + prefix.ids))
        except ConstructionRefusal as exc:
            return finish("construction_refusal", str(exc))
        if decision.status != "ok":
            return finish("invalid_wire", decision.status)
        remaining = ids[len(prefix.ids):]
        matches = [row for row in decision.verdicts if row.weight > 0
                   and remaining[:len(row.packet)] == row.packet]
        if len(matches) != 1:
            partial = any(row.weight > 0 and len(remaining) < len(row.packet)
                          and row.packet[:len(remaining)] == remaining for row in decision.verdicts)
            return finish("invalid_wire", "truncated_packet" if partial else "ineligible_packet")
        row = matches[0]
        if row.reason != "accepted":
            return finish("invalid_wire", "rejected_packet")
        support, packets = _law(decision, ordering_key, envelope, prefix)
        symbol = packets.index(row.packet)
        try:
            step = receiver.take(support, symbol)
        except CapacityLimit as exc:
            return finish("insufficient_capacity", exc.reason)
        traces.append(_trace(prefix, decision, support, row.packet, step))
        replay.append((support, symbol))
        prefix = bundle.gate.take(prefix, decision, row.packet)
        if not wire.startswith(prefix.raw):
            return finish("invalid_wire", "prefix_bytes_mismatch")
        recovered = receiver.recovered_bits
        if len(recovered) >= required:
            if len(prefix.ids) != len(ids) or prefix.raw != wire:
                return finish("invalid_wire", "trailing_after_frame")
            frame = exact_bytes(recovered[:required])
            try:
                payload = open_frame(frame, key, envelope)
            except InvalidTag:
                return finish("authentication_failed", "invalid_tag")
            except ContractError as exc:
                return finish("invalid_frame", str(exc))
            # The receiver uses only its own observed-prefix laws, never a sender trace.
            deterministic = Sender(_input_bits(frame, tail_key, envelope, config), config)
            for index, (observed_law, observed_symbol) in enumerate(replay):
                try:
                    expected = deterministic.take(observed_law)
                except CapacityLimit:
                    return finish("invalid_wire", "noncanonical_embedding_budget")
                if expected.symbol != observed_symbol:
                    return finish("invalid_wire", "noncanonical_embedding")
                if len(deterministic.resolved_bits) >= required and index != len(replay) - 1:
                    return finish("invalid_wire", "late_stop")
            if len(deterministic.resolved_bits) < required:
                return finish("insufficient_capacity", "deterministic_replay_incomplete")
            return finish("authenticated_delivery", payload=payload)
    return finish("insufficient_capacity", "wire_ended_before_complete_frame")
