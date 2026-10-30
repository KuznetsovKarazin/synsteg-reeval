"""Pure bounded-stop composition; no native/model/launcher side effects.

Both arms stop at the first committed symbol resolving 272 bits, or at 768
symbols. The fresh receiver verifies that boundary and the entire supplied
wire, including surplus bits in its last symbol. Existing v1 code is immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import framed_interval_llm_v1 as fixed
from .canonical_channel_v2 import ContractError, Prefix
from .integer_interval_channel_v1 import (
    CapacityLimit, Config, ContractError as IntervalContractError, Law, Receiver,
    Sender, advance, bits_from_bytes, exact_bytes,
)

VERSION = "framed-interval-bounded-v1"
DOMAIN = VERSION.encode("ascii") + b"\x00"
THRESHOLD_BITS, TOKEN_CAP, PAYLOAD_SLOT, FRAME_BYTES = 272, 768, 16, 34
DEFAULT_CONFIG = Config(32, TOKEN_CAP, 4096, 65536)
DEFAULT_ORDINARY_SEED = hashlib.sha256(DOMAIN + b"ordinary-control").digest()
Endpoint, Distribution, BackendFailure = fixed.Endpoint, fixed.Distribution, fixed.BackendFailure
canonical_json = fixed.canonical_json


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _static(endpoint):
    return {"schema": VERSION + "-envelope", "experiment_id": VERSION,
        "resource_id": endpoint.resource_id, "resource_sha256": endpoint.resource_digest,
        "provider_sha256": endpoint.provider_digest, "codec_sha256": endpoint.gate.codec.resource_digest,
        "context_tokens": endpoint.context_tokens, "threshold_bits": THRESHOLD_BITS,
        "token_cap": TOKEN_CAP, "max_payload_bytes": PAYLOAD_SLOT, "length_bytes": 2,
        "tag_bytes": 16, "frame_bits": THRESHOLD_BITS,
        "frame_layout": "AEAD(encrypted-u16-byte-length||payload||zero-pad-to-16)-tag16",
        "canonical_gate": "canonical-channel-v2-singleton",
        "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
        "order": "HKDF-matched-arm-HMAC-SHA256-token-order-v1",
        "ordinary": "independent-public-seed-HMAC-bits-through-same-integer-transform-v1",
        "stop": "first-committed-symbol-resolving272bits-or-cap768;no-flush;verify-entire-wire"}


def make_envelope(endpoint, nonce, *, arm, prompt_id, protocol_digest, prompt_ids=(),
                  config=DEFAULT_CONFIG, max_wire_bytes=65536,
                  ordinary_seed=DEFAULT_ORDINARY_SEED):
    if type(nonce) is not bytes or len(nonce) != 12:
        raise ContractError("nonce must be exactly12 bytes")
    if type(ordinary_seed) is not bytes or len(ordinary_seed) != 32:
        raise ContractError("ordinary public seed must be32 bytes")
    envelope = dict(_static(endpoint), arm=arm, prompt_id=prompt_id,
        prompt_ids=list(prompt_ids), protocol_sha256=protocol_digest, nonce_hex=nonce.hex(),
        ordinary_seed_hex=ordinary_seed.hex(), interval_config=asdict(config),
        max_wire_bytes=max_wire_bytes)
    validate_envelope(envelope, endpoint)
    return envelope


def validate_envelope(envelope, endpoint):
    expected = _static(endpoint)
    variable = {"arm", "prompt_id", "prompt_ids", "protocol_sha256", "nonce_hex",
                "ordinary_seed_hex", "interval_config", "max_wire_bytes"}
    if type(envelope) is not dict or set(envelope) != set(expected) | variable:
        raise ContractError("bounded envelope fields differ")
    if any(type(envelope[k]) is not type(v) or envelope[k] != v for k, v in expected.items()):
        raise ContractError("public resources/algorithm/stop binding mismatch")
    if envelope["arm"] not in ("stego", "ordinary") or type(envelope["prompt_id"]) is not str or not envelope["prompt_id"]:
        raise ContractError("invalid arm/prompt")
    # Reused endpoint's own begin guard is fixed512; enforce the stronger cap
    # here, before it can call a supplied backend's start callback.
    prompt = fixed._ids(envelope["prompt_ids"])
    if len(prompt) + TOKEN_CAP > endpoint.context_tokens:
        raise ContractError("prompt plus bounded output exceeds context")
    fixed._digest(envelope["protocol_sha256"]); fixed._digest(envelope["ordinary_seed_hex"])
    if type(envelope["nonce_hex"]) is not str or re.fullmatch("[0-9a-f]{24}", envelope["nonce_hex"]) is None:
        raise ContractError("invalid nonce")
    if type(envelope["max_wire_bytes"]) is not int or not 1 <= envelope["max_wire_bytes"] <= 65536:
        raise ContractError("invalid byte limit")
    conf = envelope["interval_config"]
    if type(conf) is not dict or set(conf) != {"state_bits", "max_symbols", "max_pending_bits", "max_stream_bits"}:
        raise ContractError("invalid interval config")
    config = Config(**conf)
    if (config.state_bits != 32 or config.max_symbols != TOKEN_CAP
            or config.max_pending_bits > 4096 or config.max_stream_bits > 65536):
        raise ContractError("bounded precision/resource limits differ")
    canonical_json(envelope)
    return config


def ordering_context_digest(envelope):
    return hashlib.sha256(canonical_json({k: v for k, v in envelope.items()
        if k not in {"arm", "nonce_hex", "ordinary_seed_hex"}})).digest()


def derive_keys(key, envelope):
    if type(key) is not bytes or len(key) != 32:
        raise ContractError("shared key must be32 bytes")
    material = HKDF(algorithm=hashes.SHA256(), length=64,
        salt=bytes.fromhex(envelope["nonce_hex"]),
        info=DOMAIN + b"transport-subkeys\x00" + hashlib.sha256(canonical_json(envelope)).digest()).derive(key)
    ordering = HKDF(algorithm=hashes.SHA256(), length=32, salt=ordering_context_digest(envelope),
        info=DOMAIN + b"matched-order-subkey\x00").derive(key)
    return material[:32], ordering, material[32:]


def _aad(envelope):
    raw = canonical_json(envelope)
    return DOMAIN + b"aead\x00" + len(raw).to_bytes(4, "big") + raw


def seal_frame(payload, key, envelope):
    if type(payload) is not bytes or len(payload) > PAYLOAD_SLOT:
        raise ContractError("payload exceeds fixed16-byte slot")
    encryption, _, _ = derive_keys(key, envelope)
    plain = len(payload).to_bytes(2, "big") + payload + bytes(PAYLOAD_SLOT-len(payload))
    return ChaCha20Poly1305(encryption).encrypt(bytes.fromhex(envelope["nonce_hex"]), plain, _aad(envelope))


def open_frame(frame, key, envelope):
    if type(frame) is not bytes or len(frame) != FRAME_BYTES:
        raise ContractError("frame must be34 bytes")
    encryption, _, _ = derive_keys(key, envelope)
    plain = ChaCha20Poly1305(encryption).decrypt(bytes.fromhex(envelope["nonce_hex"]), frame, _aad(envelope))
    count = int.from_bytes(plain[:2], "big")
    if count > PAYLOAD_SLOT or any(plain[2+count:]):
        raise ContractError("decrypted length/padding invalid")
    return plain[2:2+count]


def _prf_bits(seed, envelope, label, count):
    context = hashlib.sha256(canonical_json(envelope)).digest()
    raw = b"".join(hmac.digest(seed, DOMAIN + label + b"\x00" + context + n.to_bytes(8, "big"), "sha256")
        for n in range((count+255)//256))
    return bits_from_bytes(raw)[:count]


def input_bits(frame, tail_key, envelope, config):
    bits = bits_from_bytes(frame)
    return bits + _prf_bits(tail_key, envelope, b"stego-tail", max(0, config.max_stream_bits-len(bits)))


def ordinary_bits(envelope, config):
    return _prf_bits(bytes.fromhex(envelope["ordinary_seed_hex"]), envelope,
        b"ordinary-input", config.max_stream_bits)


def ordered_law(decision, ordering_key, envelope, prefix):
    context, prefix_raw = ordering_context_digest(envelope), canonical_json(list(prefix.ids))
    def score(row):
        token_raw = canonical_json(row[0])
        message = (DOMAIN + b"order\x00" + context + len(prefix_raw).to_bytes(4, "big") + prefix_raw
            + len(token_raw).to_bytes(4, "big") + token_raw)
        return hmac.digest(ordering_key, message, "sha256"), row[0]
    return Law(tuple(sorted(decision.weights, key=score)))


@dataclass(frozen=True)
class GenerateResult:
    status: str
    reason: str | None
    arm: str
    wire: bytes
    emitted_tokens: int
    resolved_bits: int
    consumed_bits: int
    pending_bits: int
    first_threshold_token: int | None
    frame_sha256: str | None
    authenticated_delivery: bool
    traces: tuple[dict, ...]


@dataclass(frozen=True)
class ReceiveResult:
    status: str
    reason: str | None
    payload: bytes | None
    wire_tokens: int | None
    processed_tokens: int
    resolved_bits: int
    pending_bits: int
    first_threshold_token: int | None
    authenticated_delivery: bool
    traces: tuple[dict, ...]


def generate(key, envelope, endpoint, *, payload=None):
    config = validate_envelope(envelope, endpoint)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    arm = envelope["arm"]
    if arm == "stego":
        frame = seal_frame(payload, key, envelope)
        stream = input_bits(frame, tail_key, envelope, config)
    else:
        if payload is not None: raise ContractError("ordinary has no payload")
        frame, stream = None, ordinary_bits(envelope, config)
    sender = Sender(stream, config)
    prefix, traces, first, stats = Prefix(), [], None, (0, sender.consumed_stream_bits, 0)
    def finish(status, reason=None):
        return GenerateResult(status, reason, arm, prefix.raw, len(prefix.ids), *stats,
            first, _sha(frame) if frame is not None else None, False, tuple(traces))
    try:
        endpoint.begin(envelope["prompt_ids"])
        for _ in range(TOKEN_CAP):
            distribution = endpoint.distribution(prefix)
            decision = endpoint.gate.admit(prefix, distribution.weights)
            if decision.status != "ok":
                traces.append({"step": len(prefix.ids), "admission": decision.summary(), "transmitted": False})
                return finish("support_refusal", decision.status)
            law = ordered_law(decision, ordering_key, envelope, prefix)
            try: step = sender.take(law)
            except CapacityLimit as exc: return finish("capacity_limit", exc.reason)
            advanced = endpoint.gate.take(prefix, decision, step.symbol)
            if len(advanced.raw) > envelope["max_wire_bytes"]:
                traces.append({"step": len(prefix.ids), "refused_token": step.symbol,
                    "reason": "max_wire_bytes", "transmitted": False})
                return finish("wire_limit_exceeded", "max_wire_bytes")
            record = fixed._trace(prefix, decision, distribution, law, step, config)
            endpoint.observe(step.symbol)
            traces.append(record); prefix = advanced
            stats = len(sender.resolved_bits), sender.consumed_stream_bits, sender.state.pending
            if stats[0] >= THRESHOLD_BITS:
                first = len(prefix.ids)
                return finish("threshold_pending_receiver" if arm == "stego" else "ordinary_threshold_generated")
    except BackendFailure as exc: return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc: return finish("contract_error", str(exc))
    return finish("insufficient_capacity" if arm == "stego" else "ordinary_cap_generated",
        "cap768_without_threshold")


def receive(wire, key, envelope, endpoint):
    """No sender trace/plaintext/ciphertext input; own laws and full-wire replay."""
    config = validate_envelope(envelope, endpoint)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    receiver = Receiver(config)
    prefix, traces, replay, first, wire_count = Prefix(), [], [], None, None
    def finish(status, reason=None, payload=None):
        return ReceiveResult(status, reason, payload, wire_count, len(prefix.ids),
            len(receiver.recovered_bits), receiver.state.pending, first,
            status == "authenticated_delivery", tuple(traces))
    if type(wire) is not bytes: raise ContractError("wire must be exact bytes")
    if len(wire) > envelope["max_wire_bytes"]: return finish("invalid_wire", "max_wire_bytes")
    try:
        ids = endpoint.gate.codec.encode(wire)
        wire_count = len(ids)
        if endpoint.gate.codec.render(ids) != wire: return finish("invalid_wire", "noncanonical_bytes")
    except UnicodeDecodeError: return finish("invalid_wire", "invalid_utf8")
    except ContractError as exc: return finish("contract_error", str(exc))
    if len(ids) > TOKEN_CAP: return finish("invalid_wire", "too_many_tokens")
    try:
        endpoint.begin(envelope["prompt_ids"])
        for token in ids:
            distribution = endpoint.distribution(prefix)
            decision = endpoint.gate.admit(prefix, distribution.weights)
            if decision.status != "ok" or token not in dict(decision.weights):
                traces.append({"step": len(prefix.ids), "admission": decision.summary(), "observed_token": token})
                return finish("invalid_wire", "ineligible_token")
            law = ordered_law(decision, ordering_key, envelope, prefix)
            try: step = advance(receiver.state, law, token, config)
            except CapacityLimit as exc: return finish("capacity_limit", exc.reason)
            advanced = endpoint.gate.take(prefix, decision, token)
            if not wire.startswith(advanced.raw): return finish("invalid_wire", "prefix_bytes_mismatch")
            record = fixed._trace(prefix, decision, distribution, law, step, config)
            endpoint.observe(token)
            if receiver.take(law, token) != step: raise AssertionError("pure transition changed")
            traces.append(record); replay.append((law, token)); prefix = advanced
            if len(receiver.recovered_bits) >= THRESHOLD_BITS:
                first = len(prefix.ids)
                if first != len(ids): return finish("invalid_wire", "trailing_tokens_after_first_threshold")
    except BackendFailure as exc: return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc: return finish("contract_error", str(exc))
    if prefix.raw != wire: return finish("invalid_wire", "final_bytes_mismatch")
    if first is None and len(ids) != TOKEN_CAP: return finish("incomplete_wire", "before_public_stop")
    if envelope["arm"] == "ordinary":
        stream, payload = ordinary_bits(envelope, config), None
    else:
        if first is None: return finish("insufficient_capacity", "cap768_without_threshold")
        frame = exact_bytes(receiver.recovered_bits[:THRESHOLD_BITS])
        try: payload = open_frame(frame, key, envelope)
        except InvalidTag: return finish("authentication_failed", "invalid_tag")
        except ContractError as exc: return finish("invalid_frame", str(exc))
        stream = input_bits(frame, tail_key, envelope, config)
    # AEAD alone does not authenticate surplus bits in the final symbol. Do
    # not release payload until all observed decisions match this full stream.
    deterministic = Sender(stream, config)
    for law, token in replay:
        try: expected = deterministic.take(law)
        except CapacityLimit as exc: return finish("invalid_wire", "replay_" + exc.reason)
        if expected.symbol != token: return finish("invalid_wire", "noncanonical_fullwire_embedding")
    if envelope["arm"] == "ordinary":
        return finish("ordinary_threshold_replay_verified" if first else "ordinary_cap_replay_verified")
    return finish("authenticated_delivery", payload=payload)
