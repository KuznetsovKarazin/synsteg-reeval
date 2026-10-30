"""Fixed-512 singleton canonical channel with interval coding and library AEAD.

No model is loaded here. A caller supplies an already prepared endpoint. The
module has no launcher, network, secret lookup, compilation, or automatic run.
Unlike the frozen stop-first composition, delivery requires the complete wire
and deterministic replay of all 512 decisions, including the post-frame tail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import hmac
import json
import re
from typing import Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .canonical_channel_v2 import CanonicalGate, ContractError, Prefix
from .integer_interval_channel_v1 import (
    CapacityLimit, Config, ContractError as IntervalContractError, Law, Receiver, Sender,
    advance, bits_from_bytes, exact_bytes, integer_bins,
)

VERSION = "framed-interval-llm-v1"
DOMAIN = VERSION.encode("ascii") + b"\x00"
OUTPUT_TOKENS, PAYLOAD_SLOT, FRAME_BYTES = 512, 16, 34
FRAME_BITS = FRAME_BYTES * 8
DEFAULT_CONFIG = Config(32, OUTPUT_TOKENS, 4096, 65536)
DEFAULT_ORDINARY_SEED = hashlib.sha256((VERSION + ":ordinary-control").encode()).digest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ContractError("public SHA256 identity required")


def _ids(value):
    if type(value) not in (list, tuple) or any(type(t) is not int or t < 0 for t in value):
        raise ContractError("token IDs must be nonnegative integers")
    return tuple(value)


class BackendFailure(RuntimeError):
    """Endpoint failure; never interpreted as canonical rejection or capacity."""


@dataclass(frozen=True)
class Distribution:
    weights: tuple[tuple[int, int], ...]
    evidence: dict = field(default_factory=dict)


class Endpoint:
    """One-use sequential backend; codec callbacks must not mutate model state.

    distribution_callback receives exact prompt+observed output IDs. start and
    observe are called once per prompt/token, including the last output token.
    Independent receiver endpoints reconstruct their own distributions.
    """
    def __init__(self, gate: CanonicalGate, *, resource_id: str, resource_digest: str,
                 provider_digest: str, distribution_callback: Callable,
                 start_callback: Callable = lambda ids: None,
                 observe_callback: Callable = lambda token: None, context_tokens: int = 2048):
        if (not isinstance(gate, CanonicalGate) or type(resource_id) is not str or not resource_id
                or type(context_tokens) is not int or not OUTPUT_TOKENS < context_tokens <= 16384
                or not all(callable(fn) for fn in (distribution_callback, start_callback, observe_callback))):
            raise ContractError("invalid endpoint contract")
        _digest(resource_digest); _digest(provider_digest)
        self.gate, self.resource_id = gate, resource_id
        self.resource_digest, self.provider_digest = resource_digest, provider_digest
        self.context_tokens = context_tokens
        self._distribution, self._start, self._observe = distribution_callback, start_callback, observe_callback
        self._started, self._prompt, self._observed = False, (), ()

    def begin(self, prompt_ids):
        prompt = _ids(prompt_ids)
        if self._started:
            raise ContractError("endpoint is one-use; create a fresh receiver/backend")
        if len(prompt) + OUTPUT_TOKENS > self.context_tokens:
            raise ContractError("prompt plus fixed output exceeds context")
        self._started = True
        self._prompt = prompt
        try:
            self._start(prompt)
        except Exception as exc:
            raise BackendFailure("start:" + type(exc).__name__) from exc

    def distribution(self, prefix: Prefix):
        if not self._started or prefix.ids != self._observed:
            raise ContractError("provider history differs from committed output")
        try:
            row = self._distribution(self._prompt + self._observed)
        except Exception as exc:
            raise BackendFailure("distribution:" + type(exc).__name__) from exc
        if not isinstance(row, Distribution) or type(row.weights) is not tuple or type(row.evidence) is not dict:
            raise ContractError("backend must return Distribution with tuple weights and dict evidence")
        try:
            evidence = json.loads(canonical_json(row.evidence))
            weights = tuple(tuple(pair) for pair in row.weights)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError("backend law/evidence must have a finite JSON snapshot") from exc
        return Distribution(weights, evidence)

    def observe(self, token):
        if not self._started or type(token) is not int or token < 0:
            raise ContractError("invalid output observation")
        try:
            self._observe(token)
        except Exception as exc:
            raise BackendFailure("observe:" + type(exc).__name__) from exc
        self._observed += (token,)


def make_envelope(endpoint: Endpoint, nonce: bytes, *, arm: str, prompt_id: str,
                  protocol_digest: str, prompt_ids: tuple[int, ...] = (),
                  config: Config = DEFAULT_CONFIG, max_wire_bytes: int = 65536,
                  ordinary_seed: bytes = DEFAULT_ORDINARY_SEED):
    if type(nonce) is not bytes or len(nonce) != 12:
        raise ContractError("nonce must be exactly 12 bytes")
    if type(ordinary_seed) is not bytes or len(ordinary_seed) != 32:
        raise ContractError("ordinary public control seed must be 32 bytes")
    envelope = {"schema": VERSION + "-fixed512-envelope", "experiment_id": VERSION,
        "arm": arm, "prompt_id": prompt_id, "prompt_ids": list(prompt_ids),
        "protocol_sha256": protocol_digest, "resource_id": endpoint.resource_id,
        "resource_sha256": endpoint.resource_digest, "provider_sha256": endpoint.provider_digest,
        "codec_sha256": endpoint.gate.codec.resource_digest, "context_tokens": endpoint.context_tokens,
        "nonce_hex": nonce.hex(), "ordinary_seed_hex": ordinary_seed.hex(),
        "interval_config": asdict(config), "output_tokens": OUTPUT_TOKENS,
        "max_payload_bytes": PAYLOAD_SLOT, "length_bytes": 2, "tag_bytes": 16,
        "frame_layout": "AEAD(encrypted-u16-byte-length||payload||zero-pad-to-16)-tag16",
        "canonical_gate": "canonical-channel-v2-singleton",
        "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
        "order": "HKDF-matched-arm-HMAC-SHA256-token-order-v1",
        "ordinary": "independent-public-seed-HMAC-bits-through-same-integer-transform-v1",
        "stop": "exactly512tokens-or-declared-failure;no-flush;verify-entire-wire",
        "max_wire_bytes": max_wire_bytes}
    validate_envelope(envelope, endpoint)
    return envelope


def validate_envelope(envelope, endpoint):
    expected = {"schema": VERSION + "-fixed512-envelope", "experiment_id": VERSION,
        "resource_id": endpoint.resource_id, "resource_sha256": endpoint.resource_digest,
        "provider_sha256": endpoint.provider_digest, "codec_sha256": endpoint.gate.codec.resource_digest,
        "context_tokens": endpoint.context_tokens, "output_tokens": OUTPUT_TOKENS,
        "max_payload_bytes": PAYLOAD_SLOT, "length_bytes": 2, "tag_bytes": 16,
        "frame_layout": "AEAD(encrypted-u16-byte-length||payload||zero-pad-to-16)-tag16",
        "canonical_gate": "canonical-channel-v2-singleton",
        "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
        "order": "HKDF-matched-arm-HMAC-SHA256-token-order-v1",
        "ordinary": "independent-public-seed-HMAC-bits-through-same-integer-transform-v1",
        "stop": "exactly512tokens-or-declared-failure;no-flush;verify-entire-wire"}
    variable = {"arm", "prompt_id", "prompt_ids", "protocol_sha256", "nonce_hex",
                "ordinary_seed_hex", "interval_config", "max_wire_bytes"}
    if type(envelope) is not dict or set(envelope) != set(expected) | variable:
        raise ContractError("envelope fields differ from fixed512 protocol")
    if any(envelope[k] != v or type(envelope[k]) is not type(v) for k, v in expected.items()):
        raise ContractError("public resources/algorithm/stop binding mismatch")
    if envelope["arm"] not in ("stego", "ordinary") or type(envelope["prompt_id"]) is not str or not envelope["prompt_id"]:
        raise ContractError("invalid arm or prompt ID")
    prompt = _ids(envelope["prompt_ids"])
    if len(prompt) + OUTPUT_TOKENS > endpoint.context_tokens:
        raise ContractError("prompt plus fixed output exceeds context")
    _digest(envelope["protocol_sha256"])
    _digest(envelope["ordinary_seed_hex"])
    if type(envelope["nonce_hex"]) is not str or re.fullmatch("[0-9a-f]{24}", envelope["nonce_hex"]) is None:
        raise ContractError("invalid nonce")
    if type(envelope["max_wire_bytes"]) is not int or not 1 <= envelope["max_wire_bytes"] <= 65536:
        raise ContractError("invalid wire budget")
    conf = envelope["interval_config"]
    if type(conf) is not dict or set(conf) != {"state_bits", "max_symbols", "max_pending_bits", "max_stream_bits"}:
        raise ContractError("invalid interval configuration")
    config = Config(**conf)
    if (config.state_bits != 32 or config.max_symbols != OUTPUT_TOKENS
            or config.max_pending_bits > 4096 or config.max_stream_bits > 65536):
        raise ContractError("fixed512 precision/resource limits differ")
    canonical_json(envelope)
    return config


def ordering_context_digest(envelope):
    """Same conditional ordering law for both arms, independent of their RNG.

    AEAD still binds every envelope field. Only token ordering excludes arm,
    nonce and ordinary seed, so equal prefix+interval states get identical bins.
    """
    common = {k: v for k, v in envelope.items() if k not in {"arm", "nonce_hex", "ordinary_seed_hex"}}
    return hashlib.sha256(canonical_json(common)).digest()


def derive_keys(key, envelope):
    if type(key) is not bytes or len(key) != 32:
        raise ContractError("shared key must be exactly 32 bytes")
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
        raise ContractError("payload exceeds the fixed16-byte slot")
    encryption, _, _ = derive_keys(key, envelope)
    plain = len(payload).to_bytes(2, "big") + payload + bytes(PAYLOAD_SLOT-len(payload))
    return ChaCha20Poly1305(encryption).encrypt(bytes.fromhex(envelope["nonce_hex"]), plain, _aad(envelope))


def open_frame(frame, key, envelope):
    if type(frame) is not bytes or len(frame) != FRAME_BYTES:
        raise ContractError("ciphertext must fill the complete34-byte frame")
    encryption, _, _ = derive_keys(key, envelope)
    plain = ChaCha20Poly1305(encryption).decrypt(bytes.fromhex(envelope["nonce_hex"]), frame, _aad(envelope))
    count = int.from_bytes(plain[:2], "big")
    if count > PAYLOAD_SLOT or any(plain[2+count:]):
        raise ContractError("decrypted length/padding invalid")
    return plain[2:2+count]


def _prf_bits(seed, envelope, label, count):
    context = hashlib.sha256(canonical_json(envelope)).digest()
    chunks = []
    for counter in range((count+255)//256):
        chunks.append(hmac.digest(seed, DOMAIN+label+b"\x00"+context+counter.to_bytes(8, "big"), "sha256"))
    return bits_from_bytes(b"".join(chunks))[:count]


def input_bits(frame, tail_key, envelope, config):
    """Sufficient explicit bits for the entire permitted read budget; no zeros."""
    bits = bits_from_bytes(frame)
    return bits + _prf_bits(tail_key, envelope, b"stego-tail", max(0, config.max_stream_bits-len(bits)))


def ordinary_bits(envelope, config):
    return _prf_bits(bytes.fromhex(envelope["ordinary_seed_hex"]), envelope,
                     b"ordinary-input", config.max_stream_bits)


def ordered_law(decision, ordering_key, envelope, prefix):
    context = ordering_context_digest(envelope)
    prefix_raw = canonical_json(list(prefix.ids))
    def score(row):
        token, _ = row
        token_raw = canonical_json(token)
        message = (DOMAIN+b"order\x00"+context+len(prefix_raw).to_bytes(4, "big")+prefix_raw
                   +len(token_raw).to_bytes(4, "big")+token_raw)
        return hmac.digest(ordering_key, message, "sha256"), token
    return Law(tuple(sorted(decision.weights, key=score)))


def _trace(prefix, decision, distribution, law, step, config):
    bins = integer_bins(step.before, law, config)
    widths = {token: hi-lo+1 for token, lo, hi in bins}
    R, S = step.parent_integer_width, law.total
    tv = Fraction(sum(abs(widths[t]*S-w*R) for t, w in law.rows), 2*R*S)
    return {"step": step.before.symbols, "prefix_ids": list(prefix.ids),
        "prefix_sha256": _sha(prefix.raw), "admission": decision.summary(),
        "base_law_evidence": json.loads(canonical_json(distribution.evidence)),
        "ordered_integer_law": [list(row) for row in law.rows],
        "integer_bins": [{"token_id": t, "low": lo, "high": hi, "width": hi-lo+1} for t, lo, hi in bins],
        "bin_law_vs_weights_TV": {"numerator": tv.numerator, "denominator": tv.denominator},
        "observed_token": step.symbol, "interval_before": asdict(step.before), "interval_after": asdict(step.after),
        "emitted_bits": list(step.emitted_bits), "selected_integer_width": step.selected_integer_width,
        "parent_integer_width": R, "selected_weight": step.selected_weight, "weight_total": S}


@dataclass(frozen=True)
class GenerateResult:
    status: str
    reason: str | None
    arm: str
    wire: bytes
    emitted_tokens: int
    required_frame_bits: int | None
    resolved_bits: int
    consumed_bits: int
    pending_bits: int
    first_frame_resolved_token: int | None
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
    first_frame_resolved_token: int | None
    authenticated_delivery: bool
    traces: tuple[dict, ...]


def generate(key: bytes, envelope: dict, endpoint: Endpoint, *, payload: bytes | None = None):
    config = validate_envelope(envelope, endpoint)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    arm = envelope["arm"]
    if arm == "stego":
        frame = seal_frame(payload, key, envelope)
        stream = input_bits(frame, tail_key, envelope, config)
    else:
        if payload is not None:
            raise ContractError("ordinary has no payload or delivery outcome")
        frame, stream = None, ordinary_bits(envelope, config)
    sender = Sender(stream, config)
    prefix, traces, first = Prefix(), [], None
    stats = (0, sender.consumed_stream_bits, 0)
    def finish(status, reason=None):
        return GenerateResult(status, reason, arm, prefix.raw, len(prefix.ids),
            FRAME_BITS if arm == "stego" else None, *stats, first,
            _sha(frame) if frame is not None else None, False, tuple(traces))
    try:
        endpoint.begin(envelope["prompt_ids"])
        for _ in range(OUTPUT_TOKENS):
            distribution = endpoint.distribution(prefix)
            decision = endpoint.gate.admit(prefix, distribution.weights)
            if decision.status != "ok":
                traces.append({"step": len(prefix.ids), "admission": decision.summary(), "transmitted": False})
                return finish("support_refusal", decision.status)
            law = ordered_law(decision, ordering_key, envelope, prefix)
            try:
                step = sender.take(law)
            except CapacityLimit as exc:
                return finish("capacity_limit", exc.reason)
            advanced = endpoint.gate.take(prefix, decision, step.symbol)
            if len(advanced.raw) > envelope["max_wire_bytes"]:
                traces.append({"step": len(prefix.ids), "refused_token": step.symbol,
                               "reason": "max_wire_bytes", "transmitted": False})
                return finish("wire_limit_exceeded", "max_wire_bytes")
            record = _trace(prefix, decision, distribution, law, step, config)
            # Commit model and wire only after the tentative token passes all byte limits.
            endpoint.observe(step.symbol)
            traces.append(record)
            prefix = advanced
            stats = (len(sender.resolved_bits), sender.consumed_stream_bits, sender.state.pending)
            if arm == "stego" and first is None and stats[0] >= FRAME_BITS:
                first = len(prefix.ids)
    except BackendFailure as exc:
        return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc:
        return finish("contract_error", str(exc))
    if arm == "ordinary":
        return finish("ordinary_generated")
    if stats[0] < FRAME_BITS:
        return finish("insufficient_capacity", "fixed512_without_complete_frame")
    return finish("fixed512_pending_receiver")


def receive(wire: bytes, key: bytes, envelope: dict, endpoint: Endpoint):
    """Fresh endpoint; no sender IDs, trace, candidate table, or stream input."""
    config = validate_envelope(envelope, endpoint)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    receiver = Receiver(config)
    prefix, traces, replay, first, wire_count = Prefix(), [], [], None, None
    def finish(status, reason=None, payload=None):
        return ReceiveResult(status, reason, payload, wire_count, len(prefix.ids),
            len(receiver.recovered_bits), receiver.state.pending, first,
            status == "authenticated_delivery", tuple(traces))
    if type(wire) is not bytes:
        raise ContractError("wire must be exact bytes")
    if len(wire) > envelope["max_wire_bytes"]:
        return finish("invalid_wire", "max_wire_bytes")
    try:
        ids = endpoint.gate.codec.encode(wire)
    except UnicodeDecodeError:
        return finish("invalid_wire", "invalid_utf8")
    except ContractError as exc:
        return finish("contract_error", str(exc))
    wire_count = len(ids)
    try:
        if endpoint.gate.codec.render(ids) != wire:
            return finish("invalid_wire", "noncanonical_bytes")
    except ContractError as exc:
        return finish("contract_error", str(exc))
    if len(ids) > OUTPUT_TOKENS:
        return finish("invalid_wire", "too_many_tokens")
    try:
        endpoint.begin(envelope["prompt_ids"])
        for token in ids:
            distribution = endpoint.distribution(prefix)
            decision = endpoint.gate.admit(prefix, distribution.weights)
            if decision.status != "ok" or token not in dict(decision.weights):
                traces.append({"step": len(prefix.ids), "admission": decision.summary(), "observed_token": token})
                return finish("invalid_wire", "ineligible_token")
            law = ordered_law(decision, ordering_key, envelope, prefix)
            # Public transition is checked without committing receiver counters until
            # byte and backend checks succeed (advance is pure, Receiver.take repeats it).
            try:
                step = advance(receiver.state, law, token, config)
            except CapacityLimit as exc:
                return finish("capacity_limit", exc.reason)
            advanced = endpoint.gate.take(prefix, decision, token)
            if not wire.startswith(advanced.raw):
                return finish("invalid_wire", "prefix_bytes_mismatch")
            record = _trace(prefix, decision, distribution, law, step, config)
            endpoint.observe(token)
            committed = receiver.take(law, token)
            if committed != step:
                raise AssertionError("pure receiver transition changed")
            traces.append(record)
            replay.append((law, token))
            prefix = advanced
            if envelope["arm"] == "stego" and first is None and len(receiver.recovered_bits) >= FRAME_BITS:
                first = len(prefix.ids)
    except BackendFailure as exc:
        return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc:
        return finish("contract_error", str(exc))
    if prefix.raw != wire:
        return finish("invalid_wire", "final_bytes_mismatch")
    if len(ids) != OUTPUT_TOKENS:
        return finish("incomplete_wire", "fixed512_token_count_mismatch")
    # No plaintext or success can leave this function before the entire wire is checked.
    if envelope["arm"] == "ordinary":
        stream, payload = ordinary_bits(envelope, config), None
    else:
        if len(receiver.recovered_bits) < FRAME_BITS:
            return finish("insufficient_capacity", "fixed512_without_complete_frame")
        frame = exact_bytes(receiver.recovered_bits[:FRAME_BITS])
        try:
            payload = open_frame(frame, key, envelope)
        except InvalidTag:
            return finish("authentication_failed", "invalid_tag")
        except ContractError as exc:
            return finish("invalid_frame", str(exc))
        stream = input_bits(frame, tail_key, envelope, config)
    deterministic = Sender(stream, config)
    for law, token in replay:
        try:
            expected = deterministic.take(law)
        except CapacityLimit as exc:
            return finish("invalid_wire", "replay_" + exc.reason)
        if expected.symbol != token:
            return finish("invalid_wire", "noncanonical_fullwire_embedding")
    if envelope["arm"] == "ordinary":
        return finish("ordinary_replay_verified")
    return finish("authenticated_delivery", payload=payload)
