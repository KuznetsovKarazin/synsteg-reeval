"""Message-independent public-spec composition; no model or filesystem effects.

The receiver needs wire, key, public envelope/spec/context, and a trusted local
endpoint implementing the public resource descriptors. No experiment fixture,
activation, freeze, readback, or expected plaintext is a decoder input.

This pure core does NOT enforce nonce uniqueness. A sender must reserve each
(key, nonce) before calling generate; the companion durable ledger/CLI does so.
Resource digests bind declarations; file integrity and actual backend behavior
remain the responsibility of the endpoint adapter. No native or security claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
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

VERSION = "framed-interval-public-ending-v2"
DOMAIN = VERSION.encode("ascii") + b"\x00"
THRESHOLD_BITS, TOKEN_CAP, PAYLOAD_SLOT, FRAME_BYTES = 272, 768, 16, 34
MAX_TAIL_TOKENS = 96
DEFAULT_CONFIG = Config(32, TOKEN_CAP, 4096, 65536)
DEFAULT_ORDINARY_SEED = hashlib.sha256(DOMAIN + b"ordinary-control").digest()
Endpoint, Distribution, BackendFailure = fixed.Endpoint, fixed.Distribution, fixed.BackendFailure
canonical_json = fixed.canonical_json

ALGORITHMS = {
    "interval": "integer-interval-channel-v1",
    "frame": "AEAD(encrypted-u16-byte-length||payload||zero-pad-to-16)-tag16",
    "aead": "ChaCha20Poly1305",
    "subkeys": "HKDF-SHA256-public-v1",
    "canonical_gate": "canonical-channel-v2-singleton",
    "tail": "HKDF-separated-HMAC-SHA256-counter-v1",
    "order": "HKDF-matched-arm-HMAC-SHA256-token-order-v1",
    "ordinary": "independent-public-seed-HMAC-bits-through-same-integer-transform-v1",
    "stop": "first-terminal-ascii-punctuation-after-first272bits;tail96-total768;no-flush;verify-entire-wire",
    "ending": "exact-utf8-wire-last-byte-in-.!?;no-rstrip-no-repair-v2",
}
FIXED_LIMITS = {"threshold_bits": 272, "token_cap": 768, "max_payload_bytes": 16,
                "length_bytes": 2, "tag_bytes": 16, "frame_bits": 272, "max_tail_tokens": 96}
RESOURCE_FIELDS = {"resource_id", "resource_sha256", "provider_sha256", "codec_sha256", "context_tokens"}
ENVELOPE_FIELDS = {"schema", "spec_sha256", "context_sha256", "arm", "nonce_hex", "ordinary_seed_hex"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_digest(value):
    return _sha(canonical_json(value))


def load_public_json(raw):
    """Reject duplicate keys and non-finite values, including nested objects."""
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ContractError("public JSON must be bytes at most1MiB")
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out: raise ContractError("duplicate JSON field")
            out[key] = value
        return out
    def reject(value):
        raise ContractError("non-finite JSON value")
    try:
        return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ContractError("invalid public JSON") from exc


def _strict_equal(actual, expected):
    if type(actual) is not type(expected): return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_strict_equal(actual[k], expected[k]) for k in expected)
    if type(expected) is list:
        return len(actual) == len(expected) and all(_strict_equal(a,b) for a,b in zip(actual,expected))
    return actual == expected


def _resource_descriptor(endpoint):
    return {"resource_id": endpoint.resource_id, "resource_sha256": endpoint.resource_digest,
            "provider_sha256": endpoint.provider_digest, "codec_sha256": endpoint.gate.codec.resource_digest,
            "context_tokens": endpoint.context_tokens}


def make_public_spec(endpoint, *, config=DEFAULT_CONFIG, max_wire_bytes=65536):
    if type(config) is not Config: raise ContractError("exact Config required")
    spec = {"schema": VERSION + "-public-spec", "version": VERSION,
            "resources": _resource_descriptor(endpoint), "algorithms": dict(ALGORITHMS),
            "limits": dict(FIXED_LIMITS, interval_config=asdict(config), max_wire_bytes=max_wire_bytes)}
    validate_public_spec(spec, endpoint)
    return spec


def validate_public_spec(spec, endpoint=None):
    if type(spec) is not dict or set(spec) != {"schema", "version", "resources", "algorithms", "limits"}:
        raise ContractError("strict public specification fields required")
    if spec["schema"] != VERSION + "-public-spec" or spec["version"] != VERSION:
        raise ContractError("public specification version mismatch")
    if not _strict_equal(spec["algorithms"], ALGORITHMS):
        raise ContractError("public algorithms mismatch")
    resource = spec["resources"]
    if type(resource) is not dict or set(resource) != RESOURCE_FIELDS:
        raise ContractError("strict public resource fields required")
    if (type(resource["resource_id"]) is not str or not 1 <= len(resource["resource_id"]) <= 128
            or any(ord(c) < 32 or ord(c) > 126 for c in resource["resource_id"])):
        raise ContractError("bounded ASCII resource identifier required")
    for name in ("resource_sha256", "provider_sha256", "codec_sha256"): fixed._digest(resource[name])
    if type(resource["context_tokens"]) is not int or not TOKEN_CAP <= resource["context_tokens"] <= 16384:
        raise ContractError("invalid public context limit")
    if endpoint is not None and not _strict_equal(resource, _resource_descriptor(endpoint)):
        raise ContractError("endpoint differs from public resource declaration")
    limits = spec["limits"]
    if type(limits) is not dict or set(limits) != set(FIXED_LIMITS) | {"interval_config", "max_wire_bytes"}:
        raise ContractError("strict public limit fields required")
    if any(not _strict_equal(limits[k], v) for k,v in FIXED_LIMITS.items()):
        raise ContractError("public stop/frame limits mismatch")
    if type(limits["max_wire_bytes"]) is not int or not 1 <= limits["max_wire_bytes"] <= 65536:
        raise ContractError("invalid public wire byte limit")
    conf = limits["interval_config"]
    if type(conf) is not dict or set(conf) != {"state_bits", "max_symbols", "max_pending_bits", "max_stream_bits"}:
        raise ContractError("strict interval configuration required")
    try: config = Config(**conf)
    except (TypeError, IntervalContractError) as exc: raise ContractError("invalid interval configuration") from exc
    if (config.state_bits != 32 or config.max_symbols != TOKEN_CAP
            or config.max_pending_bits > 4096 or config.max_stream_bits > 65536):
        raise ContractError("public precision/resource limit mismatch")
    canonical_json(spec)
    return config


def make_public_context(prompt_ids=(), *, context_id):
    context = {"schema": VERSION + "-public-context", "context_id": context_id,
               "prompt_ids": list(fixed._ids(prompt_ids))}
    validate_public_context(context)
    return context


def validate_public_context(context, spec=None):
    if type(context) is not dict or set(context) != {"schema", "context_id", "prompt_ids"}:
        raise ContractError("strict public context fields required")
    if context["schema"] != VERSION + "-public-context": raise ContractError("public context schema mismatch")
    name = context["context_id"]
    if (type(name) is not str or not 1 <= len(name) <= 128
            or any(ord(c) < 32 or ord(c) > 126 for c in name)):
        raise ContractError("bounded ASCII context identifier required")
    if type(context["prompt_ids"]) is not list: raise ContractError("public prompt must be a list")
    prompt = fixed._ids(context["prompt_ids"])
    if len(prompt) > 16384 or any(t >= 2147483648 for t in prompt):
        raise ContractError("public prompt exceeds identifier/length limit")
    if spec is not None and len(prompt) + TOKEN_CAP > spec["resources"]["context_tokens"]:
        raise ContractError("prompt plus bounded output exceeds context")
    canonical_json(context)
    return prompt


def make_envelope(public_spec, public_context, nonce, *, arm="stego", ordinary_seed=DEFAULT_ORDINARY_SEED):
    validate_public_spec(public_spec); validate_public_context(public_context, public_spec)
    if type(nonce) is not bytes or len(nonce) != 12: raise ContractError("nonce must be exactly12 bytes")
    if type(ordinary_seed) is not bytes or len(ordinary_seed) != 32: raise ContractError("ordinary seed must be32 bytes")
    env = {"schema": VERSION + "-envelope", "spec_sha256": public_digest(public_spec),
           "context_sha256": public_digest(public_context), "arm": arm,
           "nonce_hex": nonce.hex(), "ordinary_seed_hex": ordinary_seed.hex()}
    validate_envelope(env)
    return env


def validate_envelope(envelope):
    if type(envelope) is not dict or set(envelope) != ENVELOPE_FIELDS:
        raise ContractError("strict public envelope fields required")
    if envelope["schema"] != VERSION + "-envelope": raise ContractError("public envelope schema mismatch")
    if type(envelope["arm"]) is not str or envelope["arm"] not in ("stego", "ordinary"):
        raise ContractError("invalid public arm")
    for name in ("spec_sha256", "context_sha256", "ordinary_seed_hex"): fixed._digest(envelope[name])
    if type(envelope["nonce_hex"]) is not str or re.fullmatch("[0-9a-f]{24}", envelope["nonce_hex"]) is None:
        raise ContractError("invalid public nonce")
    canonical_json(envelope)


def validate_bindings(envelope, endpoint, *, public_spec, public_context):
    config = validate_public_spec(public_spec, endpoint)
    validate_public_context(public_context, public_spec); validate_envelope(envelope)
    if envelope["spec_sha256"] != public_digest(public_spec) or envelope["context_sha256"] != public_digest(public_context):
        raise ContractError("public specification/context digest mismatch")
    return config


def _snapshot_bindings(envelope, endpoint, public_spec, public_context):
    config = validate_bindings(envelope, endpoint, public_spec=public_spec, public_context=public_context)
    # The backend cannot mutate caller-owned dictionaries after validation.
    return (json.loads(canonical_json(envelope)), json.loads(canonical_json(public_spec)),
            json.loads(canonical_json(public_context)), config)


def metadata_costs(envelope, public_spec, public_context, *, wire_bytes, payload_bytes):
    """Logical compact transport bytes; does not include shared model/runtime/key setup."""
    validate_public_spec(public_spec); validate_public_context(public_context, public_spec); validate_envelope(envelope)
    if envelope["spec_sha256"] != public_digest(public_spec) or envelope["context_sha256"] != public_digest(public_context):
        raise ContractError("metadata binding differs")
    if type(wire_bytes) is not int or wire_bytes < 0 or type(payload_bytes) is not int or not 0 <= payload_bytes <= PAYLOAD_SLOT:
        raise ContractError("invalid transport counts")
    env_size, spec_size, context_size = map(lambda x: len(canonical_json(x)), (envelope, public_spec, public_context))
    return {"wire_bytes": wire_bytes, "compact_envelope_bytes": env_size, "public_spec_once_bytes": spec_size,
            "public_context_once_bytes": context_size, "key_once_bytes": 32, "nonce_bytes_already_in_envelope": 12,
            "payload_bytes": payload_bytes, "per_message_with_pre_shared_spec_context_bytes": wire_bytes + env_size,
            "first_message_without_pre_shared_spec_context_bytes": wire_bytes + env_size + spec_size + context_size,
            "model_runtime_distribution_bytes": None, "key_in_transport_totals": False}


def ordering_context_digest(envelope):
    return hashlib.sha256(canonical_json({k: v for k, v in envelope.items()
        if k not in {"arm", "nonce_hex", "ordinary_seed_hex"}})).digest()


def derive_keys(key, envelope):
    validate_envelope(envelope)
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



def terminal_punctuation(wire):
    """Exact-byte shape predicate only, not grammatical/semantic completeness.

    No strip/repair: a space or quote after punctuation does not satisfy it.
    Abbreviations, decimals and initials are intentional documented limitations.
    """
    if type(wire) is not bytes:
        raise ContractError("exact wire bytes required")
    return bool(wire) and wire[-1] in b".!?"


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
    ending_token: int | None
    tail_tokens: int | None
    has_terminal_punctuation: bool
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
    ending_token: int | None
    tail_tokens: int | None
    has_terminal_punctuation: bool
    authenticated_delivery: bool
    traces: tuple[dict, ...]


def generate(key, envelope, endpoint, *, public_spec, public_context, payload=None):
    envelope, public_spec, public_context, config = _snapshot_bindings(envelope, endpoint, public_spec, public_context)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    arm = envelope["arm"]
    if arm == "stego":
        frame = seal_frame(payload, key, envelope)
        stream = input_bits(frame, tail_key, envelope, config)
    else:
        if payload is not None: raise ContractError("ordinary has no payload")
        frame, stream = None, ordinary_bits(envelope, config)
    sender = Sender(stream, config)
    prefix, traces, first, ending = Prefix(), [], None, None
    stats = (0, sender.consumed_stream_bits, 0)
    def finish(status, reason=None):
        return GenerateResult(status, reason, arm, prefix.raw, len(prefix.ids), *stats,
            first, ending, None if first is None else len(prefix.ids)-first,
            terminal_punctuation(prefix.raw), _sha(frame) if frame is not None else None, False, tuple(traces))
    try:
        endpoint.begin(public_context["prompt_ids"])
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
            if len(advanced.raw) > public_spec["limits"]["max_wire_bytes"]:
                traces.append({"step": len(prefix.ids), "refused_token": step.symbol,
                    "reason": "max_wire_bytes", "transmitted": False})
                return finish("ending_not_reached" if first is not None else "wire_limit_exceeded", "max_wire_bytes")
            record = fixed._trace(prefix, decision, distribution, law, step, config)
            endpoint.observe(step.symbol)
            traces.append(record); prefix = advanced
            stats = len(sender.resolved_bits), sender.consumed_stream_bits, sender.state.pending
            if first is None and stats[0] >= THRESHOLD_BITS:
                first = len(prefix.ids)
            if first is not None:
                # A legal ending exactly at the tail or total cap is accepted.
                if terminal_punctuation(prefix.raw):
                    ending = len(prefix.ids)
                    return finish("ending_pending_receiver" if arm == "stego" else "ordinary_ending_generated")
                if len(prefix.ids)-first >= MAX_TAIL_TOKENS:
                    return finish("ending_not_reached", "tail96_without_terminal_punctuation")
    except BackendFailure as exc: return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc: return finish("contract_error", str(exc))
    if first is not None:
        return finish("ending_not_reached", "cap768_after_threshold_without_terminal_punctuation")
    return finish("insufficient_capacity" if arm == "stego" else "ordinary_cap_generated", "cap768_without_threshold")


def receive(wire, key, envelope, endpoint, *, public_spec, public_context):
    """Receiver has no sender trace/plaintext; validate first stop and FULL wire.

    The recovered plaintext stays local until the entire observed path agrees
    with the frame plus keyed deterministic tail. AEAD alone is insufficient.
    """
    envelope, public_spec, public_context, config = _snapshot_bindings(envelope, endpoint, public_spec, public_context)
    _, ordering_key, tail_key = derive_keys(key, envelope)
    receiver = Receiver(config)
    prefix, traces, replay, first, ending, wire_count = Prefix(), [], [], None, None, None
    def finish(status, reason=None, payload=None):
        return ReceiveResult(status, reason, payload, wire_count, len(prefix.ids),
            len(receiver.recovered_bits), receiver.state.pending, first, ending,
            None if first is None else len(prefix.ids)-first, terminal_punctuation(prefix.raw),
            status == "authenticated_delivery", tuple(traces))
    if type(wire) is not bytes: raise ContractError("wire must be exact bytes")
    if len(wire) > public_spec["limits"]["max_wire_bytes"]: return finish("invalid_wire", "max_wire_bytes")
    try:
        ids = endpoint.gate.codec.encode(wire)
        wire_count = len(ids)
        if endpoint.gate.codec.render(ids) != wire: return finish("invalid_wire", "noncanonical_bytes")
    except UnicodeDecodeError: return finish("invalid_wire", "invalid_utf8")
    except ContractError as exc: return finish("contract_error", str(exc))
    if len(ids) > TOKEN_CAP: return finish("invalid_wire", "too_many_tokens")
    try:
        endpoint.begin(public_context["prompt_ids"])
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
            if first is None and len(receiver.recovered_bits) >= THRESHOLD_BITS:
                first = len(prefix.ids)
            if first is not None:
                if terminal_punctuation(prefix.raw):
                    ending = len(prefix.ids)
                    if ending != len(ids): return finish("invalid_wire", "trailing_tokens_after_first_ending")
                elif len(prefix.ids)-first >= MAX_TAIL_TOKENS:
                    return finish("ending_not_reached" if len(prefix.ids)==len(ids) else "invalid_wire",
                                  "tail96_without_terminal_punctuation")
    except BackendFailure as exc: return finish("backend_error", str(exc))
    except (ContractError, IntervalContractError) as exc: return finish("contract_error", str(exc))
    if prefix.raw != wire: return finish("invalid_wire", "final_bytes_mismatch")
    if first is None:
        if len(ids) != TOKEN_CAP: return finish("incomplete_wire", "before_public_threshold")
        if envelope["arm"] == "stego": return finish("insufficient_capacity", "cap768_without_threshold")
    elif ending is None:
        if len(ids) == TOKEN_CAP or len(wire) == public_spec["limits"]["max_wire_bytes"]:
            return finish("ending_not_reached", "public_cap_without_terminal_punctuation")
        return finish("incomplete_wire", "before_public_ending")
    if envelope["arm"] == "ordinary":
        stream, payload = ordinary_bits(envelope, config), None
    else:
        frame = exact_bytes(receiver.recovered_bits[:THRESHOLD_BITS])
        try: payload = open_frame(frame, key, envelope)
        except InvalidTag: return finish("authentication_failed", "invalid_tag")
        except ContractError as exc: return finish("invalid_frame", str(exc))
        stream = input_bits(frame, tail_key, envelope, config)
    deterministic = Sender(stream, config)
    for law, token in replay:
        try: expected = deterministic.take(law)
        except CapacityLimit as exc: return finish("invalid_wire", "replay_" + exc.reason)
        if expected.symbol != token: return finish("invalid_wire", "noncanonical_fullwire_embedding")
    if envelope["arm"] == "ordinary":
        return finish("ordinary_ending_replay_verified" if ending is not None else "ordinary_cap_replay_verified")
    return finish("authenticated_delivery", payload=payload)
