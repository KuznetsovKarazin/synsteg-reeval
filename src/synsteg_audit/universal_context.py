"""Finite local-orbit eligibility, with a deliberately limited lexical gate.

This is a new mechanical protocol, not a semantic synonymy certificate. Every
potential dictionary position is expanded before any eligibility filtering.
The receiver needs only stego, key, and the serialized public resources.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from synsteg_audit import shared_dictionary as base
from synsteg_audit import invariant_context as old

POLICY_SCHEMA = "synsteg-universal-local-policy-v1"
PROTOCOL_VERSION = "synsteg-universal-local-context-v1"
SKELETON_VERSION = "all-dictionary-class-skeleton-v1"
VIRTUALIZATION_VERSION = "universal-protected-token-surface-hex-v1"
EXTERNAL_TEXT_VERSION = "reject-u-e000-u-e001-v1"
PREDICATE = {
    "version": "exact-class-lowercase-no-adjacent-indefinite-article-v1",
    "class_rule": "same-class-id-and-entire-sorted-word-list-in-public-reference",
    "case_rule": "lowercase-focus-only",
    "article_rule": "reject-if-immediately-preceding-token-core-is-a-or-an",
    "article_core_rule": "single-ASCII-word-with-Unicode-P-punctuation-except-underscore",
    "scope": "lexical-and-adjacent-article-ablation-not-semantic-certification",
}
_ALL = base.Protocol(1, 1, "plain_length16")


class UniversalContextError(base.CodecError):
    pass


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class OrbitResult:
    orbit_size: int
    evaluated_variants: int
    accepted: bool
    reason: str


def local_orbit_acceptance(options: Sequence[Sequence[str]], predicate: Callable[[tuple[str, ...]], bool], *, cap: int) -> OrbitResult:
    """Universally evaluate a finite product, or fail closed.

    The caller must construct options independently of observed class aliases.
    Arbitrary callables are for local proof tests only; they are never loaded
    from a policy or invoked by the serialized receiver. Acceptance requires
    strict boolean True for every combination. Rejection may short-circuit.
    """
    if type(cap) is not int or not 1 <= cap <= 1_000_000:
        raise UniversalContextError("Orbit cap must be an integer in 1..1000000")
    if not isinstance(options, (tuple, list)) or not options or not callable(predicate):
        raise UniversalContextError("Nonempty finite options and a predicate are required")
    normalized = []
    for values in options:
        if (not isinstance(values, (tuple, list)) or not values
                or any(not isinstance(value, str) or not value for value in values)):
            raise UniversalContextError("Every orbit coordinate requires nonempty strings")
        normalized.append(tuple(sorted(set(values))))
    size = math.prod(len(values) for values in normalized)
    if size > cap:
        return OrbitResult(size, 0, False, "orbit_cap_exceeded")
    evaluated = 0
    for variant in itertools.product(*normalized):
        evaluated += 1
        try:
            verdict = predicate(variant)
        except Exception:
            return OrbitResult(size, evaluated, False, "predicate_error")
        if type(verdict) is not bool:
            return OrbitResult(size, evaluated, False, "predicate_error")
        if not verdict:
            return OrbitResult(size, evaluated, False, "predicate_rejected")
    return OrbitResult(size, evaluated, True, "accepted")


@dataclass(frozen=True)
class UniversalPolicy:
    dictionary_sha256: str
    strict_dictionary: base.PublicDictionary
    version: str = "lexical-adjacent-article-v1"
    radius: int = 1
    orbit_cap: int = 4096

    def __post_init__(self):
        if not isinstance(self.dictionary_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.dictionary_sha256):
            raise UniversalContextError("Canonical carrier dictionary hash required")
        if not isinstance(self.strict_dictionary, base.PublicDictionary):
            raise UniversalContextError("Validated public reference dictionary required")
        if not isinstance(self.version, str) or not self.version:
            raise UniversalContextError("Policy version must be nonempty")
        if type(self.radius) is not int or not 0 <= self.radius <= 2:
            raise UniversalContextError("Radius must be an integer in 0..2")
        if type(self.orbit_cap) is not int or not 1 <= self.orbit_cap <= 1_000_000:
            raise UniversalContextError("Orbit cap must be an integer in 1..1000000")

    @classmethod
    def from_dictionaries(cls, dictionary, strict_dictionary, *, version="lexical-adjacent-article-v1", radius=1, orbit_cap=4096):
        if not isinstance(dictionary, base.PublicDictionary):
            raise UniversalContextError("Validated carrier dictionary required")
        return cls(dictionary.sha256, strict_dictionary, version, radius, orbit_cap)

    def validate_for(self, dictionary):
        if not isinstance(dictionary, base.PublicDictionary) or dictionary.sha256 != self.dictionary_sha256:
            raise UniversalContextError("Policy belongs to a different carrier dictionary")

    @property
    def reference_dictionary_sha256(self):
        return self.strict_dictionary.sha256

    def admitted_class_ids(self, dictionary):
        self.validate_for(dictionary)
        reference = {entry.class_id: entry.words for entry in self.strict_dictionary.classes}
        return tuple(entry.class_id for entry in dictionary.classes if reference.get(entry.class_id) == entry.words)

    def to_dict(self):
        return {"schema": POLICY_SCHEMA, "version": self.version,
                "dictionary_sha256": self.dictionary_sha256,
                "reference_dictionary_sha256": self.reference_dictionary_sha256,
                "strict_dictionary": self.strict_dictionary.to_dict(),
                "radius": self.radius, "orbit_cap": self.orbit_cap,
                "skeleton_version": SKELETON_VERSION,
                "predicate": dict(PREDICATE),
                "window_rule": "symmetric-whitespace-token-radius-plus-immediate-predecessor",
                "expansion_rule": "all-potential-dictionary-slots-independent-occurrences",
                "evaluation_rule": "sorted-full-cartesian-product-strict-bool-fail-closed"}

    @classmethod
    def from_dict(cls, data, dictionary):
        if not isinstance(data, dict):
            raise UniversalContextError("Public policy must be an object")
        try:
            reference = base.PublicDictionary.from_dict(data["strict_dictionary"])
            value = cls(data["dictionary_sha256"], reference, data["version"], data["radius"], data["orbit_cap"])
        except (KeyError, TypeError, base.CodecError) as error:
            raise UniversalContextError("Invalid public policy") from error
        value.validate_for(dictionary)
        if value.to_dict() != data:
            raise UniversalContextError("Unsupported or noncanonical public policy")
        return value

    @property
    def sha256(self):
        return _digest(self.to_dict())


@dataclass(frozen=True)
class UniversalProtocol:
    min_group_size: int = 1
    max_bits_per_slot: int = 1
    framing: str = "aes_siv"

    def __post_init__(self):
        self.base_protocol()

    def base_protocol(self):
        return base.Protocol(self.min_group_size, self.max_bits_per_slot, self.framing)

    def to_dict(self):
        return {"version": PROTOCOL_VERSION, "skeleton_version": SKELETON_VERSION,
                "virtualization_version": VIRTUALIZATION_VERSION,
                "external_text_version": EXTERNAL_TEXT_VERSION,
                "min_group_size": self.min_group_size, "max_bits_per_slot": self.max_bits_per_slot,
                "framing": self.framing}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise UniversalContextError("Public protocol must be an object")
        try:
            value = cls(data["min_group_size"], data["max_bits_per_slot"], data["framing"])
        except (KeyError, TypeError, base.CodecError) as error:
            raise UniversalContextError("Invalid public protocol") from error
        if value.to_dict() != data:
            raise UniversalContextError("Unsupported or noncanonical public protocol")
        return value

    @property
    def sha256(self):
        return _digest(self.to_dict())

    @property
    def frame_overhead_bits(self):
        return self.base_protocol().frame_overhead_bits


DEFAULT_PROTOCOL = UniversalProtocol()


@dataclass(frozen=True)
class SlotDecision:
    token_index: int
    class_id: str
    orbit_size: int
    evaluated_variants: int
    accepted: bool
    reason: str


@dataclass(frozen=True)
class UniversalAnalysis:
    candidate_slot_count: int
    protected_slot_count: int
    eligible_slot_count: int
    raw_capacity_bits: int
    capacity_bits: int
    frame_overhead_bits: int
    protected_token_indices: tuple[int, ...]
    eligible_token_indices: tuple[int, ...]
    skeleton_sha256: str
    virtual_normalization_sha256: str
    slot_decisions: tuple[SlotDecision, ...]

    @property
    def max_payload_bytes(self):
        if self.capacity_bits < self.frame_overhead_bits:
            return None
        return min(base.MAX_PAYLOAD_BYTES, (self.capacity_bits - self.frame_overhead_bits) // 8)


@dataclass(frozen=True)
class UniversalEncodeResult:
    stego_text: str
    payload_bytes: int
    capacity_bits: int
    raw_capacity_bits: int
    protected_slot_count: int
    frame_bits: int
    consumed_slots: int
    padding_bits: int
    changed_slots: int
    dictionary_sha256: str
    policy_sha256: str
    protocol_sha256: str
    skeleton_sha256: str
    virtual_normalization_sha256: str


@dataclass(frozen=True)
class _Carrier:
    original_text: str
    virtual_text: str
    skeleton_text: str
    protected_original_tokens: tuple[tuple[int, str], ...]
    analysis: UniversalAnalysis


def _marker(index, surface):
    return "\ue000uc1:" + str(index) + ":" + surface.encode("ascii").hex() + "\ue001"


def _prepare(text, dictionary, policy, protocol):
    if not isinstance(text, str) or not isinstance(protocol, UniversalProtocol):
        raise UniversalContextError("Text and validated universal protocol required")
    if any(character in text for character in old.RESERVED_CHARACTERS):
        raise UniversalContextError("External text contains reserved U+E000 or U+E001")
    if not isinstance(policy, UniversalPolicy):
        raise UniversalContextError("Validated universal policy required")
    policy.validate_for(dictionary)
    raw = base.analyze_text(text, dictionary, _ALL)
    tokens = list(re.finditer(r"\S+", text))
    by_token = {slot.token_index: slot for slot in raw.slots}
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    admitted = set(policy.admitted_class_ids(dictionary))
    options = []
    skeleton_changes = []
    for i, token in enumerate(tokens):
        slot = by_token.get(i)
        if slot is None:
            options.append((token.group(),))
            continue
        left, right = slot.start - token.start(), slot.end - token.start()
        values = tuple(token.group()[:left] + base._render_case(word, slot.case_style) + token.group()[right:]
                       for word in classes[slot.class_id])
        options.append(values)
        skeleton_changes.append((slot.start, slot.end, base._render_case(classes[slot.class_id][0], slot.case_style)))
    decisions = []
    for slot in raw.slots:
        i = slot.token_index
        # Radius zero still includes the predecessor required by the fixed
        # article predicate; no context dependency is silently omitted.
        start, stop = max(0, i - max(1, policy.radius)), min(len(tokens), i + policy.radius + 1)
        focus = i - start
        exact_class = slot.class_id in admitted
        lowercase = slot.case_style == "lower"
        def predicate(variant):
            if not exact_class or not lowercase:
                return False
            previous = old._literal_atom(variant[focus - 1]) if focus else None
            return previous is None or previous.value not in ("a", "an")
        result = local_orbit_acceptance(options[start:stop], predicate, cap=policy.orbit_cap)
        reason = result.reason
        if reason == "predicate_rejected":
            reason = "class_not_exact_reference" if not exact_class else "case_not_lowercase" if not lowercase else "adjacent_indefinite_article"
        decisions.append(SlotDecision(i, slot.class_id, result.orbit_size,
                                      result.evaluated_variants, result.accepted, reason))
    protected = {decision.token_index for decision in decisions if not decision.accepted}
    virtual_changes = [(slot.start, slot.end, _marker(slot.token_index, slot.surface))
                       for slot in raw.slots if slot.token_index in protected]
    skeleton = old._replace_ranges(text, skeleton_changes)
    virtual = old._replace_ranges(text, virtual_changes)
    final = base.analyze_text(virtual, dictionary, protocol.base_protocol())
    raw_capacity = base.analyze_text(text, dictionary, protocol.base_protocol()).capacity_bits
    eligible = tuple(slot.token_index for slot in final.slots)
    if (set(eligible) & protected or any(index not in by_token for index in eligible)
            or final.capacity_bits > raw_capacity):
        raise UniversalContextError("Virtualization violated public eligibility")
    analysis = UniversalAnalysis(len(raw.slots), len(protected), len(eligible), raw_capacity,
        final.capacity_bits, protocol.frame_overhead_bits, tuple(sorted(protected)), eligible,
        hashlib.sha256(skeleton.encode("utf-8")).hexdigest(), final.normalization_sha256, tuple(decisions))
    return _Carrier(text, virtual, skeleton, tuple((i, tokens[i].group()) for i in sorted(protected)), analysis)


def analyze_text(text, dictionary, policy, protocol=DEFAULT_PROTOCOL):
    return _prepare(text, dictionary, policy, protocol).analysis


def _derived_key(key, dictionary, policy, protocol):
    if not isinstance(key, bytes) or len(key) != 32:
        raise UniversalContextError("Master key must be exactly 32 bytes")
    salt = hashlib.sha256(b"synsteg/universal-local-context/salt/v1\x00" + bytes.fromhex(dictionary.sha256)).digest()
    info = _canonical({"domain": PROTOCOL_VERSION, "dictionary_sha256": dictionary.sha256,
                       "policy_sha256": policy.sha256, "protocol": protocol.to_dict(), "predicate": PREDICATE})
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(key)


def encode(cover_text, payload, key, dictionary, policy, protocol=DEFAULT_PROTOCOL, *, selector="canonical"):
    if selector != "canonical":
        raise UniversalContextError("Only canonical selection is supported")
    carrier = _prepare(cover_text, dictionary, policy, protocol)
    result = base.encode(carrier.virtual_text, payload, _derived_key(key, dictionary, policy, protocol),
                         dictionary, protocol.base_protocol(), selector="canonical")
    stego = old._restore(carrier, result.stego_text)
    check = _prepare(stego, dictionary, policy, protocol)
    if check.analysis != carrier.analysis or check.protected_original_tokens != carrier.protected_original_tokens:
        raise UniversalContextError("Eligibility or virtual normalization changed during encoding")
    a = carrier.analysis
    return UniversalEncodeResult(stego, len(payload), result.capacity_bits, a.raw_capacity_bits,
        a.protected_slot_count, result.frame_bits, result.consumed_slots, result.padding_bits,
        result.changed_slots, dictionary.sha256, policy.sha256, protocol.sha256,
        a.skeleton_sha256, a.virtual_normalization_sha256)


def decode(stego_text, key, dictionary, policy, protocol=DEFAULT_PROTOCOL):
    carrier = _prepare(stego_text, dictionary, policy, protocol)
    return base.decode(carrier.virtual_text, _derived_key(key, dictionary, policy, protocol),
                       dictionary, protocol.base_protocol())


def receiver_main():
    import socket
    import subprocess
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    warm = AESSIV(bytes(64))
    warm.decrypt(warm.encrypt(b"", [b"public-warmup"]), [b"public-warmup"])
    def guard(event, args):
        if event in {"open", "os.listdir", "os.scandir", "os.system", "os.posix_spawn"} or event.startswith(("socket.", "subprocess.")):
            raise PermissionError("Receiver external I/O forbidden")
    sys.addaudithook(guard)
    checks = {}
    for name, operation in (("file", lambda: open("/receiver-forbidden")), ("socket", socket.socket),
                            ("subprocess", lambda: subprocess.Popen(["receiver-forbidden"]))):
        try:
            operation()
        except PermissionError:
            checks[name] = True
        else:
            raise RuntimeError("Receiver I/O guard self-check failed")
    request = json.loads(sys.stdin.buffer.read())
    allowed = {"key_hex", "stego_text", "public_dictionary", "public_policy", "public_protocol"}
    if not isinstance(request, dict) or set(request) != allowed:
        raise UniversalContextError("Receiver accepts only key, stego and public resources")
    dictionary = base.PublicDictionary.from_dict(request["public_dictionary"])
    policy = UniversalPolicy.from_dict(request["public_policy"], dictionary)
    protocol = UniversalProtocol.from_dict(request["public_protocol"])
    payload = decode(request["stego_text"], bytes.fromhex(request["key_hex"]), dictionary, policy, protocol)
    print(json.dumps({"status": "decoded", "dictionary_sha256": dictionary.sha256,
                      "payload_hex": payload.hex(), "input_keys": sorted(request),
                      "io_guard_self_checks": checks, "policy_sha256": policy.sha256,
                      "protocol_sha256": protocol.sha256,
                      "isolation_scope": "inspected-program Python audit guard, not an OS sandbox"}, sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] != ["receiver"]:
        raise SystemExit("Usage: universal_context.py receiver")
    receiver_main()
