"""Experimental public phrase protection on an invariant dictionary skeleton.

This separate protocol reuses the frozen shared-dictionary cryptographic core.
Protection is computed before encoding from ALL dictionary classes, never from
the sender's chosen synonym. It is not a contextual synonymy or POS model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

# Support a fresh isolated receiver process without importing experiment code.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from synsteg_audit import shared_dictionary as base

POLICY_SCHEMA = "synsteg-invariant-phrase-policy-v1"
PROTOCOL_VERSION = "synsteg-invariant-context-v1"
SKELETON_VERSION = "all-dictionary-class-skeleton-v1"
VIRTUALIZATION_VERSION = "protected-token-surface-hex-v1"
EXTERNAL_TEXT_VERSION = "reject-u-e000-u-e001-v1"
RESERVED_CHARACTERS = ("\ue000", "\ue001")
_LITERAL = re.compile(r"[a-z]+\Z", re.ASCII)
_PHRASE = re.compile(r"[a-z]+(?: [a-z]+)*\Z", re.ASCII)
_TOKEN = re.compile(r"(?P<prefix>[^A-Za-z]*)(?P<word>[A-Za-z]+)(?P<suffix>[^A-Za-z]*)\Z")
_ALL_SLOTS = base.Protocol(min_group_size=1, max_bits_per_slot=1, framing="plain_length16")


class ContextError(base.CodecError):
    """Invalid public context policy or violation of the wrapper invariants."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, order=True)
class Atom:
    kind: str
    value: str

    def __post_init__(self):
        if self.kind not in ("class", "literal") or not isinstance(self.value, str) or not self.value:
            raise ContextError("Policy atoms must be nonempty class or literal values")
        if self.kind == "literal" and not _LITERAL.fullmatch(self.value):
            raise ContextError("Literal atoms must be lowercase ASCII whole words")

    def to_dict(self):
        return {"kind": self.kind, "value": self.value}


def _compiled_patterns(dictionary, phrases):
    lookup = {word: entry.class_id for entry in dictionary.classes for word in entry.words}
    return tuple(sorted({tuple(Atom("class", lookup[word]) if word in lookup else Atom("literal", word)
                               for word in phrase.split(" ")) for phrase in phrases}))


@dataclass(frozen=True)
class ProtectedPhrasePolicy:
    """Canonical, dictionary-bound public policy; no private sender metadata.

    Source phrases are retained and hashed for provenance. Equivalent compiled
    class patterns are deduplicated, so aliases cannot change matching order.
    """
    version: str
    dictionary_sha256: str
    phrases: tuple[str, ...]
    patterns: tuple[tuple[Atom, ...], ...]

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version:
            raise ContextError("Policy version must be nonempty")
        if not isinstance(self.dictionary_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", self.dictionary_sha256):
            raise ContextError("Policy requires a canonical public dictionary hash")
        if (not isinstance(self.phrases, tuple) or any(not isinstance(p, str) or not _PHRASE.fullmatch(p)
                or len(p.split()) > 16 for p in self.phrases)
                or self.phrases != tuple(sorted(set(self.phrases)))):
            raise ContextError("Policy phrases must be sorted unique canonical ASCII word sequences")
        if (not isinstance(self.patterns, tuple) or any(not isinstance(p, tuple) or not 1 <= len(p) <= 16
                or any(not isinstance(a, Atom) for a in p) for p in self.patterns)
                or self.patterns != tuple(sorted(set(self.patterns)))):
            raise ContextError("Compiled policy patterns must be sorted unique atom sequences")

    @classmethod
    def from_phrases(cls, dictionary, phrases, *, version):
        if not isinstance(dictionary, base.PublicDictionary):
            raise ContextError("Validated public dictionary required")
        if not isinstance(phrases, (list, tuple)) or any(not isinstance(p, str) for p in phrases):
            raise ContextError("Phrases must be an explicit sequence of strings")
        normalized = tuple(sorted(set(" ".join(p.lower().split()) for p in phrases)))
        if any(not _PHRASE.fullmatch(p) or len(p.split()) > 16 for p in normalized):
            raise ContextError("Phrases must contain one to sixteen ASCII words without punctuation")
        return cls(version, dictionary.sha256, normalized, _compiled_patterns(dictionary, normalized))

    @classmethod
    def from_dict(cls, data, dictionary):
        expected = {"schema", "canonicalization_version", "version", "dictionary_sha256", "phrases", "patterns"}
        if (not isinstance(data, dict) or set(data) != expected or data["schema"] != POLICY_SCHEMA
                or data["canonicalization_version"] != SKELETON_VERSION
                or not isinstance(data["phrases"], list) or not isinstance(data["patterns"], list)):
            raise ContextError("Unsupported or noncanonical context policy")
        patterns = []
        for phrase in data["patterns"]:
            if not isinstance(phrase, list):
                raise ContextError("Policy pattern must be a list")
            atoms = []
            for atom in phrase:
                if not isinstance(atom, dict) or set(atom) != {"kind", "value"}:
                    raise ContextError("Invalid policy atom")
                atoms.append(Atom(atom["kind"], atom["value"]))
            patterns.append(tuple(atoms))
        policy = cls(data["version"], data["dictionary_sha256"], tuple(data["phrases"]), tuple(patterns))
        policy.validate_for(dictionary)
        if policy.to_dict() != data:
            raise ContextError("Policy is not in canonical serialized form")
        return policy

    def validate_for(self, dictionary):
        if not isinstance(dictionary, base.PublicDictionary) or dictionary.sha256 != self.dictionary_sha256:
            raise ContextError("Policy belongs to a different public dictionary")
        if _compiled_patterns(dictionary, self.phrases) != self.patterns:
            raise ContextError("Policy patterns do not match public class-level compilation")

    def to_dict(self):
        return {"schema": POLICY_SCHEMA, "canonicalization_version": SKELETON_VERSION,
                "version": self.version, "dictionary_sha256": self.dictionary_sha256,
                "phrases": list(self.phrases),
                "patterns": [[atom.to_dict() for atom in pattern] for pattern in self.patterns]}

    @property
    def sha256(self):
        return _digest(self.to_dict())


@dataclass(frozen=True)
class ContextProtocol:
    min_group_size: int = 1
    max_bits_per_slot: int = 1
    framing: str = "aes_siv"

    def __post_init__(self):
        self.base_protocol()

    def base_protocol(self):
        return base.Protocol(min_group_size=self.min_group_size,
                             max_bits_per_slot=self.max_bits_per_slot, framing=self.framing)

    def to_dict(self):
        return {"version": PROTOCOL_VERSION, "skeleton_version": SKELETON_VERSION,
                "virtualization_version": VIRTUALIZATION_VERSION,
                "external_text_version": EXTERNAL_TEXT_VERSION,
                "min_group_size": self.min_group_size, "max_bits_per_slot": self.max_bits_per_slot,
                "framing": self.framing}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ContextError("Public context protocol must be an object")
        try:
            protocol = cls(data["min_group_size"], data["max_bits_per_slot"], data["framing"])
        except KeyError as error:
            raise ContextError("Incomplete public context protocol") from error
        if protocol.to_dict() != data:
            raise ContextError("Unsupported or noncanonical context protocol")
        return protocol

    @property
    def sha256(self):
        return _digest(self.to_dict())

    @property
    def frame_overhead_bits(self):
        return self.base_protocol().frame_overhead_bits


DEFAULT_PROTOCOL = ContextProtocol()


@dataclass(frozen=True)
class ContextAnalysis:
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

    @property
    def max_payload_bytes(self):
        if self.capacity_bits < self.frame_overhead_bits:
            return None
        return min(base.MAX_PAYLOAD_BYTES, (self.capacity_bits - self.frame_overhead_bits) // 8)


@dataclass(frozen=True)
class ContextEncodeResult:
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
    analysis: ContextAnalysis


def _replace_ranges(text, replacements):
    pieces, cursor = [], 0
    for start, end, value in replacements:
        if not cursor <= start <= end <= len(text):
            raise ContextError("Invalid structural replacement ranges")
        pieces.extend([text[cursor:start], value])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _literal_atom(token):
    match = _TOKEN.fullmatch(token)
    if match is None:
        return None  # Opaque tokens are barriers, never policy wildcards.
    affixes = match.group("prefix") + match.group("suffix")
    if any(c == "_" or not unicodedata.category(c).startswith("P") for c in affixes):
        return None
    return Atom("literal", match.group("word").lower())


def _marker(token_index, surface):
    # PUA characters are category Co, not punctuation: frozen v1 cannot admit
    # this token. No whitespace is inserted; no marker is found by text search.
    return "\ue000ctx1:" + str(token_index) + ":" + surface.encode("ascii").hex() + "\ue001"


def _prepare(text, dictionary, policy, protocol):
    if not isinstance(text, str) or not isinstance(protocol, ContextProtocol):
        raise ContextError("Text and validated context protocol required")
    # Without an accepted-domain restriction, replacing a protected raw word
    # with its literal virtual marker would alias the same authenticated text.
    # Reject even partial marker characters on every external API path.
    if any(character in text for character in RESERVED_CHARACTERS):
        raise ContextError("External text contains reserved U+E000 or U+E001")
    if not isinstance(policy, ProtectedPhrasePolicy):
        raise ContextError("Validated public phrase policy required")
    policy.validate_for(dictionary)
    # All classes have >=2 members. Fixed min_group_size=1 / one bit includes
    # every potential dictionary slot, even ones later ineligible for framing.
    raw = base.analyze_text(text, dictionary, _ALL_SLOTS)
    tokens = list(re.finditer(r"\S+", text))
    by_token = {slot.token_index: slot for slot in raw.slots}
    stream = tuple(Atom("class", by_token[i].class_id) if i in by_token else _literal_atom(token.group())
                   for i, token in enumerate(tokens))
    protected = set()
    for pattern in policy.patterns:
        for start in range(len(stream) - len(pattern) + 1):
            if stream[start:start + len(pattern)] == pattern:
                protected.update(i for i in range(start, start + len(pattern)) if i in by_token)
    representatives = {entry.class_id: entry.words[0] for entry in dictionary.classes}
    skeleton_changes, virtual_changes = [], []
    for slot in raw.slots:
        canonical_word = representatives[slot.class_id]
        if slot.case_style == "upper":
            canonical_word = canonical_word.upper()
        elif slot.case_style == "title":
            canonical_word = canonical_word.capitalize()
        skeleton_changes.append((slot.start, slot.end, canonical_word))
        if slot.token_index in protected:
            virtual_changes.append((slot.start, slot.end, _marker(slot.token_index, slot.surface)))
    skeleton = _replace_ranges(text, skeleton_changes)
    virtual = _replace_ranges(text, virtual_changes)
    virtual_analysis = base.analyze_text(virtual, dictionary, protocol.base_protocol())
    raw_capacity = base.analyze_text(text, dictionary, protocol.base_protocol()).capacity_bits
    eligible = tuple(slot.token_index for slot in virtual_analysis.slots)
    if set(eligible) & protected or any(index not in by_token for index in eligible) or virtual_analysis.capacity_bits > raw_capacity:
        raise ContextError("Virtual carrier introduced or retained a protected slot")
    analysis = ContextAnalysis(len(raw.slots), len(protected), len(eligible), raw_capacity,
        virtual_analysis.capacity_bits, protocol.frame_overhead_bits, tuple(sorted(protected)), eligible,
        hashlib.sha256(skeleton.encode("utf-8")).hexdigest(), virtual_analysis.normalization_sha256)
    return _Carrier(text, virtual, skeleton,
                    tuple((i, tokens[i].group()) for i in sorted(protected)), analysis)


def analyze_text(text, dictionary, policy, protocol=DEFAULT_PROTOCOL):
    return _prepare(text, dictionary, policy, protocol).analysis


def _derived_key(key, dictionary, policy, protocol):
    if not isinstance(key, bytes) or len(key) != 32:
        raise ContextError("Master key must be exactly 32 bytes")
    salt = hashlib.sha256(b"synsteg/invariant-context/salt/v1\x00" + bytes.fromhex(dictionary.sha256)).digest()
    info = _canonical({"domain": PROTOCOL_VERSION, "dictionary_sha256": dictionary.sha256,
                       "policy_sha256": policy.sha256, "protocol": protocol.to_dict()})
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(key)


def _restore(carrier, encoded_virtual):
    before = list(re.finditer(r"\S+", carrier.virtual_text))
    after = list(re.finditer(r"\S+", encoded_virtual))
    if len(before) != len(after):
        raise ContextError("Encoder changed token boundaries")
    for i in range(len(before) + 1):
        left_before = before[i - 1].end() if i else 0
        right_before = before[i].start() if i < len(before) else len(carrier.virtual_text)
        left_after = after[i - 1].end() if i else 0
        right_after = after[i].start() if i < len(after) else len(encoded_virtual)
        if carrier.virtual_text[left_before:right_before] != encoded_virtual[left_after:right_after]:
            raise ContextError("Encoder changed literal whitespace")
    changes = []
    for index, original_token in carrier.protected_original_tokens:
        if before[index].group() != after[index].group():
            raise ContextError("Encoder altered an ineligible protected marker")
        changes.append((after[index].start(), after[index].end(), original_token))
    return _replace_ranges(encoded_virtual, changes)


def encode(cover_text, payload, key, dictionary, policy, protocol=DEFAULT_PROTOCOL, *, selector="canonical"):
    """Encode using public invariant protection; canonical selector only.

    Protected word spellings remain unchanged. AES-SIV authenticates those exact
    protected spellings through virtual literal hex markers. Other invariance
    and unauthenticated plain-frame caveats of the frozen core still apply.
    """
    if selector != "canonical":
        raise ContextError("This protocol only supports the canonical selector")
    carrier = _prepare(cover_text, dictionary, policy, protocol)
    derived = _derived_key(key, dictionary, policy, protocol)
    encoded = base.encode(carrier.virtual_text, payload, derived, dictionary, protocol.base_protocol(), selector="canonical")
    stego = _restore(carrier, encoded.stego_text)
    check = _prepare(stego, dictionary, policy, protocol)
    if (check.analysis != carrier.analysis
            or check.protected_original_tokens != carrier.protected_original_tokens):
        raise ContextError("Context or virtual normalization changed during encoding")
    a = carrier.analysis
    return ContextEncodeResult(stego, len(payload), encoded.capacity_bits, a.raw_capacity_bits,
        a.protected_slot_count, encoded.frame_bits, encoded.consumed_slots, encoded.padding_bits,
        encoded.changed_slots, dictionary.sha256, policy.sha256, protocol.sha256,
        a.skeleton_sha256, a.virtual_normalization_sha256)


def decode(stego_text, key, dictionary, policy, protocol=DEFAULT_PROTOCOL):
    """Decode with stego/key/public resources only; no cover or mask supplied."""
    carrier = _prepare(stego_text, dictionary, policy, protocol)
    derived = _derived_key(key, dictionary, policy, protocol)
    return base.decode(carrier.virtual_text, derived, dictionary, protocol.base_protocol())


def receiver_main():
    """Auditable minimal receiver, with the same scope as the frozen pilot guard."""
    import socket
    import subprocess
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    warm = AESSIV(bytes(64))
    warm.decrypt(warm.encrypt(b"", [b"public-warmup"]), [b"public-warmup"])

    def guard(event, args):
        if event in {"open", "os.listdir", "os.scandir", "os.system", "os.posix_spawn"} or event.startswith(("socket.", "subprocess.")):
            raise PermissionError("receiver external I/O forbidden")

    sys.addaudithook(guard)
    # Import and warm-up precede the guard; decoding performs no resource I/O.
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
        raise ContextError("Receiver accepts only key, stego and public resources")
    result = {"input_keys": sorted(request), "io_guard_self_checks": checks,
              "isolation_scope": "inspected-program Python audit guard, not an OS sandbox"}
    try:
        dictionary = base.PublicDictionary.from_dict(request["public_dictionary"])
        policy = ProtectedPhrasePolicy.from_dict(request["public_policy"], dictionary)
        protocol = ContextProtocol.from_dict(request["public_protocol"])
        payload = decode(request["stego_text"], bytes.fromhex(request["key_hex"]), dictionary, policy, protocol)
        result.update(status="decoded", payload_hex=payload.hex(), dictionary_sha256=dictionary.sha256,
                      policy_sha256=policy.sha256, protocol_sha256=protocol.sha256)
        code = 0
    except (base.CodecError, ValueError, TypeError) as error:
        result.update(status="rejected", error_type=type(error).__name__)
        code = 3
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    if sys.argv[1:] != ["receiver"]:
        raise SystemExit("Only the minimal receiver command is supported")
    raise SystemExit(receiver_main())
