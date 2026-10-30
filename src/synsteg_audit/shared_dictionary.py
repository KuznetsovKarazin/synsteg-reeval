"""Authenticated synonym channel with a shared public dictionary, not a cover.

This is a separate research codec. AES-SIV authenticates the extracted message
and invariant text, not every synonym spelling: edits preserving all labels and
the invariant representation intentionally remain valid. No claim about natural
language quality or resistance to steganalysis follows from this construction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


DICTIONARY_SCHEMA = "synsteg-public-dictionary-v1"
PROTOCOL_VERSION = "synsteg-shared-dictionary-aes-siv-v1"
FRAME_OVERHEAD_BITS = 16 + 128
MAX_PAYLOAD_BYTES = (1 << 16) - 1
_WORD = re.compile(r"[a-z]{2,}\Z", re.ASCII)
_TOKEN = re.compile(r"(?P<prefix>[^A-Za-z]*)(?P<word>[A-Za-z]+)(?P<suffix>[^A-Za-z]*)\Z")


class CodecError(ValueError):
    """Invalid public configuration, key, payload or selector input."""


class CapacityError(CodecError):
    """The complete authenticated frame cannot fit; no output is produced."""


class DecodeError(CodecError):
    """The supplied text is not a complete valid frame for this receiver."""


class AuthenticationError(DecodeError):
    """AES-SIV authentication failed."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class DictionaryClass:
    class_id: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class PublicDictionary:
    """Canonical public vocabulary; descriptive build metadata is not hashed.

    The identifier binds schema, version, every class id and every member word.
    Keep a separate whole-file hash when archiving source/build metadata.
    """

    classes: tuple[DictionaryClass, ...]
    version: str = "fixture-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise CodecError("dictionary version must be a nonempty string")
        if not isinstance(self.classes, tuple) or not self.classes:
            raise CodecError("dictionary classes must be a nonempty tuple")
        seen_ids: set[str] = set()
        seen_words: set[str] = set()
        for entry in self.classes:
            if not isinstance(entry, DictionaryClass):
                raise CodecError("each class must be a DictionaryClass")
            if not isinstance(entry.class_id, str) or not entry.class_id or entry.class_id in seen_ids:
                raise CodecError("class ids must be unique nonempty strings")
            seen_ids.add(entry.class_id)
            if not isinstance(entry.words, tuple) or len(entry.words) < 2:
                raise CodecError("each class must contain at least two words")
            for word in entry.words:
                if not isinstance(word, str) or not _WORD.fullmatch(word):
                    raise CodecError("dictionary words must be lowercase ASCII whole words of at least two letters")
                if word in seen_words:
                    raise CodecError("dictionary words must be unique across and within classes")
                seen_words.add(word)
        normalized = tuple(sorted(
            (DictionaryClass(entry.class_id, tuple(sorted(entry.words))) for entry in self.classes),
            key=lambda entry: entry.class_id,
        ))
        object.__setattr__(self, "classes", normalized)

    @classmethod
    def from_mapping(cls, classes: Mapping[str, Sequence[str]], version: str = "fixture-v1") -> PublicDictionary:
        if not isinstance(classes, Mapping):
            raise CodecError("classes must be a mapping")
        entries = []
        for class_id, words in classes.items():
            if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
                raise CodecError("class words must be a sequence of strings")
            entries.append(DictionaryClass(class_id, tuple(words)))
        return cls(tuple(entries), version)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PublicDictionary:
        if not isinstance(data, Mapping) or data.get("schema") != DICTIONARY_SCHEMA:
            raise CodecError("unsupported public dictionary schema")
        entries = data.get("classes")
        if not isinstance(entries, list):
            raise CodecError("public dictionary classes must be a list")
        built = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("words"), list):
                raise CodecError("invalid public dictionary class")
            built.append(DictionaryClass(entry.get("id"), tuple(entry["words"])))
        return cls(tuple(built), data.get("version"))

    def to_dict(self) -> dict:
        return {"schema": DICTIONARY_SCHEMA, "version": self.version,
                "classes": [{"id": entry.class_id, "words": list(entry.words)} for entry in self.classes]}

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True)
class Protocol:
    """Public parameters: width and eligibility never depend on a cover choice."""

    min_group_size: int = 2
    max_bits_per_slot: int = 2
    framing: str = "aes_siv"

    def __post_init__(self) -> None:
        if type(self.min_group_size) is not int or self.min_group_size < 1:
            raise CodecError("min_group_size must be a positive integer")
        if type(self.max_bits_per_slot) is not int or self.max_bits_per_slot not in (1, 2):
            raise CodecError("max_bits_per_slot must be 1 or 2")
        if self.framing not in ("aes_siv", "plain_length16"):
            raise CodecError("framing must be aes_siv or plain_length16")

    def to_dict(self) -> dict:
        return {"version": PROTOCOL_VERSION, "min_group_size": self.min_group_size,
                "max_bits_per_slot": self.max_bits_per_slot, "framing": self.framing}

    @property
    def frame_overhead_bits(self) -> int:
        return FRAME_OVERHEAD_BITS if self.framing == "aes_siv" else 16

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    def width(self, class_size: int) -> int:
        groups = class_size // self.min_group_size
        return min(self.max_bits_per_slot, max(0, groups.bit_length() - 1))


DEFAULT_PROTOCOL = Protocol()


@dataclass(frozen=True)
class Slot:
    index: int
    token_index: int
    start: int
    end: int
    surface: str
    class_id: str
    case_style: str
    width: int


@dataclass(frozen=True)
class TextAnalysis:
    slots: tuple[Slot, ...]
    normalization_sha256: str
    capacity_bits: int
    frame_overhead_bits: int

    @property
    def max_payload_bytes(self) -> int | None:
        if self.capacity_bits < self.frame_overhead_bits:
            return None
        return min(MAX_PAYLOAD_BYTES, (self.capacity_bits - self.frame_overhead_bits) // 8)


@dataclass(frozen=True)
class EncodeResult:
    stego_text: str
    payload_bytes: int
    capacity_bits: int
    frame_bits: int
    consumed_slots: int
    padding_bits: int
    changed_slots: int
    dictionary_sha256: str
    protocol_sha256: str
    normalization_sha256: str


def analyze_text(text: str, dictionary: PublicDictionary, protocol: Protocol = DEFAULT_PROTOCOL) -> TextAnalysis:
    """Locate invariant whole-word slots while preserving literal characters.

    Only lower/title/upper ASCII cores in whitespace-delimited tokens qualify.
    Surrounding Unicode-category-P punctuation except underscore is allowed;
    contractions, compounds, mixed case, other affixes do not qualify. These are shared rules, not
    a sender-side semantic filter.
    """
    if not isinstance(text, str):
        raise CodecError("text must be a string")
    if not isinstance(dictionary, PublicDictionary) or not isinstance(protocol, Protocol):
        raise CodecError("validated public dictionary and protocol required")
    lookup = {word: entry for entry in dictionary.classes for word in entry.words}
    slots: list[Slot] = []
    normalized: list[list[str]] = []
    cursor = 0
    for token_index, token_match in enumerate(re.finditer(r"\S+", text)):
        token = _TOKEN.fullmatch(token_match.group())
        if token is None:
            continue
        affixes = token.group("prefix") + token.group("suffix")
        if any(char == "_" or not unicodedata.category(char).startswith("P") for char in affixes):
            continue
        surface = token.group("word")
        entry = lookup.get(surface.lower())
        if entry is None:
            continue
        if surface.islower():
            case_style = "lower"
        elif surface.isupper():
            case_style = "upper"
        elif surface.istitle():
            case_style = "title"
        else:
            continue
        width = protocol.width(len(entry.words))
        if width == 0:
            continue
        start = token_match.start() + token.start("word")
        end = token_match.start() + token.end("word")
        normalized.append(["literal", text[cursor:start]])
        normalized.append(["class", entry.class_id, case_style])
        slots.append(Slot(len(slots), token_index, start, end, surface, entry.class_id, case_style, width))
        cursor = end
    normalized.append(["literal", text[cursor:]])
    return TextAnalysis(tuple(slots), _sha(normalized), sum(slot.width for slot in slots), protocol.frame_overhead_bits)


def _keys(key: bytes, dictionary_sha: str, protocol: Protocol) -> tuple[bytes, bytes]:
    if not isinstance(key, bytes) or len(key) != 32:
        raise CodecError("secret key must be exactly 32 bytes; use cryptographic randomness")
    salt = hashlib.sha256(b"synsteg/shared-dictionary/hkdf-salt/v1\x00" + bytes.fromhex(dictionary_sha)).digest()
    def derive(label: str, length: int) -> bytes:
        info = _canonical([PROTOCOL_VERSION, protocol.sha256, label])
        return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(key)
    return derive("aead-aes-siv-512", 64), derive("mapping-hmac-sha256", 32)


def _groups(slot: Slot, words: tuple[str, ...], mapping_key: bytes, dictionary_sha: str,
            protocol_sha: str, normalization_sha: str) -> tuple[tuple[str, ...], ...]:
    context = [PROTOCOL_VERSION, "word-rank", dictionary_sha, protocol_sha, normalization_sha,
               slot.index, slot.token_index, slot.class_id, slot.width]
    ranked = sorted(words, key=lambda word: (hmac.new(mapping_key, _canonical(context + [word]), hashlib.sha256).digest(), word))
    group_count = 1 << slot.width
    return tuple(tuple(ranked[label::group_count]) for label in range(group_count))


def _render_case(word: str, style: str) -> str:
    return word.upper() if style == "upper" else word.capitalize() if style == "title" else word


def _aad(dictionary_sha: str, protocol: Protocol, normalization_sha: str, payload_bytes: int) -> list[bytes]:
    return [b"synsteg/shared-dictionary/authenticated-frame/v1",
            _canonical({"version": PROTOCOL_VERSION, "dictionary_sha256": dictionary_sha,
                        "protocol": protocol.to_dict(), "normalization_sha256": normalization_sha,
                        "payload_bytes": payload_bytes})]


def encode(cover_text: str, payload: bytes, key: bytes, dictionary: PublicDictionary,
           protocol: Protocol = DEFAULT_PROTOCOL, *, selector: str = "canonical",
           candidate_cost: Callable[[Slot, str], float] | None = None) -> EncodeResult:
    """Embed a complete byte payload or raise CapacityError.

    ``quality`` chooses the minimum finite candidate_cost(slot, rendered_word)
    inside the assigned label group. It cannot skip/change public eligibility.
    The caller owns the meaning of that cost; this function makes no quality
    guarantee. AES-SIV is deterministic and exposes repeated-message equality
    under the same key and associated invariant text. It needs no nonce.
    The explicit ``plain_length16`` diagnostic provides neither encryption nor
    authentication, including of the invariant text, header or payload.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD_BYTES:
        raise CodecError("payload must be bytes with length at most 65535")
    if selector not in ("canonical", "quality"):
        raise CodecError("selector must be canonical or quality")
    if selector == "quality" and not callable(candidate_cost):
        raise CodecError("quality selector requires a candidate_cost callable")
    if selector == "canonical" and candidate_cost is not None:
        raise CodecError("canonical selector does not accept candidate_cost")
    analysis = analyze_text(cover_text, dictionary, protocol)
    dictionary_sha, protocol_sha = dictionary.sha256, protocol.sha256
    aead_key, mapping_key = _keys(key, dictionary_sha, protocol)
    frame_bits = protocol.frame_overhead_bits + 8 * len(payload)
    if analysis.capacity_bits < frame_bits:
        raise CapacityError(f"complete frame requires {frame_bits} bits; text has {analysis.capacity_bits}")
    ciphertext = (AESSIV(aead_key).encrypt(payload, _aad(dictionary_sha, protocol, analysis.normalization_sha256, len(payload)))
                  if protocol.framing == "aes_siv" else payload)
    frame = len(payload).to_bytes(2, "big") + ciphertext
    bits = "".join(f"{byte:08b}" for byte in frame)
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    fragments: list[str] = []
    cursor = bit_cursor = consumed_slots = changed_slots = padding_bits = 0
    for slot in analysis.slots:
        if bit_cursor >= len(bits):
            break
        slot_bits = bits[bit_cursor:bit_cursor + slot.width]
        padding_bits = slot.width - len(slot_bits)
        label = int(slot_bits.ljust(slot.width, "0"), 2)
        groups = _groups(slot, classes[slot.class_id], mapping_key, dictionary_sha, protocol_sha, analysis.normalization_sha256)
        candidates = sorted(_render_case(word, slot.case_style) for word in groups[label])
        if selector == "quality":
            scored = []
            for candidate in candidates:
                cost = candidate_cost(slot, candidate)
                if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
                    raise CodecError("candidate_cost must return finite numeric costs for every group member")
                scored.append((float(cost), candidate))
            chosen = min(scored)[1]
        else:
            chosen = candidates[0]
        fragments.extend([cover_text[cursor:slot.start], chosen])
        cursor = slot.end
        bit_cursor += slot.width
        consumed_slots += 1
        changed_slots += int(chosen != slot.surface)
    fragments.append(cover_text[cursor:])
    stego_text = "".join(fragments)
    check = analyze_text(stego_text, dictionary, protocol)
    if (check.normalization_sha256, check.capacity_bits) != (analysis.normalization_sha256, analysis.capacity_bits):
        raise CodecError("internal invariant violation during rendering")
    return EncodeResult(stego_text, len(payload), analysis.capacity_bits, frame_bits, consumed_slots,
                        padding_bits, changed_slots, dictionary_sha, protocol_sha, analysis.normalization_sha256)


def decode(stego_text: str, key: bytes, dictionary: PublicDictionary,
           protocol: Protocol = DEFAULT_PROTOCOL) -> bytes:
    """Recover bytes with only text, key, and installed public resources.

    The decoder never requires the cover, a document-specific dictionary,
    selector, externally supplied length, nonce or an EncodeResult. It returns
    no partial plaintext if framing or authentication fails. Explicit diagnostic
    plain_length16 mode has no authentication and may accept incorrect bytes.
    """
    analysis = analyze_text(stego_text, dictionary, protocol)
    dictionary_sha, protocol_sha = dictionary.sha256, protocol.sha256
    aead_key, mapping_key = _keys(key, dictionary_sha, protocol)
    if analysis.capacity_bits < protocol.frame_overhead_bits:
        raise DecodeError("text cannot contain a complete frame")
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    parts: list[str] = []
    observed_bits = 0
    frame_bits: int | None = None
    payload_bytes: int | None = None
    for slot in analysis.slots:
        groups = _groups(slot, classes[slot.class_id], mapping_key, dictionary_sha, protocol_sha, analysis.normalization_sha256)
        label = next(label for label, words in enumerate(groups) if slot.surface.lower() in words)
        parts.append(f"{label:0{slot.width}b}")
        observed_bits += slot.width
        if frame_bits is None and observed_bits >= 16:
            payload_bytes = int("".join(parts)[:16], 2)
            frame_bits = protocol.frame_overhead_bits + 8 * payload_bytes
            if frame_bits > analysis.capacity_bits:
                raise DecodeError("declared complete frame exceeds text capacity")
        if frame_bits is not None and observed_bits >= frame_bits:
            break
    if frame_bits is None or observed_bits < frame_bits:
        raise DecodeError("incomplete authenticated frame")
    bits = "".join(parts)
    if any(bit != "0" for bit in bits[frame_bits:]):
        raise DecodeError("nonzero final-slot padding")
    frame = bytes(int(bits[i:i + 8], 2) for i in range(0, frame_bits, 8))
    if protocol.framing == "plain_length16":
        return frame[2:]
    try:
        payload = AESSIV(aead_key).decrypt(frame[2:], _aad(dictionary_sha, protocol, analysis.normalization_sha256, payload_bytes))
    except InvalidTag as exc:
        raise AuthenticationError("authenticated frame verification failed") from exc
    if len(payload) != payload_bytes:
        raise DecodeError("authenticated payload length mismatch")
    return payload
