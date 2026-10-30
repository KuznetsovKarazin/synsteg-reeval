"""Experimental fixed-codebook codec; this module makes no security claim.

Candidate admission, bit groups and quality scores are fixed before a payload
is supplied. Only single English lexical words are admitted. This restriction
trades capacity for exact parsing and avoids the old multiword substitutions.
The 64-bit length/CRC frame detects accidental corruption, not an adversary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Callable, Iterable, Sequence
import zlib

VERSION = "fixed-codebook-v1"
FRAME_OVERHEAD_BITS = 64
_WORD = re.compile(r"[A-Za-z]+\Z")
_TOKEN = re.compile(r"([^A-Za-z0-9_]*)([A-Za-z]+)([^A-Za-z0-9_]*)\Z")


class CodecError(ValueError):
    """Malformed input, altered text, or an invalid frame."""


class CapacityError(CodecError):
    """The whole framed payload cannot fit; no prefix is emitted."""


@dataclass(frozen=True)
class CandidateContext:
    cover: str
    token_index: int
    original: str
    candidate: str
    candidate_text: str


@dataclass(frozen=True)
class Position:
    original: str
    groups: tuple[tuple[str, ...], ...]
    # Scores align with groups. Lower is better; each is cover-context-only.
    scores: tuple[tuple[float, ...], ...]

    @property
    def width(self) -> int:
        return (len(self.groups).bit_length() - 1) if self.groups else 0


@dataclass(frozen=True)
class FrozenCodebook:
    version: str
    cover_sha256: str
    seed: str
    max_bits_per_position: int
    min_group_size: int
    protect_capitalized: bool
    candidate_filter_id: str
    scorer_id: str
    separators: tuple[str, ...]
    positions: tuple[Position, ...]

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise CodecError("Unsupported codebook version")
        if type(self.seed) is not str or not isinstance(self.candidate_filter_id, str) or not isinstance(self.scorer_id, str):
            raise CodecError("Codebook identities must be strings")
        if type(self.max_bits_per_position) is not int or not 1 <= self.max_bits_per_position <= 16:
            raise CodecError("max_bits_per_position must be in [1, 16]")
        if type(self.min_group_size) is not int or self.min_group_size < 1:
            raise CodecError("min_group_size must be positive")
        if type(self.protect_capitalized) is not bool:
            raise CodecError("protect_capitalized must be boolean")
        if type(self.separators) is not tuple or type(self.positions) is not tuple:
            raise CodecError("Codebook collections must be immutable tuples")
        if len(self.separators) != len(self.positions) + 1:
            raise CodecError("Invalid separator count")
        if any(type(s) is not str or (s and not s.isspace()) for s in self.separators):
            raise CodecError("Separators must contain only whitespace")
        if any(not s for s in self.separators[1:-1]):
            raise CodecError("Adjacent positions require a separator")
        for position in self.positions:
            if not position.original or any(c.isspace() for c in position.original):
                raise CodecError("Invalid original token")
            n = len(position.groups)
            if type(position.groups) is not tuple or type(position.scores) is not tuple:
                raise CodecError("Position collections must be immutable tuples")
            if n and (n < 2 or n & (n - 1) or position.width > self.max_bits_per_position):
                raise CodecError("Groups must be a supported power of two")
            if len(position.scores) != n:
                raise CodecError("Missing group scores")
            all_forms = []
            for group, scores in zip(position.groups, position.scores):
                if type(group) is not tuple or type(scores) is not tuple:
                    raise CodecError("Groups and scores must be immutable tuples")
                if len(group) < self.min_group_size or len(group) != len(scores):
                    raise CodecError("Invalid group size or score count")
                if any(type(x) is not str or not x or any(c.isspace() for c in x) for x in group):
                    raise CodecError("Whitespace inside a candidate is not supported")
                if any(not math.isfinite(s) for s in scores):
                    raise CodecError("Candidate scores must be finite")
                original_match = _TOKEN.fullmatch(position.original)
                if original_match is None:
                    raise CodecError("A nonlexical original cannot carry bits")
                for form in group:
                    candidate_match = _TOKEN.fullmatch(form)
                    if candidate_match is None or (candidate_match.group(1), candidate_match.group(3)) != (original_match.group(1), original_match.group(3)):
                        raise CodecError("Candidates must preserve surrounding punctuation")
                all_forms.extend(group)
            if len(all_forms) != len(set(all_forms)):
                raise CodecError("Candidate collision across or within groups")
        if hashlib.sha256(self.cover.encode("utf-8")).hexdigest() != self.cover_sha256:
            raise CodecError("Codebook cover hash mismatch")

    @property
    def cover(self) -> str:
        return self.render(tuple(p.original for p in self.positions))

    def render(self, tokens: Sequence[str]) -> str:
        if len(tokens) != len(self.positions):
            raise CodecError("Wrong number of positions")
        return "".join(s + token for s, token in zip(self.separators, tokens)) + self.separators[-1]

    @property
    def capacity_bits(self) -> int:
        return sum(p.width for p in self.positions)

    @property
    def payload_capacity_bits(self) -> int:
        return max(0, self.capacity_bits - FRAME_OVERHEAD_BITS)

    @property
    def eligibility(self) -> dict[str, int]:
        return {
            "positions": len(self.positions),
            "eligible_positions": sum(bool(p.width) for p in self.positions),
            "groups": sum(len(p.groups) for p in self.positions),
            "singleton_groups": sum(len(g) == 1 for p in self.positions for g in p.groups),
            "groups_with_quality_choice": sum(len(g) > 1 for p in self.positions for g in p.groups),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict) -> "FrozenCodebook":
        values = dict(data)
        values["separators"] = tuple(values["separators"])
        values["positions"] = tuple(Position(p["original"], tuple(tuple(g) for g in p["groups"]),
                                             tuple(tuple(s) for s in p["scores"])) for p in values["positions"])
        return cls(**values)


@dataclass(frozen=True)
class EncodedMessage:
    text: str
    embedded_bits: int
    frame_bits: int
    padding_bits: int
    capacity_bits: int
    used_positions: int
    codebook_sha256: str
    selection: str
    framing: str
    external_metadata_bits: int


def _split(text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, int], ...]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    spans = tuple((m.start(), m.end()) for m in re.finditer(r"\S+", text))
    tokens = tuple(text[start:end] for start, end in spans)
    cuts = (0,) + tuple(end for _, end in spans)
    separators = tuple(text[cuts[i]:start] for i, (start, _) in enumerate(spans)) + (text[cuts[-1]:],)
    return tokens, separators, spans


def freeze_codebook(
    cover: str,
    candidates: Sequence[Iterable[str]],
    *,
    seed: str = "0",
    max_bits_per_position: int = 1,
    min_group_size: int = 2,
    protect_capitalized: bool = True,
    candidate_filter: Callable[[CandidateContext], bool] | None = None,
    candidate_filter_id: str = "none",
    scorer: Callable[[CandidateContext], float] | None = None,
    scorer_id: str = "none",
) -> FrozenCodebook:
    """Freeze admitted words, seeded groups and scores without seeing payload.

    ``candidates`` follows ``cover.split()`` positions. The original word is
    included automatically. Candidates must be ASCII alphabetic words; phrases,
    hyphenations and apostrophes are discarded. Case-normalized duplicates are
    removed before assigning codes. Capitalized source tokens are protected by
    default as a conservative entity heuristic, not a complete entity detector.

    The optional filter must independently enforce context/meaning constraints.
    No semantic equivalence is inferred here. The scorer is called once per
    admitted form on a single replacement in the unchanged cover. It never sees
    the payload or a progressively encoded text. Lower scores are preferred.
    """
    tokens, separators, spans = _split(cover)
    if not tokens:
        raise CodecError("Cover must contain a token")
    if len(candidates) != len(tokens):
        raise CodecError("Candidate lists must align exactly with cover positions")
    if type(seed) is not str:
        raise CodecError("seed must be a string")
    if type(max_bits_per_position) is not int or not 1 <= max_bits_per_position <= 16:
        raise CodecError("max_bits_per_position must be in [1, 16]")
    if type(min_group_size) is not int or min_group_size < 1:
        raise CodecError("min_group_size must be positive")
    if candidate_filter is not None and candidate_filter_id in ("", "none"):
        raise CodecError("A supplied filter requires a recorded identity")
    if scorer is not None and scorer_id in ("", "none"):
        raise CodecError("A supplied scorer requires a recorded identity")
    cover_hash = hashlib.sha256(cover.encode("utf-8")).hexdigest()
    positions = []
    for index, (token, options, (start, end)) in enumerate(zip(tokens, candidates, spans)):
        match = _TOKEN.fullmatch(token)
        if not match:
            positions.append(Position(token, (), ()))
            continue
        prefix, core, suffix = match.groups()
        # An interior apostrophe/hyphen would otherwise be swallowed as punctuation.
        if any(c.isalpha() for c in prefix + suffix) or (protect_capitalized and not core.islower()):
            positions.append(Position(token, (), ()))
            continue
        if not (core.islower() or core.isupper() or core.istitle()):
            positions.append(Position(token, (), ()))
            continue
        if isinstance(options, str):
            raise CodecError("Each position requires a list of candidate words, not a string")
        raw = list(options)
        if any(not isinstance(x, str) for x in raw):
            raise CodecError("Candidate words must be strings")
        normalized = sorted({x.lower() for x in [core, *raw] if _WORD.fullmatch(x)})
        admitted: dict[str, CandidateContext] = {}
        for word in normalized:
            surface = word.upper() if core.isupper() else word.capitalize() if core.istitle() else word
            rendered = prefix + surface + suffix
            context = CandidateContext(cover, index, core, surface, cover[:start] + rendered + cover[end:])
            if candidate_filter is None or candidate_filter(context):
                admitted[rendered] = context
        width = min(max_bits_per_position, max(0, (len(admitted) // min_group_size).bit_length() - 1))
        if width == 0:
            positions.append(Position(token, (), ()))
            continue
        def seeded_order(form: str) -> tuple[str, str]:
            value = json.dumps([VERSION, seed, cover_hash, index, form], separators=(",", ":"))
            return hashlib.sha256(value.encode("utf-8")).hexdigest(), form
        ordered = sorted(admitted, key=seeded_order)
        groups = tuple(tuple(ordered[group::2**width]) for group in range(2**width))
        scores = []
        for group in groups:
            group_scores = tuple(float(scorer(admitted[x])) if scorer else 0.0 for x in group)
            if any(not math.isfinite(x) for x in group_scores):
                raise CodecError("Scorer returned a non-finite value")
            scores.append(group_scores)
        positions.append(Position(token, groups, tuple(scores)))
    return FrozenCodebook(VERSION, cover_hash, seed, max_bits_per_position, min_group_size,
                          protect_capitalized, candidate_filter_id, scorer_id, separators, tuple(positions))


def candidates_from_legacy_key(cover: str, secret_key: Sequence[dict]) -> list[list[str]]:
    """Extract archived candidate values, never their payload-dependent codes.

    Intended for development diagnostics only: historical generation itself may
    have seen the message. Calling this adapter cannot certify that old inputs
    were generated independently of a payload.
    """
    tokens = cover.split()
    if len(tokens) != len(secret_key):
        raise CodecError("Legacy key does not align with cover token count")
    result = []
    for token, entry in zip(tokens, secret_key):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise CodecError("Legacy positions require one original-word mapping")
        original, mapping = next(iter(entry.items()))
        match = _TOKEN.fullmatch(token)
        core = match.group(2) if match else token
        # Hyphenated/numeric tokens remain literal in the new codec, but the
        # legacy key legitimately strips their surrounding punctuation.
        punctuation_stripped = re.sub(r"^\W+|\W+$", "", token)
        if original.casefold() not in (core.casefold(), token.casefold(), punctuation_stripped.casefold()):
            raise CodecError("Legacy original word does not match the aligned cover")
        if not isinstance(mapping, dict) or any(not isinstance(x, str) for x in mapping.values()):
            raise CodecError("Invalid legacy candidate mapping")
        result.append(list(mapping.values()))
    return result


def _validate_bits(bits: str) -> None:
    if not isinstance(bits, str) or set(bits) - {"0", "1"}:
        raise CodecError("Payload must be a binary string")
    if len(bits) >= 2**32:
        raise CodecError("Payload length exceeds the 32-bit frame")


def _checksum(bits: str) -> str:
    # Include exact bit length so trailing zero bits are not byte-padding aliases.
    padded = bits + "0" * (-len(bits) % 8)
    packed = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
    crc = zlib.crc32(len(bits).to_bytes(4, "big") + packed)
    return f"{crc:032b}"


def encode_bits(codebook: FrozenCodebook, payload_bits: str, *, selection: str = "quality",
                framing: str = "length_crc32") -> EncodedMessage:
    """Encode completely without mutating the key or calling a scorer.

    Default framing adds 32 length bits and a 32-bit CRC inside the text.
    ``external_length`` sends only payload bits but requires a separate 32-bit
    length field supplied to the decoder as ``expected_bits``. This explicit
    metadata is additional to the document-specific codebook; it is not free.
    External-length mode provides no CRC or other integrity check.
    """
    _validate_bits(payload_bits)
    if selection not in ("canonical", "quality"):
        raise CodecError("selection must be canonical or quality")
    if framing not in ("length_crc32", "external_length"):
        raise CodecError("Unsupported framing mode")
    frame = (f"{len(payload_bits):032b}" + payload_bits + _checksum(payload_bits)
             if framing == "length_crc32" else payload_bits)
    if len(frame) > codebook.capacity_bits:
        raise CapacityError(f"Need {len(frame)} embedded bits ({len(payload_bits)} payload, {framing}); "
                            f"capacity is {codebook.capacity_bits}")
    tokens = []
    offset = used_positions = padding = 0
    for position in codebook.positions:
        width = position.width
        if not width or offset >= len(frame):
            tokens.append(position.original)
            continue
        chunk = frame[offset:offset + width]
        padding = width - len(chunk)
        group_index = int(chunk.ljust(width, "0"), 2)
        group = position.groups[group_index]
        if selection == "canonical":
            chosen = group[0]
        else:
            # Stable tie-break: retaining the original is preferred when admitted.
            chosen = min(zip(group, position.scores[group_index]),
                         key=lambda x: (x[1], x[0] != position.original, x[0]))[0]
        tokens.append(chosen)
        offset += width
        used_positions += 1
    return EncodedMessage(codebook.render(tokens), len(payload_bits), len(frame), padding,
                          codebook.capacity_bits, used_positions, codebook.sha256, selection,
                          framing, 32 if framing == "external_length" else 0)


def decode_bits(codebook: FrozenCodebook, stego_text: str, *, framing: str = "length_crc32",
                expected_bits: int | None = None) -> str:
    """Parse exact positions, length and CRC; reject altered separators or tail."""
    if framing not in ("length_crc32", "external_length"):
        raise CodecError("Unsupported framing mode")
    if framing == "external_length":
        if type(expected_bits) is not int or not 0 <= expected_bits < 2**32:
            raise CodecError("External-length mode needs a separate nonnegative 32-bit expected_bits field")
        if expected_bits > codebook.capacity_bits:
            raise CodecError("External payload length exceeds capacity")
    elif expected_bits is not None:
        raise CodecError("Default framing reads its own length; do not supply expected_bits")
    tokens, separators, _ = _split(stego_text)
    if len(tokens) != len(codebook.positions) or separators != codebook.separators:
        raise CodecError("Text layout differs from the frozen codebook")
    stream = ""
    expected = expected_bits if framing == "external_length" else None
    for token, position in zip(tokens, codebook.positions):
        if not position.width or (expected is not None and len(stream) >= expected):
            if token != position.original:
                raise CodecError("A literal or unused tail position was modified")
            continue
        inverse = {form: f"{i:0{position.width}b}" for i, group in enumerate(position.groups) for form in group}
        if token not in inverse:
            raise CodecError("Unknown candidate at a framed position")
        stream += inverse[token]
        if framing == "length_crc32" and expected is None and len(stream) >= 32:
            expected = int(stream[:32], 2) + FRAME_OVERHEAD_BITS
            if expected > codebook.capacity_bits:
                raise CodecError("Frame declares a payload larger than capacity")
    if expected is None or len(stream) < expected:
        raise CodecError("Incomplete frame")
    if set(stream[expected:]) - {"0"}:
        raise CodecError("Nonzero final padding bits")
    if framing == "external_length":
        return stream[:expected]
    payload = stream[32:expected - 32]
    if stream[expected - 32:expected] != _checksum(payload):
        raise CodecError("Frame checksum mismatch")
    return payload
