"""Public finite codecs for synthetic delivery tests, never natural-language models."""
from __future__ import annotations

import hashlib

from .canonical_channel_v2 import ContractError, PublicCodec
from .canonical_packet_v3 import CanonicalPacketGate, ConstructionRefusal
from .framed_interval_packet_v1 import PublicBundle, canonical_json

PROFILES = ("silent16", "adaptive4", "singleton", "empty_support", "empty_input", "construction_refusal")
PROMPT_IDS = (9000, 9001)
SPEC = {
    "schema": "framed-interval-packet-v1-public-finite-resources",
    "alphabet": "ABCDEFGHIJKLMNOP", "silent_token": 0, "visible_ids": list(range(1, 17)),
    "special_ids": [99], "canonical_encoding": "each character Xi becomes token pair (0,i)",
    "render": "silent token0 emits nothing; IDs1..16 emit their ASCII letter; invalid structure may render but fails canonical gate",
    "max_packet_tokens": 2, "prompt_ids": list(PROMPT_IDS),
    "profiles": {
        "silent16": "all16 packets(0,i) with integerweight1",
        "adaptive4": "packets(0,1)..(0,4); baseweights[1,2,3,4] rotated by sum(previousvisibleIDs) modulo4",
        "singleton": "only packet(0,1) weight1",
        "empty_support": "only forbidden-special packet(99,) weight1; gate refuses",
        "empty_input": "no candidates",
        "construction_refusal": "public provider raises ConstructionRefusal before returning candidates",
    },
    "is_language_model": False,
}


def public_bundle(profile: str) -> PublicBundle:
    if profile not in PROFILES:
        raise ContractError("unknown public synthetic resource profile")
    alphabet = SPEC["alphabet"]
    def encode(text):
        result = []
        for char in text:
            if char not in alphabet:
                raise ValueError("wire contains a character outside the finite alphabet")
            result.extend((0, alphabet.index(char) + 1))
        return result
    def render(ids):
        out = []
        for token in ids:
            if token == 0:
                continue
            if 1 <= token <= 16:
                out.append(alphabet[token - 1])
            elif token == 99:
                out.append("!")
            else:
                raise ValueError("unknown finite token")
        return "".join(out).encode("ascii")
    resource_digest = hashlib.sha256(canonical_json(SPEC)).hexdigest()
    codec_spec = {k: SPEC[k] for k in ("alphabet", "silent_token", "visible_ids", "special_ids", "canonical_encoding", "render")}
    codec_digest = hashlib.sha256(canonical_json(codec_spec)).hexdigest()
    provider_spec = {"profile": profile, "description": SPEC["profiles"][profile], "prompt_ids": list(PROMPT_IDS)}
    provider_digest = hashlib.sha256(canonical_json(provider_spec)).hexdigest()
    policy_digest = hashlib.sha256(canonical_json({"gate": "canonical-packet-v3", "max_packet_tokens": 2})).hexdigest()
    codec = PublicCodec(encode, render, frozenset({99}), codec_digest)
    gate = CanonicalPacketGate(codec, max_packet_tokens=2, policy_digest=policy_digest)
    def provider(context):
        if tuple(context[:len(PROMPT_IDS)]) != PROMPT_IDS:
            raise ContractError("synthetic public prompt mismatch")
        output = tuple(context[len(PROMPT_IDS):])
        if len(output) % 2 or any(output[i] != 0 or not 1 <= output[i + 1] <= 16
                                 for i in range(0, len(output), 2)):
            raise ContractError("provider called outside a canonical output boundary")
        if profile == "silent16":
            return tuple(((0, i), 1) for i in range(1, 17))
        if profile == "adaptive4":
            shift = sum(output[1::2]) % 4
            return tuple(((0, i + 1), (i + shift) % 4 + 1) for i in range(4))
        if profile == "singleton":
            return (((0, 1), 1),)
        if profile == "empty_support":
            return (((99,), 1),)
        if profile == "empty_input":
            return ()
        raise ConstructionRefusal("predefined_synthetic_construction_budget_refusal")
    return PublicBundle(gate, provider, profile, resource_digest, provider_digest)
