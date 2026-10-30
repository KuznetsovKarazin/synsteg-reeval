"""Public fake byte backend for unit checks; never a neural probability model."""
import hashlib

from .canonical_channel_v2 import CanonicalGate, PublicCodec
from .framed_interval_llm_v1 import Distribution, Endpoint

PROMPT_IDS = (10000, 10001)
CODEC_DIGEST = hashlib.sha256(b"fixed512 synthetic byte tokenizer-v1").hexdigest()


def fake_endpoint(profile="uniform16", *, observe_callback=None, start_callback=None,
                  distribution_callback=None):
    if profile not in {"uniform16", "uniform8", "adaptive4", "singleton", "empty_support", "empty_input", "partial_utf8"}:
        raise ValueError("unknown synthetic profile")
    codec = PublicCodec(lambda text: list(text.encode("utf-8")), lambda ids: bytes(ids),
                        frozenset({999}), CODEC_DIGEST)
    def distribution(model_prefix):
        if tuple(model_prefix[:len(PROMPT_IDS)]) != PROMPT_IDS:
            raise ValueError("synthetic public prompt changed")
        output = model_prefix[len(PROMPT_IDS):]
        if profile == "uniform16": rows = tuple((t, 1) for t in range(65, 81))
        elif profile == "uniform8": rows = tuple((t, 1) for t in range(65, 73))
        elif profile == "adaptive4":
            shift = (len(output)+sum(output)) % 4
            rows = tuple((65+i, (i+shift) % 4+1) for i in range(4))
        elif profile == "singleton": rows = ((65, 1),)
        elif profile == "empty_support": rows = ((999, 1),)
        elif profile == "empty_input": rows = ((65, 0),)
        else: rows = ((195, 1),)
        return Distribution(rows, {"source": "public-synthetic-law", "profile": profile,
                                   "full_lm_mass": None, "neural_calls": 0})
    ident = "fixed512-fake-"+profile
    return Endpoint(CanonicalGate(codec), resource_id=ident,
        resource_digest=hashlib.sha256((ident+":resource").encode()).hexdigest(),
        provider_digest=hashlib.sha256((ident+":provider").encode()).hexdigest(),
        distribution_callback=distribution_callback or distribution,
        start_callback=start_callback or (lambda ids: None),
        observe_callback=observe_callback or (lambda token: None))
