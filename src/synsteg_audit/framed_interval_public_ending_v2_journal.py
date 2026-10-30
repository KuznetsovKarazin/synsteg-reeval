"""Ending-v2 durable observer. No model or file access at import.

The byte/hash-chain journal primitives are reused unchanged. All cryptographic
and interval-law derivations below are the NEW ending-v2 derivations, never
the fixed512 derivation. A prepared native STEP without commit stays UNKNOWN.
"""
import json

from .canonical_channel_v2 import CanonicalGate, ContractError
from .framed_interval_public_ending_v2 import (
    BackendFailure, THRESHOLD_BITS, TOKEN_CAP, MAX_TAIL_TOKENS, terminal_punctuation, canonical_json, derive_keys,
    ordered_law, validate_bindings,
)
from .framed_interval_llm_v1 import _trace
from .framed_interval_llm_v1_journal import (
    TokenJournal, atomic_json, fsync_directory, recover_journal, result_record, sha,
)
from .integer_interval_channel_v1 import Receiver, advance

class _GateTap(CanonicalGate):
    def __init__(self, base):
        self.base, self.codec, self.tentative = base, base.codec, None

    def validate(self, prefix):
        return self.base.validate(prefix)

    def admit(self, prefix, candidates):
        return self.base.admit(prefix, candidates)

    def take(self, prefix, decision, token):
        advanced = self.base.take(prefix, decision, token)
        self.tentative = (prefix, decision, token, advanced)
        return advanced


class DurableEndpoint:
    """Observe public interval transitions without changing sampling or forwarding."""
    def __init__(self, base, envelope, key, directory, *, public_spec, public_context):
        self.base = base
        for field in ("resource_id", "resource_digest", "provider_digest", "context_tokens"):
            setattr(self, field, getattr(base, field))
        self.gate = _GateTap(base.gate)
        self.envelope = json.loads(canonical_json(envelope))
        self.order_key = derive_keys(key, envelope)[1]
        self.config = validate_bindings(envelope, base, public_spec=public_spec, public_context=public_context)
        self.public_context = json.loads(canonical_json(public_context))
        self.shadow = Receiver(self.config)
        self.journal = TokenJournal(directory)
        self.last_distribution = None
        self.terminal_failure = False
        self.first_threshold_token = None

    def begin(self, prompt_ids):
        if list(prompt_ids) != self.public_context["prompt_ids"]:
            raise ContractError("prompt differs from bound envelope")
        self.base.begin(prompt_ids)

    def distribution(self, prefix):
        if self.terminal_failure:
            raise ContractError("terminal endpoint must never request another distribution")
        if (len(self.journal.ids) >= TOKEN_CAP or
            (self.first_threshold_token is not None and
             (terminal_punctuation(self.journal.raw) or len(self.journal.ids)-self.first_threshold_token >= MAX_TAIL_TOKENS))):
            raise ContractError("public message-independent stop already reached")
        self.last_distribution = self.base.distribution(prefix)
        return self.last_distribution

    def observe(self, token):
        if self.terminal_failure or self.gate.tentative is None or self.last_distribution is None:
            raise ContractError("durable observer lacks gate transition")
        prefix, decision, selected, advanced = self.gate.tentative
        if selected != token or prefix.ids != self.journal.ids:
            raise ContractError("durable observation differs from committed IDs")
        law = ordered_law(decision, self.order_key, self.envelope, prefix)
        step = advance(self.shadow.state, law, token, self.config)
        trace = _trace(prefix, decision, self.last_distribution, law, step, self.config)
        try:
            self.journal.prepare(token, advanced.raw[len(prefix.raw):], trace)
        except Exception as exc:
            self.terminal_failure = True
            raise BackendFailure("journal_prepare:"+type(exc).__name__) from exc
        try:
            self.base.observe(token)
        except Exception as exc:
            self.terminal_failure = True
            try: self.journal.model_error(exc)
            except Exception: pass  # Prepared record remains authoritative UNKNOWN.
            raise
        try:
            self.journal.commit()
            self.shadow.take(law, token)
            if self.first_threshold_token is None and len(self.shadow.recovered_bits) >= THRESHOLD_BITS:
                self.first_threshold_token = len(self.journal.ids)
        except Exception as exc:
            self.terminal_failure = True
            raise BackendFailure("journal_commit:"+type(exc).__name__) from exc
        self.gate.tentative = None

    def close(self):
        self.journal.close()

