"""Durable token write-ahead journal around the frozen fixed512 endpoint API.

No model/process calls occur at import. A prepared native step without a durable
commit is UNKNOWN and is never retried. The committed journal, not a potentially
lagging convenience raw file, defines the recoverable output prefix.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .canonical_channel_v2 import CanonicalGate, ContractError
from .framed_interval_llm_v1 import BackendFailure, _trace, canonical_json, derive_keys, ordered_law
from .integer_interval_channel_v1 import Config, Receiver, advance


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".writing")
    with temp.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(temp, path)
    else:
        os.link(temp, path)  # Atomic no-clobber publication, including concurrent creation.
        temp.unlink()
    fsync_directory(path.parent)


class TokenJournal:
    def __init__(self, directory):
        directory = Path(directory)
        # The launcher creates/fsyncs the complete ancestor chain first.
        if not directory.parent.is_dir():
            raise ValueError("journal parent must already exist durably")
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)
        self.path = directory / "token-journal.jsonl"
        self.raw_path = directory / "wire.raw"
        self.stream = self.path.open("xb")
        try:
            self.raw_stream = self.raw_path.open("xb")
            fsync_directory(directory)
        except Exception:
            self.stream.close()
            raise
        self.previous, self.sequence = "0" * 64, 0
        self.ids, self.raw, self.pending = (), b"", None
        self.terminal_unknown = False

    def _append(self, event):
        body = {"sequence": self.sequence, "previous_sha256": self.previous, **event}
        digest = sha(canonical_json(body))
        data = canonical_json({**body, "event_sha256": digest}) + b"\n"
        self.stream.write(data); self.stream.flush(); os.fsync(self.stream.fileno())
        self.previous = digest; self.sequence += 1
        return digest

    def prepare(self, token, suffix, trace):
        if (self.terminal_unknown or self.pending is not None or type(token) is not int or token < 0
                or type(suffix) is not bytes or not suffix):
            raise ContractError("invalid pending journal step")
        if trace["step"] != len(self.ids) or trace["prefix_ids"] != list(self.ids) or trace["observed_token"] != token:
            raise ContractError("journal trace differs from committed history")
        digest = self._append({"kind": "prepared", "token_index": len(self.ids), "token_id": token,
            "wire_prefix_sha256": sha(self.raw), "suffix_hex": suffix.hex(), "trace": trace})
        self.pending = (digest, token, suffix)

    def model_error(self, error):
        if self.pending is None or self.terminal_unknown:
            raise ContractError("terminal native error requires one pending step")
        self.terminal_unknown = True
        self._append({"kind": "native_step_error", "token_index": len(self.ids),
                      "prepared_sha256": None if self.pending is None else self.pending[0],
                      "error_type": type(error).__name__, "error": str(error)[:512],
                      "commit_state": "unknown; never retry"})

    def commit(self):
        if self.terminal_unknown or self.pending is None:
            raise ContractError("journal has no prepared native step")
        digest, token, suffix = self.pending
        combined = self.raw + suffix
        self._append({"kind": "committed", "token_index": len(self.ids), "token_id": token,
                      "prepared_sha256": digest, "wire_after_sha256": sha(combined)})
        self.ids += (token,); self.raw = combined; self.pending = None
        # The journal is authoritative if this convenience file lags after a crash.
        self.raw_stream.write(suffix); self.raw_stream.flush(); os.fsync(self.raw_stream.fileno())

    def close(self):
        self.stream.close(); self.raw_stream.close()


def recover_journal(path):
    """Read only. Torn last record is retained as unknown; interior corruption fails."""
    previous, ids, raw, pending, traces = "0"*64, [], b"", None, []
    lines = Path(path).read_bytes().splitlines(keepends=True)
    torn, terminal_unknown = False, False
    for sequence, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if sequence != len(lines)-1:
                raise ValueError("interior torn journal record")
            torn = True; break
        row = json.loads(line)
        digest = row.pop("event_sha256")
        if (row.get("sequence") != sequence or row.get("previous_sha256") != previous
                or sha(canonical_json(row)) != digest):
            raise ValueError("journal sequence/hash chain differs")
        previous = digest
        kind = row.get("kind")
        if terminal_unknown:
            raise ValueError("journal continued after terminal unknown native step")
        if kind == "prepared":
            token, suffix, trace = row["token_id"], bytes.fromhex(row["suffix_hex"]), row["trace"]
            if (pending is not None or type(token) is not int or token < 0 or not suffix
                    or row["token_index"] != len(ids) or row["wire_prefix_sha256"] != sha(raw)
                    or trace["step"] != len(ids) or trace["prefix_ids"] != ids or trace["observed_token"] != token):
                raise ValueError("prepared record does not extend committed prefix")
            pending = (digest, token, suffix, trace)
        elif kind == "committed":
            if (pending is None or row["prepared_sha256"] != pending[0]
                    or row["token_id"] != pending[1] or row["token_index"] != len(ids)
                    or row["wire_after_sha256"] != sha(raw+pending[2])):
                raise ValueError("commit does not match prepared native step")
            ids.append(pending[1]); raw += pending[2]; traces.append(pending[3]); pending = None
        elif kind == "native_step_error":
            if pending is None or row["prepared_sha256"] != pending[0] or row["token_index"] != len(ids):
                raise ValueError("native step error lacks matching prepared record")
            terminal_unknown = True
        else:
            raise ValueError("unknown journal event kind")
    return {"committed_token_ids": ids, "committed_wire": raw, "committed_traces": traces,
            "last_event_sha256": previous, "unknown_native_step": terminal_unknown or pending is not None or torn,
            "torn_last_record": torn, "retry_permitted": False}


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
    def __init__(self, base, envelope, key, directory):
        self.base = base
        for field in ("resource_id", "resource_digest", "provider_digest", "context_tokens"):
            setattr(self, field, getattr(base, field))
        self.gate = _GateTap(base.gate)
        self.envelope = json.loads(canonical_json(envelope))
        self.order_key = derive_keys(key, envelope)[1]
        self.config = Config(**envelope["interval_config"])
        self.shadow = Receiver(self.config)
        self.journal = TokenJournal(directory)
        self.last_distribution = None
        self.terminal_failure = False

    def begin(self, prompt_ids):
        self.base.begin(prompt_ids)

    def distribution(self, prefix):
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
        except Exception as exc:
            self.terminal_failure = True
            raise BackendFailure("journal_commit:"+type(exc).__name__) from exc
        self.gate.tentative = None

    def close(self):
        self.journal.close()


def result_record(result):
    row = asdict(result)
    raw = row.pop("wire", None)
    if raw is not None:
        row["wire_bytes"], row["wire_sha256"] = len(raw), sha(raw)
    if "payload" in row:
        payload = row.pop("payload")
        row["payload_hex"] = None if payload is None else payload.hex()
    return row
