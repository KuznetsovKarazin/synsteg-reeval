"""Durable candidate evidence gate; no model, tokenizer or provider imports.

New ID, not a repair of candidate-native-v1. A complete trace is an evidence
property, not a shape/semantic acceptance. Consumers validate format only on
the bytes returned by verify_bundle(), using an externally pinned terminal SHA.
Locks are advisory. Hashes detect change relative to pins; they do not make an
untrusted filesystem immutable or prove that an actual model executed a STEP.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ID = "candidate-journal-guard-v2"
ZERO = "0" * 64
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MEMBERS = ("writer.lock", "binding.json", "journal.jsonl", "response.raw", "result.json")


class EvidenceError(ValueError):
    """Evidence is incomplete, changed, inconsistent or outside the contract."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _need(ok, message):
    if not ok:
        raise EvidenceError(message)


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _integer(value, maximum=2**31 - 1):
    return type(value) is int and 0 <= value <= maximum


def _closed(value, keys):
    _need(type(value) is dict and set(value) == set(keys), "closed schema differs")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise EvidenceError("non-finite JSON value")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        _need(canonical(value) == raw, "noncanonical JSON")
        return value
    except (UnicodeError, RecursionError, TypeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid JSON evidence") from exc


def validate_binding(binding):
    _closed(binding, ("experiment_id", "attempt_id", "request_sha256",
                      "resource_sha256", "protocol_sha256", "token_cap"))
    for key in ("experiment_id", "attempt_id"):
        _need(type(binding[key]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", binding[key]),
              "invalid binding name")
    for key in ("request_sha256", "resource_sha256", "protocol_sha256"):
        _need(_hash(binding[key]), "invalid binding digest")
    _need(_integer(binding["token_cap"], 65536) and binding["token_cap"] > 0,
          "invalid token cap")
    return json.loads(canonical(binding))


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd, raw):
    view = memoryview(raw)
    while view:
        n = os.write(fd, view)
        _need(n > 0, "short write made no progress")
        view = view[n:]


def _exclusive(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync_dir(Path(path).parent)


def _identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _read_stable(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
              "non-regular or oversized artifact")
        blocks = []
        while block := os.read(fd, 1024 * 1024):
            blocks.append(block)
            _need(sum(map(len, blocks)) <= MAX_FILE_BYTES, "growing artifact")
        after = os.fstat(fd)
        _need(_identity(before) == _identity(after) == _identity(os.lstat(path)),
              "artifact changed during read")
        return b"".join(blocks)
    finally:
        os.close(fd)


def _state():
    return {"rows": 0, "head": ZERO, "opened": False, "prompt_ids": [],
            "prompt_committed": False, "ids": [], "raw": b"", "pending": None,
            "terminal": False, "stop_label": None}


def _advance(state, row, binding):
    _need(not state["terminal"], "event after terminal")
    _closed(row, ("sequence", "previous_sha256", "event", "event_sha256"))
    _need(type(row["sequence"]) is int and row["sequence"] == state["rows"], "sequence differs")
    _need(row["previous_sha256"] == state["head"], "chain previous differs")
    body = {k: row[k] for k in ("sequence", "previous_sha256", "event")}
    _need(_hash(row["event_sha256"]) and sha(canonical(body)) == row["event_sha256"], "chain digest differs")
    event = row["event"]
    _need(type(event) is dict and type(event.get("kind")) is str, "event schema")
    kind = event["kind"]
    if kind == "opened":
        _closed(event, ("kind", "binding_sha256"))
        _need(not state["opened"] and state["rows"] == 0, "duplicate opened")
        _need(event["binding_sha256"] == sha(canonical(binding)), "binding differs")
        state["opened"] = True
    elif kind == "prompt_prepared":
        _closed(event, ("kind", "prompt_ids"))
        ids = event["prompt_ids"]
        _need(state["opened"] and not state["prompt_ids"] and state["pending"] is None,
              "prompt out of order")
        _need(type(ids) is list and 0 < len(ids) <= 65536 and all(_integer(i) for i in ids), "prompt IDs")
        state["prompt_ids"] = list(ids)
        state["pending"] = (kind, row["event_sha256"], None)
    elif kind == "prompt_committed":
        _closed(event, ("kind", "prepared_sha256", "evaluated_tokens"))
        _need(state["pending"] is not None and state["pending"][:2] ==
              ("prompt_prepared", event["prepared_sha256"]), "prompt acknowledgement differs")
        _need(type(event["evaluated_tokens"]) is int and event["evaluated_tokens"] == len(state["prompt_ids"]),
              "prompt counter differs")
        state["prompt_committed"] = True
        state["pending"] = None
    elif kind == "step_prepared":
        _closed(event, ("kind", "token_index", "token_id", "intended_raw_hex", "sample_evidence"))
        _need(state["prompt_committed"] and state["pending"] is None, "STEP before prompt or pending acknowledgement")
        _need(type(event["token_index"]) is int and event["token_index"] == len(state["ids"])
              and len(state["ids"]) < binding["token_cap"], "STEP index/cap differs")
        _need(_integer(event["token_id"]), "invalid token ID")
        raw_hex = event["intended_raw_hex"]
        _need(type(raw_hex) is str and len(raw_hex) <= 2 * MAX_RESPONSE_BYTES
              and re.fullmatch(r"(?:[0-9a-f]{2})*", raw_hex) is not None, "invalid rendered bytes")
        _need(type(event["sample_evidence"]) is dict, "sample evidence must be a JSON object")
        state["pending"] = (kind, row["event_sha256"], (event["token_id"], bytes.fromhex(raw_hex)))
    elif kind == "step_committed":
        _closed(event, ("kind", "prepared_sha256", "evaluated_tokens"))
        _need(state["pending"] is not None and state["pending"][:2] ==
              ("step_prepared", event["prepared_sha256"]), "STEP acknowledgement differs")
        _need(type(event["evaluated_tokens"]) is int and event["evaluated_tokens"] ==
              len(state["prompt_ids"]) + len(state["ids"]) + 1, "STEP counter differs")
        token, raw = state["pending"][2]
        state["ids"].append(token)
        state["raw"] = raw
        state["pending"] = None
    elif kind == "trace_complete":
        _closed(event, ("kind", "output_ids", "response_sha256", "response_bytes", "stop_label"))
        _need(state["prompt_committed"] and state["pending"] is None and state["ids"],
              "cannot complete an unacknowledged/empty trace")
        _need(type(event["output_ids"]) is list and all(_integer(t) for t in event["output_ids"])
              and event["output_ids"] == state["ids"], "terminal IDs differ")
        _need(type(event["response_bytes"]) is int and event["response_bytes"] == len(state["raw"])
              and event["response_sha256"] == sha(state["raw"]), "terminal response differs")
        _need(type(event["stop_label"]) is str and re.fullmatch(r"[a-z0-9_-]{1,64}", event["stop_label"]),
              "invalid stop label")
        state["terminal"] = True
        state["stop_label"] = event["stop_label"]
    else:
        raise EvidenceError("unknown event")
    state["rows"] += 1
    state["head"] = row["event_sha256"]


def verify_trace(raw, binding, *, require_terminal=True):
    binding = validate_binding(binding)
    _need(type(raw) is bytes and raw and len(raw) <= MAX_FILE_BYTES and raw.endswith(b"\n"),
          "missing/torn/oversized journal")
    state = _state()
    for line in raw.splitlines():
        _advance(state, strict_json(line), binding)
    if require_terminal:
        _need(state["terminal"] and state["pending"] is None, "incomplete trace: evidence UNKNOWN")
    return state


def _result(trace, journal, binding):
    return {"schema": ID + "-result", "status": "trace_complete",
            "binding_sha256": sha(canonical(binding)), "output_ids": trace["ids"],
            "committed_output_tokens": len(trace["ids"]), "prompt_tokens": len(trace["prompt_ids"]),
            "response_bytes": len(trace["raw"]), "response_sha256": sha(trace["raw"]),
            "journal_bytes": len(journal), "journal_sha256": sha(journal),
            "journal_rows": trace["rows"], "journal_head_sha256": trace["head"],
            "stop_label": trace["stop_label"], "shape_or_semantics_accepted": False}


def _verify_members(directory, expected_binding):
    binding = validate_binding(expected_binding)
    members = {name: _read_stable(directory / name) for name in MEMBERS}
    _need(members["binding.json"] == canonical(binding), "expected request/resource/protocol binding differs")
    lease = strict_json(members["writer.lock"])
    _closed(lease, ("schema", "binding_sha256"))
    _need(lease == {"schema": ID + "-lease", "binding_sha256": sha(canonical(binding))}, "lease binding differs")
    trace = verify_trace(members["journal.jsonl"], binding)
    _need(members["response.raw"] == trace["raw"], "response does not match committed trace")
    _need(members["result.json"] == canonical(_result(trace, members["journal.jsonl"], binding)),
          "result does not match committed trace")
    return members, trace


def verify_bundle(directory, expected_binding, *, expected_terminal_sha256):
    """Read-only acceptance gate; requires a trusted terminal digest.

    The caller obtains/pins this digest through its durable Git receipt or other
    trusted channel. Reading a digest from the same mutable directory is not an
    independent integrity guarantee. On any exception, accept no candidate.
    """
    _need(_hash(expected_terminal_sha256), "external terminal digest required")
    directory = Path(directory)
    before_dir = os.lstat(directory)
    _need(stat.S_ISDIR(before_dir.st_mode), "attempt directory must not be symlink")
    lease_fd = os.open(directory / "writer.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _need(stat.S_ISREG(os.fstat(lease_fd).st_mode), "writer lease is not a regular file")
        try:
            fcntl.flock(lease_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidenceError("writer still active") from exc
        _need(_identity(os.fstat(lease_fd)) == _identity(os.lstat(directory / "writer.lock")), "lease replaced")
        terminal_raw = _read_stable(directory / "terminal.json")
        _need(sha(terminal_raw) == expected_terminal_sha256, "terminal digest differs")
        terminal = strict_json(terminal_raw)
        _closed(terminal, ("schema", "binding_sha256", "files", "journal_head_sha256", "journal_rows"))
        members, trace = _verify_members(directory, expected_binding)
        expected = _manifest(members, trace, expected_binding)
        # Canonical bytes preserve JSON types: Python dict equality would accept
        # true == 1 or 1.0 == 1 in byte/row counts of a malformed pinned manifest.
        _need(terminal_raw == canonical(expected), "terminal manifest differs")
        _need(set(p.name for p in directory.iterdir()) == set(MEMBERS) | {"terminal.json"},
              "unexpected or partial artifact")
        _need(_identity(os.lstat(directory)) == _identity(before_dir), "directory changed during read")
        return {"status": "trace_complete", "response": trace["raw"], "output_ids": trace["ids"],
                "committed_tokens": len(trace["ids"]), "terminal_sha256": expected_terminal_sha256,
                "shape_or_semantics_accepted": False}
    finally:
        os.close(lease_fd)


def _manifest(members, trace, binding):
    return {"schema": ID + "-terminal", "binding_sha256": sha(canonical(binding)),
            "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in members.items()},
            "journal_head_sha256": trace["head"], "journal_rows": trace["rows"]}


class AttemptWriter:
    """One new attempt only. No resume, overwrite, retry or native invocation.

    Call prepare_* before the physical operation and commit_* only after its
    acknowledgement. A prepared event without acknowledgement is never known.
    Always close on error; leave the directory untouched for later inspection.
    """
    def __init__(self, directory, binding):
        self.directory = Path(directory)
        self.binding = validate_binding(binding)
        self._state = _state()
        self._closed = False
        self._poisoned = False
        self._fd = self._lease = None
        self._bytes = 0
        self.directory.mkdir(exist_ok=False)
        _sync_dir(self.directory.parent)
        self._directory_identity = (self.directory.stat().st_dev, self.directory.stat().st_ino)
        try:
            self._lease = os.open(self.directory / "writer.lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_all(self._lease, canonical({"schema": ID + "-lease", "binding_sha256": sha(canonical(self.binding))}))
            os.fsync(self._lease)
            _exclusive(self.directory / "binding.json", canonical(self.binding))
            self._fd = os.open(self.directory / "journal.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            _sync_dir(self.directory)
            self._append({"kind": "opened", "binding_sha256": sha(canonical(self.binding))})
        except BaseException:
            self.close()
            raise

    def _append(self, event):
        _need(not self._closed and not self._poisoned, "closed/failed writer cannot continue")
        try:
            d = os.lstat(self.directory)
            _need(stat.S_ISDIR(d.st_mode) and (d.st_dev, d.st_ino) == self._directory_identity, "directory replaced")
            st = os.fstat(self._fd)
            _need(_identity(st) == _identity(os.lstat(self.directory / "journal.jsonl"))
                  and st.st_size == self._bytes, "journal replaced/truncated/appended externally")
            _need(_identity(os.fstat(self._lease)) == _identity(os.lstat(self.directory / "writer.lock")),
                  "writer lease replaced")
            body = {"sequence": self._state["rows"], "previous_sha256": self._state["head"], "event": event}
            row = {**body, "event_sha256": sha(canonical(body))}
            next_state = copy.deepcopy(self._state)
            _advance(next_state, row, self.binding)
            raw = canonical(row) + b"\n"
            _need(self._bytes + len(raw) <= MAX_FILE_BYTES, "journal cap exceeded")
            _write_all(self._fd, raw)
            os.fsync(self._fd)
            self._bytes += len(raw)
            self._state = next_state
            return row["event_sha256"]
        except BaseException:
            self._poisoned = True
            raise

    def prepare_prompt(self, ids):
        return self._append({"kind": "prompt_prepared", "prompt_ids": ids})

    def commit_prompt(self, prepared_sha256, evaluated_tokens):
        return self._append({"kind": "prompt_committed", "prepared_sha256": prepared_sha256,
                             "evaluated_tokens": evaluated_tokens})

    def prepare_step(self, token_id, intended_raw, *, sample_evidence):
        _need(type(intended_raw) is bytes, "rendered response must be exact bytes")
        return self._append({"kind": "step_prepared", "token_index": len(self._state["ids"]),
                             "token_id": token_id, "intended_raw_hex": intended_raw.hex(),
                             "sample_evidence": sample_evidence})

    def commit_step(self, prepared_sha256, evaluated_tokens):
        return self._append({"kind": "step_committed", "prepared_sha256": prepared_sha256,
                             "evaluated_tokens": evaluated_tokens})

    def seal(self, output_ids, response, *, stop_label):
        """Publish only after independent-from-memory replay of on-disk bytes."""
        try:
            _need(type(response) is bytes and response == self._state["raw"], "producer response differs")
            self._append({"kind": "trace_complete", "output_ids": output_ids,
                          "response_sha256": sha(response), "response_bytes": len(response), "stop_label": stop_label})
            os.close(self._fd)
            self._fd = None
            journal = _read_stable(self.directory / "journal.jsonl")
            trace = verify_trace(journal, self.binding)
            _need(trace["ids"] == output_ids and trace["raw"] == response, "on-disk trace differs")
            _exclusive(self.directory / "response.raw", response)
            _exclusive(self.directory / "result.json", canonical(_result(trace, journal, self.binding)))
            members, trace = _verify_members(self.directory, self.binding)
            _need(set(p.name for p in self.directory.iterdir()) == set(MEMBERS),
                  "unexpected artifact before publication")
            manifest_raw = canonical(_manifest(members, trace, self.binding))
            # link() gives atomic no-replace publication; a crash leaves an
            # explicit incomplete staging file and never a partial final name.
            staged = self.directory / "terminal.pending"
            _exclusive(staged, manifest_raw)
            os.link(staged, self.directory / "terminal.json", follow_symlinks=False)
            staged.unlink()
            _sync_dir(self.directory)
            self.close()
            return verify_bundle(self.directory, self.binding, expected_terminal_sha256=sha(manifest_raw))
        except BaseException:
            self._poisoned = True
            self.close()
            raise

    def close(self):
        for name in ("_fd", "_lease"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
