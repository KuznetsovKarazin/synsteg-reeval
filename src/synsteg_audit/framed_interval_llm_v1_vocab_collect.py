"""Bounded vocabulary-only preflight collector; import never loads native code.

Production activation remains disabled until exact source/build/resource receipts
are frozen. The C++ companion loads GGUF vocabulary metadata with vocab_only;
it has no model context, sampling, logits or decode commands.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time

from .framed_interval_llm_v1_activation import (
    PUBLIC_PATH, PUBLIC_SHA, RESOURCE_ROOT, checked_bytes, current_runtime, digest,
    safe_path, sha_file, verify_resource_files,
)
from .framed_interval_llm_v1_journal import atomic_json
from .framed_interval_llm_v1_orchestration import mkdir_durable, write_once
from .framed_interval_llm_v1_preflight import (
    ARCHIVE_SHA256, CPU_SHA256, MODEL_SHA256, RUNTIME_COMMIT, RECEIPT_SCHEMA,
    canonical_json, chatml_prompt, fixed_fixtures, validate_inventory, validate_preflight_receipt,
)

INTENT = "collect-frozen-qwen-vocabulary-only-preflight"
PLAN_SHA256 = "86d3c68c44a6ed35e3a9e39b05b4ec4047262bb549bf7934de98a969a026c9a5"
SOURCE_PATHS = (
    "src/synsteg_audit/__init__.py",
    "src/synsteg_audit/framed_interval_llm_v1.py",
    "src/synsteg_audit/framed_interval_llm_v1_qwen.py",
    "src/synsteg_audit/canonical_channel_v2.py",
    "src/synsteg_audit/integer_interval_channel_v1.py",
    "src/synsteg_audit/seeded_pair_llm.py",
    "scripts/restore_framed_interval_llm_v1_resources.py",
    "scripts/framed_interval_llm_vocab_bridge.cpp",
    "src/synsteg_audit/framed_interval_llm_v1_vocab_collect.py",
    "src/synsteg_audit/framed_interval_llm_v1_preflight.py",
    "src/synsteg_audit/framed_interval_llm_v1_activation.py",
    "src/synsteg_audit/framed_interval_llm_v1_journal.py",
    "src/synsteg_audit/framed_interval_llm_v1_orchestration.py",
    "experiments/framed-interval-llm-v1/collect_vocab_preflight.py",
)
MAX_LINE = 131104


def _integer(text, maximum=2_000_000):
    if type(text) is not str or re.fullmatch("0|[1-9][0-9]*", text) is None:
        raise ValueError("canonical unsigned decimal required")
    value = int(text)
    if value > maximum: raise ValueError("native integer exceeds cap")
    return value


def _fields(line):
    if type(line) is not str or not line or len(line) > MAX_LINE or line.strip() != line or "  " in line:
        raise ValueError("invalid native line spacing/length")
    if any(ord(c) < 32 or ord(c) > 126 for c in line): raise ValueError("native line must be ASCII")
    return line.split(" ")


class VocabClient:
    """Real protocol parser; tests may inject a fake transcript transport."""
    def __init__(self, transport):
        self.transport, self.terminal, self.inventory_read = transport, False, False
        self.commands = []
        fields = _fields(transport.readline())
        if len(fields) != 4 or fields[0] != "READY_VOCAB": raise ValueError("vocab-only READY required")
        size, masked, context = map(_integer, fields[1:])
        if not 0 < masked < size <= 2_000_000 or context != 2048: raise ValueError("vocab-only metadata invalid")
        self.ready = {"vocabulary_size": size, "masked_tokens": masked, "context_tokens": context}

    def _send(self, command):
        if self.terminal: raise ValueError("terminal vocabulary process; retry forbidden")
        self.commands.append(command)
        self.transport.send(command)

    def inventory(self):
        if self.inventory_read: raise ValueError("inventory already attempted; retry forbidden")
        self.inventory_read = True
        try:
            self._send("INVENTORY")
            size = self.ready["vocabulary_size"]
            if _fields(self.transport.readline()) != ["INVENTORY", str(size)]: raise ValueError("inventory header differs")
            rows = []
            for index in range(size):
                row = _fields(self.transport.readline())
                if len(row) != 5 or row[:2] != ["ROW", str(index)] or any(f not in {"0", "1"} for f in row[2:]):
                    raise ValueError("complete ordered native predicate scan required")
                rows.append({"token_id": index, "is_control": row[2] == "1", "is_eog": row[3] == "1", "piece_empty": row[4] == "1"})
            if _fields(self.transport.readline()) != ["END_INVENTORY", str(size), str(self.ready["masked_tokens"])]:
                raise ValueError("inventory footer differs")
            validate_inventory(rows, vocabulary_size=size, masked_tokens=self.ready["masked_tokens"])
            return rows
        except Exception:
            self.terminal = True; raise

    def tokenize(self, text, add_special=False, parse_special=False):
        if type(text) is not str or type(add_special) is not bool or type(parse_special) is not bool:
            raise ValueError("strict tokenizer arguments required")
        raw = text.encode("utf-8", "strict")
        if len(raw) > 65536: raise ValueError("tokenizer input exceeds byte cap")
        try:
            self._send(f"TOKENIZE {int(add_special)} {int(parse_special)} {raw.hex() or '-'}")
            fields = _fields(self.transport.readline())
            if len(fields) < 2 or fields[0] != "TOK": raise ValueError("invalid TOK response")
            count = _integer(fields[1], 2048)
            ids = [_integer(x, self.ready["vocabulary_size"]-1) for x in fields[2:]]
            if len(ids) != count: raise ValueError("TOK count differs")
            return ids
        except Exception:
            self.terminal = True; raise

    def render(self, ids):
        if (type(ids) not in {list, tuple} or len(ids) > 2048
                or any(type(i) is not int or not 0 <= i < self.ready["vocabulary_size"] for i in ids)):
            raise ValueError("bounded valid native IDs required")
        try:
            self._send("TEXT "+str(len(ids))+(" "+" ".join(map(str, ids)) if ids else ""))
            fields = _fields(self.transport.readline())
            if len(fields) != 2 or fields[0] != "TEXT": raise ValueError("invalid TEXT response")
            if fields[1] == "-": return b""
            if re.fullmatch("(?:[0-9a-f]{2})+", fields[1]) is None: raise ValueError("canonical native raw hex required")
            raw = bytes.fromhex(fields[1])
            if len(raw) > 65536: raise ValueError("native render exceeds raw byte cap")
            return raw
        except Exception:
            self.terminal = True; raise

    def quit(self):
        if self.terminal: return
        try:
            self._send("QUIT")
            if self.transport.readline() != "BYE": raise ValueError("invalid vocab-only close response")
        finally: self.terminal = True


class NativeTransport:
    """One owned subprocess, no implicit retries and one monotonic deadline."""
    def __init__(self, binary, model, libraries, directory, deadline):
        self.deadline, self.buffer, self.process = deadline, b"", None
        self.stderr = self.stdout_log = self.command_log = None
        directory = Path(directory)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
        environment["LD_LIBRARY_PATH"] = str(libraries)
        try:
            self.stderr = (directory / "native.stderr.log").open("xb")
            self.stdout_log = (directory / "native.stdout.raw").open("xb")
            self.command_log = (directory / "native.commands.raw").open("xb")
            self.process = subprocess.Popen([str(binary), str(model)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, env=environment, start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
        except Exception:
            try: self.close()
            except Exception: pass
            raise

    def send(self, line):
        _fields(line)
        if time.monotonic() >= self.deadline: raise TimeoutError("vocabulary collection deadline")
        raw = line.encode("ascii")+b"\n"
        self.command_log.write(raw); self.command_log.flush(); os.fsync(self.command_log.fileno())
        position, descriptor = 0, self.process.stdin.fileno()
        while position < len(raw):
            remaining = self.deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError("vocabulary command write deadline")
            _, writable, _ = select.select([], [descriptor], [], remaining)
            if not writable: raise TimeoutError("vocabulary command pipe stalled")
            try: count = os.write(descriptor, raw[position:position+65536])
            except BlockingIOError: continue
            if count <= 0: raise BrokenPipeError("vocabulary command write made no progress")
            position += count

    def readline(self):
        if time.monotonic() >= self.deadline: raise TimeoutError("vocabulary collection deadline")
        while b"\n" not in self.buffer:
            remaining = self.deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError("vocabulary collection deadline")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready: raise TimeoutError("vocabulary response timeout")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block: raise EOFError("native vocabulary process ended before response")
            self.stdout_log.write(block); self.stdout_log.flush(); os.fsync(self.stdout_log.fileno())
            self.buffer += block
            if len(self.buffer.split(b"\n", 1)[0]) > MAX_LINE: raise ValueError("native line exceeds cap")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("ascii", "strict")

    def close(self):
        errors = []
        if self.process is not None:
            try: os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            except Exception as exc: errors.append(exc)
            try: self.process.wait(timeout=3)
            except Exception as exc: errors.append(exc)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    try: stream.close()
                    except Exception as exc: errors.append(exc)
        for stream in (self.stderr, self.stdout_log, self.command_log):
            if stream is not None:
                try: stream.close()
                except Exception as exc: errors.append(exc)
        if errors: raise errors[0]

    def finish(self):
        """After BYE, require native exit0 before publishing a passed receipt."""
        remaining = self.deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError("vocabulary finish deadline")
        if self.process.wait(timeout=min(3, remaining)) != 0:
            raise RuntimeError("vocabulary process exited unsuccessfully after BYE")
        if self.buffer: raise ValueError("unexpected native stdout after BYE")
        while True:
            remaining = self.deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError("vocabulary EOF deadline")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready: raise TimeoutError("native stdout remains open after BYE")
            extra = os.read(self.process.stdout.fileno(), 65536)
            if not extra: break
            self.stdout_log.write(extra); self.stdout_log.flush(); os.fsync(self.stdout_log.fileno())
            raise ValueError("unexpected native stdout after BYE")


def _case(service, spec, specials):
    raw = bytes.fromhex(spec["input_hex"])
    if spec["kind"] == "invalid_utf8":
        try: raw.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return {"id": spec["id"], "input_hex": spec["input_hex"], "status": "invalid_utf8", "tokenizer_calls": 0}
        raise ValueError("invalid UTF8 fixture unexpectedly valid")
    ids = (list(spec["input_token_ids"]) if spec["kind"] == "token_path"
           else service.tokenize(raw.decode("utf-8", "strict"), False, False))
    if len(ids) > 128: raise ValueError("fixed fixture exceeded reference token cap")
    prefixes, reasons, previous, prior_canonical = [], [], b"", True
    for length in range(len(ids)+1):
        prefix = ids[:length]
        rendered = service.render(prefix)
        if type(rendered) is not bytes or len(rendered) > 4096: raise ValueError("fixed render exceeded byte cap")
        try: text = rendered.decode("utf-8", "strict"); valid = True
        except UnicodeDecodeError: valid = False
        reencoded = service.tokenize(text, False, False) if valid else None
        canonical = valid and reencoded == prefix
        prefixes.append({"token_ids": prefix, "rendered_hex": rendered.hex(), "strict_utf8": valid,
                         "reencoded_ids": reencoded})
        if length:
            if not prior_canonical: reason = "invalid_prefix"
            elif ids[length-1] in specials: reason = "special_token"
            elif not rendered.startswith(previous): reason = "rewrites_prior_bytes"
            elif rendered == previous: reason = "empty_extension"
            elif not valid: reason = "invalid_utf8"
            elif not canonical: reason = "retokenization_mismatch"
            else: reason = "accepted"
            reasons.append(reason)
        previous, prior_canonical = rendered, canonical
    return {"id": spec["id"], "input_hex": spec["input_hex"], "status": "observed",
        "tokenize_flags": {"add_special": False, "parse_special": False},
        "reference_ids": ids, "prefixes": prefixes, "gate_reasons": reasons}


def collect_evidence(service, plan_bytes, protocol, resources, collector_source_sha256, inventory_path, on_record=None):
    """One fixed collection. Callback(kind, id, value) publishes incremental evidence.

    Backend exceptions propagate immediately; prior callback records remain and
    the CLI records all unattempted fixtures. No service call is ever retried.
    """
    emit = on_record or (lambda kind, name, value: None)
    plan = json.loads(plan_bytes)
    if canonical_json(plan.get("fixtures")) != canonical_json(fixed_fixtures()):
        raise ValueError("exact fixed15 tokenizer fixtures required")
    ready = service.ready
    emit("inventory_started", "inventory", {"status": "started"})
    rows = service.inventory()
    summary = validate_inventory(rows, vocabulary_size=ready["vocabulary_size"], masked_tokens=ready["masked_tokens"])
    inventory = canonical_json(rows)+b"\n"
    emit("inventory", "inventory", inventory)
    cases = []
    for spec in plan["fixtures"]:
        emit("case_started", spec["id"], {"status": "started"})
        case = _case(service, spec, set(summary["special_ids"]))
        cases.append(case); emit("case", spec["id"], case)
    prompts = {}
    for name in ("station", "library"):
        emit("prompt_started", name, {"status": "started"})
        text = chatml_prompt(protocol, name)
        ids = service.tokenize(text, True, True)
        repeated = service.tokenize(text, True, True)
        prompts[name] = {"text_sha256": digest(text.encode("utf-8")),
            "tokenize_flags": {"add_special": True, "parse_special": True},
            "token_ids": ids, "repeat_token_ids": repeated}
        emit("prompt", name, prompts[name])
    receipt = {"schema": RECEIPT_SCHEMA, "status": "passed", "plan_sha256": digest(plan_bytes),
        "resources": resources, "collector": {"source_sha256": collector_source_sha256,
            "mode": "pinned-native-vocab-only", "complete_native_id_scan": True}, "ready": ready,
        "inventory": {"path": inventory_path, "bytes": len(inventory), "sha256": digest(inventory),
            "canonical_sha256": summary["inventory_canonical_sha256"], "rows": len(rows),
            "special_ids": summary["special_ids"], "special_ids_sha256": summary["special_ids_sha256"]},
        "cases": cases, "prompts": prompts,
        "guards": {"forward_calls": 0, "prompt_calls": 0, "step_calls": 0, "distribution_calls": 0,
                   "paid_calls": 0, "closed_test_accessed": False, "model_weights_loaded": False}}
    validate_preflight_receipt(receipt, plan_bytes=plan_bytes, inventory_bytes=inventory,
        expected_plan_sha256=digest(plan_bytes), expected_resource_identity_sha256=resources["resource_identity_sha256"],
        expected_native_libraries=resources["native_libraries_sha256"],
        expected_collector_source_sha256=collector_source_sha256, expected_ready=ready, prompt_protocol=protocol)
    return receipt, inventory


def load_collection_activation(root, path, expected_sha, intent, *, resource_verifier=verify_resource_files,
                               runtime_reader=current_runtime):
    if intent != INTENT: raise ValueError("explicit frozen vocabulary-only collection intent required")
    raw = checked_bytes(root, {"path": path, "sha256": expected_sha})
    obj = json.loads(raw)
    required = {"schema", "enabled", "freeze_commit", "coordinator_readback", "sources_sha256", "plan",
        "public_protocol", "binary", "build_receipt", "resource_identity_sha256", "runtime",
        "native_system_libraries_sha256", "timeout_seconds"}
    if (type(obj) is not dict or set(obj) != required or obj["schema"] != "framed-interval-llm-v1-vocab-activation-v1"
            or obj["enabled"] is not True or type(obj["freeze_commit"]) is not str
            or re.fullmatch("[0-9a-f]{40}", obj["freeze_commit"]) is None):
        raise ValueError("complete enabled frozen vocabulary activation required")
    if obj["coordinator_readback"] != {"commit": obj["freeze_commit"], "all_files_exact": True}:
        raise ValueError("coordinator exact source readback required")
    if type(obj["coordinator_readback"].get("all_files_exact")) is not bool:
        raise ValueError("exact boolean readback required")
    sources = obj["sources_sha256"]
    if type(sources) is not dict or not set(SOURCE_PATHS).issubset(sources):
        raise ValueError("complete vocabulary collector source freeze required")
    for name, expected in sources.items(): checked_bytes(root, {"path": name, "sha256": expected})
    bundle = digest(canonical_json(sources))
    if obj["public_protocol"] != {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA}:
        raise ValueError("fixed public protocol required")
    protocol = json.loads(checked_bytes(root, obj["public_protocol"]))
    plan = checked_bytes(root, obj["plan"])
    if digest(plan) != PLAN_SHA256: raise ValueError("exact fixed tokenizer plan bytes required before process launch")
    if canonical_json(json.loads(plan).get("fixtures")) != canonical_json(fixed_fixtures()):
        raise ValueError("fixed15 plan required before process launch")
    report = resource_verifier(root)
    if report["frozen_resource_identity_sha256"] != obj["resource_identity_sha256"]:
        raise ValueError("old41 resource identity differs")
    if runtime_reader() != obj["runtime"]: raise ValueError("current collector runtime differs")
    binary = safe_path(root, obj["binary"]["path"])
    checked_bytes(root, obj["binary"])
    if not os.access(binary, os.X_OK): raise ValueError("frozen vocab-only binary must be executable")
    build = json.loads(checked_bytes(root, obj["build_receipt"]))
    if (build.get("schema") != "framed-interval-llm-v1-vocab-build-v1"
            or build.get("source_bundle_sha256") != bundle or build.get("runtime_commit") != RUNTIME_COMMIT
            or build.get("binary_sha256") != obj["binary"]["sha256"]
            or build.get("native_model_executions") != 0 or type(build.get("native_model_executions")) is not int):
        raise ValueError("exact reviewed vocabulary build receipt required")
    for name in ("compiler",):
        entry = build.get(name, {})
        if set(entry) != {"path", "sha256"} or not Path(entry["path"]).is_absolute() or sha_file(entry["path"]) != entry["sha256"]:
            raise ValueError("current compiler identity differs from build receipt")
    if (type(build.get("headers")) is not dict or not build["headers"]
            or type(build.get("build_command")) is not list or not build["build_command"]
            or any(type(part) is not str or not part for part in build["build_command"])):
        raise ValueError("exact header identities and recorded build command required")
    for name, expected in build["headers"].items(): checked_bytes(root, {"path": name, "sha256": expected})
    systems = obj["native_system_libraries_sha256"]
    if type(systems) is not dict or not systems: raise ValueError("native system library identities required")
    for name, expected in systems.items():
        if not Path(name).is_absolute() or sha_file(name) != expected: raise ValueError("native system file changed")
    if type(obj["timeout_seconds"]) is not int or obj["timeout_seconds"] != 120:
        raise ValueError("fixed120-second vocabulary-only budget required")
    return obj, report, plan, protocol, bundle


def verify_collection_resident(root, activation, report, pid, *, maps_text=None, file_hasher=sha_file):
    if type(pid) is not int or pid <= 0: raise ValueError("owned collector PID required")
    if maps_text is None: maps_text = Path(f"/proc/{pid}/maps").read_text()
    resource_root = safe_path(root, RESOURCE_ROOT).resolve()
    binary = safe_path(root, activation["binary"]["path"]).resolve()
    observed, executable = {}, set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"): continue
        if fields[5].endswith(" (deleted)"): raise ValueError("deleted collector mapping")
        path = Path(fields[5]).resolve()
        if path == binary: expected = activation["binary"]["sha256"]
        elif resource_root in path.parents:
            relative = path.relative_to(resource_root).as_posix()
            if relative not in report["verified_file_identities"]: raise ValueError("unknown mapped resource")
            if relative.endswith(".gguf"): raise ValueError("vocabulary collector mapped model tensor file")
            if "libggml-cpu-" in path.name and path.name != "libggml-cpu-haswell.so":
                raise ValueError("unexpected CPU backend mapped in vocab-only process")
            expected = report["verified_file_identities"][relative]["sha256"]
        elif "x" in fields[1]:
            expected = activation["native_system_libraries_sha256"].get(str(path))
            if expected is None: raise ValueError("unregistered collector executable mapping")
        else: continue
        if str(path) not in observed and file_hasher(path) != expected: raise ValueError("collector mapped-file hash differs")
        observed[str(path)] = expected
        if "x" in fields[1]: executable.add(str(path))
    if str(binary) not in executable or not any(Path(p).name.startswith("libllama.so") for p in executable):
        raise ValueError("collector/libllama executable mapping missing")
    return {"status": "passed", "pid": pid, "mapped_files_sha256": observed,
            "scope": "current mapped paths/file bytes; no historical system-memory attestation"}


def run_collection(root, activation_path, expected_sha, output_path, intent, *,
                   activation_loader=load_collection_activation, transport_factory=NativeTransport,
                   resident_verifier=verify_collection_resident):
    started = time.monotonic()
    root = Path(root).resolve()
    activation, report, plan, protocol, bundle = activation_loader(root, activation_path, expected_sha, intent)
    if not output_path.startswith("results/framed-interval-llm-v1-tokenizer-preflight/"):
        raise ValueError("dedicated new tokenizer-only output required")
    output = safe_path(root, output_path); mkdir_durable(output, exclusive=True)
    status = {"schema": "framed-interval-llm-v1-vocab-collection-status-v1", "activation_sha256": expected_sha,
        "state": "started", "retry_permitted": False, "planned_fixtures": 15, "planned_prompts": 2,
        "inventory_status": "not_started",
        "fixtures": [{"id": row["id"], "status": "not_started"} for row in fixed_fixtures()],
        "prompts": {name: {"status": "not_started"} for name in ("station", "library")}}
    atomic_json(output / "status-started.json", status)
    transport, client, completed = None, None, None
    try:
        resource_root = safe_path(root, RESOURCE_ROOT)
        transport = transport_factory(safe_path(root, activation["binary"]["path"]),
            resource_root / "qwen2.5-1.5b-instruct-q4_k_m.gguf", resource_root / "runtime/llama-b11791",
            output, time.monotonic()+activation["timeout_seconds"])
        client = VocabClient(transport)
        resident = resident_verifier(root, activation, report, transport.process.pid)
        atomic_json(output / "resident.json", resident)
        def publish(kind, name, record):
            if kind.endswith("_started"):
                if kind == "inventory_started": status["inventory_status"] = "started"
                elif kind == "case_started": next(row for row in status["fixtures"] if row["id"] == name)["status"] = "started"
                else: status["prompts"][name] = {"status": "started"}
                atomic_json(output / (kind+"-"+name+".json"), record)
            elif kind == "inventory":
                write_once(output / "inventory.json", record)
                status["inventory_status"] = "observed"
            else:
                atomic_json(output / (kind+"-"+name+".json"), record)
                if kind == "case":
                    next(row for row in status["fixtures"] if row["id"] == name)["status"] = record["status"]
                else: status["prompts"][name] = {"status": "observed"}
        resources = {"model_sha256": MODEL_SHA256, "runtime_commit": RUNTIME_COMMIT,
            "archive_sha256": ARCHIVE_SHA256, "cpu_backend_sha256": CPU_SHA256,
            "resource_identity_sha256": report["frozen_resource_identity_sha256"],
            "native_libraries_sha256": {name: row["sha256"] for name, row in report["verified_file_identities"].items() if name.startswith("runtime/")}}
        receipt, inventory = collect_evidence(client, plan, protocol, resources, bundle,
            output_path+"/inventory.json", publish)
        client.quit()
        if hasattr(transport, "finish"): transport.finish()  # Production transport always has this check.
        completed = receipt, inventory
        status["state"] = "evidence_collected_pending_close"
    except Exception as exc:
        status.update(state="failed_no_retry", error={"type": type(exc).__name__, "message": str(exc)[:1024]})
        if status["inventory_status"] == "started": status["inventory_status"] = "failed_unknown"
        for row in status["fixtures"]:
            if row["status"] == "not_started": row["status"] = "not_attempted_after_failure"
            elif row["status"] == "started": row["status"] = "failed_unknown"
        for row in status["prompts"].values():
            if row["status"] == "not_started": row["status"] = "not_attempted_after_failure"
            elif row["status"] == "started": row["status"] = "failed_unknown"
    finally:
        if transport is not None:
            try: transport.close()
            except Exception as exc:
                status.update(state="failed_no_retry", close_error={"type": type(exc).__name__, "message": str(exc)[:1024]})
    if completed is not None and status["state"] == "evidence_collected_pending_close":
        receipt, inventory = completed
        try:
            atomic_json(output / "receipt.json", receipt)
            status.update(state="passed", inventory_sha256=digest(inventory), receipt_sha256=sha_file(output / "receipt.json"))
        except Exception as exc:
            status.update(state="failed_no_retry", publication_error={"type": type(exc).__name__, "message": str(exc)[:1024]})
    status.update(wall_seconds=time.monotonic()-started, forward_calls=0, model_weights_loaded=False,
                  native_commands=[] if client is None else [command.split(" ", 1)[0] for command in client.commands])
    try: atomic_json(output / "status-final.json", status)
    except Exception as exc:
        status.update(state="failed_no_retry", publication_error={"type": type(exc).__name__, "message": str(exc)[:1024]})
        atomic_json(output / "status-publication-failure.json", status)
    return status
