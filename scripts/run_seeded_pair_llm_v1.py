#!/usr/bin/env python3
"""Run one frozen, local, fixed-budget LLM mechanics pilot, never retries."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import seeded_pair_llm as channel

STAGE = Path("experiments/seeded_pair_llm_v1")
RESOURCES = Path("resources/seeded-pair-llm-v1")
RUNTIME = RESOURCES / "runtime/llama-b11791"
CODE_PATHS = [
    "src/synsteg_audit/seeded_pair_llm.py",
    "scripts/run_seeded_pair_llm_v1.py",
    "scripts/seeded_pair_llama_bridge.cpp",
    "tests/test_seeded_pair_llm.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_protocol():
    return json.loads((ROOT / STAGE / "protocol.json").read_text())


def prepare():
    """Fingerprint compiled resources only; no model process or inference."""
    protocol = load_protocol()
    model = ROOT / RESOURCES / protocol["model"]["filename"]
    binary = ROOT / RESOURCES / "seeded_pair_llama_bridge"
    cpu = ROOT / RUNTIME / protocol["runtime"]["forced_cpu_backend"]
    archive = ROOT / RESOURCES / "llama-b11791-bin-ubuntu-x64.tar.gz"
    if model.stat().st_size != protocol["model"]["bytes"] or sha(model) != protocol["model"]["sha256"]:
        raise ValueError("pinned model mismatch")
    if sha(archive) != protocol["runtime"]["archive_sha256"]:
        raise ValueError("pinned runtime archive mismatch")
    if sha(cpu) != protocol["runtime"]["forced_cpu_backend_sha256"]:
        raise ValueError("pinned CPU backend mismatch")
    # Runtime aliases are hashed as bytes; no transient download files included.
    paths = [model, binary, archive] + sorted((ROOT / RUNTIME).glob("*.so*"))
    resources = {
        "files": {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths},
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "libc": platform.libc_ver(), "runtime_parameters": protocol["runtime"],
    }
    frozen = {p: sha(ROOT / p) for p in CODE_PATHS}
    frozen[(STAGE / "protocol.json").as_posix()] = sha(ROOT / STAGE / "protocol.json")
    manifest = {"schema": "seeded-pair-llm-execution-manifest-v1",
                "channel_resources": resources, "resources_sha256": channel.digest(resources),
                "frozen_files": frozen, "model_inference_calls": 0,
                "bridge_path": binary.relative_to(ROOT).as_posix(),
                "model_path": model.relative_to(ROOT).as_posix(),
                "cpu_backend_path": cpu.relative_to(ROOT).as_posix()}
    save(ROOT / STAGE / "execution_manifest.json", manifest)
    return manifest


def verify():
    manifest = json.loads((ROOT / STAGE / "execution_manifest.json").read_text())
    if manifest["schema"] != "seeded-pair-llm-execution-manifest-v1":
        raise ValueError("unsupported execution manifest")
    if channel.digest(manifest["channel_resources"]) != manifest["resources_sha256"]:
        raise ValueError("resource manifest hash mismatch")
    for path, expected in manifest["frozen_files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("frozen source changed: " + path)
    for path, row in manifest["channel_resources"]["files"].items():
        file = ROOT / path
        if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError("runtime artifact changed: " + path)
    expected = manifest["channel_resources"]
    if (expected["python"], expected["platform"], expected["machine"], expected["libc"]) != (
            sys.version, platform.platform(), platform.machine(), list(platform.libc_ver())):
        raise ValueError("runtime fingerprint differs")
    return manifest


def make_envelope(protocol, manifest, prompt_id, arm):
    return {
        "version": channel.VERSION, "prompt_id": prompt_id, "arm": arm,
        "nonce_hex": hashlib.sha256(channel.canonical([channel.VERSION, prompt_id, arm])).digest()[:16].hex(),
        "units": protocol["output_tokens"],
        "protocol_sha256": sha(ROOT / STAGE / "protocol.json"),
        "resources_sha256": manifest["resources_sha256"],
    }


def validate_envelope(envelope, protocol, manifest):
    if not isinstance(envelope, dict) or set(envelope) != {
            "version", "prompt_id", "arm", "nonce_hex", "units", "protocol_sha256", "resources_sha256"}:
        raise ValueError("invalid public envelope fields")
    if envelope["prompt_id"] not in protocol["prompt_ids"] or envelope["arm"] not in protocol["arms"]:
        raise ValueError("unexpected prompt or arm")
    if envelope != make_envelope(protocol, manifest, envelope["prompt_id"], envelope["arm"]):
        raise ValueError("public envelope differs from frozen plan")


def validate_receiver_input(item):
    if not isinstance(item, dict) or set(item) != {"envelope", "key_hex", "stego_text"}:
        raise ValueError("receiver accepts text/key/public envelope only")
    if not isinstance(item["stego_text"], str):
        raise ValueError("receiver text must be UTF8 string")
    key = bytes.fromhex(item["key_hex"])
    if len(key) != 32:
        raise ValueError("receiver key length")
    return key


def endpoint(mode, input_path, directory):
    """Separate process: receiver input contains no sender state."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "endpoint_started", "mode": mode}
    save(directory / "started.json", result)
    bridge = None
    try:
        manifest, protocol = verify(), load_protocol()
        item = json.loads(input_path.read_text())
        envelope = item["envelope"]
        validate_envelope(envelope, protocol, manifest)
        key = validate_receiver_input(item) if mode == "receive" else bytes.fromhex(item["key_hex"])
        if mode == "receive" and envelope["arm"] != "stego":
            raise ValueError("ordinary arm has no receiver")
        deadline = started + protocol["per_endpoint_timeout_seconds"]
        bridge = channel.NativeBridge(ROOT / manifest["bridge_path"], ROOT / manifest["model_path"],
                                      ROOT / manifest["cpu_backend_path"], directory / "native.log", deadline)
        with (directory / "trace.jsonl").open("x") as trace:
            if mode == "receive":
                result = channel.receive(bridge, protocol, envelope, key, item["stego_text"], trace=trace)
            else:
                payload = bytes.fromhex(item["payload_hex"]) if envelope["arm"] == "stego" else None
                with (directory / "output.raw").open("xb") as raw:
                    result = channel.generate(bridge, protocol, envelope, key, payload,
                                              trace=trace, raw_stream=raw)
                if result.get("stego_text") is not None:
                    (directory / "output.txt").write_text(result["stego_text"], encoding="utf-8")
        result.update(native_vocabulary=bridge.vocabulary_size, native_masked_tokens=bridge.masked_tokens,
                      resources_sha256=manifest["resources_sha256"])
    except Exception as exc:
        result = {"status": "endpoint_error", "error_type": type(exc).__name__, "error": str(exc), "mode": mode}
    finally:
        if bridge is not None:
            bridge.close()
    result["wall_seconds"] = time.monotonic() - started
    trace_path = directory / "trace.jsonl"
    result["trace_steps_saved"] = sum(1 for _ in trace_path.open()) if trace_path.exists() else 0
    save(directory / "result.json", result)
    print(json.dumps({"status": result["status"], "trace_steps_saved": result["trace_steps_saved"]}), flush=True)
    return result


def run_child(mode, input_path, directory, timeout):
    command = [sys.executable, str(Path(__file__)), "endpoint", "--mode", mode,
               "--input", str(input_path), "--output", str(directory)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout + 10)
        process_result = {"returncode": process.returncode, "stdout": stdout.decode(errors="replace"),
                          "stderr": stderr.decode(errors="replace"), "timeout": False}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        process_result = {"returncode": process.returncode, "stdout": stdout.decode(errors="replace"),
                          "stderr": stderr.decode(errors="replace"), "timeout": True}
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        directory.mkdir(parents=True, exist_ok=True)
        save(directory / "interrupted.json", {"returncode": process.returncode,
             "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace"),
             "partial_artifacts_retained": True, "automatic_retry_allowed": False})
        raise
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "process.json", process_result)
    result_path = directory / "result.json"
    if not result_path.exists():
        result = {"status": "process_timeout" if process_result["timeout"] else "process_error",
                  "partial_trace_preserved": (directory / "trace.jsonl").exists(),
                  "partial_raw_preserved": (directory / "output.raw").exists()}
        save(result_path, result)
    return json.loads(result_path.read_text())


def trace_compare(sender_dir, receiver_dir):
    sp, rp = sender_dir / "trace.jsonl", receiver_dir / "trace.jsonl"
    if not sp.exists() or not rp.exists():
        return {"compared_steps": 0, "all_compared_weights_identical": None}
    send = [json.loads(line) for line in sp.read_text().splitlines()]
    recv = [json.loads(line) for line in rp.read_text().splitlines()]
    equal = [s["weights_sha256"] == r["weights_sha256"] and s["prefix_sha256"] == r["prefix_sha256"]
             for s, r in zip(send, recv)]
    return {"compared_steps": len(equal), "all_compared_weights_identical": all(equal),
            "first_mismatch": next((i for i, value in enumerate(equal) if not value), None),
            "sender_trace_steps": len(send), "receiver_trace_steps": len(recv)}


def execute(output, freeze_commit):
    if len(freeze_commit) != 40 or any(c not in "0123456789abcdef" for c in freeze_commit):
        raise ValueError("explicit full Git freeze SHA required")
    manifest, protocol = verify(), load_protocol()
    output.mkdir(parents=True, exist_ok=False)
    plan = []
    for prompt in protocol["prompt_ids"]:
        for arm in protocol["arms"]:
            plan.append({"prompt_id": prompt, "arm": arm, "status": "not_attempted"})
    save(output / "plan.json", plan)
    save(output / "execution.json", {"freeze_commit": freeze_commit, "protocol_sha256": sha(ROOT / STAGE / "protocol.json"),
         "execution_manifest_sha256": sha(ROOT / STAGE / "execution_manifest.json"),
         "resources_sha256": manifest["resources_sha256"], "new_generated_texts_planned": 4,
         "new_generated_tokens_planned": 2048, "fresh_receivers_planned": 2, "paid_calls": 0})
    ledger = []
    for row in plan:
        name = row["prompt_id"] + "-" + row["arm"]
        envelope = make_envelope(protocol, manifest, row["prompt_id"], row["arm"])
        payload = hashlib.sha256((channel.VERSION + ":payload:" + row["prompt_id"]).encode()).digest()[:protocol["payload_bytes"]]
        request = {"envelope": envelope, "key_hex": protocol["fixture_key_hex"]}
        if row["arm"] == "stego":
            request["payload_hex"] = payload.hex()
        send_input = output / (name + ".sender-input.json")
        save(send_input, request)
        sender_dir = output / name
        sent = run_child("generate", send_input, sender_dir, protocol["per_endpoint_timeout_seconds"])
        print(json.dumps({"event": "sender_done", "path": str(sender_dir),
                          "status": sent["status"]}), flush=True)
        outcome = {**row, "status": sent["status"], "sender_directory": name,
                   "generated_tokens": sent.get("generated_tokens", sent.get("trace_steps_saved", 0)),
                   "frame_completed": sent.get("frame_completed"), "text_roundtrip_exact": sent.get("text_roundtrip_exact"),
                   "capacity_bits": sent.get("capacity_bits"), "sender_wall_seconds": sent.get("wall_seconds"),
                   "full_exact_delivery": False, "receiver_process_attempted": False}
        if row["arm"] == "stego":
            if isinstance(sent.get("stego_text"), str):
                recipient = {"envelope": envelope, "key_hex": protocol["fixture_key_hex"],
                             "stego_text": sent["stego_text"]}
                recv_input = output / (name + ".receiver-input.json")
                save(recv_input, recipient)
                recv_dir = output / (name + "-receiver")
                received = run_child("receive", recv_input, recv_dir, protocol["per_endpoint_timeout_seconds"])
                outcome.update(receiver_process_attempted=True, receiver_status=received["status"],
                               receiver_trace_steps=received.get("trace_steps_saved", 0),
                               receiver_wall_seconds=received.get("wall_seconds"),
                               receiver_directory=recv_dir.name,
                               full_exact_delivery=received.get("status") == "ok" and received.get("payload_hex") == payload.hex(),
                               trace_comparison=trace_compare(sender_dir, recv_dir))
            else:
                outcome["receiver_status"] = "not_attempted_no_valid_UTF8_text"
        ledger.append(outcome)
        save(output / (name + ".outcome.json"), outcome)
        print(json.dumps({"event": "case_done", "path": str(sender_dir), **outcome}), flush=True)
    summary = {"experiment_id": channel.VERSION, "planned_generated_texts": 4, "attempted_generated_texts": len(ledger),
               "completed_512_token_texts": sum(r["generated_tokens"] == 512 for r in ledger),
               "actual_generated_tokens": sum(r["generated_tokens"] for r in ledger),
               "planned_stego_deliveries": 2, "full_exact_deliveries": sum(r["full_exact_delivery"] for r in ledger),
               "receiver_processes": sum(r["receiver_process_attempted"] for r in ledger),
               "receiver_trace_steps": sum(r.get("receiver_trace_steps", 0) for r in ledger),
               "sender_statuses": dict(Counter(r["status"] for r in ledger)),
               "receiver_statuses": dict(Counter(r.get("receiver_status") for r in ledger if r["arm"] == "stego")),
               "ledger": ledger, "paid_calls": 0, "closed_test_opened": False,
               "semantic_or_security_validated": False}
    save(output / "summary.json", summary)
    save(output / "output_manifest.json", {str(p.relative_to(output)): sha(p)
         for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "execute", "endpoint"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--mode", choices=["generate", "receive"])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.command == "verify":
        print(json.dumps({"status": "verified", "resources_sha256": verify()["resources_sha256"]}))
    elif args.command == "execute":
        if args.output is None or args.freeze_commit is None:
            parser.error("execute needs --output and --freeze-commit")
        def terminated(signum, frame):
            raise KeyboardInterrupt("termination requested; preserve partials and stop owned process group")
        previous = signal.signal(signal.SIGTERM, terminated)
        try:
            print(json.dumps(execute(args.output, args.freeze_commit), indent=2))
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        if args.output is None or args.input is None or args.mode is None:
            parser.error("endpoint needs --output, --input, --mode")
        endpoint(args.mode, args.input, args.output)


if __name__ == "__main__":
    main()
