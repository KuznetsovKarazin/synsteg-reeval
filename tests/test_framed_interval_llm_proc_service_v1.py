"""Focused fake bridge lifecycle/routing; no native process or model is used."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_llm_proc_service_v1 as service
from synsteg_audit import seeded_pair_llm as frozen_bridge


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = {"events": [], "failure": None, "ticket_closes": 0, "stderr": None}
    process = SimpleNamespace(pid=8, stdin=io.BytesIO(), stdout=io.BytesIO(), returncode=None)
    process.poll = lambda: process.returncode
    def terminate():
        if state["failure"] == "cleanup": raise OSError("fake cleanup failure")
        process.returncode = 0
    process.terminate = terminate
    process.kill = lambda: setattr(process, "returncode", -9)
    process.wait = lambda timeout: process.returncode
    def popen(*args, **kwargs):
        state["events"].append("spawn"); state["stderr"] = kwargs["stderr"]
        if state["failure"] == "spawn": raise OSError("fake spawn failure")
        return process
    monkeypatch.setattr(frozen_bridge, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired))
    ticket = SimpleNamespace(process=process, parent={}, child={}, expected_executable={}, closed=False)
    def close():
        if not ticket.closed: state["ticket_closes"] += 1; ticket.closed = True
    ticket.close = close
    def capture(*args):
        state["events"].append("capture")
        assert args[0] is process
        if state["failure"] in {"capture", "cleanup"}: raise ValueError("fake ownership failure")
        return ticket
    def publish(*args):
        state["events"].append("publish")
        if state["failure"] == "publish": raise OSError("fake publication failure")
    def line(self):
        state["events"].append("read")
        if state["failure"] == "read": raise EOFError("fake READY failure")
        return "INVALID" if state["failure"] == "ready" else "READY 151936 23 2048"
    monkeypatch.setattr(service, "capture_owned_child", capture)
    monkeypatch.setattr(service, "atomic_json", publish)
    monkeypatch.setattr(frozen_bridge.NativeBridge, "_line", line)
    def make():
        return service.OwnedNativeBridge("binary", "model", "cpu", tmp_path / "stderr", 99,
            executable_sha256="a" * 64, adapter_readback={"path": "public-proof", "sha256": "b" * 64})
    return state, process, make


def test_capture_precedes_first_ready_and_occurs_once(harness):
    state, process, make = harness
    bridge = make()
    assert state["events"] == ["spawn", "capture", "publish", "read"]
    bridge._line()
    assert state["events"].count("capture") == 1
    bridge.close(); bridge.close()
    assert process.returncode == 0 and state["ticket_closes"] == 1
    assert state["stderr"].closed and process.stdin.closed and process.stdout.closed
    with pytest.raises(ValueError): bridge._line()


@pytest.mark.parametrize("failure", ["spawn", "capture", "publish", "read", "ready", "cleanup"])
def test_constructor_failures_close_owned_handles_and_preserve_error(harness, failure):
    state, process, make = harness; state["failure"] = failure
    with pytest.raises((ValueError, OSError, EOFError)) as error: make()
    assert state["stderr"].closed
    if failure != "spawn": assert process.stdin.closed and process.stdout.closed
    if failure in {"publish", "read", "ready"}: assert state["ticket_closes"] == 1
    if failure in {"capture", "publish", "cleanup"}: assert "read" not in state["events"]
    if failure == "cleanup":
        assert str(error.value) == "fake ownership failure"
        assert any("cleanup" in note for note in error.value.__notes__)


def test_all_model_and_tokenizer_commands_remain_inherited():
    for name in ("command", "tokenize", "render", "start", "step", "distribution"):
        assert getattr(service.OwnedNativeBridge, name) is getattr(frozen_bridge.NativeBridge, name)


def test_service_construction_is_inert(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "OwnedNativeBridge", lambda *a, **k: pytest.fail("constructor launched bridge"))
    obj = service.ProcNativeService(SimpleNamespace(), tmp_path, 99, adapter_readback={})
    assert obj.bridge is None and obj.process_attempted is False


@pytest.mark.parametrize("changed", [False, True])
def test_resident_uses_owned_maps_and_rechecks_after_hashes(monkeypatch, tmp_path, changed):
    process = SimpleNamespace(pid=8)
    identity = {"identity": "stable"}
    ticket = SimpleNamespace(process=process, maps_snapshot=lambda: ("OWNED", {"after": identity}),
        snapshot=lambda: {"identity": "changed"} if changed else identity)
    bridge = SimpleNamespace(process=process, identity_ticket=ticket)
    calls = []
    monkeypatch.setattr(service, "verify_resident_maps", lambda *a, **k: calls.append((a, k)) or {"status": "passed"})
    obj = service.ProcNativeService(SimpleNamespace(), tmp_path, 99, adapter_readback={})
    if changed:
        with pytest.raises(ValueError): obj._resident(obj.activation, bridge)
    else: assert obj._resident(obj.activation, bridge)["post_resident_identity"] == identity
    assert calls[0][1] == {"maps_text": "OWNED"}


def load_cli():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/framed-interval-llm-v1/run_fixed512_proc_namespace_v1.py"
    spec = importlib.util.spec_from_file_location("fake_proc_service_cli", path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def test_private_path_rejected_before_source_hash_reads(monkeypatch):
    cli = load_cli()
    proof = {"schema": "framed-interval-llm-proc-service-v1-git-readback", "commit": "a" * 40,
        "all_files_exact": True, "files_sha256": {name: "b" * 64 for name in cli.EXACT_PUBLIC_PATHS}}
    proof["files_sha256"]["private-sender.json"] = "c" * 64
    calls = []
    def read(root, item):
        calls.append(item["path"])
        if item["path"] != "proof.json": pytest.fail("private proof mapping was opened")
        return json.dumps(proof).encode()
    monkeypatch.setattr(cli, "checked_bytes", read)
    with pytest.raises(ValueError): cli.verify_readback(Path("."), "proof.json", "d" * 64)
    assert calls == ["proof.json"]


def test_worker_injects_service_and_preserves_runtime_loader(monkeypatch):
    cli = load_cli(); module = SimpleNamespace()
    def worker(args):
        assert module.load_activation is cli.load_run_activation
        assert module.execute_endpoint.func is cli.execute_endpoint
        factory = module.execute_endpoint.keywords["service_factory"]
        assert factory.func is cli.ProcNativeService
        assert factory.keywords["adapter_readback"] == {"path": "proof", "sha256": "a" * 64}
        return 7
    module.worker = worker
    monkeypatch.setattr(cli, "_private_module", lambda *a: module)
    assert cli.worker(SimpleNamespace(adapter_readback="proof", adapter_readback_sha256="a" * 64)) == 7


def test_spawn_route_reuses_frozen_lifecycle_and_targets_new_worker(tmp_path):
    cli = load_cli()
    runner = cli.worker_runner(SimpleNamespace(adapter_readback="proof", adapter_readback_sha256="a" * 64))
    globals_ = runner.func.__globals__
    calls = []
    process = SimpleNamespace(pid=123, returncode=0, wait=lambda timeout: 0)
    def popen(command, **kwargs): calls.append((command, kwargs)); return process
    globals_["subprocess"] = SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired)
    globals_["os"] = SimpleNamespace(environ={}, killpg=lambda *args: calls.append(args))
    endpoint = tmp_path / "endpoint"; endpoint.mkdir()
    result = runner(tmp_path, "activation", "b" * 64, "campaign", "endpoint", "request", 10)
    assert result["status"] == "exited"
    command, options = calls[0]
    assert command[1] == str(tmp_path / cli.SELF) and command[2] == "worker"
    assert command[-4:] == ["--adapter-readback", "proof", "--adapter-readback-sha256", "a" * 64]
    assert options["start_new_session"] is True and options["env"]["PYTHONPATH"] == str(tmp_path / "src")
    assert calls[1][0] == 123
