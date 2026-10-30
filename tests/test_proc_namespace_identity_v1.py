"""Fake procfs/Popen state only: no child, native, tokenizer or model launch."""
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import proc_namespace_identity_v1 as identity


class FakeProc:
    def __init__(self, root, *, flat=False):
        self.root = root / "proc"; self.root.mkdir()
        self.exe = root / "collector"; self.exe.write_bytes(b"fake ELF identity only")
        self.sha = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        self.parent_pid, self.child_pid = (100, 200) if flat else (5, 8)
        self.parent_ns = [100] if flat else [100, 5]
        self.child_ns = [200] if flat else [200, 8]
        self.make(100, 90, self.parent_ns, 10)
        self.make(200, 100, self.child_ns, 20)
        (self.root / "self").symlink_to("100")
        task = self.root / "100/task/100"; task.mkdir(parents=True)
        (task / "children").write_text("200\n")
        self.process = SimpleNamespace(pid=self.child_pid, poll=lambda: None)

    def make(self, pid, ppid, nspid, start, *, namespace="pid:[123]", maps="fake stable maps\n"):
        base = self.root / str(pid); base.mkdir(exist_ok=True)
        (base / "status").write_text(f"Pid:\t{pid}\nPPid:\t{ppid}\nNSpid:\t" + "\t".join(map(str, nspid)) + "\n")
        (base / "stat").write_text(f"{pid} (fake child (safe)) S {ppid} " + "0 " * 17 + str(start) + " 0\n")
        (base / "ns").mkdir(exist_ok=True)
        for name, target in (("ns/pid", namespace), ("exe", str(self.exe))):
            path = base / name
            if path.is_symlink(): path.unlink()
            path.symlink_to(target)
        (base / "maps").write_text(maps)

    def capture(self):
        return identity.capture_owned_child(self.process, self.exe, self.sha,
            proc_root=self.root, current_pid=self.parent_pid)


@pytest.mark.parametrize("flat", [False, True])
def test_exact_owned_child_flat_and_nested(tmp_path, flat):
    tree = FakeProc(tmp_path, flat=flat)
    ticket = tree.capture()
    try:
        maps, proof = ticket.maps_snapshot()
        assert maps == "fake stable maps\n"
        assert proof["popen_pid"] == tree.child_pid and proof["proc_visible_pid"] == 200
        assert proof["before"] == proof["after"]
        assert proof["before"]["executable"]["sha256"] == tree.sha
    finally: ticket.close()
    ticket.close()
    with pytest.raises(identity.IdentityError): ticket.snapshot()


@pytest.mark.parametrize("mutation", ["not_child", "wrong_parent", "namespace", "depth", "older",
                                     "parent_pid", "wrong_hash", "wrong_exe", "exited", "ambiguous"])
def test_capture_rejects_non_owned_or_ambiguous_identity(tmp_path, mutation):
    tree = FakeProc(tmp_path)
    if mutation == "not_child": (tree.root / "100/task/100/children").write_text("")
    if mutation == "wrong_parent": tree.make(200, 101, [200, 8], 20)
    if mutation == "namespace": tree.make(200, 100, [200, 8], 20, namespace="pid:[999]")
    if mutation == "depth": tree.make(200, 100, [200, 7, 8], 20)
    if mutation == "older": tree.make(200, 100, [200, 8], 9)
    if mutation == "parent_pid": tree.parent_pid = 6
    if mutation == "wrong_hash": tree.sha = "0" * 64
    if mutation == "wrong_exe":
        other = tmp_path / "other"; other.write_bytes(tree.exe.read_bytes()); tree.exe = other
    if mutation == "exited": tree.process.poll = lambda: 0
    if mutation == "ambiguous":
        tree.make(201, 100, [201, 8], 21)
        (tree.root / "100/task/100/children").write_text("200 201")
    with pytest.raises(identity.IdentityError): tree.capture()


@pytest.mark.parametrize("mutation", ["starttime", "namespace", "executable", "parent", "exited"])
def test_ticket_rechecks_before_maps(tmp_path, mutation):
    tree = FakeProc(tmp_path); ticket = tree.capture()
    try:
        if mutation == "starttime": tree.make(200, 100, [200, 8], 21)
        if mutation == "namespace": tree.make(200, 100, [200, 8], 20, namespace="pid:[999]")
        if mutation == "executable": tree.exe.write_bytes(b"modified executable")
        if mutation == "parent": tree.make(100, 90, [100, 5], 11)
        if mutation == "exited": tree.process.poll = lambda: 1
        with pytest.raises(identity.IdentityError): ticket.maps_snapshot()
    finally: ticket.close()


def test_identity_change_during_maps_is_rejected(tmp_path, monkeypatch):
    tree = FakeProc(tmp_path); ticket = tree.capture()
    original = identity._read_at
    def changed(fd, name, maximum=identity.MAX_SMALL_BYTES):
        result = original(fd, name, maximum)
        if name == "maps": tree.make(200, 100, [200, 8], 22)
        return result
    monkeypatch.setattr(identity, "_read_at", changed)
    try:
        with pytest.raises(identity.IdentityError): ticket.maps_snapshot()
    finally: ticket.close()


def test_open_proc_directory_pins_original_record(tmp_path):
    tree = FakeProc(tmp_path); ticket = tree.capture()
    try:
        (tree.root / "200").rename(tree.root / "old-record")
        tree.make(200, 100, [200, 8], 99, maps="wrong replacement maps")
        assert ticket.maps_snapshot()[0] == "fake stable maps\n"
    finally: ticket.close()


def load_cli():
    path = Path(__file__).resolve().parents[1] / "experiments/framed-interval-llm-v1/vocab_proc_namespace_identity_v1.py"
    spec = importlib.util.spec_from_file_location("fake_proc_adapter_cli", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["capture", "publication"])
def test_factory_failure_closes_child_before_return(tmp_path, monkeypatch, failure):
    cli = load_cli(); calls = []
    base = SimpleNamespace(process=SimpleNamespace(pid=8), close=lambda: calls.append("base_closed"))
    ticket = SimpleNamespace(parent={}, child={}, expected_executable={}, close=lambda: calls.append("ticket_closed"))
    monkeypatch.setattr(cli, "NativeTransport", lambda *args: base)
    def capture(*args):
        if failure == "capture": raise identity.IdentityError("fake capture failure")
        return ticket
    monkeypatch.setattr(cli, "capture_owned_child", capture)
    monkeypatch.setattr(cli, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("fake publish failure")))
    factory, _ = cli.owned_bindings(tmp_path, "0" * 64, {})
    with pytest.raises((identity.IdentityError, OSError)): factory("b", "m", "l", tmp_path, 1)
    assert calls == (["base_closed"] if failure == "capture" else ["base_closed", "ticket_closed"])
    with pytest.raises(ValueError): factory("b", "m", "l", tmp_path, 1)


def test_resident_uses_owned_maps_and_rechecks_after_frozen_gate(tmp_path, monkeypatch):
    cli = load_cli(); calls = []
    state = {"identity": "same"}
    ticket = SimpleNamespace(parent={}, child={}, expected_executable={}, close=lambda: None,
        maps_snapshot=lambda: ("OWNED MAPS", {"after": state}), snapshot=lambda: state)
    base = SimpleNamespace(process=SimpleNamespace(pid=8), close=lambda: None)
    monkeypatch.setattr(cli, "NativeTransport", lambda *args: base)
    monkeypatch.setattr(cli, "capture_owned_child", lambda *args: ticket)
    monkeypatch.setattr(cli, "atomic_json", lambda *args: None)
    def verify(*args, **kwargs):
        calls.append((args[-1], kwargs)); return {"status": "passed"}
    monkeypatch.setattr(cli, "verify_collection_resident", verify)
    factory, resident = cli.owned_bindings(tmp_path, "0" * 64, {})
    with pytest.raises(ValueError): resident(tmp_path, {}, {}, 8)
    factory("b", "m", "l", tmp_path, 1)
    with pytest.raises(ValueError): resident(tmp_path, {}, {}, 9)
    result = resident(tmp_path, {}, {}, 8)
    assert calls == [(8, {"maps_text": "OWNED MAPS"})]
    assert result["post_resident_identity"] == state


def test_no_native_apis_in_identity_module():
    assert "subprocess" not in identity.__dict__
