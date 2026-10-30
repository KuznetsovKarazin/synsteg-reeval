"""Pidfd protocol faults using fake procfs and a fake pidfd; no native child."""
import pytest

from synsteg_audit import proc_namespace_identity_v2 as identity
from test_proc_namespace_identity_v1 import FakeProc


class FakePidfd:
    def __init__(self):
        self.raw = b"pos:\t0\nino:\t777\nPid:\t200\nNSpid:\t200\t8\n"
        self.readiness = []
        self.closed = []
        self.opened = []
        self.reads = 0

    def open(self, pid): self.opened.append(pid); return 900
    def info(self, fd): self.reads += 1; return self.raw
    def events(self, fd): return self.readiness
    def close(self, fd): self.closed.append(fd)


def capture(tree, api):
    return identity.capture_owned_child(tree.process, tree.exe, tree.sha,
        proc_root=tree.root, current_pid=tree.parent_pid, pidfd_api=api)


def test_pidfd_resolves_without_any_children_file(tmp_path):
    tree = FakeProc(tmp_path); api = FakePidfd()
    (tree.root / "100/task/100/children").unlink()
    ticket = capture(tree, api)
    maps, proof = ticket.maps_snapshot()
    assert maps == "fake stable maps\n" and proof["schema"] == "proc-namespace-identity-v2"
    assert proof["pidfd_identity"] == {"proc_pid": 200, "namespace_pids": [200, 8], "inode": 777}
    assert api.opened == [8]
    ticket.close(); ticket.close()
    assert api.closed == [900]
    before = api.reads
    with pytest.raises(identity.IdentityError): ticket.snapshot()
    assert api.reads == before


@pytest.mark.parametrize("fault", ["exit", "duplicate", "missing", "wrong_inner", "wrong_outer",
    "error_event", "wrong_parent", "namespace", "wrong_exe", "process_exit"])
def test_capture_rejects_invalid_pidfd_or_target_and_closes(tmp_path, fault):
    tree = FakeProc(tmp_path); api = FakePidfd()
    if fault == "exit": api.raw = b"ino:777\nPid:-1\nNSpid:-1\n"
    if fault == "duplicate": api.raw += b"Pid:200\n"
    if fault == "missing": api.raw = b"Pid:200\nNSpid:200 8\n"
    if fault == "wrong_inner": api.raw = b"ino:777\nPid:200\nNSpid:200 9\n"
    if fault == "wrong_outer": api.raw = b"ino:777\nPid:201\nNSpid:200 8\n"
    if fault == "error_event": api.readiness = [(900, 32)]
    if fault == "wrong_parent": tree.make(200, 101, [200, 8], 20)
    if fault == "namespace": tree.make(200, 100, [200, 8], 20, namespace="pid:[999]")
    if fault == "wrong_exe": tree.sha = "0" * 64
    if fault == "process_exit": tree.process.poll = lambda: 1
    with pytest.raises(identity.IdentityError): capture(tree, api)
    assert api.closed == ([] if fault == "process_exit" else [900])


@pytest.mark.parametrize("fault", ["pidfd_exit", "inode", "namespace_ids", "popen_pid", "starttime", "exe"])
def test_snapshot_rejects_change_after_capture(tmp_path, fault):
    tree = FakeProc(tmp_path); api = FakePidfd(); ticket = capture(tree, api)
    try:
        if fault == "pidfd_exit": api.readiness = [(900, 1)]
        if fault == "inode": api.raw = api.raw.replace(b"777", b"778")
        if fault == "namespace_ids": api.raw = api.raw.replace(b"200\t8", b"200\t9")
        if fault == "popen_pid": tree.process.pid = 9
        if fault == "starttime": tree.make(200, 100, [200, 8], 99)
        if fault == "exe": tree.exe.write_bytes(b"changed executable")
        with pytest.raises(identity.IdentityError): ticket.maps_snapshot()
    finally: ticket.close()
    assert api.closed == [900]


def test_pidfd_exit_during_maps_is_rejected(tmp_path, monkeypatch):
    from synsteg_audit import proc_namespace_identity_v1 as prior
    tree = FakeProc(tmp_path); api = FakePidfd(); ticket = capture(tree, api)
    original = prior._read_at
    def read(fd, name, maximum=prior.MAX_SMALL_BYTES):
        raw = original(fd, name, maximum)
        if name == "maps": api.readiness = [(900, 1)]
        return raw
    monkeypatch.setattr(prior, "_read_at", read)
    try:
        with pytest.raises(identity.IdentityError): ticket.maps_snapshot()
    finally: ticket.close()
