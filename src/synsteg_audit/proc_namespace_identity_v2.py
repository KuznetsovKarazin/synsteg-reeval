"""proc-namespace-identity-v2: pidfd-owned procfs identity, without children files.

The actual Popen namespace PID is opened as a kernel pidfd. Its mounted-procfs
fdinfo identifies the outer PID; no process list or offset guessing is used.
Existing v1 identity/stat/exe validation is reused without changing v1 bytes.
No process is created by importing or calling this resolver.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import select

from .proc_namespace_identity_v1 import (
    IdentityError, MAX_SMALL_BYTES, OwnedChild, _alive, _directory, _executable,
    _identity, _positive,
)

SCHEMA = "proc-namespace-identity-v2"


class KernelPidfd:
    def open(self, pid): return os.pidfd_open(pid, 0)

    def info(self, descriptor):
        with (Path("/proc/self/fdinfo") / str(descriptor)).open("rb") as stream:
            raw = stream.read(MAX_SMALL_BYTES + 1)
        if len(raw) > MAX_SMALL_BYTES: raise IdentityError("pidfd info exceeds byte cap")
        return raw

    def events(self, descriptor):
        poll = select.poll()
        poll.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
        return poll.poll(0)

    def close(self, descriptor): os.close(descriptor)


def _pidfd_info(raw):
    values = {}
    for line in raw.decode("ascii", "strict").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Pid", "NSpid", "ino"}:
            if key in values: raise IdentityError("duplicate pidfd identity field")
            words = value.split()
            if not words or any(re.fullmatch("[1-9][0-9]*", word) is None for word in words):
                raise IdentityError("pidfd is exited or has invalid identity numbers")
            values[key] = [int(word) for word in words]
    if set(values) != {"Pid", "NSpid", "ino"} or len(values["Pid"]) != 1 or len(values["ino"]) != 1:
        raise IdentityError("complete pidfd identity required")
    if values["Pid"][0] != values["NSpid"][0]:
        raise IdentityError("pidfd mounted-proc PID differs from first namespace PID")
    return {"proc_pid": values["Pid"][0], "namespace_pids": values["NSpid"], "inode": values["ino"][0]}


@dataclass
class PidfdOwnedChild(OwnedChild):
    pidfd: int | None = None
    pidfd_api: object = None
    pidfd_identity: dict | None = None
    _pidfd_closed: bool = False

    def _check_pidfd(self):
        if self._closed or self._pidfd_closed: raise IdentityError("closed pidfd identity ticket")
        if self.process.pid != self.child["namespace_pids"][-1]:
            raise IdentityError("Popen PID changed after pidfd capture")
        _alive(self.process)
        if self.pidfd_api.events(self.pidfd): raise IdentityError("owned pidfd signals exit/error")
        if _pidfd_info(self.pidfd_api.info(self.pidfd)) != self.pidfd_identity:
            raise IdentityError("pidfd identity changed")
        if self.pidfd_api.events(self.pidfd): raise IdentityError("owned pidfd exited during identity read")
        _alive(self.process)

    def snapshot(self):
        self._check_pidfd()
        result = super().snapshot()
        self._check_pidfd()
        return result

    def maps_snapshot(self):
        maps, proof = super().maps_snapshot()
        return maps, {**proof, "schema": SCHEMA, "pidfd_identity": self.pidfd_identity,
            "pidfd_live_before_and_after_maps": True,
            "scope": "actual Popen pidfd and pinned proc directory; stable live identity/exe around maps; no process enumeration"}

    def close(self):
        try:
            super().close()
        finally:
            if not self._pidfd_closed:
                self._pidfd_closed = True
                self.pidfd_api.close(self.pidfd)


def capture_owned_child(process, expected_executable_path, expected_executable_sha256,
                        *, proc_root=Path("/proc"), current_pid=None, pidfd_api=None):
    """Capture before READY. Overrides are solely for deterministic fake tests."""
    inner_pid = _positive(process.pid, "owned Popen PID")
    current_pid = os.getpid() if current_pid is None else _positive(current_pid, "parent PID")
    if re.fullmatch("[0-9a-f]{64}", expected_executable_sha256) is None:
        raise IdentityError("exact expected executable SHA256 required")
    expected_path = str(Path(expected_executable_path).resolve(strict=True))
    api = KernelPidfd() if pidfd_api is None else pidfd_api
    pidfd = parent_fd = child_fd = None
    try:
        _alive(process)
        pidfd = api.open(inner_pid)
        if type(pidfd) is not int or pidfd < 0: raise IdentityError("invalid owned pidfd descriptor")
        if api.events(pidfd): raise IdentityError("owned pidfd is not live")
        info = _pidfd_info(api.info(pidfd))
        if info["namespace_pids"][-1] != inner_pid:
            raise IdentityError("pidfd namespace PID differs from actual Popen PID")
        parent_fd = _directory(Path(proc_root) / "self")
        parent = _identity(parent_fd)
        if parent["namespace_pids"][-1] != current_pid:
            raise IdentityError("proc self namespace PID differs from current process")
        child_fd = _directory(Path(proc_root) / str(info["proc_pid"]))
        child = _identity(child_fd)
        if (child["proc_pid"] != info["proc_pid"] or child["namespace_pids"] != info["namespace_pids"]
                or child["proc_ppid"] != parent["proc_pid"]
                or child["pid_namespace"] != parent["pid_namespace"]
                or len(child["namespace_pids"]) != len(parent["namespace_pids"])
                or child["starttime_ticks"] < parent["starttime_ticks"]):
            raise IdentityError("pidfd target is not same-namespace immediate child")
        executable = _executable(child_fd)
        if executable["path"] != expected_path or executable["sha256"] != expected_executable_sha256:
            raise IdentityError("pidfd child is not expected executable bytes")
        ticket = PidfdOwnedChild(process, child_fd, parent_fd, parent, child, executable,
            pidfd=pidfd, pidfd_api=api, pidfd_identity=info)
        ticket.snapshot()
        pidfd = parent_fd = child_fd = None
        return ticket
    finally:
        try:
            if child_fd is not None: os.close(child_fd)
        finally:
            try:
                if parent_fd is not None: os.close(parent_fd)
            finally:
                if pidfd is not None: api.close(pidfd)
