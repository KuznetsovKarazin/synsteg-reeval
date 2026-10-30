"""proc-namespace-identity-v1: bind an owned immediate Popen child to procfs.

The mounted procfs may expose outer PIDs while Popen returns inner PIDs. Never
scan arbitrary processes or guess a PID offset. Retain a descriptor to the
selected child's proc directory and recheck ownership around maps inspection.
No subprocess is created by this module. The caller owns and closes the child.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

SCHEMA = "proc-namespace-identity-v1"
MAX_TASKS = 256
MAX_CHILDREN = 1024
MAX_SMALL_BYTES = 1024 * 1024
MAX_MAPS_BYTES = 16 * 1024 * 1024


class IdentityError(ValueError):
    pass


def _positive(value, label):
    if type(value) is not int or value <= 0: raise IdentityError("invalid " + label)
    return value


def _read_at(directory_fd, name, maximum=MAX_SMALL_BYTES):
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory_fd)
    try:
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, maximum + 1 - size))
            if not block: break
            chunks.append(block); size += len(block)
            if size > maximum: raise IdentityError("proc record exceeds fixed byte cap")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _status(raw):
    values = {}
    for line in raw.decode("utf-8", "strict").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Pid", "PPid", "NSpid"}:
            if key in values: raise IdentityError("duplicate proc identity field")
            words = value.split()
            if not words or any(re.fullmatch("[1-9][0-9]*", word) is None for word in words):
                raise IdentityError("invalid proc identity number")
            values[key] = [int(word) for word in words]
    if set(values) != {"Pid", "PPid", "NSpid"} or len(values["Pid"]) != 1 or len(values["PPid"]) != 1:
        raise IdentityError("complete proc identity fields required")
    return {"proc_pid": values["Pid"][0], "proc_ppid": values["PPid"][0], "namespace_pids": values["NSpid"]}


def _stat(raw):
    text = raw.decode("utf-8", "strict").strip()
    head, separator, tail = text.rpartition(") ")
    if not separator or " (" not in head: raise IdentityError("invalid proc stat")
    pid = head.split(" (", 1)[0]
    fields = tail.split()
    if (not pid.isdecimal() or len(fields) < 20 or not fields[1].isdecimal()
            or not fields[19].isdecimal() or int(fields[19]) <= 0):
        raise IdentityError("incomplete proc stat")
    return {"proc_pid": int(pid), "proc_ppid": int(fields[1]), "starttime_ticks": int(fields[19])}


def _identity(directory_fd):
    status = _status(_read_at(directory_fd, "status"))
    stat = _stat(_read_at(directory_fd, "stat"))
    if any(status[name] != stat[name] for name in ("proc_pid", "proc_ppid")):
        raise IdentityError("proc status/stat race or inconsistency")
    namespace = os.readlink("ns/pid", dir_fd=directory_fd)
    if re.fullmatch(r"pid:\[[0-9]+\]", namespace) is None:
        raise IdentityError("invalid PID namespace identity")
    if status["namespace_pids"][0] != status["proc_pid"]:
        raise IdentityError("proc PID not first namespace PID")
    return {**status, "starttime_ticks": stat["starttime_ticks"], "pid_namespace": namespace}


def _executable(directory_fd):
    path = os.readlink("exe", dir_fd=directory_fd)
    if not Path(path).is_absolute() or path.endswith(" (deleted)"):
        raise IdentityError("absolute live executable identity required")
    fd = os.open("exe", os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory_fd)
    try:
        stat = os.fstat(fd)
        if stat.st_size <= 0 or stat.st_size > 512 * 1024 * 1024:
            raise IdentityError("executable size outside fixed identity cap")
        h = hashlib.sha256()
        while block := os.read(fd, 1024 * 1024): h.update(block)
        after = os.fstat(fd)
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise IdentityError("executable changed during hashing")
        return {"path": path, "sha256": h.hexdigest(), "device": stat.st_dev,
                "inode": stat.st_ino, "bytes": stat.st_size}
    finally:
        os.close(fd)


def _directory(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def _alive(process):
    if process.poll() is not None: raise IdentityError("owned Popen child has exited")


@dataclass
class OwnedChild:
    process: object
    directory_fd: int
    parent_directory_fd: int
    parent: dict
    child: dict
    expected_executable: dict
    _closed: bool = False

    def snapshot(self):
        if self._closed: raise IdentityError("closed proc identity ticket")
        _alive(self.process)
        if _identity(self.parent_directory_fd) != self.parent:
            raise IdentityError("parent identity changed")
        current = _identity(self.directory_fd)
        if current != self.child: raise IdentityError("owned child identity/starttime changed")
        executable = _executable(self.directory_fd)
        if executable != self.expected_executable: raise IdentityError("owned child executable changed")
        _alive(self.process)
        return {"identity": current, "executable": executable}

    def maps_snapshot(self):
        before = self.snapshot()
        maps = _read_at(self.directory_fd, "maps", MAX_MAPS_BYTES).decode("utf-8", "strict")
        after = self.snapshot()
        if before != after: raise IdentityError("child changed around maps read")
        return maps, {"schema": SCHEMA, "parent": self.parent, "popen_pid": self.process.pid,
            "proc_visible_pid": self.child["proc_pid"], "before": before, "after": after,
            "maps_sha256": hashlib.sha256(maps.encode("utf-8")).hexdigest(),
            "scope": "owned immediate Popen child, pinned proc directory, stable namespace/starttime/exe around maps read"}

    def close(self):
        if not self._closed:
            self._closed = True
            try: os.close(self.directory_fd)
            finally: os.close(self.parent_directory_fd)


def capture_owned_child(process, expected_executable_path, expected_executable_sha256,
                        *, proc_root=Path("/proc"), current_pid=None):
    """Called immediately after the actual Popen, before consuming native READY.

proc_root/current_pid exist for deterministic fake procfs tests. Production
entrypoint supplies neither override. Never accepts a bare pid as ownership.
"""
    inner_pid = _positive(process.pid, "owned Popen PID")
    current_pid = os.getpid() if current_pid is None else _positive(current_pid, "parent PID")
    if re.fullmatch("[0-9a-f]{64}", expected_executable_sha256) is None:
        raise IdentityError("exact expected executable SHA256 required")
    expected_path = str(Path(expected_executable_path).resolve(strict=True))
    _alive(process)
    proc_root = Path(proc_root)
    parent_fd = _directory(proc_root / "self")
    selected_fd = None
    try:
        parent = _identity(parent_fd)
        if parent["namespace_pids"][-1] != current_pid:
            raise IdentityError("proc self namespace PID does not match current process")
        task_root = proc_root / "self" / "task"
        tasks = sorted(task_root.iterdir(), key=lambda p: p.name)
        if not tasks or len(tasks) > MAX_TASKS or any(not p.name.isdecimal() for p in tasks):
            raise IdentityError("parent task count/IDs outside fixed cap")
        children = set()
        for task in tasks:
            with (task / "children").open("rb") as stream: raw = stream.read(MAX_SMALL_BYTES + 1)
            if len(raw) > MAX_SMALL_BYTES: raise IdentityError("children record exceeds cap")
            for word in raw.decode("ascii", "strict").split():
                if re.fullmatch("[1-9][0-9]*", word) is None: raise IdentityError("invalid immediate-child PID")
                children.add(int(word))
            if len(children) > MAX_CHILDREN: raise IdentityError("immediate-child count exceeds cap")
        matches = []
        for candidate in sorted(children):
            try: fd = _directory(proc_root / str(candidate))
            except FileNotFoundError: continue  # unrelated owned child exited
            try:
                identity = _identity(fd)
                if identity["namespace_pids"][-1] != inner_pid: continue
                if (identity["proc_pid"] != candidate or identity["proc_ppid"] != parent["proc_pid"]
                        or identity["pid_namespace"] != parent["pid_namespace"]
                        or len(identity["namespace_pids"]) != len(parent["namespace_pids"])
                        or identity["starttime_ticks"] < parent["starttime_ticks"]):
                    raise IdentityError("matching PID is not same-namespace immediate child")
                matches.append(identity)
                if len(matches) > 1: raise IdentityError("ambiguous owned namespace PID")
                selected_fd, fd = fd, None
            finally:
                if fd is not None: os.close(fd)
        if len(matches) != 1: raise IdentityError("exactly one owned namespace PID required")
        executable = _executable(selected_fd)
        if executable["path"] != expected_path or executable["sha256"] != expected_executable_sha256:
            raise IdentityError("owned child is not expected executable bytes")
        ticket = OwnedChild(process, selected_fd, parent_fd, parent, matches[0], executable)
        ticket.snapshot()
        selected_fd = parent_fd = None
        return ticket
    finally:
        if selected_fd is not None: os.close(selected_fd)
        if parent_fd is not None: os.close(parent_fd)
