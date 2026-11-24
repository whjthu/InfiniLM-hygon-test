"""Classify denied procfs accesses without substituting for the holder audit."""

import errno
import os
from pathlib import PurePosixPath
import re


OPERATIONS = {"process_stat", "fd_enumeration", "fd_readlink"}
RELATIONSHIPS = {"process_uid_unresolved", "process_identity_unresolved", "supervisor_self",
                 "owned_surviving_standby", "same_numbered_step_unowned",
                 "same_uid_outside_numbered_step", "same_uid_step_unresolved"}


class HolderAccessDenied(PermissionError):
    def __init__(self, error, operation, relationship, in_kfd):
        super().__init__(error.errno, "FD_HOLDER_AUDIT_DENIED")
        self.diagnostic = {"operation": operation, "errno": error.errno,
                           "process_relationship": relationship, "process_in_KFD_listing": in_kfd,
                           "holder_audit_complete": False, "target_clean": False}


def terminal_diagnostic(terminal, *, request, step, scope_sha, runtime_sha):
    """Return only the diagnostic fields after the caller pins the terminal file."""
    if (terminal.get("request"), terminal.get("step"), terminal.get("scope_SHA256"),
            terminal.get("runtime_SHA256")) != (request, step, scope_sha, runtime_sha):
        raise ValueError("FD_AUDIT_TERMINAL_IDENTITY")
    row = terminal.get("holder_audit_denial")
    if row is None:
        return None
    if (not isinstance(row, dict) or row.get("operation") not in OPERATIONS
            or row.get("boundary") not in {"pre_witness", "pre_restore"}
            or row.get("process_relationship") not in RELATIONSHIPS
            or row.get("errno") not in {errno.EACCES, errno.EPERM}
            or type(row.get("process_in_KFD_listing")) is not bool
            or row.get("holder_audit_complete") is not False or row.get("target_clean") is not False
            or terminal.get("restored_eight_proved") is not False
            or terminal.get("error", {}).get("type") != "HolderAccessDenied"
            or (row["boundary"] == "pre_witness" and "witness" in terminal)):
        raise ValueError("FD_AUDIT_TERMINAL_DIAGNOSTIC")
    keys = ("boundary", "operation", "process_relationship", "errno", "process_in_KFD_listing",
            "holder_audit_complete", "target_clean")
    return {key: row[key] for key in keys}


def names_step(raw, step):
    if not re.fullmatch(r"[0-9]+\.[0-9]+", step):
        raise ValueError("numbered step required")
    job, ordinal = step.split(".")
    wanted = (f"job_{job}", f"step_{ordinal}")
    paths = []
    for line in raw.decode("utf-8").splitlines():
        hierarchy, _, pathname = line.split(":", 2)
        path = PurePosixPath(pathname)
        if not hierarchy.isdecimal() or not path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid cgroup record")
        paths.append(path.parts)
    if not paths:
        raise ValueError("empty cgroup record")
    return any(wanted in zip(parts, parts[1:]) for parts in paths)


def cgroup_bytes(process):
    with (process / "cgroup").open("rb") as stream:
        raw = stream.read(16385)
    if not 0 < len(raw) <= 16384:
        raise ValueError("cgroup record size")
    return raw


class Scan:
    def __init__(self, proc, uid, supervisor_pid, step, expected, kfd, identity):
        self.proc, self.uid, self.supervisor_pid = proc, uid, str(supervisor_pid)
        self.step, self.expected, self.kfd, self.identity = step, expected, kfd, identity

    def relationship(self, process, same_uid):
        if not same_uid:
            return "process_uid_unresolved"
        try:
            if process.stat().st_uid != self.uid:
                return "process_uid_unresolved"
            before = self.identity(process)
            if before is None or before.state in ("Z", "X"):
                return "process_identity_unresolved"
            expected = self.expected.get(process.name)
            if expected is not None:
                if before.receipt() != expected.receipt():
                    return "process_identity_unresolved"
                return "owned_surviving_standby"
            if process.name == self.supervisor_pid:
                return "supervisor_self"
            if not names_step(cgroup_bytes(self.proc / self.supervisor_pid), self.step):
                return "same_uid_step_unresolved"
            matches = names_step(cgroup_bytes(process), self.step)
            after = self.identity(process)
            if after is None or after.state in ("Z", "X") or after.receipt() != before.receipt():
                return "process_identity_unresolved"
            if process.stat().st_uid != self.uid:
                return "process_uid_unresolved"
            return "same_numbered_step_unowned" if matches else "same_uid_outside_numbered_step"
        except (OSError, UnicodeError, ValueError, IndexError):
            return "same_uid_step_unresolved"

    def denied(self, error, operation, process, same_uid):
        if error.errno not in (errno.EACCES, errno.EPERM):
            raise error
        return HolderAccessDenied(error, operation, self.relationship(process, same_uid),
                                  process.name in self.kfd)

    def uid_matches(self, process):
        try:
            return process.stat().st_uid == self.uid
        except PermissionError as error:
            raise self.denied(error, "process_stat", process, False) from None

    def fd_entries(self, process):
        try:
            yield from (process / "fd").iterdir()
        except PermissionError as error:
            raise self.denied(error, "fd_enumeration", process, True) from None

    def fd_target(self, process, entry):
        try:
            return os.readlink(entry)
        except PermissionError as error:
            raise self.denied(error, "fd_readlink", process, True) from None
