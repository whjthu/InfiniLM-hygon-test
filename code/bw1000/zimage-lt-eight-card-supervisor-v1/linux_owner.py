"""Observe and retire only process sessions launched by this Slurm-step supervisor."""

from dataclasses import asdict, dataclass
import ctypes
import os
from pathlib import Path
import re
import signal
import subprocess
import time


PROC = Path("/proc")
DRM = Path("/sys/class/drm")
KFD = Path("/sys/class/kfd/kfd/proc")


@dataclass(frozen=True)
class Identity:
    pid: int
    state: str
    parent: int
    group: int
    session: int
    start_ticks: int

    def receipt(self):
        return {key: value for key, value in asdict(self).items() if key != "state"}


def identity(root):
    try:
        record = (root / "stat").read_text()
    except FileNotFoundError:
        return None
    fields = record[record.rindex(")") + 2:].split()
    return Identity(int(root.name), fields[0], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19]))


def process_rows():
    return [row for path in PROC.iterdir() if path.name.isdecimal() and (row := identity(path)) is not None]


def become_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")


class Child:
    def __init__(self, process, directory, role, pci, event):
        self.process, self.directory, self.role, self.pci = process, directory, role, pci
        self.identity = identity(PROC / str(process.pid))
        if self.identity is None or (self.identity.parent, self.identity.group, self.identity.session) != (
                os.getpid(), process.pid, process.pid):
            raise ValueError("OWNED_SESSION_IDENTITY")
        self.event = event
        self.event["identity"] = self.identity.receipt()

    def exited(self):
        if self.process.returncode is not None:
            return True
        return os.waitid(os.P_PID, self.identity.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def signal(self, signum):
        # waitid holds the session leader's PID unreaped until retire() verifies the group.
        current = identity(PROC / str(self.identity.pid))
        if current is None or current.receipt() != self.identity.receipt():
            raise ValueError("OWNED_SESSION_CHANGED")
        try:
            os.killpg(self.identity.group, signum)
        except ProcessLookupError:
            pass

    def descendants(self):
        return [row for row in process_rows()
                if row.session == self.identity.session and row.pid != self.identity.pid]

    def retire(self, deadline):
        while time.monotonic() < deadline:
            descendants = self.descendants()
            for row in descendants:
                if row.state == "Z" and row.parent == os.getpid():
                    os.waitpid(row.pid, os.WNOHANG)
            if self.exited() and not descendants:
                self.event["returncode"] = self.process.wait(timeout=0.2)
                self.event["group_clean"] = True
                return self.event["returncode"]
            time.sleep(0.1)
        raise ValueError("OWNED_SESSION_NOT_CLEAN")


def spawn(command, directory, role, pci, environment, event):
    with (directory / "stdout.private").open("xb") as out, (directory / "stderr.private").open("xb") as err:
        process = subprocess.Popen(command, env=environment, start_new_session=True, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err)
    try:
        return Child(process, directory, role, pci, event)
    except (OSError, ValueError):
        process.kill()
        event["returncode"] = process.wait(timeout=1)
        raise


def audit_no_other_children(role_map):
    expected = {child.identity.pid: child.identity for child in role_map.values()}
    supervisor = os.getpid()
    actual = {row.pid: row for row in process_rows() if row.parent == supervisor}
    if set(actual) != set(expected) or any(actual[pid].state in ("Z", "X") or actual[pid].receipt() != row.receipt()
                                           for pid, row in expected.items()):
        raise ValueError("UNEXPECTED_SUPERVISOR_CHILD")


def audit_device_holders(standbys, expected_pcis):
    """Reject any device holder outside the surviving one-PCI torch children."""
    audit_no_other_children(standbys)
    expected = {str(child.identity.pid): pci for pci, child in standbys.items()}
    render = {entry.name: (entry / "device").resolve(strict=True).name.lower()
              for entry in DRM.iterdir() if re.fullmatch(r"(?:renderD|card)[0-9]+", entry.name)}
    render_pcis = {pci for name, pci in render.items() if name.startswith("renderD")}
    if render_pcis != expected_pcis:
        raise ValueError("DRM_PCI_SET_DIFFERS")
    kfd = {item.name for item in KFD.iterdir()}
    if not kfd <= set(expected):
        raise ValueError("UNOWNED_KFD_HOLDER")
    render_holders = {}
    for process in PROC.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            for entry in (process / "fd").iterdir():
                try:
                    target = os.readlink(entry)
                except FileNotFoundError:
                    continue
                if target == "/dev/kfd" and process.name not in expected:
                    raise ValueError("UNOWNED_PROJECT_KFD_FD")
                if target.startswith("/dev/dri/"):
                    pci = render.get(Path(target).name)
                    if pci is None or expected.get(process.name) != pci:
                        raise ValueError("UNOWNED_OR_CROSS_PCI_RENDER_FD")
                    render_holders[process.name] = pci
        except FileNotFoundError:
            continue
    if not kfd or set(render_holders) != set(expected):
        raise ValueError("SIBLING_DRIVER_HOLDER_UNPROVED")
    return {"sibling_driver_PID_to_PCI": render_holders, "KFD_PIDs": sorted(kfd), "DRM_PCIs": sorted(render_pcis)}
