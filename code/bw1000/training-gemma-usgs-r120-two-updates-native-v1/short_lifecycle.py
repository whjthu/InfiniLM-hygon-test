"""Observe only top-level metadata in this owner's verified short mount."""
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import time


def observation(path, uid, *, max_members=128):
    root = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISDIR(root.st_mode)
            or stat.S_IMODE(root.st_mode) != 0o700 or root.st_uid != uid):
        raise ValueError("R46_SHORT_IPC_OBSERVATION_ROOT")
    root_key = root.st_dev, root.st_ino
    result = {}
    for child in path.iterdir():
        if len(result) >= max_members:
            raise ValueError("R46_SHORT_IPC_OBSERVATION_BOUND")
        try:
            meta = child.lstat()
        except FileNotFoundError:
            continue
        category = ("SCOPE" if child.name == ".scope" else
                    "PYMP" if re.fullmatch(r"pymp-[a-z0-9_]{8}", child.name) else
                    "TORCHINDUCTOR" if child.name.startswith("torchinductor_") else "OTHER")
        kind = ("DIRECTORY" if stat.S_ISDIR(meta.st_mode) else "SYMLINK" if stat.S_ISLNK(meta.st_mode)
                else "SOCKET" if stat.S_ISSOCK(meta.st_mode) else "REGULAR" if stat.S_ISREG(meta.st_mode)
                else "OTHER")
        key = sha256(os.fsencode(child.name)).hexdigest()
        identity = sha256(f"{meta.st_dev}:{meta.st_ino}:{meta.st_uid}".encode()).hexdigest()
        result[key] = dict(category=category, kind=kind, same_UID=meta.st_uid == uid,
                           same_mount=meta.st_dev == root.st_dev, private_mode=oct(stat.S_IMODE(meta.st_mode)),
                           identity_SHA256=identity)
    current = path.lstat()
    if root_key != (current.st_dev, current.st_ino):
        raise ValueError("R46_SHORT_IPC_OBSERVATION_ROOT_CHANGED")
    return result


class Lifecycle:
    def __init__(self, path, uid):
        self.path, self.uid = Path(path), uid
        self.started = time.monotonic()
        self.current = {}
        self.total_samples = 0
        self.errors = []
        self.events = []
        self.truncated_events = 0
        self.checkpoints = {}

    def sample(self, stage):
        self.total_samples += 1
        try:
            now = observation(self.path, self.uid)
        except (OSError, ValueError) as exc:
            if len(self.errors) < 16:
                self.errors.append(dict(stage=stage, error_class=type(exc).__name__,
                                        errno=getattr(exc,"errno",None)))
            return
        elapsed_ms = int((time.monotonic()-self.started)*1000)
        for key in sorted(self.current.keys() | now.keys()):
            previous, current = self.current.get(key), now.get(key)
            if previous == current:
                continue
            if len(self.events) >= 128:
                self.truncated_events += 1
                continue
            event = "ADDED" if previous is None else "DISAPPEARED" if current is None else "CHANGED"
            self.events.append(dict(stage=stage, elapsed_ms=elapsed_ms, event=event,
                                    basename_SHA256=key, current=current, previous=previous))
        self.current = now
        if stage != "RUNNING":
            self.checkpoints[stage] = now

    def evidence(self):
        return dict(schema="GEMMA_R46_REQUEST_SHORT_DIRECTORY_LIFECYCLE_V1",samples=self.total_samples,
                    observed_only=True,creator_attribution=False,children_traversed=False,raw_paths=False,
                    errors=self.errors,events=self.events,truncated_events=self.truncated_events,
                    checkpoints=self.checkpoints)


def receipt_domains(*, exact_rank_results, returncode, failure, cleanup_errors, held, tree, retirement):
    training = len(exact_rank_results) == 8 and returncode == 0 and failure is None
    owner = (not cleanup_errors and held.get("reaped") is True and held.get("termination_needed") is False
             and tree.get("owned_processes_remaining") == 0 and tree.get("termination_needed") is False
             and retirement.get("state") == "MARKER_ONLY")
    return dict(native_two_updates_witnessed=training, owner_cleanup_passed=owner, passed=training and owner)
