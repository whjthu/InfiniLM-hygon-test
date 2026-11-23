"""Install the hashed HunyuanVideo observer into an isolated common framework checkout."""

import hashlib
import json
from pathlib import PurePosixPath
import stat
import os

from framework_binding import COMMIT, MANIFEST_SHA256, TREE
from render_t09_phase_profile import render


BACKEND = "generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"
OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_phase_profile.py"
OBSERVER_SHA = "a237fb3930ae5b60c03e576082ccfbb52720e95d89421bc203dfb434f229576a"
PATCHED_SHA = "614f4d8598ef8b5d913183242d4760d6a56ae20520974f6c8073a54d104439e8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_pin(data, mode):
    return {"bytes": len(data), "SHA256": sha(data), "mode": mode}


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def original_inventory(manifest):
    raw = manifest.read_bytes()
    if len(raw) != 112315 or sha(raw) != MANIFEST_SHA256:
        raise ValueError("Frozen parent manifest differs")
    value = json.loads(raw)
    if (value["framework_commit"], value["framework_tree"], len(value["files"])) != (COMMIT, TREE, 393):
        raise ValueError("Frozen parent revision differs")
    return value["files"]


def read_member(root, name, pin):
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or member.as_posix() != name:
        raise ValueError("Invalid frozen framework member")
    path = root.joinpath(*member.parts)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != pin["bytes"] or path.resolve(strict=True) != path:
        raise ValueError("Frozen framework member type differs")
    with path.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    expected = {key: pin[key] for key in ("bytes", "mode", "SHA256")}
    if source_pin(raw, stat.S_IMODE(before.st_mode)) != expected or stamp(before) != stamp(path.lstat()):
        raise ValueError("Frozen framework member bytes differ")
    if "git_blob" in pin:
        blob = hashlib.sha1(f"blob {len(raw)}".encode() + b"\0" + raw).hexdigest()
        if blob != pin["git_blob"]:
            raise ValueError("Frozen framework Git blob differs")
    return raw


def verify_staged(parent, staged, manifest, observer):
    if sha(observer) != OBSERVER_SHA:
        raise ValueError("Frozen observer source differs")
    pins = original_inventory(manifest)
    patched = render(read_member(parent, BACKEND, pins[BACKEND]))
    if sha(patched) != PATCHED_SHA or OBSERVER in pins:
        raise ValueError("Frozen diagnostic overlay differs")
    pins[BACKEND] = source_pin(patched, pins[BACKEND]["mode"])
    pins[OBSERVER] = source_pin(observer, 0o644)
    leaves = set()
    for path in staged.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Unexpected file type in staged framework")
        if path.is_file():
            leaves.add(path.relative_to(staged).as_posix())
    if leaves != set(pins):
        raise ValueError("Staged framework inventory differs")
    for name, pin in pins.items():
        read_member(staged, name, pin)
    return {
        "common_parent": {"framework_commit": COMMIT, "framework_tree": TREE,
                          "manifest_SHA256": MANIFEST_SHA256, "framework_files": len(pins) - 1},
        "unchanged_members": len(pins) - 2,
        "total_members": len(pins),
        "changes": {name: pins[name] for name in (BACKEND, OBSERVER)},
        "inventory_SHA256": sha(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()),
    }


def stage(parent, manifest, destination, observer):
    if sha(observer) != OBSERVER_SHA:
        raise ValueError("Frozen observer source differs")
    original = original_inventory(manifest)
    if OBSERVER in original:
        raise ValueError("Observer already present in the baseline")
    destination.mkdir(mode=0o700)
    for name, pin in original.items():
        raw = read_member(parent, name, pin)
        data = render(raw) if name == BACKEND else raw
        target = destination / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        os.chmod(target, pin["mode"])
    with (destination / OBSERVER).open("xb") as stream:
        stream.write(observer)
    os.chmod(destination / OBSERVER, 0o644)
    return verify_staged(parent, destination, manifest, observer)
