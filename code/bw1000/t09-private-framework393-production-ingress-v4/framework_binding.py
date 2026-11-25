"""Verify the full target framework used by the fixed T09 replay."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat

COMMIT = "a68952cb2285d6593b665147249d14ec1f8cecbc"
TREE = "7373ac3f6eee95828af4eb2e457ff4ea56a5d888"
MANIFEST_SHA256 = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
DEPLOYMENT = "training-qwen-a68952c-swift-initial-CPU-v1"
REPLAY_ENTRY_SHA256 = "8236aeb91210b0ec971b7470b78775519657b0b6049d890f2146451a6b9dc0c3"


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def exact_bytes(path, expected_size):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        raise ValueError(f"Unexpected framework file type or length: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(expected_size + 1)
    if len(data) != expected_size or stamp(before) != stamp(path.lstat()):
        raise ValueError(f"Framework file changed while reading: {path.name}")
    return data, before


def verify_framework(root, manifest_path):
    raw, _ = exact_bytes(manifest_path, 112315)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Deployed framework manifest differs")
    manifest = json.loads(raw)
    files = manifest["files"]
    if manifest["framework_commit"] != COMMIT or manifest["framework_tree"] != TREE or len(files) != 393:
        raise ValueError("Deployed framework revision differs")
    entry = files.get("tools/hygon_hunyuanvideo_first_update.py", {})
    if entry.get("SHA256") != REPLAY_ENTRY_SHA256:
        raise ValueError("Fixed-input HunyuanVideo replay entry differs")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Expected the canonical framework directory")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink inside the deployed framework tree")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("Special file inside the deployed framework tree")
    if actual != set(files):
        raise ValueError("Deployed source inventory differs")
    for name, pin in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Malformed framework manifest path")
        data, info = exact_bytes(root.joinpath(*relative.parts), pin["bytes"])
        if (hashlib.sha256(data).hexdigest() != pin["SHA256"]
                or hashlib.sha1(f"blob {len(data)}".encode() + b"\0" + data).hexdigest() != pin["git_blob"]
                or stat.S_IMODE(info.st_mode) != pin["mode"]):
            raise ValueError(f"Deployed framework digest or mode differs: {name}")
    return dict(framework_commit=COMMIT, framework_tree=TREE, framework_files=len(files),
                manifest_SHA256=MANIFEST_SHA256, replay_entry_SHA256=REPLAY_ENTRY_SHA256)


if __name__ == "__main__":
    import sys

    print(json.dumps(verify_framework(Path(sys.argv[1]), Path(sys.argv[2]))))
