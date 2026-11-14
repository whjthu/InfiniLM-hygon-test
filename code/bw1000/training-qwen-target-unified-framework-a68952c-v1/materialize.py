"""Materialize the frozen shared training framework from three verified sources."""
import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

COMMIT = "a68952cb2285d6593b665147249d14ec1f8cecbc"
TREE = "7373ac3f6eee95828af4eb2e457ff4ea56a5d888"
MANIFEST_SHA256 = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
COUNTS = {"qualified_target_parent": 304, "published_delta": 22, "new_delta": 67}


def checked_file(root, relative, pin):
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in source path {relative}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != pin["bytes"]:
        raise ValueError(f"Source file type or size differs: {relative}")
    data = path.read_bytes()
    if (len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["SHA256"]
            or hashlib.sha1(f"blob {len(data)}".encode() + b"\0" + data).hexdigest() != pin["git_blob"]):
        raise ValueError(f"Source file digest differs: {relative}")
    return data


def materialize(manifest_path, target_parent, published, new, output):
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Unexpected framework manifest digest")
    manifest = json.loads(raw)
    roots = dict(qualified_target_parent=target_parent, published_delta=published, new_delta=new)
    if (manifest["framework_commit"] != COMMIT or manifest["framework_tree"] != TREE
            or manifest["source_counts"] != COUNTS or len(manifest["files"]) != sum(COUNTS.values())):
        raise ValueError("Unexpected framework identity or inventory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    sources = []
    for name, pin in manifest["files"].items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or '..' in relative.parts or '\\' in name
                or relative.as_posix() != name or pin["mode"] not in (0o644, 0o755)):
            raise ValueError(f"Unexpected framework path or mode: {name}")
        sources.append((relative, pin["mode"], checked_file(roots[pin["source"]], relative, pin)))
    output.mkdir(mode=0o700)
    for relative, mode, data in sources:
        dest = output.joinpath(*relative.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(data)
        dest.chmod(mode)
    return {"framework_commit": COMMIT, "framework_tree": TREE,
            "framework_files": len(sources), "manifest_SHA256": MANIFEST_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "target_parent", "published", "new", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.manifest, args.target_parent, args.published, args.new, args.output)))


if __name__ == "__main__":
    main()
