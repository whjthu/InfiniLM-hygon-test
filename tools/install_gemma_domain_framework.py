"""Verify the 393-file shared base and build the private Gemma runtime tree."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from prepare_gemma_domain_pair import candidate
from training_framework_identity import git_object, manifest_tree


BASE_MANIFEST = (
    "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json",
    "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
)
BASE_COMMIT = "a68952cb2285d6593b665147249d14ec1f8cecbc"
MAX_FILE = 32 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def base_files(raw):
    data = json.loads(raw)
    if (digest(raw) != BASE_MANIFEST[1] or data["framework_commit"] != BASE_COMMIT
            or len(data["files"]) != 393 or manifest_tree(data["files"]) != data["framework_tree"]):
        raise ValueError("base is not the pinned 393-file framework")
    return data["files"]


def candidate_files(base, overlay):
    files = {path: dict(row) for path, row in base.items()}
    for path, data in overlay.items():
        files[path] = dict(bytes=len(data), SHA256=digest(data), git_blob=git_object("blob", data),
                           mode=files.get(path, {}).get("mode", 0o644), source="gemma_paired_domain_overlay")
    tree = manifest_tree(files)
    return files, tree


def exact_source(root, name, pin):
    source = root / name
    if (source.is_symlink() or source.resolve(strict=True) != source or not source.is_file()
            or not 0 <= pin["bytes"] <= MAX_FILE):
        raise ValueError(f"framework member is unsafe or oversized: {name}")
    with source.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    if len(raw) != pin["bytes"] or digest(raw) != pin["SHA256"] or git_object("blob", raw) != pin["git_blob"]:
        raise ValueError(f"framework bytes differ: {name}")
    return raw


def write_tree(source, destination, base, overlay, final):
    for name, record in final.items():
        raw = exact_source(source, name, base[name]) if name in base else None
        if name in overlay:
            raw = overlay[name]
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(record["mode"])
        if os.name == "posix" and path.stat().st_mode & 0o777 != record["mode"]:
            raise ValueError(f"framework executable mode not preserved: {name}")
        exact_source(destination, name, record)


def install(repo, base_path, output, target, models, plugins, port):
    base_root = base_path.resolve(strict=True)
    destination = output.resolve()
    if destination.is_relative_to(base_root) or base_root.is_relative_to(destination):
        raise ValueError("installation must stay separate from the verified framework base")
    if output.is_symlink():
        raise ValueError("fresh installation namespace must not be a symbolic link")
    base = base_files((repo / BASE_MANIFEST[0]).read_bytes())
    overlay, plans = candidate(repo, target, models, plugins, port)
    files, tree = candidate_files(base, overlay)
    destination.mkdir(mode=0o700)
    write_tree(base_root, destination / "framework", base, overlay, files)
    for arm, plan in plans.items():
        (destination / f"{arm}-private.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    private_manifest = json.dumps(dict(base_commit=BASE_COMMIT, framework_tree=tree, files=files),
                                  sort_keys=True, separators=(",", ":")).encode()
    (destination / "candidate-framework-private.json").write_bytes(private_manifest)
    receipt = dict(original_base_files_read=len(base), installed_files_verified=len(files),
                   candidate_framework_tree=tree, candidate_manifest_SHA256=digest(private_manifest),
                   exact_Git_commit_deployed=False, actual_updates=0, training_ready=False)
    with (destination / "installation-complete.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("base", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("target", "models", "plugins"):
        parser.add_argument("--" + name, type=PurePosixPath, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.repo, args.base, args.output, args.target, args.models, args.plugins, args.port),
                     sort_keys=True))


if __name__ == "__main__":
    main()
