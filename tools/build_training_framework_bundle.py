"""Export and verify the exact common training source used by the DP2 recovery."""
import argparse
import gzip
from hashlib import sha1, sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile


DESCRIPTOR = "coordination/training/hunyuanvideo-dp2-integrated-framework-tree-v1.json"
REGISTRY = "results/bw1000/2026-11-17/training-common-a68952c-live-17-case-registry.json"
EXPECTED_SOURCE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
                       "training-hunyuanvideo-cached-dp2-target-preflight-v4/framework")
BRANCH = "refs/heads/framework-dp2-alignment"
MAX_METADATA = 512 << 10
MAX_SOURCE = 6 << 20


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(stream, maximum):
    raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, "Missing or oversized source input")
    return raw


def blob_identity(raw):
    return sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def check_blob(raw, pin):
    require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"]
            and blob_identity(raw) == pin["git_blob"], "Source bytes differ from the verified framework tree")


def validated_manifest(data):
    manifest = json.loads(data)
    files = manifest["files"]
    require(len(files) == 393 and sum(pin["bytes"] for pin in files.values()) <= MAX_SOURCE
            and manifest["tree"] == "0df1c52f31415d2c20576601b113f2c3baa3d45f", "Unexpected framework inventory")
    for name, pin in files.items():
        relative = PurePosixPath(name)
        require(name and not relative.is_absolute() and relative.as_posix() == name
                and ".." not in relative.parts and not re.search(r"[\x00-\x1f\\]", name)
                and pin["mode"] in (0o644, 0o755) and 0 < pin["bytes"] <= MAX_SOURCE, "Invalid source member")
    return manifest


def prepare(repo):
    from training_framework_identity import manifest_tree, verify_overlay_source

    def pinned(name, expected):
        path = (repo / name).resolve(strict=True)
        require(path.is_relative_to(repo), "Source reference escaped the project")
        with path.open("rb") as stream:
            raw = bounded(stream, MAX_METADATA)
        require(sha256(raw).hexdigest() == expected, "Pinned project evidence changed")
        return raw

    descriptor = json.loads((repo / DESCRIPTOR).read_bytes())
    verified = verify_overlay_source(descriptor, pinned)
    base = descriptor["base_manifest"]
    files = json.loads(pinned(base["path"], base["SHA256"]))["files"]
    for row in descriptor["replacements"]:
        files[row["path"]] = row
    files = {name: {key: row[key] for key in ("mode", "bytes", "SHA256", "git_blob")} for name, row in files.items()}
    require(manifest_tree(files) == verified["framework_tree"], "Reconstructed Git tree differs")
    registry = json.loads((repo / REGISTRY).read_bytes())
    cases = registry["result"]["registry_cases"]
    require(len(cases) == len(set(cases)) == 17, "Frozen registry is incomplete")
    raw = json.dumps(dict(tree=verified["framework_tree"], files=files, registry_cases=cases,
                          original_framework_commit=descriptor["base_framework_commit"]),
                     sort_keys=True, separators=(",", ":")).encode()
    validated_manifest(raw)
    return raw


def export_source(expected_sha):
    manifest = bounded(sys.stdin.buffer, MAX_METADATA)
    require(sha256(manifest).hexdigest() == expected_sha, "Staged framework manifest changed")
    files = validated_manifest(manifest)["files"]
    source = EXPECTED_SOURCE.resolve(strict=True)
    require(source == EXPECTED_SOURCE and source.is_dir(), "Qualified framework mount differs")
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
            for name, pin in sorted(files.items()):
                path = source / name
                require(path.resolve(strict=True) == path and path.is_file(), "Source member is missing or linked")
                with path.open("rb") as member:
                    before = os.fstat(member.fileno())
                    raw = bounded(member, pin["bytes"])
                    after = os.fstat(member.fileno())
                stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                require(stamp(before) == stamp(after) == stamp(path.lstat()), "Source member changed during read")
                check_blob(raw, pin)
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(raw), pin["mode"]
                tar.addfile(info, io.BytesIO(raw))
    sys.stdout.buffer.write(stream.getvalue())


def archive_blobs(archive, files):
    blobs = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            require(member.isfile() and member.name in files and member.name not in blobs,
                    "Unexpected file in exported source inventory")
            pin = files[member.name]
            require(member.size == pin["bytes"] and member.mode == pin["mode"], "Exported file metadata differs")
            with tar.extractfile(member) as source:
                raw = bounded(source, member.size)
            check_blob(raw, pin)
            blobs[member.name] = raw
    require(set(blobs) == set(files), "Missing files in exported source inventory")
    return blobs


def make_bundle(manifest, archive, destination):
    files = manifest["files"]
    blobs = archive_blobs(archive, files)
    destination.mkdir()
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)

    def run(argv, data=None):
        return subprocess.run(argv, input=data, capture_output=True, timeout=35, check=True, env=env).stdout

    source_repo = destination / "source.git"
    verifier = destination / "verified.git"
    bundle = destination / "common-training-dp2.bundle"
    run(["git", "init", "--bare", str(source_repo)])
    message = ("Reconstruct the common training framework with the verified HunyuanVideo DP2 providers\n\n"
               f"Frozen original framework: {manifest['original_framework_commit']}\n").encode()
    stream = bytearray(f"commit {BRANCH}\n".encode()
                       + b"committer whjthu <haojie0429@gmail.com> 1795415143 +0000\n"
                       + f"data {len(message)}\n".encode() + message + b"\n")
    for name, raw in sorted(blobs.items()):
        mode = 0o100000 | files[name]["mode"]
        quoted = json.dumps(name, ensure_ascii=False)
        stream.extend(f"M {mode:o} inline {quoted}\ndata {len(raw)}\n".encode() + raw + b"\n")
    stream.extend(b"\ndone\n")
    source_git = ["git", "--git-dir", str(source_repo)]
    run([*source_git, "fast-import", "--quiet"], bytes(stream))
    run([*source_git, "bundle", "create", str(bundle), BRANCH])

    run(["git", "init", "--bare", str(verifier)])
    verified_git = ["git", "--git-dir", str(verifier)]
    run([*verified_git, "bundle", "verify", str(bundle)])
    run([*verified_git, "fetch", str(bundle), f"{BRANCH}:{BRANCH}"])
    actual_tree = run([*verified_git, "rev-parse", BRANCH + "^{tree}"]).decode().strip()
    require(actual_tree == manifest["tree"], "Bundle imported a different source tree")
    commit = run([*verified_git, "rev-parse", BRANCH]).decode().strip()
    run([*verified_git, "fsck", "--strict", "--no-reflogs", commit])
    checkout = destination / "framework"
    checkout.mkdir()
    run([*verified_git, "--work-tree", str(checkout), "checkout", BRANCH, "--", "."])
    registry_output = run([sys.executable, "-B", "-s", str(checkout / "tools/hygon_training_recipe.py"), "list"])
    cases = json.loads(registry_output)
    require(cases == manifest["registry_cases"], "The imported framework registry differs")
    with bundle.open("rb") as stream:
        bundle_pin = sha256(bounded(stream, 8 << 20)).hexdigest()
    return dict(candidate_commit=commit, git_tree=actual_tree, source_files=len(files), registered_cases=cases,
                verified_by_fresh_bundle_import=True, bundle_sha256=bundle_pin, bundle_bytes=bundle.stat().st_size,
                bundle_branch=BRANCH, executed_GPU_steps=0, new_case_acceptance=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    manifest_parser = modes.add_parser("prepare")
    manifest_parser.add_argument("repo", type=Path)
    source_parser = modes.add_parser("export")
    source_parser.add_argument("manifest_sha256")
    bundle_parser = modes.add_parser("bundle")
    for name in ("manifest", "archive", "destination"):
        bundle_parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        sys.stdout.buffer.write(prepare(args.repo.resolve(strict=True)))
    elif args.mode == "export":
        export_source(args.manifest_sha256)
    else:
        with args.manifest.open("rb") as stream:
            manifest = validated_manifest(bounded(stream, MAX_METADATA))
        result = make_bundle(manifest, args.archive.resolve(strict=True), args.destination.resolve())
        with (args.destination / "result.json").open("x", encoding="utf8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps(result, sort_keys=True))
