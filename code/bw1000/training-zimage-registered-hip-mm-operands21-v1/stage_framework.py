"""Stage two pinned profiler sources on the qualified target framework tree."""

from base64 import b64decode
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
BASE = PROJECT / "results/training-common-zimage-complete-window-v1"
OLD_RECEIPT = "372c280ff6075002a26cfd2f35106b0a13174de0ca7f45ca831aa5dd8ea701fa"
OLD_MANIFEST = "b9073489c0e89650e3037dc65590f253249cc90f7c4ace53b90711e16b8a8991"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def exact(path, pin):
    require(path.is_file() and path.resolve(strict=True) == path
            and path.stat().st_size == pin["bytes"], "Wrong target source extent")
    data = path.read_bytes()
    require(digest(data) == pin["SHA256"], "Wrong target source identity")
    return data


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def stage(scope, payload):
    framework = scope["framework"]
    receipt_raw = (BASE / "source-stage-receipt.json").read_bytes()
    old_raw = (BASE / "candidate-manifest.json").read_bytes()
    require(digest(receipt_raw) == OLD_RECEIPT and digest(old_raw) == OLD_MANIFEST,
            "Original common framework source receipt changed")
    previous = json.loads(old_raw)
    require(previous["commit"] == framework["parent_commit"] and len(previous["files"]) == 396,
            "Previous source tree differs")
    mraw = b64decode(payload["manifest"], validate=True)
    require(digest(mraw) == framework["private_artifacts"]["candidate-manifest.json"]["SHA256"],
            "New framework manifest changed")
    manifest = json.loads(mraw)
    require((manifest["commit"], manifest["tree"], len(manifest["files"])) ==
            (framework["candidate_commit"], framework["git_tree"], 397), "New framework identity differs")
    changed = payload["changed"]
    require(set(changed) == set(framework["changed"]) and len(changed) == 2, "Unexpected source delta")
    verified = {}
    for name, pin in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name,
                "Invalid framework member path")
        if name in changed:
            raw = b64decode(changed[name], validate=True)
            require(pin == framework["changed"][name] and len(raw) == pin["bytes"] and
                    digest(raw) == pin["SHA256"], "Observer source differs")
            compile(raw, name, "exec")
        else:
            require(pin == previous["files"][name], "Unexpected inherited framework change")
            raw = exact(BASE / "source-tree" / name, pin)
        verified[name] = raw
    require(sum(map(len, verified.values())) <= 8 * 1024 * 1024, "New framework source extent exceeds budget")
    destination = PROJECT / scope["framework_target"]
    os.umask(0o077)
    destination.mkdir()
    tree = destination / "source-tree"
    tree.mkdir()
    write(destination / "candidate-manifest.json", mraw)
    for name, raw in verified.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, raw)
        path.chmod(manifest["files"][name]["mode"])
    for name, pin in manifest["files"].items():
        exact(tree / name, pin)
    command = [sys.executable, "-B", "-X", "utf8", "tools/hygon_training_recipe.py"]
    registry = json.loads(subprocess.check_output([*command, "list"], cwd=tree, timeout=5))
    require(registry == framework["registry"] and len(registry) == 17, "Framework registry differs")
    help_text = subprocess.check_output([*command, "T07.sft", "--native-engineering", "--help"], cwd=tree, timeout=5)
    require(b"--profile-resumed21" in help_text and b"--measure-complete-window" in help_text,
            "Registered profile or complete window options differ")
    result = dict(request=scope["request"], candidate_commit=manifest["commit"],
                  source_files=len(verified), inherited_source_files=len(verified) - len(changed),
                  target_source_rehash_verified=True, profile21_option_verified=True,
                  operand_observer_verified=True, complete_window_option_verified=True,
                  case_count=len(registry), source_publication=payload["publication"], Slurm=0, GPU=0)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    write(destination / "stage-result.json", raw)
    print(json.dumps(dict(result, stage_result_sha256=digest(raw)), sort_keys=True), flush=True)


if __name__ == "__main__":
    stage(BOUND_SCOPE, json.loads(gzip.decompress(b64decode(BOUND_BUNDLE, validate=True))))
