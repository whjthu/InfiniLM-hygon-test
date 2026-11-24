"""Reconstruct the immutable MiniCPM recovery controller files from Git blobs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess


COMMIT = "8a0780fbc8fc0c7cab79b20d038772093787d14d"
BASE = "code/bw1000"
CONTROL = "coordination/training/minicpm-usgs-target-control-files-v3.json"
MANIFESTS = {
    CONTROL: "eb4e2de22895f7ede8273e5693829bfe9d025958b3fe7abcaf96586b574060cb",
    f"{BASE}/training-minicpm-checkpoint20-first-forward-target-v1/SOURCE_FILES.json":
        "f8063e2475514d6463d9d5afd48dd6252457334d2a33ccf0490c25c0865c9fb6",
    f"{BASE}/training-minicpm-checkpoint20-first-forward-source-v1/SOURCE_FILES.json":
        "b78196e08b364f64b4755474185b9d0118931ab01f074944af67bf9078b73046",
    f"{BASE}/training-minicpm-checkpoint20-rng-bracket-source-v1/SOURCE_FILES.json":
        "90966b2ec210ac68eaceedbeb7cc7cdf59a5396070eec22936097f9ddf37fe64",
    f"{BASE}/training-minicpm-checkpoint20-rng-bracket-target-v1/SOURCE_FILES.json":
        "258be1be7b37b8ff61e6c14d08ce1ecb31dbd4d4f14030fa2488f697cb831442",
}
RESULTS = {
    "results/bw1000/2026-11-22/training-minicpm-native88-structured-v3-1171-paired-recovery-terminal.json":
        "816e3561638b1625aa18038b8dfb50fc2e9c343570501cbde6a7c29839d5be86",
    "results/bw1000/2026-11-22/training-minicpm-original1189-checkpoint20-first-forward-terminal.json":
        "a367dcb55e35aed72975f5910e09d2a2d20bbd60f005efdd988bcf48f1d9a105",
    "results/bw1000/2026-11-22/training-minicpm-original1189-first-forward-SDPA-route.json":
        "db7bcb1609050b843b8260b018e61849f683a91af5991fdb52149141e5156b4a",
}
RECIPE_FILES = {
    f"{BASE}/training-minicpm-usgs-exposure25-resume20-hip-v1/minicpm_domain_recipe.py":
        "4864bf57c7d66c8df188115c61b65d342a10f309988fdf860478cd734fd47531",
    f"{BASE}/training-minicpm-usgs-exposure25-resume20-hip-v1/original-launch.json":
        "015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789",
    f"{BASE}/training-qwen-checkpointing-lifecycle-CPU-v1/framework/tools/hygon_vlm_recipe.py":
        "71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2",
}


def blob(repo, name, expected, size=None):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe archived Git path")
    raw = subprocess.check_output(["git", "-c", "safe.directory=" + repo.as_posix(), "show",
                                   COMMIT + ":" + name], cwd=repo, timeout=10)
    if (not 0 < len(raw) <= 2 * 1024**2 or (size is not None and len(raw) != size)
            or sha256(raw).hexdigest() != expected):
        raise ValueError("archived MiniCPM dependency differs: " + name)
    return raw


def materialize(repo, destination):
    """Verify all members before creating an isolated source directory."""
    docs = {name: blob(repo, name, pin) for name, pin in MANIFESTS.items()}
    members = dict(docs)
    members.update({name: blob(repo, name, pin) for name, pin in (RESULTS | RECIPE_FILES).items()})
    members.update({name: blob(repo, name, pin) for name, pin in json.loads(docs[CONTROL]).items()})
    for name in MANIFESTS.keys() - {CONTROL}:
        folder = PurePosixPath(name).parent
        listing = json.loads(docs[name])["files"]
        for leaf, pin in listing.items():
            if PurePosixPath(leaf).name != leaf:
                raise ValueError("unexpected archived manifest path")
            path = (folder / leaf).as_posix()
            members[path] = blob(repo, path, pin["SHA256"], pin["bytes"])
    destination.mkdir(mode=0o700)
    for name, raw in sorted(members.items()):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    result = {"git_commit": COMMIT, "exact_files": len(members),
              "manifest_SHA256": MANIFESTS,
              "member_set_SHA256": sha256(json.dumps({name: sha256(raw).hexdigest() for name, raw in members.items()},
                                                  sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    with (destination / "ARCHIVED_HOST_SOURCE.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.repo.resolve(strict=True), args.out.absolute()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
