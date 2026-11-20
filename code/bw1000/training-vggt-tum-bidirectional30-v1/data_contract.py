"""Freeze and verify two training RGB-D pairs for the canonical VGGT camera head."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ADAPTER_SHA256 = "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962"
PAIRS = (
    ("near", "freiburg3_structure_texture_near",
     "b98b2473335cecd21c65be9dfb64d908b177802adfa74ce5750a5e4c4efeac50"),
    ("planar", "freiburg3_nostructure_texture_far",
     "b79cdedcf39261deb7f03cc2823785b8fdd94614187a9cfee1cf9bfb2a47e753"),
)
EXCLUDED = (
    ("freiburg3_cabinet", "eee2076fce62d51ad9e957983db96b0d4dcbf8e232fde0aa9ce46e87439b55b1"),
    ("freiburg3_teddy", "b9b364d65e11a121df251e3d3b3cef27bb486b1e0e92fdb7aa98004175a14ac7"),
    ("freiburg3_teddy", "5cfe85b6b405e0d789e02f8225cdd8f348f819d0fc79acfb4d383edb101b8f03"),
)
BLOCKED_SEQUENCES = ("freiburg3_cabinet", "freiburg3_teddy", "freiburg3_long_office_household")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or aliased frozen data")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("Frozen input SHA256 mismatch")
    return raw


def normalized(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("Invalid private bundle path")
    return path


def pair_files(root, manifest, manifest_sha):
    raw = pinned(manifest, manifest_sha)
    value = json.loads(raw)
    if value["sequence"] in BLOCKED_SEQUENCES or len(value["files_SHA256"]) != 7:
        raise ValueError("Expected training pair with seven bound files")
    files = {}
    for name, expected in value["files_SHA256"].items():
        relative = normalized(name)
        source = root.joinpath(*relative.parts)
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("Pair file escapes its source")
        files[name] = pinned(source, expected)
    return value, raw, files


def verify_bundle(root, manifest_sha):
    value = json.loads(pinned(root / "bundle.json", manifest_sha))
    expected = dict(schema="vggt-tum-two-training-pairs-v1", pairs=[
        dict(root=key, manifest=key + "/pair.json", manifest_SHA256=pin, sequence=sequence)
        for key, sequence, pin in PAIRS
    ], media_isolation_SCOPE="existing_one_train_and_three_development_pairs_exact_bytes",
        excluded_pair_manifests_SHA256=[pin for _, pin in EXCLUDED],
        original_blind_media_inspected=False, training_pair_count=2)
    if value != expected:
        raise ValueError("Training bundle contract changed")
    seen = set()
    for (key, sequence, pin), row in zip(PAIRS, value["pairs"], strict=True):
        pair, _, _ = pair_files(root / key, root / row["manifest"], pin)
        hashes = {v for n, v in pair["files_SHA256"].items() if n.endswith(".png")}
        if pair["sequence"] != sequence or len(hashes) != 4 or hashes & seen:
            raise ValueError("Repeated training media or wrong scene")
        seen.update(hashes)
    return value


def build(near, planar, excluded, output):
    if output.exists():
        raise FileExistsError(output)
    exclusions = set()
    for (sequence, pin), path in zip(EXCLUDED, excluded, strict=True):
        value = json.loads(pinned(path, pin))
        if value["sequence"] != sequence:
            raise ValueError("Development exclusion manifest differs")
        exclusions.update(v for n, v in value["files_SHA256"].items() if n.endswith(".png"))
    buffers = {}
    known = set()
    for (key, sequence, pin), (root, manifest) in zip(PAIRS, (near, planar), strict=True):
        value, raw, files = pair_files(root, manifest, pin)
        media = {v for n, v in value["files_SHA256"].items() if n.endswith(".png")}
        if value["sequence"] != sequence or len(media) != 4 or media & (known | exclusions):
            raise ValueError("Training scene or excluded media collision")
        known.update(media)
        buffers[key + "/pair.json"] = raw
        buffers.update({key + "/" + name: raw for name, raw in files.items()})
    value = dict(schema="vggt-tum-two-training-pairs-v1", pairs=[
        dict(root=key, manifest=key + "/pair.json", manifest_SHA256=pin, sequence=sequence)
        for key, sequence, pin in PAIRS
    ], media_isolation_SCOPE="existing_one_train_and_three_development_pairs_exact_bytes",
        excluded_pair_manifests_SHA256=[pin for _, pin in EXCLUDED],
        original_blind_media_inspected=False, training_pair_count=2)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    buffers["bundle.json"] = raw
    output.mkdir()
    for name, data in buffers.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    verify_bundle(output, sha(raw))
    return dict(bundle_SHA256=sha(raw), files=len(buffers), media=8, bytes=sum(map(len, buffers.values())),
                source_manifest_pins=[pin for _, _, pin in PAIRS], dataset_geometry_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("near_root", "near_manifest", "planar_root", "planar_manifest", "output"):
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--excluded", nargs=3, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build((args.near_root,args.near_manifest), (args.planar_root,args.planar_manifest),
                           args.excluded, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
