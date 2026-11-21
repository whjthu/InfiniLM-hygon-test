"""Stage the three-recording dataset for the existing VGGT camera-head training backend."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "code/bw1000/training-vggt-domain-eight-pairs25-v1"
BASE_BACKEND_SHA = "762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998"
ORIGINAL_SETTINGS_SHA = "6aef14e2326162edeca285a691e5629e74ca3800cb134348c668e2ff35f88816"
CONTRACT_SHA = "aef0765500140c237fb8c57adfa53d0ba024064c67a1716e9610ca3448f4d898"
ADAPTER_SHA = "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962"
COHORT_SOURCE_SHA = "0ca0e2ceb89643cc513482f9d3b100d3a1384f450d7cf52a62209adaecaa9bba"
ORIGINAL_BUNDLE_SHA = "fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c"
NEW_PAIR_SHA = "9a39e38c2acf5da7aacfc08729c60307cc048ea2293be81c83b40a69f6695cfa"
ORDER_SHA = "432012b6fce5fe874dd5883df184cd50f4f6d7dfb45f4b2699536839100c44d0"
MANIFEST = {
    "schema": "vggt-tum-three-recording-native-inputs-v1",
    "original_root": "original",
    "addition_root": "sitting_xyz",
    "input_order_SHA256": ORDER_SHA,
}
ROOT_BLOCK = '''REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "code/bw1000/training-vggt-domain-eight-pairs25-v1/data_contract.py"
ADAPTER = REPO / "code/bw1000/training-domain-pilot-v1/tum_vggt_pair.py"'''
RUNTIME_ROOT_BLOCK = '''ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "data_contract.py"
ADAPTER = ROOT / "tum_vggt_pair.py"'''
OVERLAY = '''"""Input-only variant of the unified VGGT camera-head collector."""

from .vggt_eight_pair_settings import SETTINGS as ORIGINAL_SETTINGS, check_config
from .vggt_three_recording_inputs import native_batches, read_pinned


SETTINGS = dict(ORIGINAL_SETTINGS, cached_batches=9)
INPUT_MANIFEST_SHA = {manifest_sha!r}
ORDER_SHA = {order_sha!r}


def batches(torch, config, tensor_batch, valid_depth):
    count = config.get("cached_batches")
    if type(count) is not int or count != 9:
        raise ValueError("Expected nine frozen RGB-D batches")
    root = check_config(dict(config, cached_batches=ORIGINAL_SETTINGS["cached_batches"]))
    if config["dataset_sha"] != INPUT_MANIFEST_SHA:
        raise ValueError("Three-recording dataset identity differs")
    read_pinned(root / "bundle.json", INPUT_MANIFEST_SHA, 2048)
    values, evidence = native_batches(torch, root / "original", root / "sitting_xyz", tensor_batch, valid_depth)
    if len(values) != count or evidence["input_order_SHA256"] != ORDER_SHA:
        raise ValueError("Three-recording adapter input order changed")
    return values, evidence
'''


def digest(data):
    return sha256(data).hexdigest()


def read_bound(path, expected, maximum):
    if path.is_symlink():
        raise ValueError(f"Aliased source: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum or digest(raw) != expected:
        raise ValueError(f"Frozen source differs: {path.name}")
    return raw


def emit(root, filename, data):
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)


def runtime_sources(old_backend, manifest):
    previous = read_bound(
        REPO / "tools/prepare_vggt_tum_three_recording_cohort.py", COHORT_SOURCE_SHA, 10000,
    ).decode("utf-8")
    if previous.count(ROOT_BLOCK) != 1:
        raise ValueError("Fixed cohort dependency paths changed")
    relocated = previous.replace(ROOT_BLOCK, RUNTIME_ROOT_BLOCK, 1)
    sources = {
        "backends/vggt_training.py": read_bound(old_backend, BASE_BACKEND_SHA, 70000),
        "vggt_eight_pair_settings.py": read_bound(OLD / "domain_overlay.py", ORIGINAL_SETTINGS_SHA, 10000),
        "tum_vggt_pair.py": read_bound(
            REPO / "code/bw1000/training-domain-pilot-v1/tum_vggt_pair.py", ADAPTER_SHA, 10000,
        ),
        "data_contract.py": read_bound(OLD / "data_contract.py", CONTRACT_SHA, 10000),
        "vggt_three_recording_inputs.py": relocated.encode("utf-8"),
        "domain_overlay.py": OVERLAY.format(manifest_sha=digest(manifest), order_sha=ORDER_SHA).encode("utf-8"),
    }
    for raw in sources.values():
        ast.parse(raw, feature_version=(3, 10))
    return sources


def pair_members(root, manifest_path, manifest_sha, expected_sequence):
    files = {manifest_path.as_posix(): read_bound(root / manifest_path, manifest_sha, 4096)}
    manifest = json.loads(files[manifest_path.as_posix()])
    if manifest["sequence"] != expected_sequence or len(manifest["files_SHA256"]) != 7:
        raise ValueError("Expected seven files for the frozen pair")
    directory = manifest_path.parent
    for name, pin in manifest["files_SHA256"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or rel.as_posix() != name:
            raise ValueError("Invalid RGB-D asset path")
        asset = directory / rel
        path = root / asset
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("RGB-D asset escapes its source root")
        files[asset.as_posix()] = read_bound(path, pin, 2 << 20)
    return files


def dataset_files(original, addition, manifest):
    old_raw = read_bound(original / "bundle.json", ORIGINAL_BUNDLE_SHA, 5000)
    old = json.loads(old_raw)
    result = {"bundle.json": manifest, "original/bundle.json": old_raw}
    for row in old["pairs"]:
        files = pair_members(original, PurePosixPath(row["manifest"]), row["manifest_SHA256"], row["sequence"])
        if any(f"original/{name}" in result for name in files):
            raise ValueError("Aliased row in original bundle")
        result.update({f"original/{name}": raw for name, raw in files.items()})
    extra = pair_members(addition, PurePosixPath("pair-manifest.json"), NEW_PAIR_SHA, "freiburg3_sitting_xyz")
    result.update({f"sitting_xyz/{name}": raw for name, raw in extra.items()})
    if len(old["pairs"]) != 8 or len(result) != 74:
        raise ValueError("Full 8 + 1 dataset closure differs")
    return result


def stage(old_backend, original, addition, output):
    manifest = (json.dumps(MANIFEST, sort_keys=True, indent=2) + "\n").encode()
    sources = runtime_sources(old_backend, manifest)
    assets = dataset_files(original, addition, manifest)
    output.mkdir()
    for name, raw in sources.items():
        emit(output / "gencuda", name, raw)
    for name, raw in assets.items():
        emit(output / "data", name, raw)
    return dict(state="NATIVE_INPUT_DELTA_STAGED", sdk_backend_SHA256=BASE_BACKEND_SHA,
                source_SHA256={name: digest(raw) for name, raw in sources.items()},
                manifest_SHA256=digest(manifest), logical_input_order_SHA256=ORDER_SHA, files=len(assets),
                asset_bytes=sum(map(len, assets.values())), training_updates=0, target_install_attempted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-backend", "original-bundle", "additional-pair", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.old_backend, args.original_bundle, args.additional_pair, args.output)
    emit(args.output, "stage.private.json", (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
