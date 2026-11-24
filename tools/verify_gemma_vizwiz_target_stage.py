"""Independently verify the installed VizWiz input packet and offline CPU mounts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
STAGED = RESULTS / "training-gemma-vizwiz-five-native-input-audit-v1"
PROCESSOR = RESULTS / "training-gemma-processor-assets-v1/models/gemma-3-4b-it"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
SWIFT = RESULTS / "training-vlm-swift-offline-runtime-v1"
TRANSFORMERS = RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site"
CONTAINER = PROJECT / "containers/dtk26-sandbox-r1"
APPTAINER = Path("/public/software/apps/apptainer/1.3.4/bin/apptainer")
REQUEST = "41eeb2278050498c8db75c54789637bb"
BINDING = {"bytes": 2743, "SHA256": "389c4d3d0284b153f25800ae66e78401b26b4c209ce3571f5cba2999c5fdee85"}
ROWS = {"bytes": 2164, "SHA256": "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"}
PROCESSOR_MANIFEST = "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b"
FRAMEWORK_MANIFEST = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
TERMINAL = dict(request=REQUEST, state="DATA_STAGED", completed=7, payload_bytes=1410330,
                slurm=0, gpu=0, optimizer_updates=0, training_READY=False)


def read(path, maximum, expected=None):
    def location(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size)

    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode) or not 0 < entry.st_size <= maximum:
        raise ValueError("expected a bounded non-aliased regular file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if location(opened) != location(entry):
            raise ValueError("installed file was replaced before the read")
        raw = stream.read(maximum + 1)
        final = os.fstat(stream.fileno())
    if (len(raw) != opened.st_size or location(final) != location(opened) or location(path.lstat()) != location(opened)
            or (opened.st_mtime_ns, opened.st_ctime_ns) != (final.st_mtime_ns, final.st_ctime_ns)):
        raise ValueError("source changed while being read")
    observed = {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}
    if expected is not None and observed != expected:
        raise ValueError("installed bytes did not match the original packet")
    return raw, observed


def verify_data(root, bound_data_path):
    expected_root = {"stage-intent.json", "stage-outcome.json", "data"}
    if root.is_symlink() or {p.name for p in root.iterdir()} != expected_root:
        raise ValueError("unexpected target staging namespace; inspect the original request")
    intent, _ = read(root / "stage-intent.json", 2048)
    receipt, receipt_pin = read(root / "stage-outcome.json", 2048)
    if json.loads(intent) != {"request": REQUEST, "replay_allowed": False} or json.loads(receipt) != TERMINAL:
        raise ValueError("terminal did not verify the original staged packet")
    data = root / "data"
    if data.is_symlink() or {x.name for x in data.iterdir()} != {
            "data-binding-private.json", "train-swift.jsonl", "images"}:
        raise ValueError("unexpected exact installed data members")
    plan = json.loads(read(data / "data-binding-private.json", 8192, BINDING)[0])
    read(data / "train-swift.jsonl", 8192, ROWS)
    images = data / "images"
    media = plan["media"]
    if (plan["target_root"] != bound_data_path or images.is_symlink()
            or {x.name for x in images.iterdir()} != set(media) or len(media) != 5):
        raise ValueError("the five frozen media files were not staged")
    for name, attributes in media.items():
        read(images / name, 384 * 1024, {f: attributes[f] for f in ("bytes", "SHA256")})
    return receipt_pin


def manifest(path, expected_sha256):
    raw, pin = read(path, 128 * 1024)
    if pin["SHA256"] != expected_sha256:
        raise ValueError("offline provider manifest differed")
    return json.loads(raw)["files"]


def verify_assets(processor_manifest, framework_manifest):
    processor_members = manifest(processor_manifest, PROCESSOR_MANIFEST)
    if (PROCESSOR.is_symlink() or len(processor_members) != 14
            or {p.name for p in PROCESSOR.iterdir()} != set(processor_members)):
        raise ValueError("unexpected installed processor metadata-only namespace")
    for name, pin in processor_members.items():
        if Path(name).name != name:
            raise ValueError("unexpected processor manifest member")
        read(PROCESSOR / name, pin["bytes"], pin)
    framework_members = manifest(framework_manifest, FRAMEWORK_MANIFEST)
    observer = "vlm/runtime/first_update_audit.py"
    expected = framework_members[observer]
    read(FRAMEWORK / observer, expected["bytes"], expected)
    if any(not p.is_dir() for p in (FRAMEWORK, SWIFT / "site", TRANSFORMERS, CONTAINER, Path("/usr/local/hyhal"))):
        raise ValueError("one or more original offline mount directories are unavailable")
    if not APPTAINER.is_file():
        raise ValueError("qualified Apptainer path is unavailable")
    return len(processor_members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-manifest", type=Path, required=True)
    parser.add_argument("--framework-manifest", type=Path, required=True)
    options = parser.parse_args()
    receipt = verify_data(STAGED, (STAGED / "data").as_posix())
    files = verify_assets(options.processor_manifest, options.framework_manifest)
    print(json.dumps(dict(
        request=REQUEST, independent_stage_receipt=receipt, staged_files_verified=7,
        installed_processor_metadata_files_verified=files, native_encode_calls=0,
        full_model_weights_loaded=0, slurm_steps=0, gpu=0, actual_resource_admitted=False,
        training_READY=False,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
