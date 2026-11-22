"""Check the original T03 Swift runtime on the currently owned T03 compute node."""

from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sys


def read_manifest(root, expected):
    data = (root / "SOURCE_FILES.json").read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("Frozen v5 manifest differs")
    manifest = json.loads(data)
    for name, pin in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected v5 source name")
        path = root / name
        content = path.read_bytes()
        if path.resolve(strict=True) != path or len(content) != pin["bytes"]:
            raise ValueError("Frozen v5 source path or size differs")
        if sha256(content).hexdigest() != pin["SHA256"]:
            raise ValueError("Frozen v5 source bytes differ")
    return manifest


def main():
    scope = Path(__file__).resolve().parent
    claim = json.loads((scope / "CLAIM.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    parent = os.environ["SLURM_JOB_ID"]
    step = os.environ["SLURM_STEP_ID"]
    if node != claim["node"] or parent != claim["parent"] or not step.isdecimal():
        raise ValueError("Unexpected T03 compute step")
    if any(os.environ.get(name) != "-1" for name in
           ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("GPU mask is incomplete")
    original_source = scope / "v5"
    checked = read_manifest(original_source, claim["v5_manifest_SHA256"])
    sys.path.insert(0, str(original_source))
    import native
    from scope import encoded

    runtime = native.runtime_identity()
    if runtime["torch_device_initialized"]:
        raise ValueError("Unexpected accelerator initialization")
    recipe = native.binding()
    count = len(recipe["trainable_manifest"])
    ranks = recipe["input_fingerprints_by_rank"]
    if count != 504 or set(ranks) != set(map(str, range(8))) or any(len(slots) != 8 for slots in ranks.values()):
        raise ValueError("Original Swift recipe or fingerprint denominator differs")
    from tools.hygon_vlm_recipe import build_launch

    original_launch = native.pin(original_source / "original-launch.json", expected=native.LAUNCH_SHA)
    launch = native.launch_recipe(build_launch, original_launch, package=original_source, run=scope)
    dataset = native.pin(native.DATASET, expected=claim["original_dataset_SHA256"], extent=26817)
    native.pin(native.ASSETS / "plugins/swift_callback_nvidia_baseline_v2.py",
               expected=native.BASELINE_SHA, extent=4692)
    if (native.INITIAL.resolve(strict=True) != native.INITIAL or
            native.INITIAL.stat().st_size != native.INITIAL_BYTES):
        raise ValueError("Original rank-zero initialization extent differs")
    scratch = native.private_tmp(scope / "tmp")
    unix = native.private_unix_socket()
    print(json.dumps({
        "request": claim["request"],
        "step": parent + "." + step,
        "node": node,
        "original_trainables": count,
        "original_reference_slots": sum(map(len, ranks.values())),
        "dataset_SHA256": sha256(dataset).hexdigest(),
        "live_launch_SHA256": sha256(encoded(launch)).hexdigest(),
        "training_recipe": recipe["spec"]["training"],
        "source_members": len(checked["files"]),
        "training_ready": False,
        **runtime,
        **scratch,
        **unix,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
