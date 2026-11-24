"""Report independent phases of the Gemma VizWiz stage readback without exposing paths or values."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import verify_gemma_vizwiz_target_stage as qualified


MANIFEST_BYTES = {"processor": 2107, "framework": 112315}
FRAMEWORK_OBSERVER = "vlm/runtime/first_update_audit.py"


def inspect(action):
    try:
        return {"state": "VERIFIED", **action()}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"state": "UNVERIFIED", "error_class": type(error).__name__,
                "detail_SHA256": sha256(str(error).encode()).hexdigest()}


def parse_manifests(raw):
    end = sum(MANIFEST_BYTES.values())
    if len(raw) != end:
        raise ValueError("public manifest byte count differs")
    prefix = MANIFEST_BYTES["processor"]
    result = {}
    for name, block, digest in (
        ("processor", raw[:prefix], qualified.PROCESSOR_MANIFEST),
        ("framework", raw[prefix:], qualified.FRAMEWORK_MANIFEST),
    ):
        if sha256(block).hexdigest() != digest:
            raise ValueError("public manifest SHA differs")
        result[name] = json.loads(block)["files"]
    return result


def data():
    receipt = qualified.verify_data(qualified.STAGED, (qualified.STAGED / "data").as_posix())
    return {"receipt": receipt, "staged_files_verified": 7}


def processor(files):
    root = qualified.PROCESSOR
    if root.is_symlink() or len(files) != 14 or {p.name for p in root.iterdir()} != set(files):
        raise ValueError("offline processor namespace differs")
    for name, pin in files.items():
        if Path(name).name != name:
            raise ValueError("unexpected processor name")
        qualified.read(root / name, pin["bytes"], pin)
    return {"processor_files_verified": 14}


def framework(files):
    pin = files[FRAMEWORK_OBSERVER]
    qualified.read(qualified.FRAMEWORK / FRAMEWORK_OBSERVER, pin["bytes"], pin)
    return {"framework_observer_files_verified": 1}


def mounts():
    return {"host_visibility": {
        "apptainer_executable": os.access(qualified.APPTAINER, os.X_OK),
        "container": qualified.CONTAINER.is_dir(),
        "swift_runtime": (qualified.SWIFT / "site").is_dir(),
        "transformers_runtime": qualified.TRANSFORMERS.is_dir(),
        "hyhal": Path("/usr/local/hyhal").is_dir(),
    }, "compute_mount_checks": 0}


def diagnose(raw):
    manifests = parse_manifests(raw)
    return dict(
        request=qualified.REQUEST, image_data=inspect(data),
        offline_processor=inspect(lambda: processor(manifests["processor"])),
        framework_observer=inspect(lambda: framework(manifests["framework"])),
        login_host_mount_visibility=inspect(mounts),
        slurm_steps=0, gpu=0, native_encode_calls=0, training_READY=False,
    )


def main():
    raw = sys.stdin.buffer.read(sum(MANIFEST_BYTES.values()) + 1)
    result = diagnose(raw)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
