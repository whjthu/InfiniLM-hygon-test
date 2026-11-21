"""Exercise relocated real images, and stop the exact backend at subprocess.run."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
from types import SimpleNamespace

from entry import BACKEND_SHA256, BINDING_SHA256, TRAINER_RELATIVE, run_via_common_backend


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz"
ARCHIVE_SHA = "85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f"
MEMBER = "framework/generation/src/gencuda/backends/sdxl_training.py"
FUNCTIONS = {"run_sdxl_training", "_positive_config_int", "_external_checkpoint"}


class SubprocessBoundary(Exception):
    pass


def frozen_backend(out, intercept):
    archive = ARCHIVE.read_bytes()
    assert hashlib.sha256(archive).hexdigest() == ARCHIVE_SHA
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember(MEMBER)
        assert member.isfile() and member.size == 11576
        source = tar.extractfile(member).read()
    assert hashlib.sha256(source).hexdigest() in BACKEND_SHA256
    provider_path = out / "frozen_gencuda_backend.py"
    provider_path.write_bytes(source)
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {node.name for node in nodes} == FUNCTIONS
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(provider_path), "exec")
    scope = dict(Path=Path, Any=object, re=re, os=os, sys=sys,
                 subprocess=SimpleNamespace(run=intercept, STDOUT=-2))
    exec(code, scope)
    return SimpleNamespace(run_sdxl_training=scope["run_sdxl_training"])


def main():
    origin = time.monotonic()
    local = ROOT / "local-only"
    out = local / "training-sdxl-USGS-style18-portable-entry-v1-first"
    out.mkdir(exist_ok=False)
    original = local / "training-sdxl-USGS-style18-local-v1-original/dataset/train"
    relocated = out / "relocated-train-only"
    shutil.copytree(original, relocated)
    metadata = relocated / "metadata.jsonl"
    image = next(relocated.glob("*.png"))
    source = out / "sentinel-source"
    trainer = source / TRAINER_RELATIVE
    trainer.parent.mkdir(parents=True)
    trainer.write_bytes(b"# CPU test path sentinel; no deployed trainer or native execution.\n")
    trainer_sha = hashlib.sha256(trainer.read_bytes()).hexdigest()
    weights = out / "empty-test-weights"
    weights.mkdir()
    commands = []

    def intercept(command, **kwargs):
        commands.append(list(command))
        raise SubprocessBoundary

    backend = frozen_backend(out, intercept)
    kwargs = dict(weights=weights, output_dir=out / "no-native-output", source_root=source,
                  train_data_dir=relocated, trainer_sha256=trainer_sha)

    def reject(message, **overrides):
        try:
            run_via_common_backend(backend, **{**kwargs, **overrides})
        except ValueError as error:
            assert str(error) == message, str(error)
        else:
            raise AssertionError("invalid launch was accepted")
        assert not commands

    reject("training directory must be absolute", train_data_dir=Path("train"))
    extra = relocated / "unexpected-subdirectory"
    extra.mkdir()
    reject("training directory entries differ")
    extra.rmdir()
    missing = out / "held-image"
    image.rename(missing)
    reject("training directory entries differ")
    missing.rename(image)
    for file in (image, metadata):
        raw = file.read_bytes()
        file.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        reject("training input bytes differ")
        file.write_bytes(raw)
    reject("training input bytes differ", trainer_sha256="0" * 64)

    try:
        run_via_common_backend(backend, **kwargs)
    except SubprocessBoundary:
        pass
    else:
        raise AssertionError("real common backend did not reach the intercepted subprocess")
    assert len(commands) == 1
    command = commands[0]
    values = {
        "--train_data_dir": str(relocated.resolve()),
        "--resolution": "256",
        "--max_train_steps": "25",
        "--mixed_precision": "bf16",
        "--gencuda_checkpoint_steps": "20,25",
    }
    assert all(command[command.index(flag) + 1] == value for flag, value in values.items())
    binding = json.loads((Path(__file__).with_name("DATA_BINDING.json")).read_bytes())
    images = binding["train_images_by_filename_sha256"]
    result = dict(
        schema="SDXL_STYLE18_PORTABLE_TRAIN_ENTRY_CPU_V1", passed=True,
        binding_SHA256=BINDING_SHA256, verified_train_images=len(images),
        verified_image_bytes=sum(pin["bytes"] for pin in images.values()),
        verified_metadata_bytes=binding["train_metadata"]["bytes"],
        rejected_before_subprocess=["relative_path", "extra_directory", "missing_image",
                                    "same_size_mutated_png", "same_size_mutated_metadata",
                                    "incorrect_trainer_sha"],
        accepted_common_backend_calls=len(commands), backend_archive_SHA256=ARCHIVE_SHA,
        used_test_trainer_sentinel=True, deployed_Hygon_trainer_verified=False,
        actual_native_training_subprocesses=0, actual_optimizer_updates=0,
        actual_GPU_runs=0, training_READY_increment=0, retained_original_T06_numeric_failure=True,
        wall_seconds=time.monotonic() - origin,
    )
    with (out / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
