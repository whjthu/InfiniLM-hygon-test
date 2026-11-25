"""Verify the installed Gemma pilot assets in a separate zero-GRES step."""

from hashlib import sha256
import importlib.metadata as md
import json
import os
from pathlib import Path
import socket
import stat
import sys

ORIGINAL = Path(
    "/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/"
    "training-gemma-usgs72-blind36-58925c79ef98-v1-source"
)
ORIGINAL_MANIFEST = "34380f1660e572ecbf232ee7e1ba1be672fd06c8789e2d603ca95e5c9b3a90bd"
INITIAL_BYTES = 119382791
INITIAL_SHA256 = "da110892d3f9cac70993d14e8ea1dfb39bae64631b08051007a276f94e067de7"
CHUNK_BYTES = 1 << 20


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def check_initial(path, extent, expected_sha):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != extent:
        raise ValueError("Initial adapter is not the expected regular file")
    digest = sha256()
    count = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError("Initial adapter changed between lookup and open")
        for raw in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(raw)
            count += len(raw)
            if count > extent:
                raise ValueError("Initial adapter exceeded its frozen size")
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError("Initial adapter changed while being hashed")
    if identity(path.lstat()) != identity(before) or count != extent or digest.hexdigest() != expected_sha:
        raise ValueError("Initial adapter bytes or final file identity differ")
    return {"bytes": count, "SHA256": digest.hexdigest(), "observed_file_identity": list(identity(before))}


def run():
    report = {"state": "BLOCKED", "phase": "PLACEMENT", "planned": 108, "generation_calls": 0, "gold_reads": 0}
    try:
        if not (
            socket.gethostname().split(".")[0] == "f13r1n12"
            and os.environ.get("SLURM_JOB_ID") == "1103606"
            and os.environ.get("SLURM_STEP_ID", "").isdecimal()
            and not os.environ.get("SLURM_STEP_GPUS")
            and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("HIP_VISIBLE_DEVICES") == ""
            and sys.dont_write_bytecode
        ):
            raise ValueError("Expected the allocated zero-GRES Gemma step")
        report["actual_step"] = f"1103606.{os.environ['SLURM_STEP_ID']}"
        report["phase"] = "INSTALLED_SOURCE"
        if sha256((ORIGINAL / "SOURCE_FILES.json").read_bytes()).hexdigest() != ORIGINAL_MANIFEST:
            raise ValueError("Original blind evaluator manifest differs")
        sys.path.insert(0, str(ORIGINAL / "core"))
        from bind_ready import bind_value
        from blind_inputs import INPUT_MANIFEST_SHA
        from effect_contract import DOMAIN, identity as old_identity, need, pin, read
        from infer_blind36 import ENGINE_PINS

        report["phase"] = "BIND_INPUT_AND_BOOT"
        value = bind_value()
        need(value["input_manifest"]["SHA256"] == INPUT_MANIFEST_SHA, "wrong input bundle")
        need(
            value["same_compute_host"] == socket.gethostname()
            and value["same_compute_boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "original compute host or boot differs",
        )
        report["phase"] = "MODEL_AND_ADAPTER_METADATA"
        members = value["model"]
        need(len(members) == 16, "original full-model manifest differs")
        for path, expected in members.items():
            need(old_identity(Path(path).lstat()) == expected["file_identity"], "original model lifetime differs")
        adapter_count = 0
        for checkpoint in value["checkpoints"].values():
            need(len(checkpoint["files"]) == 14, "original checkpoint manifest differs")
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                expected = checkpoint["files"][name]
                path = Path(checkpoint["root"]) / name
                need(old_identity(path.lstat()) == expected["file_identity"], "original adapter lifetime differs")
                if name.endswith(".json"):
                    read(path, expected)
                adapter_count += 1
        report["phase"] = "INITIAL_ADAPTER_SAME_FD"
        initial = value["initial"]
        need(initial["bytes"] == INITIAL_BYTES and initial["SHA256"] == INITIAL_SHA256, "original .851 pin differs")
        path = Path(initial["path"])
        need(path == DOMAIN / "continuous/train/continuous-witness/rank-00/target-fresh-initial.pt",
             "unexpected initial artifact path")
        report["initial"] = check_initial(path, INITIAL_BYTES, INITIAL_SHA256)
        report["phase"] = "FROZEN_FRAMEWORK"
        framework = json.loads(read(DOMAIN / "fresh-framework-files.json", value["framework"]))
        need(len(framework["files"]) == 396, "original framework member count differs")
        for name, expected in framework["files"].items():
            need(not Path(name).is_absolute() and ".." not in Path(name).parts, "bad framework member")
            read(Path(value["framework_root"]) / name, expected)
        report["phase"] = "RUNTIME_PROVIDERS"
        runtime = json.loads(read(DOMAIN / "runtime-binding.json", value["runtime"]))
        need({name: md.version(name) for name in runtime["versions"]} == runtime["versions"],
             "original runtime packages differ")
        for name, expected in ENGINE_PINS.items():
            need(pin(read(Path("/swift-runtime/site/swift") / name))["SHA256"] == expected,
                 "native Swift engine source differs")
        report.update(state="EXACT_ASSETS_OBSERVED", phase="COMPLETE", model16_stat=16,
                      adapter_members_stat=adapter_count, framework_files_hash=396, engine_source_hash=4,
                      input_manifest=value["input_manifest"], actual_native_provider_import=False)
    except Exception as error:
        report.update(error_type=type(error).__name__, errno=getattr(error, "errno", None))
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome, sort_keys=True))
    raise SystemExit(0 if outcome["state"] == "EXACT_ASSETS_OBSERVED" else 2)
