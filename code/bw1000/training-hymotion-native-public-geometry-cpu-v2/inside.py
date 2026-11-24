"""Run the fixed public HY-Motion geometry check inside a zero-GRES step."""

from hashlib import sha256
from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

from check_hymotion_native_public_geometry import pinned_file, run_native, verify_inputs


SOURCE = Path("/input-stage")
ORIGINAL = Path("/input-original")
OUTPUT = Path("/evidence")


def write_observation(name, value):
    with (OUTPUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    epoch = float(os.environ["INPUT_CPU_EPOCH"])
    deadline = float(os.environ["INPUT_CPU_WORK_DEADLINE"])
    slack = deadline - time.time()
    claim = json.loads((SOURCE / "CLAIM.json").read_bytes())
    if not 0 < slack <= claim["work_seconds"] or deadline - epoch != claim["work_seconds"]:
        raise ValueError("invalid original CPU deadline")
    started = time.monotonic()
    node = socket.gethostname().split(".")[0]
    job, number = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    if node != claim["node"] or job != claim["parent"] or not number.isdigit():
        raise ValueError("Slurm owner differs from published scope")
    if os.environ["SLURM_CPUS_PER_TASK"] != str(claim["cpu_cores"]):
        raise ValueError("Slurm CPU binding differs")
    if os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("GPU devices were assigned to the CPU step")

    if any(os.environ.get(key) != "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("GPU visibility differs")
    expected = claim["expected_runtime"]
    providers = {}
    for name, pin in expected["providers"].items():
        path = Path(pin["container_path"])
        raw = pinned_file(path, pin["SHA256"], pin["bytes"])
        if len(raw) != pin["bytes"]:
            raise ValueError("native CPU provider size differs")
        providers[name] = dict(path=str(path), bytes=len(raw), SHA256=sha256(raw).hexdigest())
    import numpy as np
    import torch

    for name, module in (("torch_extension", torch._C), ("torch_version", torch.version), ("numpy_module", np)):
        if str(Path(module.__file__).resolve(strict=True)) != providers[name]["path"]:
            raise ValueError("imported provider differs from the pinned target")
    if torch.cuda.is_initialized():
        raise ValueError("GPU runtime initialized during provider import")
    runtime = dict(distribution={name: distribution_version(name) for name in ("torch", "numpy")},
                   module=dict(torch=str(torch.__version__), numpy=np.__version__), HIP=torch.version.hip)
    write_observation("runtime.json", dict(node=node, step=job + "." + number, runtime=runtime, providers=providers))

    asset_dir, upstream = ORIGINAL / "assets", ORIGINAL / "upstream"
    takes, mean, safe_std = verify_inputs(asset_dir, upstream, ORIGINAL / "geometry.private.npz")
    _, result = run_native(upstream, asset_dir, takes, mean, safe_std, started + slack)
    runtime_passed = runtime == {key: expected[key] for key in ("distribution", "module", "HIP")}
    geometry_passed = all(take["passed"] for take in result.values())
    value = dict(passed=runtime_passed and geometry_passed, node=node, step=job + "." + number,
                 epoch=epoch, work_deadline=deadline, runtime=runtime, providers=providers, actual_checks=result,
                 runtime_passed=runtime_passed, geometry_passed=geometry_passed,
                 GPU_cards=0, CUDA_initialized=torch.cuda.is_initialized(),
                 accepted_training_rows=0, text_encoder_calls=0,
                 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 wall_seconds=time.monotonic() - started)
    write_observation("CPU-result.json", value)
    return int(not value["passed"])


if __name__ == "__main__":
    sys.exit(main())
