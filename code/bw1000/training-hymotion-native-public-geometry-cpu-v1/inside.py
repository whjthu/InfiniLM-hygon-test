"""Run the fixed public HY-Motion geometry check inside a zero-GRES step."""

from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

from check_hymotion_native_public_geometry import pinned_file, run_native, verify_inputs


SOURCE = Path("/input-stage")
OUTPUT = Path("/evidence")


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

    asset_dir, upstream = SOURCE / "assets", SOURCE / "upstream"
    expected_provider = claim["torch_extension"]
    expected_path = Path(expected_provider["container_path"])
    provider_raw = pinned_file(expected_path, expected_provider["SHA256"], expected_provider["bytes"])
    if len(provider_raw) != expected_provider["bytes"]:
        raise ValueError("native CPU provider size differs")
    takes, mean, safe_std = verify_inputs(asset_dir, upstream, SOURCE / "geometry.private.npz")
    version, result = run_native(upstream, asset_dir, takes, mean, safe_std, started + slack)
    import numpy as np
    import torch

    runtime = dict(torch=version, numpy=np.__version__, HIP=torch.version.hip)
    if runtime != claim["expected_runtime"]:
        raise ValueError("original HIP CPU runtime versions differ")
    provider = Path(torch._C.__file__).resolve(strict=True)
    if provider != expected_path or provider.read_bytes() != provider_raw:
        raise ValueError("unexpected CPU Torch provider")
    value = dict(passed=all(take["passed"] for take in result.values()), node=node, step=job + "." + number,
                 epoch=epoch, work_deadline=deadline, runtime=runtime, actual_checks=result,
                 torch_extension_bytes=provider.stat().st_size,
                 torch_extension_SHA256=sha256(provider_raw).hexdigest(),
                 GPU_cards=0, CUDA_initialized=torch.cuda.is_initialized(),
                 accepted_training_rows=0, text_encoder_calls=0,
                 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 wall_seconds=time.monotonic() - started)
    with (OUTPUT / "CPU-result.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return int(not value["passed"])


if __name__ == "__main__":
    sys.exit(main())
