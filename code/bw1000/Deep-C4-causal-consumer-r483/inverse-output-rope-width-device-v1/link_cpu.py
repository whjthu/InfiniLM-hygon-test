"""Link the single-card comparison with its pinned objects and CUDA provider."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess


CUDA = Path("/opt/dtk/cuda/cuda")
PROVIDER = CUDA / "lib64/libcudart.so.12"
CXX = "/usr/bin/g++"
READELF = "/usr/bin/readelf"
NM = "/usr/bin/nm"
OUT = Path("/audit-out")
SCOPE = Path(__file__).resolve().parent


def need(value, reason):
    if not value:
        raise ValueError(reason)


def file_pin(path):
    with path.open("rb") as handle:
        digest = sha256()
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
        return {"bytes": handle.tell(), "SHA256": digest.hexdigest()}


def command(name, args):
    result = subprocess.run(args, capture_output=True, timeout=25, check=False)
    with (OUT / (name + ".stdout")).open("xb") as stream:
        stream.write(result.stdout)
    with (OUT / (name + ".stderr")).open("xb") as stream:
        stream.write(result.stderr)
    need(result.returncode == 0, name + " failed")
    return result.stdout.decode("utf-8")


def main():
    claim = json.loads((SCOPE / "CLAIM.json").read_bytes())
    parent = os.environ["SLURM_JOB_ID"]
    step = os.environ["SLURM_STEP_ID"]
    node = socket.gethostname().split(".")[0]
    need(parent == claim["parent"] and node == claim["node"] and step.isdecimal(), "Unexpected CPU step")
    need(not os.environ.get("SLURM_STEP_GPUS") and all(os.environ.get(key) == "-1" for key in
         ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")), "CPU GRES mask differs")
    tools = claim["target_tools_SHA256"]
    for path in (Path(CXX), Path(READELF), Path(NM), PROVIDER, CUDA / "include/cuda_runtime.h"):
        need(file_pin(path)["SHA256"] == tools[str(path)], "Target toolchain identity differs: " + path.name)
    source = SCOPE / "compare.cpp"
    need(file_pin(source) == claim["sources"]["compare.cpp"], "Comparison source differs")
    objects = []
    for pinned in claim["objects"].values():
        path = Path("/refactor") / pinned["path"]
        need(path.resolve(strict=True) == path and file_pin(path) == pinned["pin"], "Original object differs")
        objects.append(str(path))
    executable = OUT / "compare"
    report = {"request": claim["request"], "node": node, "step": parent + "." + step,
              "target_host_translation_units": 0, "new_device_translation_units": 0,
              "new_explicit_runtime_calls": 0, "GPU_steps": 0, "passed": False}
    try:
        command("link", [CXX, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                         "-isystem", str(CUDA / "include"), str(source), *objects, str(PROVIDER),
                         "-Wl,-z,defs", "-Wl,-rpath," + str(PROVIDER.parent), "-o", str(executable)])
        report["target_host_translation_units"] = 1
        dynamic = command("dynamic", [READELF, "-d", str(executable)])
        needed = re.findall(r"Shared library: \[([^\]]+)\]", dynamic)
        need(needed.count(PROVIDER.name) == 1 and str(PROVIDER.parent) in dynamic, "Unexpected runtime link provider")
        symbols = command("symbols", [NM, "--defined-only", str(executable)])
        defined = {line.split()[-1] for line in symbols.splitlines() if line.strip()}
        expected = {"deep_inverse_rope_h1_pack32_v1", "deep_inverse_rope_h1_pack64_v1"}
        need(defined.intersection(expected) == expected, "Original entry symbols are missing")
        report.update(passed=True, executable=file_pin(executable), CUDA_provider=file_pin(PROVIDER),
                      direct_needed=needed, entries=sorted(expected))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(error=type(error).__name__, error_summary=str(error)[:160])
    finally:
        with (OUT / "original-result.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
