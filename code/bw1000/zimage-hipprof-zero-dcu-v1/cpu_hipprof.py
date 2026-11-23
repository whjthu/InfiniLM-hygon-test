"""Verify the target image's existing DTK hipprof without starting a GPU context."""

from hashlib import sha256
from pathlib import Path
import json
import os
import re
import socket
import subprocess


TOOL = Path("/opt/dtk/hip/bin/hipprof")
ALIASED = Path("/opt/dtk/bin/hipprof")
OLD_SHA256 = "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"
OLD_BYTES = 15_528_344
OPTIONS = ("-d", "-o", "--hip-trace", "--trace-off", "--show-pid", "--output-type")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def helper(argv):
    out = subprocess.run(argv, capture_output=True, timeout=10, check=False)
    raw = out.stdout + out.stderr
    require(len(raw) <= 32768, "Tool inspection exceeded output limit")
    return out.returncode, raw


def main():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(r"1103606\.[0-9]+", step) is not None
            and socket.gethostname().split(".")[0] == "f12r2n13"
            and all(os.environ.get(key) == "" for key in
                    ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
            "Unexpected zero-GRES scope")
    report = {"passed": False, "step": step, "GPU_context_initializations_attempted": 0,
              "trace_invocations_attempted": 0, "kernel_route_qualified": False}
    try:
        path = TOOL.resolve(strict=True)
        require(path == TOOL and ALIASED.resolve(strict=True) == path and path.stat().st_size <= 32 << 20,
                "DTK ELF or alias differs")
        raw = path.read_bytes()
        require(raw[:5] == b"\x7fELF\x02", "Expected a native ELF64")
        digest = sha256(raw).hexdigest()
        report["tool"] = {"path": str(path), "bytes": len(raw), "SHA256": digest,
                          "matches_proven_historical_ELF": (len(raw), digest) == (OLD_BYTES, OLD_SHA256)}
        require(report["tool"]["matches_proven_historical_ELF"], "Unverified hipprof binary")
        rc, output = helper(["ldd", str(path)])
        need = [match.group(1).decode() for match in
                re.finditer(rb"(?m)^\s*([a-zA-Z0-9_.-]+) => not found", output)]
        report["ldd"] = {"rc": rc, "missing": need, "SHA256": sha256(output).hexdigest()}
        require(rc == 0 and not need, "Tool dependencies did not resolve")
        rc, output = helper([str(path), "-h"])
        present = {flag: re.search(rb"(?m)^\s*" + re.escape(flag.encode()) + rb"(?:\s|$)", output) is not None
                   for flag in OPTIONS}
        json_mode = b"0=json" in output
        report["help"] = {"rc": rc, "SHA256": sha256(output).hexdigest(), "bytes": len(output),
                          "observed_flags": present, "JSON_mode_0": json_mode}
        require(rc == 0 and all(present.values()) and json_mode, "Required CLI options are unavailable")
        report["passed"] = True
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        report.update(error_type=type(error).__name__,
                      error=str(error) if type(error) is ValueError else None)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
