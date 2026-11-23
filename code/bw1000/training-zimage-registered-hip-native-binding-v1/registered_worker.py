"""Run the verified unified training CLI inside its owned single-card Slurm step."""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import sys
import time


ROOT = Path("/fusion-run")
FRAMEWORK = Path("/common-framework")
BINDING = Path("/binding-read/native-binding.json")
REFERENCE = Path("/original-first")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected):
    with path.open("rb") as stream:
        data = stream.read((1 << 20) + 1)
    require(0 < len(data) <= 1 << 20 and sha256(data).hexdigest() == expected, "Registered input changed")
    return data


def write(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    claim = json.loads((ROOT / "CLAIM.json").read_bytes())
    intent = json.loads((ROOT / "intent.json").read_bytes())
    epoch = intent["epoch"]
    deadline = epoch + claim["worker_seconds"]
    step = os.environ["SLURM_STEP_ID"]
    require(intent["request"] == claim["request"] and epoch <= time.time() < deadline
            and socket.gethostname().split(".")[0] == claim["node"]
            and os.environ["SLURM_JOB_ID"] == claim["parent"] and step.isdecimal()
            and int(os.environ["SLURM_CPUS_PER_TASK"]) == claim["cpus"], "Wrong registered Slurm scope")
    result = {"passed": False, "step": claim["parent"] + "." + step, "actual_optimizer_updates": 0,
              "request": claim["request"], "phase": "registered-inputs"}
    output = ROOT / "registered-output"

    def timeout(*_):
        raise TimeoutError("Registered native training epoch expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        pin = claim["registered_binding"]
        read(BINDING, pin["SHA256"])
        entry = FRAMEWORK / "tools/hygon_training_recipe.py"
        read(entry, pin["cli_SHA256"])
        read(REFERENCE / "progress.json", claim["fixed_hip_reference"]["progress"]["SHA256"])
        import torch
        require(torch.cuda.device_count() == 1, "Expected one Slurm-visible Hygon accelerator")
        pci = ctypes.create_string_buffer(24)
        hip = ctypes.CDLL("libamdhip64.so")
        require(hip.hipDeviceGetPCIBusId(pci, len(pci), 0) == 0, "Could not verify assigned Hygon PCI")
        address = pci.value.decode("ascii").lower()
        if len(address) == 7:
            address = "0000:" + address
        pre = json.loads((ROOT / "pre.json").read_bytes())
        require(pre["node"] == claim["node"] and address in pre["PCIs"], "Assigned PCI escaped the verified inventory")
        result["phase"] = "registered-cli"
        result["physical_pci"] = address
        write("registered-invocation.json", {"case": "T07.sft", "requested_updates": 1,
                                             "candidate": claim["common_candidate"]["candidate_commit"],
                                             "entry_SHA256": pin["cli_SHA256"], "binding_SHA256": pin["SHA256"]})
        argv = sys.argv
        sys.argv = [str(entry), "T07.sft", "--fixed-first-update", "--binding", str(BINDING),
                    "--reference", str(REFERENCE), "--reference-sha256", claim["fixed_hip_reference"]["progress"]["SHA256"],
                    "--output", str(output), "--epoch", str(epoch), "--deadline", str(deadline)]
        sys.path.insert(0, str(entry.parent))
        try:
            try:
                runpy.run_path(str(entry), run_name="__main__")
            except SystemExit as exit_signal:
                require(exit_signal.code in (None, 0), "Registered CLI returned failure")
        finally:
            sys.path.pop(0)
            sys.argv = argv
        data = json.loads((output / "result.json").read_bytes())
        inventory = data["original_fused_base"]
        require(data["passed"] is True and data["actual_optimizer_updates"] == 1
                and data["completed_global_steps"] == [1] and inventory["matched_tensors"] == 521
                and data["first_update_capture_complete"] is True and data["frozen_parameters_unchanged"] is True,
                "Registered first update, original fused state or capture did not complete")
        progress = output / "first-update-private/progress.json"
        write("optimizer-step-returned.private.json", {"actual_optimizer_updates": 1})
        result.update(passed=True, actual_optimizer_updates=1, phase="registered-cli-returned",
                      independent_capture_progress_SHA256=sha256(progress.read_bytes()).hexdigest(),
                      complete_result_SHA256=sha256((output/"result.json").read_bytes()).hexdigest(),
                      fused_tensors=inventory["matched_tensors"],
                      peak_allocated_bytes=data["peak_allocated_bytes"])
    except Exception as error:
        progress = output / "first-update-private/progress.json"
        if progress.is_file():
            capture = json.loads(progress.read_bytes())
            if capture.get("first_update_complete") is True:
                result["actual_optimizer_updates"] = 1
            elif capture.get("phase") in ("postclip", "updated"):
                result["actual_optimizer_updates"] = None
        result.update(error_type=type(error).__name__, error=str(error)[:220])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write("worker-result.json", result)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
