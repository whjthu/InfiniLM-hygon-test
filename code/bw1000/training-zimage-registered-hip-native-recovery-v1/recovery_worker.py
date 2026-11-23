"""Execute one registered engineering phase on its owned Hygon Slurm step."""

import ctypes
import hashlib
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
PHASES = {"continuous": list(range(1, 26)), "resume20": list(range(21, 26))}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path, max_bytes, expected=None):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            count += len(chunk)
            require(count <= max_bytes, "Artifact exceeds limit: " + path.name)
            digest.update(chunk)
    actual = digest.hexdigest()
    require(count > 0 and (expected is None or actual == expected), "Artifact hash differs: " + path.name)
    return actual


def json_read(path):
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    require(0 < len(raw) <= 1 << 20, "Invalid JSON receipt")
    return json.loads(raw)


def write(name, value):
    with (ROOT / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def main(mode):
    expected_steps = PHASES[mode]
    claim = json_read(ROOT / "CLAIM.json")
    intent = json_read(ROOT / "intent.json")
    epoch = intent["epoch"]
    deadline = epoch + claim["worker_seconds"]
    step = os.environ["SLURM_STEP_ID"]
    require(intent["request"] == claim["request"] and epoch <= time.time() < deadline
            and socket.gethostname().split(".")[0] == claim["node"] and os.environ["SLURM_JOB_ID"] == claim["parent"]
            and step.isdecimal() and int(os.environ["SLURM_CPUS_PER_TASK"]) == claim["cpus"],
            "Invalid registered engineering allocation")
    record = {"request": claim["request"], "phase": mode, "slurm": claim["parent"] + "." + step,
              "passed": False, "owner_pid": os.getpid(), "planned_global_steps": expected_steps,
              "completed_global_steps": [], "actual_optimizer_updates": 0}
    output = ROOT / mode

    def timeout(*_):
        raise TimeoutError("Original registered engineering epoch expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    try:
        pins = claim["registered_binding"]
        entry = FRAMEWORK / "tools/hygon_training_recipe.py"
        fingerprint(BINDING, 1 << 20, pins["SHA256"])
        fingerprint(entry, 1 << 20, pins["cli_SHA256"])
        import torch
        require(torch.cuda.device_count() == 1, "Expected one Slurm-visible Hygon accelerator")
        address = ctypes.create_string_buffer(24)
        hip = ctypes.CDLL("libamdhip64.so")
        require(hip.hipDeviceGetPCIBusId(address, len(address), 0) == 0, "Could not verify assigned Hygon PCI")
        pci = address.value.decode("ascii").lower()
        if len(pci) == 7:
            pci = "0000:" + pci
        inventory = json_read(ROOT / ("pre.json" if mode == "continuous" else "between.json"))
        require(inventory["node"] == claim["node"] and pci in inventory["PCIs"], "Assigned PCI escaped the inventory")
        record["physical_pci"] = pci

        extra = []
        if mode == "resume20":
            original = json_read(ROOT / "continuous-worker-result.json")
            require(original["passed"] is True and original["physical_pci"] == pci
                    and original["slurm"] != record["slurm"], "Resume requires a separate process on the same physical card")
            cp = ROOT / "continuous/checkpoint-20/state.private.pt"
            original_result = json_read(ROOT / "continuous/result.json")
            pins20 = [item for item in original_result["checkpoints"] if item["global_step"] == 20]
            require(len(pins20) == 1 and cp.stat().st_size == pins20[0]["bytes"], "Checkpoint20 is absent or different")
            record["checkpoint20_SHA256"] = fingerprint(cp, 512 << 20, pins20[0]["SHA256"])
            extra = ["--checkpoint", str(cp)]
        write(mode + "-registered-invocation.json",
              {"entry_SHA256": pins["cli_SHA256"], "binding_SHA256": pins["SHA256"],
               "registered_case": "T07.sft", "candidate": claim["common_candidate"]["candidate_commit"],
               "mode": mode, "planned_global_steps": expected_steps})
        argv = sys.argv
        sys.argv = [str(entry), "T07.sft", "--native-engineering", "--mode", mode,
                    "--binding", str(BINDING), "--output", str(output),
                    "--epoch", str(epoch), "--deadline", str(deadline), *extra]
        sys.path.insert(0, str(entry.parent))
        try:
            try:
                runpy.run_path(str(entry), run_name="__main__")
            except SystemExit as exit_signal:
                require(exit_signal.code in (None, 0), "Registered engineering entry returned failure")
        finally:
            sys.path.pop(0)
            sys.argv = argv
        result = json_read(output / "result.json")
        require(result["passed"] is True and result["completed_global_steps"] == expected_steps
                and result["actual_optimizer_updates"] == len(expected_steps)
                and result["frozen_parameters_unchanged"] is True
                and result["original_fused_base"]["matched_tensors"] == 521,
                "Registered engineering updates, frozen weights or fused state differ")
        if mode == "resume20":
            restored = result["restore_info"]
            require(restored["restored_boundary_exact"] is True and restored["metadata"]["global_step"] == 20
                    and restored["next"]["global_step"] == 21 and restored["next"]["sample_cursor"] == 0
                    and restored["rng_devices"] == [0] and restored["checkpoint_SHA256"] == record["checkpoint20_SHA256"],
                    "Fresh process did not restore the original step 20 boundary")
        record.update(passed=True, completed_global_steps=expected_steps, actual_optimizer_updates=len(expected_steps),
                      result_SHA256=fingerprint(output / "result.json", 1 << 20),
                      peak_allocated_bytes=result["peak_allocated_bytes"])
    except Exception as error:
        if (output / "progress.json").is_file():
            record["completed_global_steps"] = json_read(output / "progress.json")["completed_global_steps"]
            record["actual_optimizer_updates"] = None
        record.update(error_type=type(error).__name__, error=str(error)[:220])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write(mode + "-worker-result.json", record)
    return int(not record["passed"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
