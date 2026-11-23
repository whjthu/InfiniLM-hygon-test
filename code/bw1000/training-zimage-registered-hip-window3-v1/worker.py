"""Execute one registered engineering phase on its owned Hygon Slurm step."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time


ROOT = Path("/fusion-run")
FRAMEWORK = Path("/common-framework")
BINDING = Path("/binding-read/native-binding.json")
PHASES = {mode: list(range(1,26)) for mode in ("repeat01","repeat02","repeat03")}


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
        inventory = json_read(ROOT / "pre.json")
        require(inventory["node"] == claim["node"] and pci in inventory["PCIs"], "Assigned PCI escaped the inventory")
        record["physical_pci"] = pci

        write(mode + "-registered-invocation.json",
              {"entry_SHA256": pins["cli_SHA256"], "binding_SHA256": pins["SHA256"],
               "registered_case": "T07.sft", "candidate": claim["common_candidate"]["candidate_commit"],
               "mode": "continuous", "complete_window": True, "repeat": mode, "planned_global_steps": expected_steps})
        argv = sys.argv
        sys.argv = [str(entry), "T07.sft", "--native-engineering", "--mode", "continuous", "--measure-complete-window",
                    "--binding", str(BINDING), "--output", str(output),
                    "--epoch", str(epoch), "--deadline", str(deadline)]
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
        window=result["engineering_complete_window"]
        elapsed=(window["end_ns"]-window["start_ns"])/1_000_000_000
        require((window["clock"],window["warmup_updates"],window["measured_updates"])==("monotonic_ns",5,20)
                and window["update_ordinals"]==expected_steps[5:] and window["checkpoint_ordinals"]==[10,20,25]
                and window["seconds"]==elapsed>0 and window["updates_per_second"]==20/elapsed
                and window["includes_input_fetch_and_durable_checkpoint_io"] is True
                and window["first_update_capture"] is False and window["profile"] is False
                and result["profile_summary"] is None and result["first_update_capture_bytes"]==0
                and result["original_full_recipe"] is True and result["restore_info"] is None,
                "Registered timing mode or declared boundary differs")
        require([row["global_step"] for row in result["checkpoints"]]==[10,20,25]
                and all(row["bytes"]>0 and len(row["SHA256"])==64 for row in result["checkpoints"]),
                "Original checkpoint ordinals or content pins differ")
        record.update(window=window,load_seconds=result["load_seconds"],
                      peak_reserved_bytes=result["peak_reserved_bytes"],
                      checkpoints=[{k:row[k] for k in ("global_step","bytes","SHA256","adapter_SHA256")}
                                   for row in result["checkpoints"]])
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


def batch():
    claim=json_read(ROOT/"CLAIM.json")
    intent=json_read(ROOT/"intent.json")
    deadline=intent["epoch"]+claim["worker_seconds"]
    repeats={}
    record=dict(request=claim["request"],completed_repeats=0,planned_repeats=list(PHASES),repeats=repeats)
    pci=slurm=None
    pids=set()
    try:
        for mode in PHASES:
            remaining=deadline-time.time()-10
            require(remaining>0,"Original common repeat deadline expired")
            path=ROOT/(mode+"-process.log")
            with path.open("xb") as stream:
                child=subprocess.run([sys.executable,"-B","-u",str(ROOT/"worker.py"),mode],
                                     stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
            receipt=json_read(ROOT/(mode+"-worker-result.json"))
            repeats[mode]=receipt
            require(child.returncode==0 and receipt["passed"] is True,"Independent training process failed: "+mode)
            if pci is None:
                pci,slurm=receipt["physical_pci"],receipt["slurm"]
            require(receipt["physical_pci"]==pci and receipt["slurm"]==slurm and receipt["owner_pid"] not in pids,
                    "Repeat moved between GPUs or did not begin in a new process")
            pids.add(receipt["owner_pid"])
            record["completed_repeats"]+=1
    except Exception as error:
        record.update(error_type=type(error).__name__,error=str(error)[:220])
    finally:
        write("batch-result.json",record)
    return int(record["completed_repeats"]!=len(PHASES))


if __name__ == "__main__":
    raise SystemExit(batch() if sys.argv[1]=="batch" else main(sys.argv[1]))
