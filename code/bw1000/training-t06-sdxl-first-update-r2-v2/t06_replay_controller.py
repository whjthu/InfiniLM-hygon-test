"""Execute one GPU first update, CPU comparator, and separate clean-state checks."""

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
WORK = Path(__file__).resolve().parent
NODE = "f11r4n11"
JOB = "1087656"
PYTHON = "/usr/local/python3.12/bin/python3"
LABEL = "t06-sdxl-first-update-r2-v2"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def query(*args):
    return subprocess.check_output(args, text=True, timeout=12)


def accounting(name):
    return [line.split("|") for line in query("sacct", "-n", "-P", "-j", JOB,
            "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw").splitlines() if "|" + name + "|" in line]


def done(name):
    for _ in range(25):
        rows = accounting(name)
        require(len(rows) <= 1, "duplicate step accounting")
        if rows and rows[0][2].split()[0] in TERMINAL:
            return rows[0]
        time.sleep(0.6)
    raise RuntimeError("missing terminal accounting for " + name)


def emit(event, **fields):
    with (WORK / "events.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(dict(event=event, epoch=time.time(), **fields)) + "\n")
        log.flush()
        os.fsync(log.fileno())


def step(label, cores, memory, duration, device, command, seconds):
    require(not accounting(label), "step already submitted")
    args = ["srun", f"--jobid={JOB}", f"--job-name={label}", "--exact", "--cpu-bind=none", "--nodes=1",
            "--ntasks=1", f"--nodelist={NODE}", f"--cpus-per-task={cores}", f"--mem={memory}M",
            f"--gres={device}", "--immediate=5", f"--time={duration}", *command]
    with (WORK / (label + ".log")).open("xb") as log:
        rc = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, timeout=seconds).returncode
    emit("process_return", label=label, process_rc=rc)
    terminal = done(label)
    emit("accounting", terminal=terminal, process_rc=rc)
    return rc, terminal


def snapshot(tag):
    name = LABEL + "-" + tag
    rc, terminal = step(name, 2, 4096, "00:01:00", "none",
                        [PYTHON, "-B", str(WORK / "snapshot.py"), "cpu"], 75)
    rows = [json.loads(x) for x in (WORK / (name + ".log")).read_bytes().splitlines() if x.startswith(b"{")]
    row, = rows
    require(rc == 0 and terminal[2:4] == ["COMPLETED", "0:0"]
            and row["step"] == terminal[0] and row["kfd_holder_count"] == 0
            and row["node"] == NODE and len(row["pci"]) == 8, "unclean PCI snapshot")
    emit(tag, snapshot=row)
    return row


def container(script, gpu):
    model = ROOT / "results/training-t06-sdxl-unet-login-stage-v1/model"
    reference = ROOT / "results/training-t06-sdxl-first-reference-r2"
    deps = ROOT / "results/training-t06-sdxl-runtime-v1"
    shell = ("set -euo pipefail\nsource /opt/dtk/env.sh\n"
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "
             "OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
             "export PYTHONPATH=/training-runtime/site\n"
             "unset BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT\n")
    if not gpu:
        shell += "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1\n"
    return ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec",
            *(["--rocm"] if gpu else []), "--no-home", "--pwd", "/",
            "--bind", f"{WORK}:/t06-run:rw", "--bind", f"{model}:/model:ro",
            "--bind", f"{reference}:/reference:ro", "--bind", f"{deps}:/training-runtime:ro",
            "--bind", "/opt/hyhal:/opt/hyhal:ro", str(ROOT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", shell + script]


def main():
    before = snapshot("pre")
    gpu_cmd = container("exec python3 -B -u /t06-run/replay_entry.py\n", gpu=True)
    emit("submit_gpu", node=NODE, training_case="T06.sft")
    rc, gpu = step(LABEL, 8, 64000, "00:12:00", "dcu:1", gpu_cmd, 755)
    post = snapshot("post")
    require(post["pci"] == before["pci"] and post["boot_id"] == before["boot_id"],
            "physical device inventory changed during replay")
    if rc != 0 or gpu[2:4] != ["COMPLETED", "0:0"]:
        emit("terminal", device_step=gpu[0], numerics="UNRUN", post_clean=True)
        return 1
    assigned = json.loads((WORK / "assigned.json").read_bytes())
    require(assigned["step"] == gpu[0] and assigned["visible_cards"] == 1
            and assigned["pci"] in before["pci"], "foreign assigned PCI identity")
    compare = ("exec python3 -B -u /t06-run/source/tools/hygon_sdxl_compare_first_step.py "
               "--reference /reference/reference.pt --policy /reference/policy.json "
               "--observed /t06-run/replay/observed.pt --receipt /t06-run/replay/summary.json "
               "--output /t06-run/compare.json\n")
    rc, cpu = step(LABEL + "-compare", 4, 8192, "00:04:00", "none", container(compare, gpu=False), 270)
    report = json.loads((WORK / "compare.json").read_bytes()) if (WORK / "compare.json").is_file() else None
    passed = bool(rc == 0 and cpu[2:4] == ["COMPLETED", "0:0"] and report and report["passed"])
    emit("terminal", device_step=gpu[0], comparison_step=cpu[0], numerics="PASS" if passed else "FAIL",
         post_clean=True, checks=report.get("checks") if report else None)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
