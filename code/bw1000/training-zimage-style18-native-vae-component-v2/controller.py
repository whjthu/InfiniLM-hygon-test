"""Run one pre/device/post sequence inside the owned T07 Slurm parent."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(name, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
    with (ROOT / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def main():
    claim = json.loads((ROOT / "CLAIM.json").read_bytes())
    intent = json.loads((ROOT / "intent.json").read_bytes())
    need(claim["request"] == intent["request"] and ROOT.name == claim["label"], "Original intent differs")
    parent, node, label = claim["parent"], claim["node"], claim["label"]
    epoch, deadline = intent["epoch"], intent["deadline"]
    need(deadline == epoch + claim["whole_seconds"] and epoch <= time.time() < epoch + 10,
         "Controller is outside the original start window")
    phases = {}
    result = {"request": claim["request"], "epoch": epoch, "deadline": deadline,
              "observations": phases, "passed": False, "independent_A100_latent_parity": False,
              "gpu_srun_attempted": False, "training_ready": False, "accounting_columns": COLUMNS}

    def remaining(reserve=0):
        available = deadline - time.time() - reserve
        if available <= 0:
            raise TimeoutError("VAE controller deadline")
        return available

    def query(*argv):
        return subprocess.check_output(argv, text=True, timeout=min(10, remaining(2)))

    def accounting(name):
        raw = query("sacct", "-n", "-P", "-j", parent, "--format=" + ",".join(COLUMNS))
        matched = [line.split("|") for line in raw.splitlines() if line.split("|")[1] == name]
        need(len(matched) <= 1, "Original exact label is duplicated")
        return matched[0] if matched else None

    def wait_accounting(name, cap=20):
        until = time.monotonic() + cap
        while time.monotonic() < until:
            row = accounting(name)
            if row and row[2] in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
                                  "NODE_FAIL", "PREEMPTED", "DEADLINE", "BOOT_FAIL"}:
                return row
            time.sleep(min(.5, remaining(5)))
        return None

    def launch(phase, command, gpu=False):
        name = label + "-" + phase
        need(accounting(name) is None, "Original step label already exists")
        budget = min(360 if gpu else 55, remaining(75 if gpu else 10))
        args = ["srun", "--jobid="+parent, "--job-name="+name, "--exact", "--nodes=1", "--ntasks=1",
                "--nodelist="+node, "--immediate=5", "--cpus-per-task="+str(claim["cpus"] if gpu else 2),
                "--mem="+("16384M" if gpu else "4096M"), "--gres="+("dcu:1" if gpu else "none"),
                "--time="+str(int(budget / 60) + 1), *command]
        path = ROOT / (phase + ".log")
        with path.open("xb") as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if gpu:
                result["gpu_srun_attempted"] = True
            try:
                rc = child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=min(5, remaining(8)))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait(timeout=min(5, remaining(2)))
        row = wait_accounting(name)
        if row is None:
            raise RuntimeError("Original step accounting is not terminal")
        phases[phase] = {"child_rc": rc, "accounting": row,
                         "log_SHA256": sha256(path.read_bytes()).hexdigest()}
        need(row[0].startswith(parent + ".") and row[5] == node
             and ("gres/dcu=1" in row[6].split(",")) is gpu, "Unexpected Slurm accounting scope")
        return path, rc == 0 and row[2:4] == ["COMPLETED", "0:0"]

    def snapshot(phase):
        path, passed = launch(phase, [PYTHON, "-I", "-S", "-B", str(ROOT / "node_inventory.py"), parent, node])
        need(passed, "Independent device inventory step failed")
        [value] = [json.loads(line) for line in path.read_bytes().splitlines() if line.startswith(b"{")]
        need(value["node"] == node and value["step"] == phases[phase]["accounting"][0]
             and len(set(value["PCIs"])) == 8 and not value["kfd_process_count"]
             and not value["own_device_fd_process_count"], "Node not physically clear")
        phases[phase]["inventory_SHA256"] = save(phase + ".json", value)
        return value

    pre = None
    try:
        check = fields(query("scontrol", "show", "job", parent, "-o"))
        need(check["JobState"] == "RUNNING" and check["UserId"].endswith(f"({os.getuid()})")
             and check["NumNodes"] == "40" and "gres/dcu=320" in check["AllocTRES"].split(","),
             "Owned parent resources differ")
        need(node in query("scontrol", "show", "hostnames", check["NodeList"]).splitlines(),
             "Exact node left parent")
        for line in query("scontrol", "show", "step", parent, "-o").splitlines():
            step = fields(line)
            if step["StepId"] in (parent+".batch", parent+".extern"):
                continue
            need(re.fullmatch(parent + r"\.\d+", step["StepId"]) and step["NodeList"] not in ("", "(null)")
                 and node not in query("scontrol", "show", "hostnames", step["NodeList"]).splitlines(),
                 "Cannot establish exclusive target node")
        manifest_raw = (ROOT / "SOURCE_FILES.json").read_bytes()
        need(sha256(manifest_raw).hexdigest() == intent["manifest_SHA256"], "Target manifest mismatch")
        for relative, pin in json.loads(manifest_raw)["files"].items():
            path = ROOT / relative
            raw = path.read_bytes()
            need(ROOT in path.resolve().parents and len(raw) == pin["bytes"]
                 and sha256(raw).hexdigest() == pin["SHA256"], "Execution source drift")
        pre = snapshot("pre")
        entries = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        bindings = (
            (ROOT, "/vae-run", "rw"),
            (BASE/"results/training-zimage-style18-native12-CPU-v2/source", "/original-input-stage", "ro"),
            (BASE/"results/training-zimage-style18-native12-CPU-v2/data", "/zimage-input", "ro"),
            (BASE/"results/training-gemma-usgs-render72-native-CPU-v1/data/train/media", "/zimage-input/train/images", "ro"),
            (BASE/"results/training-zimage-target-native-meta-CPU-v1/source/upstream", "/native-source", "ro"),
            (BASE/"results/training-zimage-target-native-meta-CPU-v2/dependencies/site", "/zimage-runtime", "ro"),
            (BASE/"results/training-t06-sdxl-runtime-v1", "/training-runtime", "ro"),
            (BASE/"results/training-zimage-publisher-runtime-transfer-v1/candidate/base/vae", "/sealed-vae", "ro"),
            (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
        )
        need(all(src.is_dir() for src, _, _ in bindings), "Original approved mount absent")
        dataset_source = bindings[1][0] / "source"
        expected = {
            "tools/preflight_zimage_style18_train_input.py": claim["dataset_entry"],
            "coordination/training/zimage-style18-target-train-inputs-v1.json": {
                "bytes": 4960, "SHA256": "a06c29c8af3d56ed8192c39f63063db82bf67a14b75ba1fb2913202155d1ea09"},
        }
        for name, pin in expected.items():
            path = dataset_source / name
            need(path.resolve(strict=True) == path and path.is_file() and path.stat().st_size == pin["bytes"]
                 and sha256(path.read_bytes()).hexdigest() == pin["SHA256"], "Pinned data entry host bind differs")
        for src, dst, mode in bindings:
            entries.extend(("--bind", f"{src}:{dst}:{mode}"))
        command = shlex.join(["python3", "-B", "-u", "/vae-run/worker.py", "--epoch", repr(epoch),
                              "--deadline", repr(deadline), "--pre", "/vae-run/pre.json",
                              "--output", "/vae-run/worker-result.json"])
        env = ("set -euo pipefail; source /opt/dtk/env.sh; "
               "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
               "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/native-source:/zimage-runtime:/training-runtime/site; "
               "exec " + command)
        entries.extend((str(BASE / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", env))
        _, success = launch("device", entries, gpu=True)
        result["device_passed"] = success
        if (ROOT / "worker-result.json").is_file():
            result["worker_result_SHA256"] = sha256((ROOT / "worker-result.json").read_bytes()).hexdigest()
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:240])
    finally:
        if result["gpu_srun_attempted"]:
            try:
                need("device" in phases, "Actual terminal device accounting required before post")
                after = snapshot("post")
                result["post_clean"] = pre["PCIs"] == after["PCIs"] and pre["boot_id_SHA256"] == after["boot_id_SHA256"]
            except Exception as error:
                result.update(post_clean=False, post_error_type=type(error).__name__, post_error=str(error)[:240])
    result["passed"] = bool(result.get("device_passed") and result.get("post_clean") and not result.get("error_type"))
    save("outcome.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
