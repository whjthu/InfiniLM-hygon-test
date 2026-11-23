"""Recover the original grouped-finite COCO20 checkpoint in a new Slurm step."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time

from resume_entry import worker_source


WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent.parent
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")


def require(value, detail):
    if not value:
        raise ValueError(detail)


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())


def checked_read(root, name, digest, size_limit=262144):
    path = root / name
    require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(),
            "Original evidence path or owner differs: " + name)
    with path.open("rb") as file:
        raw = file.read(size_limit + 1)
    require(0 < len(raw) <= size_limit and sha256(raw).hexdigest() == digest,
            "Original evidence bytes differ: " + name)
    return raw


def accounting_rows(raw):
    rows = []
    for line in raw.splitlines():
        row = line.split("|")
        require(len(row) == len(COLUMNS), "Unexpected Slurm accounting columns")
        rows.append(row)
    return rows


def original_inputs(original, claim):
    source = claim["source_work"]
    pins = source["small_receipts"]
    observed = {name: checked_read(original, name, digest) for name, digest in pins.items()}
    original_claim = json.loads(observed["CLAIM.json"])
    worker = json.loads(observed["continuous/worker-result.json"])
    require(json.loads(observed["outcome.json"])["passed"] is True
            and original_claim["request"] == source["request"]
            and original_claim["parent"] == claim["parent"] and original_claim["node"] == claim["node"]
            and original_claim["cpus"] == claim["CPUs"] and original_claim["memory_MiB"] == claim["memory_MiB"]
            and worker["request"] == source["request"] and worker["step"] == source["gpu_step"]
            and worker["actual_optimizer_updates"] == 25 and worker["passed"] is True
            and worker["physical_PCI"] == claim["PCI"], "Original continuous training identity differs")
    checkpoints = json.loads(observed["continuous/checkpoint-receipts.json"])
    require([row["global_step"] for row in checkpoints] == [10, 20, 25]
            and checkpoints[1]["next_data_cursor"] == 0
            and checkpoints[1]["checkpoint_SHA256"] == source["checkpoint_SHA256"],
            "Original checkpoint receipt differs")
    checkpoint = original / "continuous/checkpoint-20.private.pt"
    require(checkpoint.resolve(strict=True) == checkpoint and checkpoint.stat().st_uid == os.getuid()
            and checkpoint.stat().st_size == checkpoints[1]["bytes"] == source["bytes_per_checkpoint"],
            "Original checkpoint metadata differs")
    files = json.loads(observed["SOURCE_FILES.json"])["files"]
    require(len(files) == 29, "Unexpected original package size")
    for name, pin in files.items():
        require(Path(name).name == name, "Unexpected package path")
        require(len(checked_read(original, name, pin["SHA256"])) == pin["bytes"], "Package size differs")
    compile(worker_source((original / "worker.py").read_bytes()), "original-resume-worker", "exec")
    return json.loads(observed["pre.json"])["PCIs"]


def run():
    claim = json.loads((WORK / "CLAIM.json").read_bytes())
    epoch = time.time()
    monotonic_end = time.monotonic() + claim["whole_seconds"]
    deadline = epoch + claim["whole_seconds"]
    original = WORK.parent / claim["source_work"]["label"]
    parent, node, label = claim["parent"], claim["node"], claim["label"]
    require(WORK.name == label and re.fullmatch("[a-f0-9]{32}", claim["request"])
            and claim["DCU"] == 1 and claim["planned_updates"] == list(range(21, 26)),
            "Invalid recovery claim")
    result = dict(request=claim["request"], source_step=claim["source_work"]["gpu_step"],
                  planned_updates=claim["planned_updates"], actual_optimizer_updates=None,
                  phases={}, accounting_columns=COLUMNS, passed=False)

    def remaining(reserve=0):
        seconds = min(deadline - time.time(), monotonic_end - time.monotonic()) - reserve
        if seconds <= 0:
            raise TimeoutError("Recovery deadline")
        return seconds

    def query(*args):
        return subprocess.check_output(args, text=True, timeout=min(8, remaining(2)))

    def rows(job):
        return accounting_rows(query("sacct", "-n", "-P", "-j", job, "--format=" + ",".join(COLUMNS)))

    def lookup(name):
        matches = [row for row in rows(parent) if row[1] == name]
        require(len(matches) <= 1, "Duplicate recovery step label")
        return matches[0] if matches else None

    def placement():
        job = query("scontrol", "show", "job", parent, "-o")
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", job))
        require(fields["JobState"] == "RUNNING" and fields["NumNodes"] == "40"
                and fields["UserId"].endswith(f"({os.getuid()})")
                and "gres/dcu=320" in fields["AllocTRES"].split(",")
                and node in query("scontrol", "show", "hostnames", fields["NodeList"]).splitlines(),
                "Target lost the original owned placement")
        for line in query("squeue", "--steps", "-h", "-j", parent, "-o", "%i|%N").splitlines():
            step, hosts = line.split("|")
            if step not in {parent+".batch", parent+".extern"}:
                require(re.fullmatch(re.escape(parent) + r"\.\d+", step) and hosts not in {"", "(null)"}
                        and node not in query("scontrol", "show", "hostnames", hosts).splitlines(),
                        "Target node is busy or its current step cannot be isolated")

    def launch(phase, command, gpu=False):
        name = label + "-" + phase
        require(lookup(name) is None, "Prior label exists")
        seconds = min(claim["maximum_gpu_seconds"] if gpu else 55, int(remaining(75 if gpu else 10)))
        require(seconds > 0, "Insufficient time for Slurm step")
        limit = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
        args = ["srun", "--jobid="+parent, "--job-name="+name, "--exact", "--nodes=1", "--ntasks=1",
                "--nodelist="+node, "--immediate=5", "--cpus-per-task="+str(claim["CPUs"] if gpu else 2),
                "--mem="+(str(claim["memory_MiB"])+"M" if gpu else "4096M"),
                "--gres="+("dcu:1" if gpu else "none"), "--time="+limit, *command]
        log = WORK / (phase + ".log")
        record = result["phases"].setdefault(phase, {})
        save(WORK/(phase+"-intent.json"), {"label": name, "gpu": gpu, "epoch": epoch})
        with log.open("xb") as stream, subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                                        start_new_session=True) as child:
            try:
                stat = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()
                owner = (int(stat[2]), int(stat[19]))
                require(owner[0] == child.pid, "Slurm child is outside the expected process group")
                save(WORK/(phase+"-owner.json"), {"PID": child.pid, "PGID": owner[0], "start_ticks": owner[1]})
                record["child_rc"] = child.wait(timeout=seconds+2)
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if child.poll() is not None:
                        break
                    stat = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()
                    require((int(stat[2]), int(stat[19])) == owner, "Slurm child ownership changed at deadline")
                    os.killpg(child.pid, sig)
                    try:
                        record["child_rc"] = child.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        continue
                record["controller_timeout"] = True
        observed = wait_terminal(lambda: lookup(name), time.monotonic() + min(18, remaining(5)),
                                 interval=1, max_observations=16)
        record.update(observed)
        record["log_SHA256"] = sha256(log.read_bytes()).hexdigest()
        require(observed["terminal_visible"], "Exact Slurm step is unconfirmed")
        row = observed["row"]
        require(re.fullmatch(re.escape(parent) + r"\.\d+", row[0]) and row[5] == node
                and ("gres/dcu=1" in row[6].split(",")) is gpu, "Unexpected Slurm resource accounting")
        return log, record.get("child_rc") == 0 and row[2:4] == ["COMPLETED", "0:0"]

    def snapshot(phase):
        path, passed = launch(phase, [PYTHON, "-I", "-S", "-B", str(original/"node_inventory.py"), parent, node])
        require(passed, "Independent node probe did not finish")
        [record] = [json.loads(row) for row in path.read_bytes().splitlines() if row.startswith(b"{")]
        require(record["step"] == result["phases"][phase]["row"][0] and record["node"] == node
                and record["PCIs"] == pcis and len(set(pcis)) == 8 and claim["PCI"] in pcis
                and record["kfd_process_count"] == record["own_device_fd_process_count"] == 0,
                "Original GPU topology or current device ownership differs")
        save(WORK / (phase+".json"), record)
        return record

    def device_command():
        prefix = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        bindings = (
            (WORK, "/recovery", "rw"),
            (original, "/fusion-run", "ro"),
            (PROJECT/"results/training-zimage-native-checkout-import-v1/source", "/native-source", "ro"),
            (PROJECT/"results/training-qwen-a68952c-swift-initial-CPU-v1/framework", "/common-framework", "ro"),
            (PROJECT/"results/training-zimage-original-coco20-media-stage-v1/inputs", "/coco20-inputs", "ro"),
            (PROJECT/"results/training-zimage-original-coco20-native-input-cpu-v1", "/coco20-preprocess", "ro"),
            (PROJECT/"results/training-zimage-native-checkout-import-v2/site", "/zimage-compat", "ro"),
            (PROJECT/"results/training-zimage-target-native-meta-CPU-v2/dependencies/site", "/zimage-runtime", "ro"),
            (PROJECT/"results/training-t06-sdxl-runtime-v1", "/training-runtime", "ro"),
            (PROJECT/"results/training-zimage-modelscope-whole-runtime-v2/runtime", "/sealed-runtime", "ro"),
            (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
        )
        need_mounts = [str(src) for src, _, _ in bindings if not src.is_dir()]
        require(not need_mounts, "Original runtime mount absent")
        for src, dst, mode in bindings:
            prefix.extend(("--bind", f"{src}:{dst}:{mode}"))
        env = ("set -euo pipefail; source /opt/dtk/env.sh; "
               "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
               "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/native-source:/zimage-compat:/zimage-runtime:"
               "/training-runtime/site:/common-framework/generation/src; "
               "exec python3 -B -u /recovery/resume_entry.py")
        return prefix + [str(PROJECT/"containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", env]

    before = None
    try:
        pcis = original_inputs(original, claim)
        sys.path.insert(0, str(original))
        wait_terminal = runpy.run_path(str(original/"slurm_accounting.py"))["wait_terminal"]
        compare = runpy.run_path(str(original/"compare.py"))["compare"]
        old_rows = [row for row in rows(claim["source_work"]["gpu_step"]) if row[0] == claim["source_work"]["gpu_step"]]
        require(len(old_rows) == 1 and old_rows[0][2:4] == ["COMPLETED", "0:0"]
                and old_rows[0][5] == node and "gres/dcu=1" in old_rows[0][6].split(","),
                "Original GPU accounting differs")
        result["original_accounting"] = old_rows[0]
        placement()
        for phase in ("pre", "resume20", "post"):
            require(lookup(label + "-" + phase) is None, "Prior recovery accounting exists")
        command = device_command()
        before = snapshot("pre")
        placement()
        (WORK/"continuous").symlink_to(original/"continuous", target_is_directory=True)
        save(WORK/"intent.json", dict(request=claim["request"], source_request=claim["source_work"]["request"],
                                     epoch=epoch, deadline=deadline, gpu_deadline=deadline-85,
                                     publication=sys.argv[1]))
        _, passed = launch("resume20", command, gpu=True)
        worker_result = WORK / "resume20/worker-result.json"
        if worker_result.is_file():
            worker_raw = worker_result.read_bytes()
            worker = json.loads(worker_raw)
            require(worker["step"] == result["phases"]["resume20"]["row"][0]
                    and worker["request"] == claim["request"], "Worker result belongs to another step")
            result.update(actual_optimizer_updates=worker["actual_optimizer_updates"],
                          optimizer_returned_steps=worker.get("optimizer_returned_steps", []),
                          original_worker_result_SHA256=sha256(worker_raw).hexdigest())
        require(passed and result["actual_optimizer_updates"] == 5
                and result["optimizer_returned_steps"] == claim["planned_updates"] and worker["passed"] is True,
                "Native recovery did not finish the complete update denominator")
        original_contract = claim["source_work"]["small_receipts"]["continuous/contract.json"]
        checked_read(WORK, "resume20/contract.json", original_contract)
        result["comparison"] = compare(WORK)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError, TimeoutError, RuntimeError) as error:
        result.update(error_type=type(error).__name__, error=str(error)[:240])
    finally:
        if result["phases"].get("resume20", {}).get("terminal_visible") is True:
            try:
                after = snapshot("post")
                result["post_clean"] = before["boot_id_SHA256"] == after["boot_id_SHA256"]
            except (ValueError, OSError, KeyError, subprocess.SubprocessError, TimeoutError) as error:
                result.update(post_clean=False, post_error_type=type(error).__name__, post_error=str(error)[:240])
        result["passed"] = bool(result.get("comparison", {}).get("passed") and result.get("post_clean")
                                and "error_type" not in result)
        save(WORK/"outcome.json", result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    require(len(sys.argv) == 2 and re.fullmatch("[a-f0-9]{40}", sys.argv[1]), "Published source commit required")
    raise SystemExit(run())
