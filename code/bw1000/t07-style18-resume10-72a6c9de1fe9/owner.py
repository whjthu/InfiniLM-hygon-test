"""Own one admitted Style18 resume run and retain original Slurm accounting."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from device_entry import (BASE, INV_SHA, LABEL, MAX_GPU, NODE, ORIGINAL, PARENT, PYTHON,
                          REQUEST, STAGE, TERMINAL, live_fields, manifest, need, proof,
                          srun_argv, write)

SOURCE_SHA = "f3d2c4f67a1c7b44d9a530d06a05fabcb4af9a8cb57fbe17b181d5ae2cafd910"
COLUMNS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
LOCK = Path(BASE / "results/.training80-dcu-standby.lock")
SCOPE_SECONDS = 900


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def utc(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


@contextmanager
def shared_lock(clock, wait=0):
    import fcntl

    fd = os.open(LOCK, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode), "EXISTING_REGULAR_SHARED_OWNER_LOCK")
        until = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise
                time.sleep(min(.2, clock.left(12)))
        cur = LOCK.lstat()
        need((cur.st_dev, cur.st_ino) == (st.st_dev, st.st_ino), "UNCHANGED_SHARED_OWNER_LOCK")
        yield
    finally:
        os.close(fd)


class Window:
    def __init__(self):
        self.epoch = time.time()
        self.end = self.epoch + SCOPE_SECONDS
        self.deadline = time.monotonic() + SCOPE_SECONDS

    def left(self, reserve=0):
        left = min(self.end - time.time(), self.deadline - time.monotonic()) - reserve
        if left <= 0:
            raise TimeoutError("ONE_RESUME10_EPOCH_DEADLINE")
        return left

    def query(self, *argv):
        raw = subprocess.check_output(argv, env=dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S"),
                                      timeout=min(7, self.left(3)))
        need(len(raw) <= 4 * 1024 * 1024, "BOUNDED_SLURM_QUERY")
        return raw.decode("utf-8").splitlines()

    def nodes(self, spec):
        if re.fullmatch(r"f\d+r\d+n\d+", spec):
            return [spec]
        need(re.fullmatch(r"[f0-9rn,\[\]-]{1,8192}", spec) is not None, "UNCLASSIFIED_SCHEDULER_NODELIST")
        rows = self.query("scontrol", "show", "hostnames", spec)
        need(0 < len(rows) <= 80 and len(set(rows)) == len(rows) and
             all(re.fullmatch(r"f\d+r\d+n\d+", x) for x in rows), "EXACT_SCHEDULER_NODELIST")
        return rows

    def live(self, permitted=None):
        job, = self.query("scontrol", "show", "job", PARENT, "--oneliner")
        j = fields(job)
        need(j["JobId"] == PARENT and j["JobState"] == "RUNNING" and
             j["UserId"].endswith(f"({os.getuid()})") and NODE in self.nodes(j["NodeList"]) and
             utc(j["EndTime"]) >= self.end and "gres/dcu=320" in j["AllocTRES"].split(","),
             "ACTUAL_OWNED_PARENT_HAS_FROZEN_NODE_AND_TIME")
        ctl = []
        for line in self.query("scontrol", "show", "step", PARENT, "--oneliner"):
            row = fields(line)
            step = row["StepId"]
            if re.fullmatch(PARENT + r"\.\d+", step) and row["State"].rstrip("+") not in TERMINAL:
                if NODE in self.nodes(row["NodeList"]):
                    ctl.append(row)
        queued = []
        for line in self.query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%j|%N"):
            step, label, spec = line.split("|")
            if re.fullmatch(PARENT + r"\.\d+", step) and NODE in self.nodes(spec):
                queued.append((step, label))
        if permitted is None:
            need(not ctl and not queued, "BOTH_SCHEDULER_VIEWS_NODE_CLEAR")
            return None
        need(all((r.get("Name", r.get("JobName")) == permitted and
                  live_fields(r.get("TRES", r.get("AllocTRES", "")), True)) for r in ctl) and
             all(label == permitted for _, label in queued), "NO_FOREIGN_STEP_IN_EXACT_GPU_GATE")
        if len(ctl) == len(queued) == 1 and ctl[0]["StepId"] == queued[0][0]:
            return queued[0][0]
        return None

    def accounting(self, phase, until):
        label = LABEL + "-" + phase
        while True:
            rows = [line.split("|") for line in self.query("sacct", "-nP", "-j", PARENT, "-o", COLUMNS) if line]
            need(all(len(row) == 9 for row in rows), "EXACT_NINE_ACCOUNTING_COLUMNS")
            found = [row for row in rows if row[1] == label]
            need(len(found) <= 1, "EXACT_LABEL_HAS_ONE_ORIGINAL_STEP")
            if found and found[0][2].rstrip("+") in TERMINAL:
                row = found[0]
                need(re.fullmatch(PARENT + r"\.\d+", row[0]) and row[5] == NODE and
                     live_fields(row[6], phase == "gpu") and
                     self.epoch - 1 <= utc(row[7]) <= utc(row[8]) <= time.time() + 1,
                     "ACTUAL_TERMINAL_SCOPE_AND_TRES")
                return row
            if time.monotonic() >= until:
                raise TimeoutError("ORIGINAL_ACCOUNTING_NOT_TERMINAL")
            time.sleep(min(.35, self.left(5)))


def wait_child(child, timeout):
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            return child.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            return child.wait(timeout=6)


def snapshot(clock, phase):
    original = Path(ORIGINAL / "node_inventory.py")
    need(original.resolve(strict=True) == original and original.stat().st_size <= 16384
         and digest(original) == INV_SHA, "PINNED_ORIGINAL_DEVICE_INVENTORY")
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL + "-" + phase, "--exact", "--nodes=1",
            "--ntasks=1", "--nodelist=" + NODE, "--immediate=5", "--cpus-per-task=2", "--mem=4096M",
            "--gres=none", "--time=00:01:00", PYTHON, "-I", "-S", "-B", str(original), PARENT, NODE]
    with (HERE / (phase + ".log")).open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rc = wait_child(child, min(30, clock.left(20)))
    row = clock.accounting(phase, time.monotonic() + 10)
    log = HERE / (phase + ".log")
    need(log.stat().st_size <= 32768 and rc == 0 and row[2:4] == ["COMPLETED", "0:0"], "ACTUAL_ZERO_GRES_SNAPSHOT")
    value, = [json.loads(line) for line in log.read_bytes().splitlines() if line.startswith(b"{")]
    need(value["node"] == NODE and value["step"] == row[0] and len(set(value["PCIs"])) == len(value["PCIs"]) == 8
         and value["kfd_process_count"] == value["own_device_fd_process_count"] == 0, "ACTUAL_SAME_NODE_EIGHT_DEVICE_CLEARANCE")
    value.update(request=REQUEST, phase=phase, parent=PARENT, observed_epoch=time.time(), accounting=row)
    proof_path = HERE / (phase + ".json")
    write(proof_path, value)
    return value, digest(proof_path)


def run():
    manifest(HERE, SOURCE_SHA)
    need(HERE == Path(STAGE), "EXACT_OWNED_ONE_SHOT_STAGE")
    auth = json.loads((HERE / "root-start.json").read_bytes())
    need(auth["request"] == REQUEST and auth["parent"] == PARENT and auth["node"] == NODE and
         auth["source_manifest_SHA256"] == SOURCE_SHA and auth["root_GPU_GO"] is True and
         0 <= time.time() - auth["sent_epoch"] <= 30,
         "EXPLICIT_EXACT_ROOT_SCOPE")
    clock = Window()
    state = {"request": REQUEST, "epoch": clock.epoch, "planned_optimizer_updates": 15, "gpu_srun_spun": False}
    try:
        with shared_lock(clock, 8):
            clock.live()
            pre, pre_sha = snapshot(clock, "pre")
            clock.live()
            a = dict(request=REQUEST, label=LABEL, parent=PARENT, node=NODE,
                     original_full_read_SHA256=auth["original_full_read_SHA256"],
                     independent_PRE_SHA256=pre_sha, boot_id_SHA256=pre["boot_id_SHA256"], PCIs=pre["PCIs"],
                     root_GPU_GO=True, root_sole_owner=True, ordinary_exclusion_until_independent_post=True,
                     one_DCU_once=True, no_prior_unknown_conflict=True, epoch=clock.epoch, execution_deadline=clock.end)
            authority = HERE / "authority.json"
            write(authority, a)
            authority_sha = digest(authority)
            proof(authority.read_bytes(), authority_sha, (HERE / "pre.json").read_bytes(), pre_sha, time.time())
            write(HERE / "gpu-intent.json", {"request": REQUEST, "pre_SHA256": pre_sha, "epoch": clock.epoch,
                                           "authority_SHA256": authority_sha, "maximum_GPU_seconds": MAX_GPU})
            with (HERE / "gpu.log").open("xb") as log:
                child = subprocess.Popen(srun_argv(a, SOURCE_SHA, authority_sha, pre_sha),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(gpu_srun_spun=True, pre=pre["accounting"])
            start = time.monotonic()
            while child.poll() is None and time.monotonic() - start <= 18:
                active = clock.live(LABEL + "-gpu")
                if active:
                    state["first_verified_running_step"] = active
                    break
                time.sleep(.35)
            if child.poll() is None and "first_verified_running_step" not in state:
                state["gpu_returncode"] = wait_child(child, min(MAX_GPU, clock.left(85)))
        if "gpu_returncode" not in state:
            state["gpu_returncode"] = wait_child(child, min(MAX_GPU - (time.monotonic() - start), clock.left(85)))
        gpu = clock.accounting("gpu", time.monotonic() + 18)
        state["gpu"] = gpu
        with shared_lock(clock, 10):
            clock.live()
            post, post_sha = snapshot(clock, "post")
            need(post["PCIs"] == pre["PCIs"] and post["boot_id_SHA256"] == pre["boot_id_SHA256"] and
                 gpu[8] <= post["accounting"][7], "SAME_BOOT_EIGHT_DEVICE_RELEASE_AFTER_ORIGINAL_GPU")
            clock.live()
        state.update(post=post["accounting"], post_clean=True, post_SHA256=post_sha)
        command = [PYTHON, "-I", "-S", "-B", str(HERE / "device_entry.py"), "independent-terminal",
                   "--source-sha", SOURCE_SHA, "--post-sha", post_sha]
        with (HERE / "independent-read.log").open("xb") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=min(45, clock.left(3)))
        state["independent_reader_rc"] = done.returncode
        state["independent_receipt_exists"] = (HERE / "work-once/independent-terminal.json").is_file()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        state["error_type"] = type(exc).__name__
        state["error_SHA256"] = sha256(str(exc).encode("utf-8")).hexdigest()
        state["original_gpu_must_be_reconciled"] = state["gpu_srun_spun"] and "gpu" not in state
    write(HERE / "controller-outcome.json", state)


if __name__ == "__main__":
    run()
