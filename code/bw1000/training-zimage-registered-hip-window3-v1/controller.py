"""Run one native HIP fusion with independent before and after device inventories."""

from hashlib import sha256
import fcntl
import json
from zipfile import ZipFile
import os
from pathlib import Path
import re
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
    need(deadline == epoch + claim["whole_seconds"] and epoch <= time.time() < epoch + 30,
         "Controller is outside the original start window")
    phases = {}
    result = {"request": claim["request"], "epoch": epoch, "deadline": deadline,
              "observations": phases, "passed": False, "independent_A100_fusion_parity": False,
              "gpu_srun_attempted": False, "training_ready": False, "accounting_columns": COLUMNS}

    monotonic_deadline = time.monotonic() + deadline - time.time()

    def remaining(reserve=0):
        available = min(deadline-time.time(),monotonic_deadline-time.monotonic())-reserve
        if available <= 0:
            raise TimeoutError("Fusion controller deadline")
        return available

    def query(*argv):
        return subprocess.check_output(argv, text=True, timeout=min(10, remaining(2)))

    def accounting(name, accounting_parent=parent):
        raw = query("sacct", "-n", "-P", "-j", accounting_parent, "--format=" + ",".join(COLUMNS))
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
        budget = min(claim["worker_seconds"]-20 if gpu else 55, remaining(75 if gpu else 10))
        args = ["srun", "--jobid="+parent, "--job-name="+name, "--exact", "--nodes=1", "--ntasks=1",
                "--nodelist="+node, "--immediate=5", "--threads=1", "--cpus-per-task="+str(claim["cpus"] if gpu else 2),
                "--mem="+(str(claim["memory_MiB"])+"M" if gpu else "4096M"), "--gres="+("dcu:1" if gpu else "none"),
                "--time="+time.strftime("%H:%M:%S", time.gmtime(int(budget))), *command]
        path = ROOT / (phase + ".log")
        if gpu:
            result.setdefault("attempted_gpu_phases",[]).append(phase)
        with path.open("xb") as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if gpu:
                result["gpu_srun_attempted"] = True
            try:
                if gpu:
                    # Keep global admission exclusive until Slurm owns the target GPU.
                    startup_end=time.monotonic()+min(40, remaining(90))
                    while child.poll() is None and time.monotonic() < startup_end:
                        try:
                            live_rows=query("scontrol","show","step",parent,"-o").splitlines()
                        except subprocess.SubprocessError:
                            time.sleep(.4)
                            continue
                        for line in live_rows:
                            live=fields(line)
                            if live.get("Name")==name and live.get("State")=="RUNNING":
                                need(live["NodeList"]==node and "gres/dcu=1" in live["TRES"].split(","),
                                     "Unexpected confirmed GPU step scope")
                                result["live_GPU_step"]=live["StepId"]
                                save("gpu-step-bound.json",dict(step=live["StepId"],name=name,node=node,request=claim["request"]))
                                fcntl.flock(handoff_lock,fcntl.LOCK_UN)
                                break
                        if "live_GPU_step" in result:
                            break
                        time.sleep(.4)
                budget=min(budget, remaining(60 if gpu else 5))
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

    def owned_standby():
        matching = []
        allowed = claim["standby_handoff"]
        for line in query("scontrol", "show", "step", parent, "-o").splitlines():
            row = fields(line)
            if row["StepId"] in (parent+".batch", parent+".extern"):
                continue
            need(re.fullmatch(parent+r"\.\d+",row["StepId"])
                 and row.get("NodeList") not in (None,"","(null)"),"Unknown numbered step scope")
            expression = row["NodeList"]
            members = [expression] if re.fullmatch(r"f\d+r\d+n\d+",expression) else query(
                "scontrol","show","hostnames",expression).splitlines()
            if node not in members or row.get("State") in {
                "COMPLETED","CANCELLED","FAILED","TIMEOUT","NODE_FAIL","PREEMPTED","DEADLINE","OUT_OF_MEMORY"
            }:
                continue
            need(members == [allowed["node"]] and row["StepId"] == allowed["step"]
                 and row["Name"] == allowed["name"] and row["State"] == "RUNNING"
                 and "gres/dcu=8" in row["TRES"].split(","), "Unrecognized live target node owner")
            matching.append(row)
        need(len(matching)<=1,"Duplicate target standby identity")
        return matching[0] if matching else None

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
    handoff_lock = (BASE/"results/.training80-dcu-standby.lock").open("ab")
    try:
        fcntl.flock(handoff_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check = fields(query("scontrol", "show", "job", parent, "-o"))
        need(check["JobState"] == "RUNNING" and check["UserId"].endswith(f"({os.getuid()})")
             and check["NumNodes"] == "40" and "gres/dcu=320" in check["AllocTRES"].split(","),
             "Owned parent resources differ")
        need(node in query("scontrol", "show", "hostnames", check["NodeList"]).splitlines(),
             "Exact node left parent")
        standby_before_static_checks = owned_standby()
        manifest_raw = (ROOT / "SOURCE_FILES.json").read_bytes()
        need(sha256(manifest_raw).hexdigest() == intent["manifest_SHA256"], "Target manifest mismatch")
        for relative, pin in json.loads(manifest_raw)["files"].items():
            path = ROOT / relative
            raw = path.read_bytes()
            need(ROOT in path.resolve().parents and len(raw) == pin["bytes"]
                 and sha256(raw).hexdigest() == pin["SHA256"], "Execution source drift")
        entry = ROOT/"native_train.py"
        need(entry.is_file() and sha256(entry.read_bytes()).hexdigest() == claim["native_train_file"]["SHA256"]
             and entry.stat().st_size == claim["native_train_file"]["bytes"], "Pinned original training entry absent")
        source = BASE / "results/training-zimage-native-checkout-import-v1/source"
        compat = BASE / "results/training-zimage-native-checkout-import-v2"
        evidence_raw = (ROOT / "NATIVE_IMPORT_EVIDENCE.json").read_bytes()
        need(sha256(evidence_raw).hexdigest() == claim["target_zero_gres_native_import"]["SHA256"],
             "Qualified zero-GRES evidence differs")
        evidence = json.loads(evidence_raw)
        row = evidence["rows"][0]
        need(row == ["1087656.1126", compat.name, "COMPLETED", "0:0", "27", "f11r4n13", "cpu=4,mem=8G,node=1"]
             and accounting(compat.name, "1087656") == row, "Qualified zero-GRES accounting differs")
        for name, pin in evidence["originals"].items():
            original = (compat / name).read_bytes()
            need(len(original) == pin["bytes"] and sha256(original).hexdigest() == pin["SHA256"],
                 "Qualified runtime receipt changed")
        compat_raw = (compat / "CLAIM.json").read_bytes()
        need(sha256(compat_raw).hexdigest() == claim["isolated_runtime_claim"]["SHA256"], "Overlay claim changed")
        for name, pin in json.loads(compat_raw)["wheel_files"].items():
            wheel = compat / name
            raw = wheel.read_bytes()
            need(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "Overlay wheel changed")
            with ZipFile(wheel) as archive:
                for member in archive.infolist():
                    path = compat / "site" / member.filename
                    need(path.is_file() and path.resolve(strict=True) == path
                         and path.stat().st_size == member.file_size and path.read_bytes() == archive.read(member),
                         "Installed overlay member changed")
        trusted = {"siglip_provider_SHA256": evidence["originals"]["stdout.private"]["records"][1]["siglip_provider_SHA256"],
                   "overlay_versions": evidence["originals"]["stdout.private"]["records"][1]["verified_overlay_versions"]}
        save("qualified-overlay.json", trusted)
        standby = owned_standby()
        if standby is not None:
            need(standby_before_static_checks is not None
                 and standby["StepId"] == standby_before_static_checks["StepId"],
                 "Standby changed during bounded preparation")
            save("standby-handoff-intent.json",dict(node=node,step=standby["StepId"],name=standby["Name"]))
            result["standby_cancel_attempted"] = True
            query("scancel","--signal=TERM",standby["StepId"])
            row = wait_accounting(standby["Name"],cap=35)
            need(row is not None and row[0]==standby["StepId"] and row[5]==node,"Original standby terminal accounting missing")
            need(owned_standby() is None,"Original standby still appears live")
            result["standby_terminal"] = row
        if standby is None:
            terminal = wait_accounting(claim["standby_handoff"]["name"],cap=12)
            need(terminal is not None and terminal[0]==claim["standby_handoff"]["step"] and terminal[5]==node,
                 "Previous exact standby has not reconciled")
            result["standby_terminal"] = terminal
        pre = snapshot("pre")
        need(sha256(",".join(sorted(pre["PCIs"])).encode()).hexdigest()==claim["expected_target_pci_sha256"],"Actual target PCI inventory differs")
        entries = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        bindings = (
            (ROOT, "/fusion-run", "rw"),
            (source, "/native-source", "ro"),
            (BASE/"results/training-zimage-original-coco20-media-stage-v1/inputs", "/coco20-inputs", "ro"),
            (BASE/"results/training-zimage-original-coco20-native-input-cpu-v1", "/coco20-preprocess", "ro"),
            (BASE/"results/training-common-zimage-complete-window-v1/source-tree", "/common-framework", "ro"),
            (BASE/"results/training-common-zimage-registered-hip-v1", "/binding-read", "ro"),
            (BASE/"results/training-zimage-coco20-fixed-full-hip-image-v2", "/prior-approved", "ro"),
            (BASE/"results/training-zimage-COCO20-native-first-step-hip-v2/first-update-private", "/original-first", "ro"),
            (compat / "site", "/zimage-compat", "ro"),
            (BASE/"results/training-zimage-target-native-meta-CPU-v2/dependencies/site", "/zimage-runtime", "ro"),
            (BASE/"results/training-t06-sdxl-runtime-v1", "/training-runtime", "ro"),
            (BASE/"results/training-zimage-modelscope-whole-runtime-v2/runtime", "/sealed-runtime", "ro"),
            (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
        )
        need(all(src.is_dir() for src, _, _ in bindings), "Original approved mount absent")
        for src, dst, mode in bindings:
            entries.extend(("--bind", f"{src}:{dst}:{mode}"))
        env = ("set -euo pipefail; source /opt/dtk/env.sh; "
               "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
               "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/native-source:/zimage-compat:/zimage-runtime:/training-runtime/site:/common-framework/generation/src; "
               "exec python3 -B -u /fusion-run/worker.py")
        entries.extend((str(BASE / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", env))
        _, finished = launch("baseline", entries[:-1] + [env+" batch"], gpu=True)
        batch_path=ROOT/"batch-result.json"
        if batch_path.is_file():
            batch_raw=batch_path.read_bytes()
            result["batch_SHA256"]=sha256(batch_raw).hexdigest()
            result["completed_repeats"]=json.loads(batch_raw)["completed_repeats"]
        result["device_passed"]=finished
        need(finished and result["completed_repeats"]==3, "Three independent complete windows did not finish")
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:240])
    finally:
        if pre is not None:
            lock_end=time.monotonic()+min(15,max(0,deadline-time.time()-45))
            while True:
                try:
                    fcntl.flock(handoff_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    if time.monotonic()>=lock_end:
                        result.update(post_clean=False,post_error_type=type(error).__name__)
                        break
                    time.sleep(.4)
                else:
                    result["post_lock_acquired"]=True
                    break
        if pre is not None and result.get("post_lock_acquired"):
            try:
                need(all(p in phases for p in result.get("attempted_gpu_phases",[])),
                     "Actual terminal accounting required before post")
                after = snapshot("post")
                result["post_clean"] = pre["PCIs"] == after["PCIs"] and pre["boot_id_SHA256"] == after["boot_id_SHA256"]
            except Exception as error:
                result.update(post_clean=False, post_error_type=type(error).__name__, post_error=str(error)[:240])
    result["passed"] = bool(result.get("device_passed") and result.get("post_clean") and not result.get("error_type"))
    handoff_lock.close()
    save("outcome.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
