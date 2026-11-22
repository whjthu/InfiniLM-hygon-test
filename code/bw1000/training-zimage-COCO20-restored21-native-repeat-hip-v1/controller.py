"""Run one native HIP fusion with independent before and after device inventories."""

from hashlib import sha256
import json
from zipfile import ZipFile
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import runpy
import sys


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


def wait_for_worker_files(root, modes, budget):
    until = time.monotonic() + budget
    while True:
        try:
            for mode in modes:
                for name in ("worker-result.json", "repeat-receipt.json", "steps.jsonl", "optimizer-journal.jsonl"):
                    path = root / mode / name
                    if path.stat().st_size == 0:
                        raise FileNotFoundError(str(path))
            return
        except FileNotFoundError:
            left = until - time.monotonic()
            if left <= 0:
                raise
            time.sleep(min(.25, left))


def load_compare(files):
    entry = ROOT / "compare.py"
    pin = files["compare.py"]
    data = entry.read_bytes()
    need(entry.resolve() == entry and len(data) == pin["bytes"] and sha256(data).hexdigest() == pin["SHA256"],
         "Pinned controller comparison source differs")
    compare = runpy.run_path(str(entry))["compare"]
    need(callable(compare), "Pinned comparison function unavailable")
    return compare


def check_entry():
    claim = json.loads((ROOT/"CLAIM.json").read_bytes())
    need(re.fullmatch(r"[a-f0-9]{32}", claim["request"]) and ROOT.name == claim["label"]
         and claim["parent"] == "1103606" and claim["node"] == "f13r2n07"
         and claim["actual_optimizer_updates_planned"] == 3
         and claim["arms"] == {"continuous": ["resident_first", "resident_repeat"], "resume20": ["fresh_process"]},
         "Frozen launch identity or training scope is invalid")
    manifest = json.loads((ROOT/"SOURCE_FILES.json").read_bytes())["files"]
    load_compare(manifest)
    print(json.dumps(dict(isolated_controller_import=True, Slurm_steps=0), sort_keys=True))


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
              "observations": phases, "passed": False, "independent_A100_fusion_parity": False,
              "gpu_srun_attempted": False, "training_ready": False, "accounting_columns": COLUMNS}

    def remaining(reserve=0):
        available = deadline - time.time() - reserve
        if available <= 0:
            raise TimeoutError("Fusion controller deadline")
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
        budget = min(claim["maximum_device_seconds"][phase] if gpu else 55, remaining(75 if gpu else 10))
        args = ["srun", "--jobid="+parent, "--job-name="+name, "--exact", "--nodes=1", "--ntasks=1",
                "--nodelist="+node, "--immediate=5", "--cpus-per-task="+str(claim["cpus"] if gpu else 2),
                "--mem="+(str(claim["memory_MiB"])+"M" if gpu else "4096M"), "--gres="+("dcu:1" if gpu else "none"),
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
        compare = load_compare(json.loads(manifest_raw)["files"])
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
             and [line.split("|") for line in query("sacct", "-n", "-P", "-j", "1087656.1126",
                    "--format="+",".join(COLUMNS)).splitlines() if line.split("|")[0]=="1087656.1126"] == [row],
             "Qualified historical zero-GRES accounting differs")
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
        pre = snapshot("pre")
        admitted = intent["single_parent_admission"]
        need(pre["boot_id_SHA256"] == admitted["boot_id_SHA256"] and pre["PCIs"] == admitted["PCIs"],
             "Physical PCI roster or boot identity changed since the admission probe")
        entries = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        bindings = (
            (ROOT, "/fusion-run", "rw"),
            (BASE/"results"/claim["original_label"], "/original-coco20", "ro"),
            (source, "/native-source", "ro"),
            (BASE/"results/training-qwen-a68952c-swift-initial-CPU-v1/framework", "/common-framework", "ro"),
            (BASE/"results/training-zimage-original-coco20-media-stage-v1/inputs", "/coco20-inputs", "ro"),
            (BASE/"results/training-zimage-original-coco20-native-input-cpu-v1", "/coco20-preprocess", "ro"),
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
        def device(mode):
            command = entries + [str(BASE / "containers/dtk26-sandbox-r1"), "/bin/bash",
                                 "--noprofile", "--norc", "-c", env + " " + mode]
            _, passed = launch(mode, command, gpu=True)
            return passed

        continuous = device("continuous")
        result["continuous_passed"] = continuous
        if continuous:
            between = snapshot("between")
            need(pre["PCIs"] == between["PCIs"] and pre["boot_id_SHA256"] == between["boot_id_SHA256"],
                 "Device inventory changed between processes")
            result["resume20_passed"] = device("resume20")
            if result["resume20_passed"]:
                wait_for_worker_files(ROOT, ("continuous", "resume20"), min(20, remaining(70)))
                comparison = compare(ROOT)
                need(comparison["worker_steps"] == {mode: phases[mode]["accounting"][0]
                                                     for mode in ("continuous", "resume20")},
                     "Worker and terminal accounting identities differ")
                result["comparison_SHA256"] = save("comparison.json", comparison)
                result["paired_passed"] = comparison["passed"]
        result["domain_workers"] = {}
        for mode in ("continuous", "resume20"):
            worker = ROOT / mode / "worker-result.json"
            if worker.is_file():
                result["domain_workers"][mode] = sha256(worker.read_bytes()).hexdigest()
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:240])
    finally:
        if result["gpu_srun_attempted"]:
            try:
                need("continuous" in phases and ("between" not in phases or "resume20" in phases), "Actual terminal device accounting required before post")
                after = snapshot("post")
                result["post_clean"] = pre["PCIs"] == after["PCIs"] and pre["boot_id_SHA256"] == after["boot_id_SHA256"]
            except Exception as error:
                result.update(post_clean=False, post_error_type=type(error).__name__, post_error=str(error)[:240])
    result["passed"] = bool(result.get("paired_passed") and result.get("post_clean") and not result.get("error_type"))
    save("outcome.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-entry"]:
        check_entry()
    elif len(sys.argv) == 1:
        raise SystemExit(main())
    else:
        raise SystemExit(2)
