"""Run and audit the complete T09 DP2 continuous and fresh-recovery scope."""
from base64 import b64decode
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-hunyuanvideo-dp2-live25-recovery-v1"
PARENT, NODE = "1103606", "f13r3n13"
PREFLIGHT = RESULTS / "training-hunyuanvideo-cached-dp2-target-preflight-v4"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PY = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
PHASES = ("continuous", "resume10", "resume20")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def need(value, detail):
    if not value:
        raise ValueError(detail)


def query(*args):
    return subprocess.check_output(args, text=True, timeout=10).splitlines()


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def record(path, value):
    return save(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def metadata(path, pin=None):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    need(0 < len(raw) <= 65536 and (pin is None or sha256(raw).hexdigest() == pin), "Bound DP2 metadata differs")
    return json.loads(raw)


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def accounting():
    return [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS)]


def node_steps():
    [row] = query("scontrol", "show", "job", "-o", PARENT)
    parent = fields(row)
    hosts = query("scontrol", "show", "hostnames", parent["NodeList"])
    need(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING"
         and parent["UserId"].endswith(f"({os.getuid()})")
         and NODE in hosts and len(hosts) == len(set(hosts)) == 40
         and "gres/dcu=320" in parent["AllocTRES"].split(","), "Current T09 allocation differs")
    rows = query("scontrol", "show", "step", PARENT, "--oneliner")
    need(rows, "Current Slurm step inventory unavailable")
    matches = []
    for line in rows:
        step = fields(line)
        handle = step["StepId"]
        if handle in (PARENT + ".batch", PARENT + ".extern"):
            continue
        need(re.fullmatch(PARENT + r"\.[0-9]+", handle), "Unexpected T09 parent step")
        members = query("scontrol", "show", "hostnames", step["NodeList"])
        need(members and set(members).issubset(hosts), "Numbered step membership differs")
        if NODE in members:
            matches.append(handle)
    return matches


def reclaim(claim, run):
    steps = node_steps()
    if not steps:
        return None
    need(len(steps) == 1, "T09 node has more than one incumbent step")
    [handle] = steps
    request = claim["previous_owned_standby_request"]
    root = RESULTS / ("training80-dcu-standby-" + request)
    expected_label = f"training80-dcu-standby-{request[:12]}-{NODE}"
    expected = metadata(root / NODE / "expected.json")
    worker = metadata(root / NODE / "worker.json")
    prior = metadata(root / "intent.json")
    rows = [row for row in accounting() if row[0] == handle]
    need(prior["request"] == request and prior["synthetic"] is True
         and expected["parent"] == PARENT and expected["node"] == NODE and expected["label"] == expected_label
         and worker["step"] == handle and len(rows) == 1 and rows[0][1] == expected_label
         and rows[0][2] == "RUNNING" and rows[0][5] == NODE and "gres/dcu=8" in rows[0][6].split(","),
         "Only the specifically verified project standby may be reclaimed")
    record(run / "reclaim-intent.json", dict(handle=handle, node=NODE, stage_label=expected_label, before=rows[0]))
    q = subprocess.run(["scancel", "--signal=TERM", handle], capture_output=True, timeout=10)
    save(run / "reclaim.stderr.private", q.stderr)
    end = time.monotonic() + 45
    while time.monotonic() < end:
        rows = [row for row in accounting() if row[0] == handle]
        if len(rows) == 1 and rows[0][2].split()[0] in TERMINAL and not node_steps():
            receipt = dict(handle=handle, reclaimed=True, signal_rc=q.returncode, terminal=rows[0])
            record(run / "reclaim-outcome.json", receipt)
            return receipt
        time.sleep(1)
    raise ValueError("Original standby terminal cleanup was not verified")


def argv(run, phase, seconds):
    gpu = phase == "gpu"
    cap = f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"
    base = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL + "-" + phase, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--time=" + cap, "--kill-on-bad-exit=1",
            "--gres=dcu:2" if gpu else "--gres=none", "--cpus-per-task=16" if gpu else "--cpus-per-task=1",
            "--mem=210000M" if gpu else "--mem=1024M"]
    if not gpu:
        return [*base, HOST_PY, "-B", str(run / "src/worker.py"), str(run), phase]
    binds = (
        f"{ROOT}:{ROOT}:ro", f"{run}:{run}:rw", "/opt/hyhal:/opt/hyhal:ro",
        f"{RESULTS}/training-t06-sdxl-runtime-v1:/training-runtime:ro",
        f"{RESULTS}/training-hunyuanvideo-original-source-v1/source:/t09-native/hunyuanvideo-1.5:ro",
        f"{RESULTS}/training-hunyuanvideo-cached-native25-target-v3/overlay/flash_attn_no_pad.py:"
        "/t09-native/hunyuanvideo-1.5/hyvideo/utils/flash_attn_no_pad.py:ro",
    )
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; exec python3 -B -u \"$@\"")
    mounts = [token for bind in binds for token in ("--bind", bind)]
    return [*base, APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/", *mounts,
            str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell, LABEL,
            str(run / "src/worker.py"), str(run), phase]


def phase_run(run, phase, seconds, deadline, release=None):
    limit = min(seconds, int(deadline - time.monotonic()) - 75)
    need(limit >= (90 if phase == "gpu" else 15), "Original DP2 time budget cannot support this phase")
    started = time.monotonic()
    with (run / f"{phase}.stdout.private").open("xb") as out, (run / f"{phase}.stderr.private").open("xb") as err:
        process = subprocess.Popen(argv(run, phase, limit), stdout=out, stderr=err)
        if release is not None:
            readiness = min(deadline, started + 25)
            while process.poll() is None and time.monotonic() < readiness:
                rows = [row for row in accounting() if row[1] == LABEL + "-" + phase]
                if len(rows) == 1 and rows[0][0] in node_steps() and rows[0][2] == "RUNNING":
                    record(run / "gpu-live-ownership.json", dict(row=rows[0], epoch=time.time()))
                    break
                time.sleep(.5)
            release()
        try:
            code = process.wait(timeout=max(1, limit + 12 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=5)
    until = min(deadline, time.monotonic() + 15)
    rows = []
    while time.monotonic() < until:
        rows = [row for row in accounting() if row[1] == LABEL + "-" + phase]
        if len(rows) == 1 and rows[0][2].split()[0] in TERMINAL:
            break
        time.sleep(.5)
    result = dict(rc=code, rows=rows, terminal=len(rows) == 1 and rows[0][2].split()[0] in TERMINAL)
    result["passed"] = (result["terminal"] and code == 0 and rows[0][2:4] == ["COMPLETED", "0:0"]
                        and re.fullmatch(PARENT + r"\.[0-9]+", rows[0][0]) is not None and rows[0][5] == NODE
                        and ("gres/dcu=2" in rows[0][6].split(",") if phase == "gpu" else "gres/" not in rows[0][6]))
    record(run / f"{phase}-receipt.json", result)
    return result


def audit_results(run, results, claim):
    sys.path.insert(0, str(run / "src"))
    from audit import audit_recovery

    bounds = {}
    row = results["gpu"]["rows"][0]
    assigned_devices = None
    for phase in PHASES:
        path = run / f"outputs/hunyuanvideo-1.5/train/{phase}"
        raw = (path / "dp2-evidence.json").read_bytes()
        need(len(raw) <= 65536, "DP2 final evidence exceeded bounds")
        bounds[phase] = (path, sha256(raw).hexdigest())
        assigned = metadata(run / f"{phase}-assignment.json")
        need(assigned["step"] == row[0] and assigned["node"] == NODE, "DP2 worker and Slurm assignment differ")
        if assigned_devices is None:
            assigned_devices = assigned["pci"]
        need(assigned["pci"] == assigned_devices, "DP2 recovery ran on a different physical rank pair")
    path = run / "independent-accounting.psv"
    pin = save(path, ("|".join([row[0], *row[2:]]) + "\n").encode())
    audit = audit_recovery(bounds["continuous"], {10: bounds["resume10"], 20: bounds["resume20"]},
                           claim["fixed_DP2_backend_SHA256"], accounting=(path, pin))
    return audit, record(run / "audit.json", audit)


def run(envelope):
    os.umask(0o077)
    start = time.time()
    raw = {key: b64decode(value, validate=True) for key, value in envelope["files"].items()}
    manifest = json.loads(raw["SOURCE_FILES.json"])
    expected = {"worker.py", "controller.py", "snapshot.py", "audit.py", "CLAIM.json"}
    need(set(raw) == expected | {"SOURCE_FILES.json"} and set(manifest["files"]) == expected, "Frozen DP2 publication source set differs")
    for name, pin in manifest["files"].items():
        need(len(raw[name]) == pin["bytes"] and sha256(raw[name]).hexdigest() == pin["SHA256"], "DP2 source bytes differ")
    claim = json.loads(raw["CLAIM.json"])
    need((claim["label"], claim["parent"], claim["node"], claim["planned_DCU"], claim["max_invocations"])
         == (LABEL, PARENT, NODE, 2, 1) and claim["GPU_GO"] is True and claim["replay_allowed"] is False
         and re.fullmatch(r"[0-9a-f]{32}", claim["request"]) and re.fullmatch(r"[0-9a-f]{40}", envelope["publication"]),
         "DP2 claim scope differs")
    dry = metadata(PREFLIGHT / "outcome.json", claim["preflight_outcome_SHA256"])
    need(dry["request"] == claim["preflight_request"] and dry["passed"] is True
         and dry["private_framework_inventory_SHA256"] == claim["private_framework_inventory_SHA256"]
         and dry["launch_SHA256"] == claim["original_launch_SHA256"], "Compute preflight is not qualified")
    deadline = time.monotonic() + claim["whole_seconds"]
    run = RESULTS / LABEL
    result = dict(request=claim["request"], publication=envelope["publication"], passed=False,
                  stages={}, planned_steps=claim["planned_steps"], replay_allowed=False)
    with (RESULTS / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any(row[1].startswith(LABEL + "-") for row in accounting()), "Previous live-label DP2 accounting exists")
        run.mkdir(mode=0o700)
        (run / "src").mkdir()
        for name, data in raw.items():
            save(run / "src" / name, data)
        result["intent_SHA256"] = record(run / "intent.json", dict(result,
             parent=PARENT, node=NODE, origin_epoch=start, deadline_epoch=start+claim["whole_seconds"],
             backend_SHA256=claim["fixed_DP2_backend_SHA256"], weight_identity_tail=dry["mounted_weight_identity"][1:],
             phase_seconds=claim["phase_seconds"]))
        try:
            result["reclaimed_standby"] = reclaim(claim, run)
            result["stages"]["pre"] = phase_run(run, "pre", 60, deadline)
            need(result["stages"]["pre"]["passed"], "DP2 independent physical-device admission failed")
            need(not node_steps(), "Another workload entered the T09 node")
            result["stages"]["gpu"] = phase_run(run, "gpu", sum(claim["phase_seconds"].values()), deadline,
                                                release=lambda: fcntl.flock(lock, fcntl.LOCK_UN))
            need(result["stages"]["gpu"]["passed"], "DP2 training or recovery phase failed")
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            result["execution_error"] = type(error).__name__
            if type(error) is ValueError:
                result["execution_message"] = str(error)
        finally:
            try:
                if "pre" in result["stages"] and not node_steps():
                    result["stages"]["post"] = phase_run(run, "post", 60, deadline)
                    if result["stages"]["post"]["passed"] and result["stages"]["pre"]["passed"]:
                        before, after = [metadata(run / f"{phase}-devices.json") for phase in ("pre", "post")]
                        result["post_clean"] = all(before[key] == after[key] for key in ("boot_id", "pci", "pci_to_kfd"))
            except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
                result["post_error"] = type(error).__name__
            fcntl.flock(lock, fcntl.LOCK_UN)
    try:
        if result.get("post_clean") is True and result["stages"].get("gpu", {}).get("passed") is True:
            audit, result["independently_read_audit_SHA256"] = audit_results(run, result["stages"], claim)
            result["independent_audit"] = audit
            result["passed"] = True
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["audit_error"] = type(error).__name__
        if type(error) is ValueError:
            result["audit_message"] = str(error)
    result["outcome_SHA256"] = record(run / "outcome.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(INPUT))
