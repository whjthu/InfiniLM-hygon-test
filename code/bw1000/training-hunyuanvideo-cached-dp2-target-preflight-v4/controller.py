"""Run host and zero-GRES compute checks against the original T09 DP2 launch."""
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-hunyuanvideo-cached-dp2-target-preflight-v4"
PARENT, NODE = "1103606", "f13r3n13"
HOST_PY = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
RUNTIME = ROOT / "results/training-t06-sdxl-runtime-v1"
ACCOUNTING_COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
SOURCE_NAMES = {
    "CLAIM.json", "controller.py", "preflight.py", "test_CPU.py",
    "hunyuanvideo_data_parallel.py", "hunyuanvideo_distributed_checkpoint.py",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=10).splitlines()


def accounting():
    lines = query("sacct", "-n", "-X", "-P", "-j", PARENT, "-o", ACCOUNTING_COLUMNS)
    parents = [line.split("|") for line in lines if line.split("|")[0] == PARENT]
    steps = query("sacct", "-n", "-P", "-j", PARENT, "-o", ACCOUNTING_COLUMNS)
    rows = [line.split("|") for line in steps if line.split("|")[1] == LABEL]
    need(len(parents) == 1 and parents[0][2] == "RUNNING", "live T09 parent accounting missing")
    return rows


def current_slot(standby_request):
    [row] = query("scontrol", "show", "job", "-o", PARENT)
    fields = dict(part.split("=", 1) for part in row.split() if "=" in part)
    nodes = query("scontrol", "show", "hostnames", fields["NodeList"])
    need(fields["JobId"] == PARENT and fields["JobState"] == "RUNNING"
         and fields["UserId"].endswith(f"({os.getuid()})")
         and len(set(nodes)) == len(nodes) == 40 and NODE in nodes
         and "gres/dcu=320" in fields["AllocTRES"].split(","), "current T09 allocation differs")
    inventory = query("scontrol", "show", "step", PARENT, "--oneliner")
    need(inventory, "live Slurm step inventory missing")
    covering = []
    for line in inventory:
        step = dict(word.split("=", 1) for word in line.split() if "=" in word)
        handle = step["StepId"]
        if handle in (PARENT + ".batch", PARENT + ".extern"):
            continue
        need(re.fullmatch(PARENT + r"\.[0-9]+", handle), "unrecognized step in T09 parent")
        members = query("scontrol", "show", "hostnames", step["NodeList"])
        need(members and set(members).issubset(nodes), "Slurm step location cannot be verified")
        if NODE in members:
            covering.append(handle)
    if not covering:
        return []
    need(len(covering) == 1 and re.fullmatch(r"[0-9a-f]{32}", standby_request),
         "T09 node already hosts another workload")
    root = ROOT / "results" / ("training80-dcu-standby-" + standby_request)
    intent = json.loads((root / "intent.json").read_bytes())
    expected = json.loads((root / NODE / "expected.json").read_bytes())
    worker = json.loads((root / NODE / "worker.json").read_bytes())
    label = f"training80-dcu-standby-{standby_request[:12]}-{NODE}"
    rows = [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", ACCOUNTING_COLUMNS)]
    matched = [r for r in rows if r[0] == covering[0]]
    need(intent["request"] == standby_request and intent["synthetic"] is True
         and expected["request"] == standby_request and expected["label"] == label
         and expected["parent"] == PARENT and worker["step"] == covering[0]
         and len(matched) == 1 and matched[0][1] == label and matched[0][2] == "RUNNING"
         and matched[0][5] == NODE and "gres/dcu=8" in matched[0][6].split(","),
         "T09 node does not have the explicitly bound project standby")
    return covering


def stash(path, raw):
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha256(raw).hexdigest()


def save(path, data):
    return stash(path, (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())


def object_output(raw, request):
    found = []
    for line in raw.splitlines():
        if line.startswith(b"{"):
            value = json.loads(line)
            if value.get("request") == request:
                found.append(value)
    need(len(found) == 1, "expected a unique preflight receipt")
    return found[0]


def worker_argv(directory, host):
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--overlap", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=2",
            "--mem=4096M", "--gres=none", "--time=00:02:00",
            APPTAINER, "exec", "--no-home", "--pwd", "/"]
    for mount in (f"{directory}:{directory}:ro",
                  f"{RUNTIME}:/training-runtime:ro", *host["mounts"]):
        argv += ["--bind", mount]
    shell = (
        "set -euo pipefail; source /opt/dtk/env.sh; "
        "export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= "
        "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; "
        "exec python3 -B -u \"$1\" mounted --a100-launch \"$2\" --asset-map \"$3\""
    )
    return [*argv, str(ROOT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", shell, "dp2",
            str(directory / "preflight.py"), str(directory / "original-launch.json"),
            str(directory / "asset-map.json")]


def run(envelope):
    os.umask(0o077)
    epoch = time.time()
    deadline = time.monotonic() + 240
    files = {name: b64decode(value, validate=True) for name, value in envelope["sources"].items()}
    manifest = json.loads(files["SOURCE_FILES.json"])
    need(set(files) == SOURCE_NAMES | {"SOURCE_FILES.json"} and set(manifest["files"]) == SOURCE_NAMES,
         "exact preflight source set differs")
    for name, pin in manifest["files"].items():
        raw = files[name]
        need(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "source bytes differ")
    claim = json.loads(files["CLAIM.json"])
    request = claim["request"]
    publication = envelope["publication"]
    launch = b64decode(envelope["original_launch"], validate=True)
    directory = ROOT / "results" / LABEL
    need(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch(r"[0-9a-f]{40}", publication)
         and (claim["parent"], claim["node"], claim["label"], claim["DCU"], claim["optimizer_updates"])
         == (PARENT, NODE, LABEL, 0, 0) and claim["GPU_GO"] is False
         and claim["maximum_invocations"] == 1 and claim["replay_allowed"] is False
         and len(launch) == claim["original_launch"]["bytes"] == 1563
         and sha256(launch).hexdigest() == claim["original_launch"]["SHA256"],
         "original launch or run identity differs")
    owned_standby = claim["owned_standby_request"]
    initial_standby = current_slot(owned_standby)
    need(not accounting(), "preflight accounting already exists")
    directory.mkdir(mode=0o700)
    intent = dict(request=request, publication=publication, epoch=epoch, deadline=epoch+240,
                  parent=PARENT, node=NODE, source_manifest_SHA256=sha256(files["SOURCE_FILES.json"]).hexdigest(),
                  launch_SHA256=sha256(launch).hexdigest(), replay_allowed=False, DCU=0)
    intent_sha = save(directory / "intent.json", intent)
    for name, raw in files.items():
        stash(directory / name, raw)
    stash(directory / "original-launch.json", launch)
    result = dict(intent, intent_SHA256=intent_sha, passed=False, slurm_requested=False, actual_GPU_steps=0)
    result["initial_owned_standby_steps"] = initial_standby
    try:
        host = subprocess.run([HOST_PY, "-I", "-S", "-B", str(directory / "preflight.py"),
                               "host", "--a100-launch", str(directory / "original-launch.json")],
                              capture_output=True, timeout=50)
        result["host_rc"] = host.returncode
        result["host_stdout_SHA256"] = stash(directory / "host.stdout.private", host.stdout)
        result["host_stderr_SHA256"] = stash(directory / "host.stderr.private", host.stderr)
        need(host.returncode == 0, "exact target asset check failed")
        host_document = object_output(host.stdout, request)
        result["asset_map_SHA256"] = save(directory / "asset-map.json", host_document["asset_map"])
        result["owned_standby_steps_at_dispatch"] = current_slot(owned_standby)
        need(not accounting(), "preflight stage appeared before dispatch")
        need(deadline - time.monotonic() > 135, "original CPU deadline too close")
        result["slurm_requested"] = True
        step = subprocess.run(worker_argv(directory, host_document),
                              capture_output=True, timeout=min(130, deadline - time.monotonic()))
        result["step_rc"] = step.returncode
        result["step_stdout_SHA256"] = stash(directory / "mounted.stdout.private", step.stdout)
        result["step_stderr_SHA256"] = stash(directory / "mounted.stderr.private", step.stderr)
        rows = accounting()
        terminal_cutoff = min(deadline, time.monotonic() + 10)
        while (time.monotonic() < terminal_cutoff
               and (not rows or any(row[2] in ("RUNNING", "COMPLETING", "PENDING") for row in rows))):
            time.sleep(0.5)
            rows = accounting()
        result["matching_accounting_rows"] = rows
        need(step.returncode == 0 and len(rows) == 1 and re.fullmatch(PARENT + r"\.[0-9]+", rows[0][0])
             and rows[0][2:4] == ["COMPLETED", "0:0"] and rows[0][5] == NODE
             and "gres/" not in rows[0][6], "actual zero-GRES compute step did not pass")
        mounted = object_output(step.stdout, request)
        plan = mounted["common_CLI_Plan"]
        config = plan["config"]
        need((plan["case"], plan["a100_launch_sha256"], plan["cards"], config["world_size"], config["sp"])
             == ("T09.dp2", intent["launch_SHA256"], 2, 2, 1)
             and config["checkpoint_steps"] == [10,20,25]
             and mounted["transformer_identity"][1:] == host_document["transformer_identity"][1:]
             and mounted["target_cache_SHA256"] == host_document["cache_sha256"]
             and mounted["optimizer_updates"] == mounted["gpu_steps_started"] == 0,
             "mounted common CLI recipe differs")
        result.update(passed=True, actual_CPU_step=rows[0][0], world_size=2, sp=1,
                      planned_rank_checkpoints=[10,20,25], framework_files=host_document["existing_framework_files_pinned"],
                      framework_files_rechecked_here=host_document["existing_framework_files_rechecked_here"],
                      private_framework_inventory_SHA256=host_document["private_framework_inventory_SHA256"],
                      target_cache_SHA256=host_document["cache_sha256"],
                      host_weight_identity=host_document["transformer_identity"],
                      mounted_weight_identity=mounted["transformer_identity"],
                      target_weight_extent=host_document["transformer_stat_bytes"])
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
        if type(error) is ValueError:
            result["failure"] = str(error)
    finally:
        digest = save(directory / "outcome.json", result)
    print(json.dumps(dict(result, outcome_SHA256=digest), sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(INPUT))
