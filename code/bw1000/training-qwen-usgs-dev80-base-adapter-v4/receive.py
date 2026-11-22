"""Launch the two isolated native Qwen development arms once."""
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-qwen-usgs-dev80-base-adapter-v4"
FAILED_GPU_SCOPE = "training-qwen-usgs-dev80-base-adapter-v3"
PREDECESSOR = "training-qwen-usgs-dev80-base-adapter-v1"
PARENT = "1103606"
RUN = RESULTS / LABEL
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
TRAINING = RESULTS / "training-qwen-usgs-table-sft2-hip-v1"
ADAPTER = TRAINING / "continuous/train/checkpoint-2"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
DATA = RESULTS / "training-domain-usgs-table-reading-v1/data/dev"


def digest(raw):
    return sha256(raw).hexdigest()


def record(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def query(*args):
    return subprocess.check_output(args, timeout=10, text=True).splitlines()


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def accounting(label):
    hits = [row.split("|") for row in query("sacct", "-nP", "-j", PARENT, "--format="+COLUMNS)
            if row.split("|")[1:2] == [label]]
    assert len(hits) <= 1
    return hits[0] if hits else None


def exact_placement(nodes):
    [row] = query("scontrol", "show", "job", PARENT, "-o")
    job = fields(row)
    assert (job["JobState"], job["NumNodes"]) == ("RUNNING", "40")
    assert job["UserId"].endswith(f"({os.getuid()})") and "gres/dcu=320" in job["AllocTRES"].split(",")
    assert nodes <= set(query("scontrol", "show", "hostnames", job["NodeList"]))
    steps = query("scontrol", "show", "step", PARENT, "-o")
    assert steps
    for line in steps:
        stage = fields(line)
        suffix = stage["StepId"].removeprefix(PARENT+".")
        if suffix in ("extern", "batch"):
            continue
        assert suffix.isdecimal()
        assert not nodes.intersection(query("scontrol", "show", "hostnames", stage["NodeList"]))


def snapshot():
    import socket
    nodes = Path("/sys/class/drm")
    pci = sorted((p/"device").resolve().name for p in nodes.glob("renderD*"))
    count = len(list(Path("/sys/class/kfd/kfd/proc").iterdir()))
    assert len(pci) == len(set(pci)) == 8 and count == 0
    print(json.dumps(dict(node=socket.gethostname().split(".")[0], pci=pci, KFD_holders=count,
                          boot_SHA256=digest(Path("/proc/sys/kernel/random/boot_id").read_bytes()))), flush=True)


def device_count_matches(row, native):
    actual = [part for part in row[6].split(",") if part.startswith("gres/")]
    return actual == (["gres/dcu=1"] if native else [])


def stage(arm, phase, node, command, maximum_s, deadline):
    label = LABEL + "-" + arm + "-" + phase
    assert accounting(label) is None
    limit = min(maximum_s, int(deadline-time.time())-20)
    assert limit >= 12
    minutes = max(1, limit // 60)
    native = phase == "gpu"
    args = ["srun", "--jobid="+PARENT, "--job-name="+label, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist="+node, "--cpus-per-task="+("8" if native else "2"),
            "--mem="+("96G" if native else "2G"), "--gres="+("dcu:1" if native else "none"),
            "--time="+str(minutes), *command]
    log = RUN/(arm+"-"+phase+".log")
    rc = None
    with log.open("xb") as out:
        child = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                rc = child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                rc = child.wait(timeout=3)
    row = None
    for _ in range(12):
        row = accounting(label)
        if row and row[2].split()[0] in TERMINAL:
            break
        if time.time() >= deadline-10:
            break
        time.sleep(1)
    passed = bool(row and rc == 0 and row[2:4] == ["COMPLETED", "0:0"] and row[5] == node
                  and device_count_matches(row, native))
    result = dict(passed=passed, row=row, client_rc=rc, log_SHA256=digest(log.read_bytes()))
    record(RUN/(arm+"-"+phase+"-accounting.json"), result)
    return result


def native(arm, deadline):
    scratch = RUN / (arm + "-apptainer-scratch")
    scratch.mkdir(mode=0o700)
    config = RUN / (arm + "-xdg-config")
    miopen = RUN / (arm + "-miopen-cache")
    for directory in (config, miopen):
        directory.mkdir(mode=0o700)
    settings = {
        "XDG_CONFIG_HOME": config,
        "MIOPEN_USER_DB_PATH": miopen,
        "MIOPEN_CUSTOM_CACHE_DIR": miopen,
        "MIOPEN_CACHE_DIR": miopen,
    }
    command = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--contain",
               "--no-home", "--workdir", str(scratch)]
    for path in (DATA, MODEL, ADAPTER):
        command.extend(("--bind", f"{path}:{path}:ro"))
    command.extend(("--bind", f"{RUN}:{RUN}:rw"))
    for path, target in (("training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
                         ("training-qwen-vision014-CPU-v2/vision", "/vision"),
                         ("training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers")):
        command.extend(("--bind", f"{RESULTS/path}:{target}:ro"))
    command.extend(("--bind", "/usr/local/hyhal:/opt/hyhal:ro"))
    worker = ["python3", "-B", "-u", str(RUN/"native_generation.py"), "--inputs", str(DATA),
              "--model", str(MODEL), "--output", str(RUN/(arm+"-outputs")), "--inventory", str(RUN/"DEV_INPUTS.json"),
              "--arm", arm, "--deadline", str(deadline)]
    if arm == "domain":
        worker.extend(("--adapter", str(ADAPTER)))
    runtime = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
               "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
               "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; "
               "export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site; exec ")
    return [*command, str(ROOT/"containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c",
            runtime.removesuffix("exec ") +
            "export " + " ".join(f"{name}={shlex.quote(str(path))}" for name, path in settings.items()) + "; exec " +
            shlex.join(worker)]


def arm_work(arm, node, deadline):
    probe = [PYTHON, "-B", str(RUN/"receive.py"), "snapshot"]
    pre = stage(arm, "pre", node, probe, 48, deadline)
    if not pre["passed"]:
        return dict(arm=arm, pre=pre, passed=False)
    gpu = stage(arm, "gpu", node, native(arm, deadline-110), 960, deadline-100)
    result = dict(arm=arm, pre=pre, gpu=gpu, passed=False)
    row = gpu["row"]
    if row and row[2].split()[0] in TERMINAL:
        post = stage(arm, "post", node, probe, 48, deadline)
        result["post"] = post
        if post["passed"]:
            a, b = [json.loads((RUN/(arm+"-"+phase+".log")).read_bytes()) for phase in ("pre", "post")]
            assert a == b and a["node"] == node
            result["post_clean"] = True
            result["passed"] = gpu["passed"]
    return result


def controller():
    intent = json.loads((RUN/"intent.json").read_bytes())
    claim = json.loads((RUN/"CLAIM.json").read_bytes())
    assert digest((TRAINING/"controller-outcome.json").read_bytes()) == claim["training_controller_SHA256"]
    for name, key in (("gpu", "training_step"), ("post", "training_cleanup")):
        original = accounting(TRAINING.name+"-"+name)
        assert original and original[0] == claim[key] and original[2:4] == ["COMPLETED","0:0"]
    for name, pin in claim["checkpoint_2"].items():
        path = ADAPTER/name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"]
        assert digest(path.read_bytes()) == pin["SHA256"]
    assert claim["placement"] == {"base":"f13r1n10", "domain":"f13r1n11"}
    exact_placement(set(claim["placement"].values()))
    from usgs_dev_input import bound_inputs
    bound_inputs(DATA, RUN/"DEV_INPUTS.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = {arm:pool.submit(arm_work, arm, node, intent["deadline"])
                for arm,node in claim["placement"].items()}
        outcome = {}
        for arm, run in runs.items():
            try:
                outcome[arm] = run.result()
            except Exception as error:
                outcome[arm] = dict(arm=arm, passed=False, error_type=type(error).__name__, error=str(error)[:180])
    record(RUN/"outcome.json", dict(request=claim["request"], results=outcome))
    return int(not all(x["passed"] for x in outcome.values()))


def submit(payload):
    os.umask(0o077)
    items = {name:base64.b64decode(raw,validate=True) for name,raw in payload["files"].items()}
    manifest = json.loads(items["SOURCE_FILES.json"])["files"]
    assert set(manifest)|{"SOURCE_FILES.json"} == set(items)
    for name, pin in manifest.items():
        assert Path(name).name == name and len(items[name]) == pin["bytes"] and digest(items[name]) == pin["SHA256"]
    claim = json.loads(items["CLAIM.json"])
    assert claim["label"] == LABEL and claim["GPU_GO"] is True and claim["training_GO"] is False
    assert claim["request"] != claim["predecessor_request"] and re.fullmatch("[0-9a-f]{32}", claim["request"])
    assert not RUN.exists() and not (RESULTS/PREDECESSOR).exists()
    for scope in (LABEL, PREDECESSOR):
        assert all(accounting(scope+"-"+arm+"-"+phase) is None
                   for arm in ("base","domain") for phase in ("pre","gpu","post"))
    previous = items["PREVIOUS_GPU_TERMINAL.json"]
    assert len(previous) == claim["previous_native_terminal"]["bytes"]
    assert digest(previous) == claim["previous_native_terminal"]["SHA256"]
    scratch = items["SCRATCH_PROOF.json"]
    assert len(scratch) == claim["scratch_mount_proof"]["bytes"] and digest(scratch) == claim["scratch_mount_proof"]["SHA256"]
    assert json.loads(scratch)["accounting"][:4] == ["1103606.1304", "training-qwen-dev80-apptainer-scratch-CPU-v1", "COMPLETED", "0:0"]
    for arm, expected in zip(("base", "domain"), claim["previous_native_failed_steps"]):
        result = accounting(FAILED_GPU_SCOPE+"-"+arm+"-gpu")
        post = accounting(FAILED_GPU_SCOPE+"-"+arm+"-post")
        assert result and result[0] == expected and result[2:4] == ["FAILED", "1:0"]
        assert post and post[2:4] == ["COMPLETED", "0:0"] and post[5] == claim["placement"][arm]
    exact_placement(set(claim["placement"].values()))
    RUN.mkdir(mode=0o700)
    for name,raw in items.items():
        with (RUN/name).open("xb") as f:
            f.write(raw)
    epoch=time.time()
    receipt=dict(request=claim["request"],publication=payload["publication"],epoch=epoch,deadline=epoch+1200)
    record(RUN/"intent.json",receipt)
    with (RUN/"controller.log").open("xb") as log:
        child=subprocess.Popen([PYTHON,"-B",str(RUN/"receive.py"),"controller"],stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt["controller_pid"] = child.pid
    proc = Path("/proc") / str(child.pid) / "stat"
    try:
        stat = proc.read_text()
        receipt["controller_startticks"] = int(stat[stat.rfind(")") + 2:].split()[19])
    except FileNotFoundError:
        receipt["controller_startticks"] = None
    record(RUN/"launch.json",receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["controller"]:
        raise SystemExit(controller())
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    else:
        submit(PAYLOAD)
