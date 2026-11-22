"""Submit the three fixed TP8 model arms under one bounded epoch."""

import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "deep-rope-real-model-tp8-diagnostic-v2"
ABORTED_SCOPE = "deep-rope-real-model-tp8-diagnostic-v1"
PARENT = "1103606"
NODES = ("f13r2n01", "f13r2n02", "f13r2n03")
ARMS = dict(zip(NODES, ("legacy", "h1-pack32", "h1-pack64")))
WALL_SECONDS = 4200
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
PREFLIGHT = PROJECT / "results/deep-rope-real-model-preflight-v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(path.read_bytes())


def transport_environment():
    return {key: value for key, value in os.environ.items()
            if key not in ("LD_PRELOAD", "LD_AUDIT", "BASH_ENV", "ENV", "PYTHONHOME", "PYTHONPATH")
            and not key.startswith(("APPTAINERENV_", "SINGULARITYENV_"))}


def args(node, name, cards, minutes, cpus, memory):
    gres = f"dcu:{cards}" if cards else "none"
    return ["srun", f"--jobid={PARENT}", f"--job-name={name}", "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", f"--nodelist={node}", "--cpu-bind=none",
            f"--cpus-per-task={cpus}", f"--mem={memory}M", f"--gres={gres}", f"--time={minutes}"]


def identity(module, node, leased):
    require(sys.platform == "linux" and re.fullmatch(r"f13r2n0[123]", node), "Invalid worker host")
    module.NODE = node
    module.STEP = re.compile(re.escape(PARENT) + r"\.\d+")
    return module.snapshot(leased)


def identity_at_host(scope, node, phase):
    import split_admission

    require(node in ARMS and phase in ("before", "gpu", "post"), "Unexpected worker scope")
    require(socket.gethostname().split(".")[0] == node, "Unexpected worker hostname")
    folder = scope / node
    current = identity(split_admission, node, phase == "gpu")
    require(current["kfd_holder_count"] == 0, "DCU process holders are present")
    save(folder / (phase + "-physical.json"), current)
    if phase != "gpu":
        return 0
    try:
        split_admission.validate_before(read(folder / "before-accounted.json"), current)
        epoch = read(scope / "intent.json")["deadline_epoch"]
        remaining = epoch - time.time()
        require(600 < remaining <= WALL_SECONDS, "Expired or changed original epoch")
        require(not any(os.environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT")), "Host runtime is preloaded")
        require(os.environ.get("SLURM_CPUS_PER_TASK") == "64", "Owner CPU allocation differs")
        arm = ARMS[node]
        old = PROJECT.parent / "hygon-inference-20261013"
        cmd = [
            "/usr/bin/timeout", "--kill-after=5s", str(int(remaining) - 20),
            "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm",
            "--no-home", "--pwd", "/", "--bind", f"{PROJECT}:/refactor:ro",
            "--bind", f"{old}:/old:ro", "--bind", f"{PROJECT.parent / 'shared'}:/shared:ro",
            "--bind", f"{scope / 'stage'}:/audit:ro", "--bind", f"{folder}:/audit-out:rw",
            "--bind", "/opt/hyhal:/opt/hyhal:ro", str(PROJECT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "/audit/inside.sh", arm, str(epoch),
        ]
        save(folder / "engine-intent.json", {"step": current["step"], "arm": arm, "deadline_epoch": epoch})
        child = subprocess.Popen(cmd, env=transport_environment())
        try:
            return child.wait()
        except BaseException:
            child.terminate()
            raise
    except BaseException as error:
        save(folder / "host-gate-failed.json", {"error_type": type(error).__name__,
                                                "message": str(error)[:180]})
        raise


def match_accounting(source, label):
    rows = [row for row in source.accounting()
            if row[1] == label and re.fullmatch(re.escape(PARENT) + r"\.\d+", row[0])]
    require(len(rows) <= 1, "Duplicate Slurm labels")
    return rows[0] if rows else None


def matching_resources(row, node, cards):
    if row is None or row[5] != node:
        return False
    fields = dict(field.split("=", 1) for field in row[6].split(",") if "=" in field)
    return (fields.get("node") == "1" and
            fields.get("cpu") == ("64" if cards else "2") and
            fields.get("gres/dcu", "0") == str(cards))


def arm_run(scope, node, deadline):
    import preflight_once as source
    from slurm_accounting import wait_terminal

    name = f"{LABEL}-{node}"
    result = {"arm": ARMS[node], "node": node}
    folder = scope / node
    script = scope / "stage/model_launch.py"

    def run(phase, allocated, minutes, cpus, memory):
        label = f"{name}-{phase}"
        if phase == "gpu":
            timeout = max(1, deadline - time.time() + 10)
        else:
            timeout = 72
        cmd = args(node, label, allocated, minutes, cpus, memory)
        cmd += [HOST_PYTHON, "-B", "-u", str(script), "worker", str(scope), node, phase]
        with (folder / (phase + "-slurm.log")).open("xb") as stream:
            try:
                client = subprocess.run(cmd, env=transport_environment(), stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                rc = client.returncode
            except subprocess.TimeoutExpired:
                rc = None
        actual = wait_terminal(lambda: match_accounting(source, label), time.monotonic() + 27,
                               interval=1, max_observations=14)
        actual["client_rc"] = rc
        save(folder / (phase + "-accounting.json"), actual)
        result[phase] = actual
        return actual["row"]

    try:
        before = run("before", 0, 1, 2, 8192)
        require(matching_resources(before, node, 0) and before[2:4] == ["COMPLETED", "0:0"],
                "Before snapshot did not pass Slurm accounting")
        physical = read(folder / "before-physical.json")
        require(physical["step"] == before[0] and physical["node"] == node,
                "Before snapshot and accounting disagree")
        physical.update(accounting_state=before[2], exit_code=before[3])
        save(folder / "before-accounted.json", physical)
        require(time.time() < deadline - WALL_SECONDS + 85, "Original dispatch admission window expired")
        save(folder / "gpu-submit-intent.json", {"node": node, "arm": ARMS[node], "label": f"{name}-gpu",
                                                 "deadline_epoch": deadline})
        gpu = run("gpu", 8, 70, 64, 224000)
        require(matching_resources(gpu, node, 8), "Actual GPU terminal or its allocation is unknown")
        post = run("post", 0, 1, 2, 8192)
        require(matching_resources(post, node, 0) and post[2:4] == ["COMPLETED", "0:0"],
                "Post snapshot did not pass Slurm accounting")
        after = read(folder / "post-physical.json")
        require(after["step"] == post[0] and after["kfd_holder_count"] == 0
                and after["boot_id"] == physical["boot_id"] and after["pci"] == physical["pci"],
                "Post snapshot changed identity or retained a DCU holder")
        result["post_clean"] = True
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)[:180]}
    save(folder / "terminal.json", result)
    return result


def validate_claim(claim):
    require(claim["label"] == LABEL and claim["parent"] == PARENT and tuple(claim["nodes"]) == NODES
            and claim["arm_by_node"] == ARMS and claim["wall_seconds"] == WALL_SECONDS
            and claim["model_GO"] is True and claim["GPU_GO"] is True
            and claim["replay_allowed"] is False and re.fullmatch(r"[0-9a-f]{32}", claim["request"])
            and (claim["TP"], claim["PP"], claim["rows"], claim["context"]) == (8, 1, 1, 32768)
            and (claim["prompt_tokens"], claim["generated_cap"], claim["free_floor_bytes"])
            == (8489, 4096, 16 * 1024**3), "Unexpected model scope")


def detached(scope):
    import preflight_once as source

    claim = read(scope / "stage/CLAIM.json")
    validate_claim(claim)
    stages = {"before", "gpu", "post"}
    labels = {f"{LABEL}-{node}-{phase}" for node in NODES for phase in stages}
    aborted = PROJECT / "results" / ABORTED_SCOPE
    require(pin((aborted / "controller.log").read_bytes()) == claim["aborted_v1_controller_log"]
            and not (aborted / "intent.json").exists()
            and not any((aborted / node / "gpu-submit-intent.json").exists() for node in NODES),
            "Prior pre-submission failure is uncertain")
    old_labels = {f"{ABORTED_SCOPE}-{node}-{phase}" for node in NODES for phase in stages}
    source.admit(labels | old_labels)
    epoch = time.time()
    deadline = epoch + WALL_SECONDS
    save(scope / "intent.json", {"request": claim["request"], "dispatch_epoch": epoch,
                                 "deadline_epoch": deadline, "nodes": NODES})
    with ThreadPoolExecutor(max_workers=len(NODES)) as pool:
        tasks = [pool.submit(arm_run, scope, node, deadline) for node in NODES]
        results = [task.result() for task in tasks]
    save(scope / "terminal.json", {"request": claim["request"], "accounting_columns": source.COLUMNS.split(","),
                                  "results": results})


def install(claim_sha):
    supplied = json.loads(sys.stdin.buffer.read(160 * 1024))
    blobs = {name: base64.b64decode(value, validate=True) for name, value in supplied.items()}
    claim_bytes = blobs.pop("CLAIM.json")
    require(pin(claim_bytes)["SHA256"] == claim_sha, "Model claim digest differs")
    claim = json.loads(claim_bytes)
    validate_claim(claim)
    require(set(blobs) == {"inside.sh", "model_owner.py", "load_probe.py", "slurm_accounting.py",
                           "split_admission.py", "model_launch.py", "preflight_once.py"},
            "Unexpected staged member set")
    for name, raw in blobs.items():
        expected = claim["materials"][name]
        require(pin(raw) == {key: expected[key] for key in ("bytes", "SHA256")}, "Invalid source: " + name)
    require(pin((PREFLIGHT / "terminal.json").read_bytes())["SHA256"] == claim["CPU_terminal_SHA256"],
            "Original three-node preflight receipt differs")
    scope = PROJECT / "results" / LABEL
    scope.mkdir(mode=0o700)
    stage = scope / "stage"
    stage.mkdir(mode=0o700)
    blobs["CLAIM.json"] = claim_bytes
    for name, raw in blobs.items():
        with (stage / name).open("xb") as stream:
            stream.write(raw)
        (stage / name).chmod(0o400)
    for node in NODES:
        (scope / node).mkdir(mode=0o700)
    # The exclusive installed scope and its original process are consumed once.
    with (scope / "controller.log").open("xb") as log:
        supervisor = subprocess.Popen(
            [HOST_PYTHON, "-B", "-u", str(stage / "model_launch.py"), "controller", str(scope)],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=transport_environment(),
        )
    stat = Path(f"/proc/{supervisor.pid}/stat").read_text()
    receipt = {"request": claim["request"], "controller_pid": supervisor.pid,
               "start_ticks": int(stat.rsplit(")", 1)[1].split()[19]), "claim_SHA256": claim_sha,
               "origin_login_identity_SHA256": sha256(socket.gethostname().encode()).hexdigest()}
    save(scope / "controller-start.json", receipt)
    return receipt


if __name__ == "__main__":
    mode, *rest = sys.argv[1:]
    if mode == "install":
        print(json.dumps(install(rest[0])))
    elif mode == "controller":
        detached(Path(rest[0]))
    elif mode == "worker":
        raise SystemExit(identity_at_host(Path(rest[0]), rest[1], rest[2]))
    else:
        raise SystemExit("Invalid control role")
