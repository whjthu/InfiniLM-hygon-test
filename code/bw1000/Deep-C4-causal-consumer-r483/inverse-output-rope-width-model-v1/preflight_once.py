"""Install one immutable package and run the three claimed zero-GRES preflights."""

import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "deep-rope-real-model-preflight-v1"
PARENT = "1103606"
NODES = ("f13r2n01", "f13r2n02", "f13r2n03")
MATERIALS = {"model_owner.py", "inside.sh", "load_probe.py", "slurm_accounting.py"}
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def command(*arguments, timeout=8):
    return subprocess.run(arguments, text=True, capture_output=True, timeout=timeout)


def query(*arguments):
    result = command(*arguments)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} returned {result.returncode}")
    return result.stdout


def field(line, name):
    match = re.search(r"(?:^|\s)" + re.escape(name) + r"=(\S+)", line)
    return match.group(1) if match else ""


def accounting():
    return [line.split("|") for line in
            query("sacct", "-n", "-P", "-j", PARENT, "--format", COLUMNS).splitlines()
            if len(line.split("|")) == 7]


def hosts(expression):
    if not expression:
        raise RuntimeError("Slurm node list is absent")
    return query("scontrol", "show", "hostnames", expression).splitlines()


def admit(labels):
    parent = query("scontrol", "show", "job", PARENT, "-o")
    allocated = hosts(field(parent, "NodeList"))
    tres = field(parent, "AllocTRES").split(",")
    if (field(parent, "JobId") != PARENT or field(parent, "JobState") != "RUNNING"
            or "gres/dcu=320" not in tres
            or len(set(allocated)) != 40 or not set(NODES) <= set(allocated)
            or not field(parent, "UserId").endswith(f"({os.getuid()})")):
        raise RuntimeError("Parent ownership or 40-node/320-DCU allocation differs")
    steps = query("scontrol", "show", "step", PARENT, "--oneliner").splitlines()
    if not steps:
        raise RuntimeError("Active Slurm steps could not be determined")
    for line in steps:
        step = field(line, "StepId")
        if step in (PARENT + ".batch", PARENT + ".extern"):
            continue
        if not re.fullmatch(PARENT + r"\.\d+", step):
            raise RuntimeError("Unexpected active step entry")
        if set(hosts(field(line, "NodeList"))) & set(NODES):
            raise RuntimeError("A claimed node already has a numbered step")
    if any(row[1] in labels for row in accounting()):
        raise RuntimeError("A step in this claim already has accounting")


def emit(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, allow_nan=False, ensure_ascii=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def run_node(root, node, wait_terminal, start):
    stage = root / "stage"
    out = root / node
    label = f"{LABEL}-{node}"
    args = (
        "srun", f"--jobid={PARENT}", "--exact", "--immediate=5", "--nodes=1",
        "--ntasks=1", f"--nodelist={node}", "--cpus-per-task=2", "--mem=4096M", "--gres=none",
        "--time=2", f"--job-name={label}", "/public/software/apps/apptainer/1.3.4/bin/apptainer",
        "exec", "--contain", "--no-home", "--pwd", "/",
        "--bind", f"{PROJECT}:/refactor:ro",
        "--bind", f"{PROJECT.parent / 'shared'}:/shared:ro",
        "--bind", f"{stage}:/audit:ro", "--bind", f"{out}:/audit-out:rw",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(PROJECT / "containers/dtk26-sandbox-r1"),
        "/bin/bash", "--noprofile", "--norc", "/audit/inside.sh",
    )
    with (out / "client.log").open("xb") as log:
        try:
            client = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(1, start + 124 - time.monotonic()))
            client_rc = client.returncode
        except subprocess.TimeoutExpired:
            client_rc = None

    def read():
        matches = [row for row in accounting()
                   if row[1] == label and re.fullmatch(PARENT + r"\.\d+", row[0])]
        if len(matches) > 1:
            raise RuntimeError("Duplicate preflight accounting")
        return matches[0] if matches else None

    observed = wait_terminal(read, start + 142, interval=1, max_observations=8)
    phase = out / "phase.jsonl"
    raw = phase.read_bytes() if phase.is_file() else b""
    if len(raw) > 8192:
        raise RuntimeError("Unexpected preflight phase log extent")
    records = [json.loads(line) for line in raw.splitlines()]
    result = {"node": node, "client_rc": client_rc, "accounting": observed, "phase": records,
              "phase_SHA256": sha256(raw).hexdigest() if raw else None}
    emit(out / "terminal.json", result)
    return result


def main():
    supplied = json.loads(sys.stdin.buffer.read(128 * 1024))
    if set(supplied) != MATERIALS | {"CLAIM.json"}:
        raise ValueError("Unexpected file set")
    data = {name: base64.b64decode(value, validate=True) for name, value in supplied.items()}
    claim = json.loads(data["CLAIM.json"])
    if (sha256(data["CLAIM.json"]).hexdigest() != sys.argv[1] or claim["label"] != LABEL
            or claim["parent"] != PARENT or tuple(claim["nodes"]) != NODES
            or claim["CPU_GO"] is not True or claim["GPU_GO"] is not False
            or claim["model_GO"] is not False or claim["planned_CPU_steps"] != 3
            or claim["cpus_per_step"] != 2 or claim["cpu_memory_mib"] != 4096
            or claim["cpu_slurm_seconds"] != 120):
        raise ValueError("Unexpected preflight claim")
    for name in MATERIALS:
        pin = claim["materials"][name]
        if (pin["bytes"], pin["SHA256"]) != (len(data[name]), sha256(data[name]).hexdigest()):
            raise ValueError("Package member differs: " + name)
    admit({f"{LABEL}-{node}" for node in NODES})
    root = PROJECT / "results" / LABEL
    root.mkdir(mode=0o700)
    stage = root / "stage"
    stage.mkdir(mode=0o700)
    for name, payload in data.items():
        path = stage / name
        with path.open("xb") as handle:
            handle.write(payload)
        path.chmod(0o400)
    for node in NODES:
        (root / node).mkdir(mode=0o700)
    emit(root / "intent.json", {"request": claim["request"], "nodes": NODES,
                               "claim_SHA256": sys.argv[1], "placed_before_submit": True})
    sys.path.insert(0, str(stage))
    from slurm_accounting import wait_terminal

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(run_node, root, node, wait_terminal, start) for node in NODES]
        results = [task.result() for task in tasks]
    terminal = {"request": claim["request"], "accounting_columns": COLUMNS.split(","),
                "rows": results}
    emit(root / "terminal.json", terminal)
    print(json.dumps(terminal, allow_nan=False, ensure_ascii=True))


if __name__ == "__main__":
    main()
