"""Launch bounded per-node standby steps from a fresh independent admission."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PARENTS = ("1103606", "1194900")
PREFIX = "training80-dcu-standby"
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
NODE = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
SHELL = (
    "set -euo pipefail; source /opt/dtk/env.sh; "
    "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    "PYTHONPATH=/training-runtime/site HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
    "exec python3 -B -u \"$1\" \"$2\""
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=10).splitlines()


def pinned_fields(row):
    return dict(part.split("=", 1) for part in row.split() if "=" in part)


def live_inventory():
    all_nodes = {}
    active = set()
    for parent in PARENTS:
        [row] = query("scontrol", "show", "job", "-o", parent)
        fields = pinned_fields(row)
        nodes = query("scontrol", "show", "hostnames", fields["NodeList"])
        require(fields["JobId"] == parent and fields["JobState"] == "RUNNING"
                and fields["UserId"].endswith(f"({os.getuid()})")
                and len(set(nodes)) == len(nodes) == 40
                and "gres/dcu=320" in fields["AllocTRES"].split(",")
                and not set(nodes).intersection(all_nodes), "current owned parent inventory differs")
        all_nodes.update({node: parent for node in nodes})
        for line in query("squeue", "--steps", "-h", "-j", parent, "-o", "%i|%N"):
            handle, expression = line.split("|", 1)
            if re.fullmatch(parent + r"\.[0-9]+", handle):
                members = query("scontrol", "show", "hostnames", expression)
                require(members and set(members).issubset(nodes), "numbered step scope is unresolved")
                active.update(members)
    return all_nodes, active


def write(directory, name, value):
    with (directory / name).open("x", encoding="utf8") as output:
        output.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def node_seconds(node, seconds, request):
    if seconds == 120:
        return seconds
    seed = sha256(f"{request}:{node}".encode("ascii")).digest()
    return seconds - int.from_bytes(seed[:2], "big") % 301


def command(node, parent, label, work_root, node_dir, seconds):
    allowed = seconds + 300
    time_limit = f"{allowed // 3600:02d}:{allowed // 60 % 60:02d}:{allowed % 60:02d}"
    return ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--immediate=6",
            "--nodes=1", "--ntasks=1", "--nodelist=" + node, "--cpus-per-task=8",
            "--mem=16384M", "--gres=dcu:8", "--time=" + time_limit, "--kill-on-bad-exit=1",
            APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
            "--bind", f"{work_root}:{work_root}:ro", "--bind", f"{node_dir}:{node_dir}:rw",
            "--bind", f"{RESULTS}/training-t06-sdxl-runtime-v1:/training-runtime:ro",
            "--bind", "/opt/hyhal:/opt/hyhal:ro", str(ROOT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", SHELL, "standby",
            str(work_root / "worker.py"), str(node_dir)]


def spawn(root, node, row, seconds, request):
    directory = root / node
    directory.mkdir()
    label = f"{PREFIX}-{request[:12]}-{node}"
    seconds = node_seconds(node, seconds, request)
    write(directory, "expected.json", dict(node=node, parent=row["parent"], pci=row["pci"],
                                           seconds=seconds, label=label, request=request))
    with (directory / "slurm.stdout.private").open("xb") as out, (directory / "slurm.stderr.private").open("xb") as err:
        process = subprocess.Popen(command(node, row["parent"], label, root, directory, seconds), stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, start_new_session=True)
    stat = Path(f"/proc/{process.pid}/stat")
    try:
        startticks = int(stat.read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        startticks = None
    record = dict(node=node, parent=row["parent"], label=label, launcher_pid=process.pid, startticks=startticks,
                  worker_seconds=seconds)
    write(directory, "submit.json", record)
    return record


def run(data):
    os.umask(0o077)
    request = data["request"]
    publication = data["publication"]
    worker = data["worker"]
    claim = data["claim"]
    required = claim["gpu_per_node"]
    require(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch(r"[0-9a-f]{40}", publication)
            and (claim["pilot_seconds"], claim["production_min_seconds"], claim["production_seconds"], claim["Slurm_cap_seconds"])
            == (120, 10500, 10800, 11100)
            and data["seconds"] in (120, 10800) and required == 8 and claim["synthetic"] is True
            and len(data["nodes"]) <= (2 if data["seconds"] == 120 else 80)
            and data["nodes"] and sha256(worker).hexdigest() == claim["worker_SHA256"],
            "standby source, default budget or scope differs")
    require(re.fullmatch(r"[0-9a-f]{64}", data["admission_SHA256"]), "invalid admission binding")
    [*nodes] = data["nodes"].items()
    for name, row in nodes:
        require(NODE.fullmatch(name) and row["parent"] in PARENTS and len(set(row["pci"])) == required
                and all(PCI.fullmatch(item) for item in row["pci"]), "admitted node PCI or parent differs")
    root = RESULTS / f"{PREFIX}-{request}"
    with (RESULTS / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners, active = live_inventory()
        require(all(owners.get(name) == row["parent"] for name, row in nodes), "parent ownership drift")
        eligible = [(name, row) for name, row in nodes if name not in active]
        require(eligible, "no clear owned nodes at dispatch")
        root.mkdir(mode=0o700)
        write(root, "intent.json", dict(request=request, publication=publication, synthetic=True,
                                        admission_SHA256=data["admission_SHA256"], nodes=[name for name, _ in eligible],
                                        seconds=data["seconds"], epoch=time.time(), replay_allowed=False))
        with (root / "worker.py").open("xb") as stream:
            stream.write(worker)
        records = []
        failures = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            planned = [(name, pool.submit(spawn, root, name, row, data["seconds"], request))
                       for name, row in eligible]
            for name, work in planned:
                try:
                    records.append(work.result())
                except OSError as exc:
                    failures.append(dict(node=name, error=type(exc).__name__))
        result = dict(request=request, publication=publication, synthetic=True, submit_records=records,
                      launch_failures=failures, changed_before_launch=sorted(active.intersection(data["nodes"])),
                      card_steps_observed=0)
        write(root, "submission-receipt.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    run(INPUT)
