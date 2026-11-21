"""Dispatch bounded, independently replaceable projection benchmarks."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from admission import fields, parent_gate
from qualified_sampler import node_map


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RUN = ROOT / "results/fleet-qwen-projection-sync-depth-v1"
HERE = Path(__file__).resolve().parent
REQUEST = "92b334901da041b3a15342d7cca88639"
PARENTS = ("1087656", "1103606")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def query(*args):
    raw = subprocess.check_output(args, timeout=10)
    if len(raw) > 4 * 1024**2:
        raise ValueError("scheduler data exceeds the fixed envelope")
    return raw.decode().splitlines()


def hosts(expression):
    return query("scontrol", "show", "hostnames", expression)


def save(path, val):
    with path.open("x", encoding="utf-8") as out:
        json.dump(val, out, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())


def allocation(job, nodes):
    rows = query("scontrol", "show", "job", job, "-o")
    if len(rows) != 1:
        raise ValueError("expected one canonical parent allocation")
    parent_gate(rows[0], job, os.getuid(), nodes, hosts)
    occupied = set()
    for line in query("scontrol", "show", "step", job, "-o"):
        row = fields(line)
        step = row["StepId"]
        if re.fullmatch(job + r"\.\d+", step):
            occupied.update(hosts(row["NodeList"]))
        elif step not in {job + ".batch", job + ".extern"}:
            raise ValueError("unrecognized parent step")
    return occupied


def account(labels):
    result = {}
    for row in query("sacct", "-n", "-P", "-j", ",".join(PARENTS), "--format=" + COLUMNS):
        values = row.split("|")
        if len(values) == 7 and values[1] in labels:
            if values[1] in result:
                raise ValueError("multiple Slurm steps share one planned label")
            result[values[1]] = values
    return result


def collect_case(root, card_count):
    reports = {}
    for node in card_count:
        src = root / node / "terminal.json"
        if not src.is_file() or src.is_symlink() or src.stat().st_size > 4096:
            continue
        row = json.loads(src.read_bytes())
        if row["node"] == node and row["planned_cards"] == 8:
            reports[node] = {k: row[k] for k in ("step", "qualified_cards")}
    return reports


def shard_nodes(selected, owners, occupied):
    """Keep each independently stoppable Slurm step inside one resource group."""
    by_group = defaultdict(list)
    for node in sorted(selected):
        parent, zone = owners[node]
        if node not in occupied[parent]:
            by_group[parent, zone].append(node)
    return {
        f"qwensync-{REQUEST}-{parent}-{zone}-{offset // 4:02d}": (parent, nodes[offset:offset + 4])
        for (parent, zone), nodes in sorted(by_group.items())
        for offset in range(0, len(nodes), 4)
    }


def main(plan_path):
    plan = json.loads(plan_path.read_bytes())
    if plan["request"] != REQUEST or not 0 <= time.time() - plan["epoch"] < 60:
        raise ValueError("wrong request or stale launch epoch")
    ledger = json.loads((HERE / "ledger.json").read_bytes())
    owners = node_map(ledger)
    selected = plan["PCI"]
    if not selected or len(selected) > 80 or any(node not in owners or len(set(pci)) != 8
                                                 for node, pci in selected.items()):
        raise ValueError("fixed card inventory differs")
    if plan["defer"] and set(selected).intersection(plan["defer"]):
        raise ValueError("protected nodes must stay out of the replacement queue")
    by_parent = {j: sorted(n for n, (job, _) in owners.items() if job == j) for j in PARENTS}
    occupied = {job: allocation(job, nodes) for job, nodes in by_parent.items()}
    groups = shard_nodes(selected, owners, occupied)
    if not groups or account(groups):
        raise ValueError("no new safe shard slots or fixed label already submitted")
    RUN.mkdir(mode=0o700, exist_ok=False)
    save(RUN / "intent.json", dict(plan=plan, groups=groups, raw_plan_SHA256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
    pci_file = RUN / "pci.json"
    save(pci_file, selected)
    logs = RUN / "nodes"
    logs.mkdir(mode=0o700)
    children = {}
    exits = {}
    image = ROOT / "containers/dtk26-sandbox-r1"
    apptainer = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
    # Each group has a distinct numbered Slurm step and can finish independently.
    for label, (parent, nodes) in groups.items():
        runtime = [apptainer, "exec", "--rocm", "--no-home", "--pwd", "/",
                   "--bind", str(RUN) + ":" + str(RUN),
                   "--bind", str(HERE) + ":" + str(HERE) + ":ro",
                   "--bind", "/opt/hyhal:/opt/hyhal:ro", str(image),
                   "/bin/bash", "--noprofile", "--norc", "-c",
                   'set -e; source /opt/dtk/env.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1; exec python3 -B -u "$@"',
                   "qwen-projection-sync-depth", str(HERE / "worker.py"), str(logs), str(plan["epoch"] + 210), str(pci_file)]
        command = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact",
                   "--nodes=" + str(len(nodes)), "--ntasks=" + str(len(nodes)), "--ntasks-per-node=1",
                   "--nodelist=" + ",".join(nodes), "--immediate=5", "--kill-on-bad-exit=0",
                   "--cpus-per-task=12", "--mem=24G", "--gres=dcu:8", "--time=00:03:30",
                   "--output=" + str(RUN / (label + "-%N.log")), *runtime]
        try:
            children[label] = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            exits[label] = type(exc).__name__
    save(RUN / "submitted.json", {label: child.pid for label, child in children.items()})
    for label, child in children.items():
        try:
            exits[label] = child.wait(timeout=max(0.1, plan["epoch"] + 240 - time.time()))
        except subprocess.TimeoutExpired:
            # A missing observation cannot establish terminal or permit another dispatch.
            exits[label] = "ACCOUNTING_REQUIRED"
    rows = account(groups)
    states = {label: rows[label][2].split()[0] if label in rows else "UNKNOWN" for label in groups}
    verified = collect_case(logs, selected)
    result = dict(request=REQUEST, selected_cards=8 * len(selected), submitted_shards=len(children),
                  grouped_nodes=len({n for _, nodes in groups.values() for n in nodes}),
                  provider="PyTorch BF16 torch.mm on HIP", synthetic_operands=True,
                  synchronization_depths=[8, 64],
                  planned_cards=8 * sum(len(group[1]) for group in groups.values()),
                  completed_node_reports=len(verified), qualified_operator_cards=sum(v["qualified_cards"] for v in verified.values()),
                  original_slurm=rows, original_exits=exits, known_terminal=all(s in TERMINAL for s in states.values()))
    save(RUN / "outcome.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["known_terminal"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    raise SystemExit(main(parser.parse_args().plan))
