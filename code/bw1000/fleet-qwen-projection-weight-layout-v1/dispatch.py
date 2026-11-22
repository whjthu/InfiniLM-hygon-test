"""Dispatch bounded, independently replaceable projection benchmarks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import statistics
import time

from admission import fields, parent_gate
from layout_scope import LABEL, PARENTS, REQUEST, owners, shards


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RUN = ROOT / "results" / LABEL
HERE = Path(__file__).resolve().parent
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


def checked_trial(value):
    mode_a, mode_b = "checkpoint_transpose_view", "pretransposed_contiguous"
    if ((value["M"], value["K"], value["N"], value["sync_depth"], value["distinct_inputs"])
            != (184,4096,1024,64,2) or value["numeric"] is not True
            or value["arm_order"] != [mode_a,mode_b,mode_b,mode_a]
            or value["layout_strides"] != {mode_a:[1,4096],mode_b:[1024,1]}
            or value["packed_logical_bytes"] != 8388608 or not math.isfinite(value["pack_wall_s"])
            or value["pack_wall_s"] < 0 or value["extra_resident_allocated_bytes"] < 8388608
            or set(value["measurements"]) != {mode_a, mode_b}):
        raise ValueError("Numeric and layout trial proof differs")
    rates = {}
    for mode, rows in value["measurements"].items():
        if len(rows) != 2 or any(
                type(r["drains"]) is not int or r["drains"] < 1 or r["projections"] != 64*r["drains"]
                or r["rows"] != 184*r["projections"] or not math.isfinite(r["seconds"]) or r["seconds"] < 5
                for r in rows):
            raise ValueError("Incomplete assigned arm or useful-row evidence")
        rates[mode] = [r["projections"]/r["seconds"] for r in rows]
    pairs = [right/left for left,right in zip(rates[mode_a],reversed(rates[mode_b]),strict=True)]
    ratio = statistics.median(rates[mode_b])/statistics.median(rates[mode_a])
    return pairs, ratio


def collect_case(root, placement, accounting):
    reports = {}
    for node, label in placement.items():
        if label not in accounting or accounting[label][2:4] != ["COMPLETED","0:0"]:
            continue
        src = root / node / "terminal.json"
        if not src.is_file() or src.is_symlink() or src.stat().st_size > 4096:
            continue
        row = json.loads(src.read_bytes())
        if row["node"] != node or row["planned_cards"] != 8 or row["step"] != accounting[label][0]:
            continue
        cards = []
        for card in range(8):
            path = root/node/f"card-{card}.json"
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 16384:
                continue
            record = json.loads(path.read_bytes())
            if ((record["node"],record["step"],record["card"],record["error"]) != (node,row["step"],card,None)
                    or len(record["cases"]) != 1):
                continue
            try:
                value = record["cases"][0]
                pairs, ratio = checked_trial(value)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            cards.append(dict(card=card,pairs=pairs,ratio=ratio,pack_wall_s=value["pack_wall_s"],
                              extra_resident_allocated_bytes=value["extra_resident_allocated_bytes"],
                              amortized_projections=value["amortized_projections"]))
        reports[node] = dict(step=row["step"], qualified_cards=len(cards), cards=cards)
    return reports


def main(plan_path):
    plan = json.loads(plan_path.read_bytes())
    if plan["request"] != REQUEST or not 0 <= time.time() - plan["epoch"] < 60:
        raise ValueError("wrong request or stale launch epoch")
    ledger = json.loads((HERE / "ledger.json").read_bytes())
    roster = owners(ledger)
    selected = plan["PCI"]
    if set(selected).intersection(plan["defer"]):
        raise ValueError("protected nodes must stay out of the replacement queue")
    groups = shards(selected, roster)
    if plan["groups"] != groups or account(groups):
        raise ValueError("Original placement or unique labels differ")
    by_parent = {j: sorted(n for n, (job, _) in roster.items() if job == j) for j in PARENTS}
    occupied = {job: allocation(job, nodes) for job, nodes in by_parent.items()}
    deferred = {label:sorted(set(unit["nodes"]) & occupied[unit["parent"]]) for label, unit in groups.items()}
    deferred = {label:nodes for label,nodes in deferred.items() if nodes}
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
    for label, unit in groups.items():
        if label in deferred:
            continue
        parent, nodes = unit["parent"], unit["nodes"]
        runtime = [apptainer, "exec", "--rocm", "--no-home", "--pwd", "/",
                   "--bind", str(RUN) + ":" + str(RUN),
                   "--bind", str(HERE) + ":" + str(HERE) + ":ro",
                   "--bind", "/opt/hyhal:/opt/hyhal:ro", str(image),
                   "/bin/bash", "--noprofile", "--norc", "-c",
                   'set -e; source /opt/dtk/env.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1; exec python3 -B -u "$@"',
                   "qwen-projection-weight-layout", str(HERE / "worker.py"), str(logs), str(plan["epoch"] + 210), str(pci_file)]
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
    placement = {node:label for label,unit in groups.items() for node in unit["nodes"]}
    verified = collect_case(logs, placement, rows)
    cards = [item for evidence in verified.values() for item in evidence["cards"]]
    ratios = [row["ratio"] for row in cards]
    result = dict(request=REQUEST, selected_cards=8 * len(selected), submitted_shards=len(children),
                  grouped_nodes=len(placement), deferred_at_controller=deferred,
                  provider="PyTorch BF16 torch.mm on HIP", synthetic_operands=True,
                  synchronization_depth=64, planned_cards=8*len(placement), planned_pairs=16*len(placement),
                  planned_arms=32*len(placement), complete_pairs=2*len(cards), complete_arms=4*len(cards),
                  completed_node_reports=len(verified), qualified_operator_cards=len(cards),
                  paired_ratio_median=statistics.median(ratios) if ratios else None,
                  cards_faster=sum(v>1 for v in ratios),
                  pack_wall_median_s=statistics.median(v["pack_wall_s"] for v in cards) if cards else None,
                  cross_node_busy_qualified=False,
                  original_slurm=rows, original_exits=exits, known_terminal=all(s in TERMINAL for s in states.values()))
    save(RUN / "outcome.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["known_terminal"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    raise SystemExit(main(parser.parse_args().plan))
