"""Collect the live Slurm evidence expected by the pinned O01 HOST contract."""

from functools import lru_cache
import re
import subprocess
import time

import host_scope as spec


def command(*argv):
    result = subprocess.run(argv, capture_output=True, timeout=8, check=False)
    spec.require(result.returncode == 0 and len(result.stdout) <= 2 * 1024**2,
                 "SLURM_COMMAND_FAILED_OR_OVERSIZE_" + argv[0])
    return result.stdout.decode("utf8")


def single(text, identity):
    result = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            spec.require(key not in result, "DUPLICATE_SLURM_FIELD")
            result[key] = value
    spec.require(result.get(identity), "MISSING_SLURM_IDENTITY_" + identity)
    return result


def collect(run=command, clock=time.time):
    start = clock()

    @lru_cache(maxsize=80)
    def expand(expression):
        spec.require(re.fullmatch(r"[a-z0-9,\[\]-]+", expression), "SLURM_NODE_EXPRESSION")
        nodes = ([expression] if re.fullmatch(r"[a-z0-9-]+", expression)
                 else run("scontrol", "show", "hostnames", expression).splitlines())
        spec.require(nodes and len(nodes) <= 80 and len(nodes) == len(set(nodes))
                     and all(re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", node) for node in nodes),
                     "SLURM_HOSTNAME_EXPANSION")
        return nodes

    parent = single(run("scontrol", "-o", "show", "job", spec.PARENT), "JobId")
    names = expand(parent["NodeList"])
    node = single(run("scontrol", "-o", "show", "node", spec.NODE), "NodeName")
    spec.require(node["NodeName"] == spec.NODE and int(node["RealMemory"]) >= 24576
                 and int(node["AllocMem"]) >= 24576 and int(node["CPUAlloc"]) >= 12
                 and not set(node["State"].split("+")) & {"DOWN", "DRAIN", "FAIL", "INVAL", "NORESP"},
                 "TARGET_ADMISSION_RESOURCES")
    records = []
    step_stdout = run("scontrol", "-o", "show", "step", spec.PARENT)
    for line in step_stdout.splitlines():
        if not line.strip():
            continue
        row = single(line, "StepId")
        step_id = row["StepId"]
        if step_id in {spec.PARENT + ".extern", spec.PARENT + ".batch", spec.PARENT + ".interactive"}:
            continue
        spec.require(spec.STEP.fullmatch(step_id), "FOREIGN_OR_UNKNOWN_STEP_FORMAT")
        records.append({"StepId": step_id, "State": row["State"], "Name": row["Name"],
                        "nodes": expand(row["NodeList"]), "TRES": row["TRES"]})
    accounting = run("sacct", "-n", "-P", "-j", spec.PARENT, "-o", ",".join(spec.COLUMNS))
    account_rows = [line.split("|") for line in accounting.splitlines() if line]
    spec.require(account_rows and all(len(row) == len(spec.COLUMNS) for row in account_rows),
                 "NO_OR_MALFORMED_ACCOUNTING_ROWS")
    spec.require(any(row[0] == spec.OLD_STEP for row in account_rows), "ORIGINAL_STEP_MISSING")
    end = clock()
    spec.require(0 <= end - start <= 22, "SCHEDULER_WINDOW_TOO_WIDE")
    result = {"epoch": end, "window_start_epoch": start, "parent": parent, "parent_nodes": names,
              "numbered_steps_complete": True, "numbered_steps": records, "accounting_complete": True,
              "accounting_columns": spec.COLUMNS, "accounting_rows": account_rows,
              "target_configured": {"RealMemory_MiB": int(node["RealMemory"]),
                                    "AllocatedMemory_MiB": int(node["AllocMem"]),
                                    "AllocatedCPUs": int(node["CPUAlloc"]),
                                    "State": node["State"]}}
    return result
