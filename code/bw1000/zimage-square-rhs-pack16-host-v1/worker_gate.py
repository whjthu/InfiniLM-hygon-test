"""Check the exact compute-node step before starting the frozen Pack16 container."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess


PARENT = "1103606"
LABEL = "O01-zimage-square-rhs-pack16-v1"
REQUEST = "a31421758fc74d319923dbce7045cc12"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"


def require(value, gate):
    if not value:
        raise ValueError(gate)


def exact_step(scope, env, node, query):
    step_number = env.get("SLURM_STEP_ID", "")
    require(scope["parent"] == env.get("SLURM_JOB_ID") == PARENT and re.fullmatch(r"[0-9]+", step_number)
            and node == scope["node"] != "f12r2n11" and env.get("SLURM_CPUS_PER_TASK") == "4",
            "PRE_TORCH_NUMERIC_STEP")
    step = PARENT + "." + step_number
    text = query(["scontrol", "show", "step", step, "--oneliner"])
    rows = [dict(piece.split("=", 1) for piece in line.split() if "=" in piece)
            for line in text.splitlines() if line.strip()]
    require(len(rows) == 1, "PRE_TORCH_STEP_NOT_UNIQUE")
    row = rows[0]
    tres = row.get("TRES", "").split(",")
    require(row.get("StepId") == step and row.get("Name") == scope["label"] == LABEL
            and row.get("NodeList") == node and row.get("State") == "RUNNING"
            and len(tres) == len(set(tres)) and set(tres) in (
                {"cpu=4", "gres/dcu=1", "mem=8G", "node=1"},
                {"cpu=4", "gres/dcu=1", "mem=8192M", "node=1"}), "PRE_TORCH_EXACT_TRES")
    return step


def query(argv):
    data = subprocess.check_output(argv, timeout=4)
    require(0 < len(data) <= 16384, "PRE_TORCH_ACCOUNTING_SIZE")
    return data.decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command
    require(argv[:3] == ["--", APPTAINER, "exec"], "PRE_TORCH_CONTAINER_ENTRY")
    require(args.scope.resolve(strict=True) == args.scope.absolute(), "PRE_TORCH_SCOPE_PATH")
    with args.scope.open("rb") as stream:
        raw = stream.read(16385)
    require(0 < len(raw) <= 16384 and sha256(raw).hexdigest() == args.scope_sha256, "PRE_TORCH_SCOPE_PIN")
    scope = json.loads(raw)
    require(scope["request"] == REQUEST and scope["root_admitted"] is True
            and scope["gfx"] == "gfx936", "PRE_TORCH_SCOPE")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == scope["boot_id"]
            and len(os.sched_getaffinity(0)) == 4, "PRE_TORCH_BOOT_AND_AFFINITY")
    step = exact_step(scope, os.environ, socket.gethostname().split(".")[0], query)
    print(json.dumps({"stage": "PRE_TORCH_EXACT_STEP", "actual_step": step, "dcu": 1, "cpus": 4,
                      "memory_MiB": 8192}), flush=True)
    # The frozen worker independently checks single-device visibility, gfx and physical PCI.
    os.execv(APPTAINER, argv[1:])


if __name__ == "__main__":
    main()
