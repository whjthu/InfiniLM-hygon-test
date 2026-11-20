"""Prove both sides of the exact two-node pilot before fleet expansion."""
import json
import os
from pathlib import Path
import socket
import sys

from worker import node_run


def main(root, deadline):
    spec = json.loads((root / "pilot-plan.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    if node not in spec["nodes"] or parent != spec["parent"] or not step.isdecimal():
        raise ValueError("EXACT_PILOT_PLACEMENT")
    identity = dict(node=node, step=parent+"."+step, launch_epoch=None, cohort=spec["nodes"])
    work = root / "pilot-workers" / node
    return node_run(work, identity, deadline)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2])))
