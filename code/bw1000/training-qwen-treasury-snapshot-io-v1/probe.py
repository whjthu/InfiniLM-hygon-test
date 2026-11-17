"""Emit one real HOST snapshot and an independent stderr marker; never import Torch or SDK."""
import os
import re
import sys
import json
import snapshot


if __name__ == "__main__":
    parent, node = sys.argv[1:]
    if os.environ.get("SLURM_STEP_GPUS") not in (None, "", "(null)"):
        raise ValueError("CPU snapshot gate assigned GPU")
    if "torch" in sys.modules:
        raise ValueError("unexpected Torch import")
    snapshot.NODE = node
    snapshot.STEP = re.compile(re.escape(parent) + r"\.[0-9]+")
    value = snapshot.snapshot(False)
    if value["kfd_holder_count"] != 0 or not value["full_fd_inventory_performed"]:
        raise ValueError("HOST is occupied or holder scan incomplete")
    print("snapshot-io-gate: retained independent stderr marker", file=sys.stderr, flush=True)
    print(json.dumps(value, allow_nan=False), flush=True)
