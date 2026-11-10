"""Run I03 device admission in the leased Slurm host process before Apptainer."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from split_admission import pinned_bytes, require, snapshot, validate_before


ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014"
OLD = "/public/home/aclsylqidf/hygon-inference-20261013"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
SHARED = "/public/home/aclsylqidf/shared"


def invoke(before_raw, inside, probe, deadline, probe_sha, *, take=snapshot,
           spawn=subprocess.run, emit=print):
    require(not os.environ.get("LD_PRELOAD") and not os.environ.get("APPTAINER_CONTAINER")
            and not os.environ.get("SINGULARITY_CONTAINER") and not Path("/.singularity.d").exists(),
            "GPU admission must begin in the unpreloaded Slurm host process")
    require(130 <= float(deadline) - time.time() <= 7200, "invalid original deadline")
    try:
        leased = take(True)
        validate_before(json.loads(before_raw), leased)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        emit(json.dumps({"kind": "i03_host_gate_failure", "stage": "host_device_admission",
                         "error_type": type(error).__name__, "errno": getattr(error, "errno", None)}),
             flush=True)
        raise
    leased["before_sha256"] = hashlib.sha256(before_raw).hexdigest()
    leased["holder_scan_location"] = "slurm_host_before_container"
    emit(json.dumps(leased), flush=True)
    with socket.socket() as port:
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(("127.0.0.1", 43882))
    output = Path(ROOT) / "results/i03-deep-r454-sparse-http-v1-private"
    output.mkdir(mode=0o700, exist_ok=False)
    command = [
        APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
        "--bind", ROOT + ":/refactor:ro", "--bind", OLD + ":/old:ro",
        "--bind", SHARED + ":/shared:ro", "--bind", "/opt/hyhal:/opt/hyhal:ro",
        "--bind", str(output) + ":/pilot-output:rw",
        ROOT + "/containers/dtk26-sandbox-r1",
        "/bin/bash", "--noprofile", "--norc", "-c", inside.decode("utf-8"),
        "deep-r454-http-pilot", deadline, probe_sha,
    ]
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("APPTAINERENV_", "SINGULARITYENV_"))}
    return spawn(command, input=probe, env=environment, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before-sha256", "inside-sha256", "probe-sha256", "deadline-epoch"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    encoded = sys.stdin.buffer.read(262145)
    require(0 < len(encoded) <= 262144, "invalid admission bundle size")
    envelope = json.loads(encoded)
    before = pinned_bytes(envelope, "before", args.before_sha256)
    inside = pinned_bytes(envelope, "inside", args.inside_sha256)
    probe = base64.b64decode(envelope["probe"], validate=True)
    require(0 < len(probe) <= 131072 and hashlib.sha256(probe).hexdigest() == args.probe_sha256,
            "wrong service bundle")
    return invoke(before, inside, probe, args.deadline_epoch, args.probe_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
