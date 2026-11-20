"""Verify the actual container, shared input pins and arm providers without device API calls."""
import json
import os
from pathlib import Path
import socket
import sys
import time

import shared
import native_arm
from verify_transport import INFINICCL_SHA256, RCCL_SHA256, PLUGIN_SHA256


def main(output):
    start = time.monotonic()
    rank = int(os.environ["SLURM_PROCID"])
    node = socket.gethostname().split(".")[0]
    if (not 0 <= rank < len(shared.NODES) or node != shared.NODES[rank]
            or [socket.gethostbyname(n) for n in shared.NODES] != shared.IPS
            or any(os.environ.get(k) != "-1" for k in (
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))):
        raise ValueError("preflight node identity or CPU device mask differs")
    pins = {}
    for name in ("SOURCE_MANIFEST.json", "HOST_MANIFEST.json"):
        path = shared.D / name
        pins[name] = shared.H(path)
        entries = json.loads(path.read_bytes())["files_SHA256"]
        if any(Path(n).name != n or shared.H(shared.D / n) != sha for n, sha in entries.items()):
            raise ValueError("installed source bytes differ")
    inp = json.loads((shared.D / "INPUT_BINDING.json").read_bytes())
    private = inp["private_plan"]
    path = shared.R / Path(private["path"]).relative_to(shared.HOST)
    if path.stat().st_size != private["bytes"]:
        raise ValueError("private plan length changed")
    plan = shared.load_pinned(path, private["SHA256"])
    shared.verify_inputs(inp, json.loads((shared.D / "TRIAL.json").read_bytes()), plan)
    model = Path(inp["model_path"])
    fixed = {
        model / "config.json": inp["model_config_SHA256"],
        model / "model.safetensors.index.json": inp["model_index_SHA256"],
        Path(inp["tokenizer"]["path"]): inp["tokenizer"]["SHA256"],
        Path("/refactor/install/hygon-glm-5f91ad7-r3672/lib/libinfiniccl.so"): INFINICCL_SHA256,
        Path("/opt/dtk/lib/librccl.so.1.0"): RCCL_SHA256,
        Path("/candidate-opt/rccl-shca-net/lib/librccl-net-shca.so.0.0.0"): PLUGIN_SHA256,
    }
    if any(shared.H(path) != pin for path, pin in fixed.items()):
        raise ValueError("actual model metadata or transport provider differs")
    library, digest = native_arm.verified_library("socket")
    if native_arm.ARMS["shca"] != native_arm.ARMS["socket"]:
        raise ValueError("transport arms must share the same Native library")
    shared.save(Path(output) / (node + "-container.json"),
                dict(passed=True, step=shared.step_id(), node=node, rank=rank,
                     Native_SHA256=digest, manifest_pins=pins, cards=0,
                     weight_tensor_payload_bytes_read=0, seconds=time.monotonic()-start))


if __name__ == "__main__":
    main(sys.argv[1])
