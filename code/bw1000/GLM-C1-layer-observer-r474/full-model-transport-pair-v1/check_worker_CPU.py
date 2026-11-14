"""Exercise the paired launch identities and actual two-leader framing on CPU."""
import hashlib
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import control
import shared
from native_arm import ARMS


def rejected(callback):
    try:
        callback()
    except (ValueError, AssertionError):
        return
    raise AssertionError("accepted mismatched evidence")


def main():
    run = "0123456789abcdef" * 2
    baseline, out_a = shared.namespace(run, "socket")
    candidate, out_b = shared.namespace(run, "shca")
    assert baseline != candidate and out_a != out_b
    rejected(lambda: shared.namespace("../other", "socket"))
    rejected(lambda: shared.namespace(run, "wrong"))
    epoch = time.time()
    for arm, out in (("socket", out_a), ("shca", out_b)):
        manifest = dict(schema="glm474-transport-worker-v1", run_id=run, arm=arm, epoch=epoch,
                        parent=shared.PARENT, nodes=shared.NODES, output=str(out),
                        Native_SHA256=ARMS[arm][1], source_manifest_SHA256="a" * 64,
                        claim_SHA256="b" * 64, transport_key_SHA256="c" * 64, GPU_GO=True, Model_execution_authorized=True,
                        Model_qualified=False)
        shared.verify_manifest(manifest, run, arm, epoch)
        other = "socket" if arm == "shca" else "shca"
        rejected(lambda: shared.verify_manifest(manifest, run, other, epoch))
        for key, value in (("GPU_GO", False), ("Model_execution_authorized", 1),
                           ("parent", "910371"), ("transport_key_SHA256", "bad"), ("Native_SHA256", "0" * 64)):
            invalid = dict(manifest, **{key: value})
            rejected(lambda: shared.verify_manifest(invalid, run, arm, epoch))
    assert ARMS["socket"] == ARMS["shca"]
    source = shared.D / "SOURCE_MANIFEST.json"
    with patch.dict(os.environ, {"SLURM_JOB_ID": shared.PARENT, "SLURM_STEP_ID": "999",
                                 "GLM474_PAIR_ARM": "socket", "GLM470_GPU_MF": "a" * 64}):
        with patch.object(control, "LABEL", baseline):
            identity = control.identity(epoch)
        assert identity["arm"] == "socket" and identity["step"] == "1103606.999"
        assert identity["source_manifest_SHA256"] == shared.H(source)
        assert identity["Native_SHA256"] == ARMS["socket"][1]
    with patch.dict(os.environ, {"SLURM_JOB_ID": "910371", "SLURM_STEP_ID": "1"}):
        rejected(shared.step_id)
    rows = [[dict(identity=identity, phase="created", rank=node * 8 + dev,
                  node=shared.NODES[node], device=dev, PID=1 + node * 8 + dev,
                  startticks="17") for dev in range(8)] for node in (0, 1)]
    left, right = socket.socketpair()
    with left, right, ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(control.exchange, left, 0, rows[0], "created", identity, epoch)
        b = pool.submit(control.exchange, right, 1, rows[1], "created", identity, epoch)
        assert a.result(timeout=3) == b.result(timeout=3)
    wrong = deepcopy(rows[0])
    wrong[0]["identity"]["arm"] = "shca"
    rejected(lambda: control.validate(wrong, 0, "created", identity))
    inp = json.loads((shared.D / "INPUT_BINDING.json").read_bytes())
    trial = json.loads((shared.D / "TRIAL.json").read_bytes())
    pins = inp["private_plan"]["requests"]
    for pin in pins:
        ids = [1] * pin["prompt_tokens"]
        pin["prompt_IDs_SHA256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    plan = dict(planned=32, requests=[
        dict(pin, body={"temperature": 0.0}, prompt_token_ids=[1] * pin["prompt_tokens"])
        for pin in pins
    ])
    shared.verify_inputs(inp, trial, plan)
    plan["requests"][0]["prompt_token_ids"][0] = 2
    rejected(lambda: shared.verify_inputs(inp, trial, plan))
    plan["requests"][0]["prompt_token_ids"][0] = 1
    trial["memory_floor_bytes"] -= 1
    rejected(lambda: shared.verify_inputs(inp, trial, plan))
    print("PASS: two arm namespaces, 16-owner leader exchange, 16 rejection cases, frozen input checks")


if __name__ == "__main__":
    main()
