"""Immutable placement and one-epoch contract for the new resident pair."""

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
HOST = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
CONTAINER = Path("/refactor")
ID = "2bf7ebc97ef74cb9a8d185f10e3d762a"
LABEL = "GLM474-prefill512-resident-pair-GPU-v1"
PARENT = "1103606"
NODES = ["f13r1n08", "f13r1n09"]
PORTS = [43943, 43944]
PHASES = ("prefill_pair_created", "prefill_pair_loaded", "prefill_pair_ready_off",
          "prefill_pair_complete_off", "prefill_pair_ready_on", "prefill_pair_complete_on",
          "prefill_pair_reset_complete", "prefill_pair_closed")
LOAD = 10800
CONSTRUCT = 1800
WORK = 12120
INSIDE = 12300
GPU = 12480
KEEP = 12900
WHOLE = 13200
PAIR_MANIFEST = "fe7dde542b2d9626f1c9d69d4ac1ef6f0f7f77d1f1b95b1098ae330abda37048"
STAGED = {
    "ready_socket_parent.py": "613c8fea553456dfc7416a6fc96d4b0b294b0b1df5b81593ffe6ee697b2176cd",
    "slurm_accounting.py": "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    "snapshot.py": "9516fae47f438d72cc7528dfc8cc634f3a96f77dfbf58b8cf64a4e87c348b904",
    "ENV.sh": "aa8d9619534a2395492d62dd7a5c6aa828539b28a389d00b98a389f80ab39d67",
    "checkpoint.py": "89fff1fb549b411f41c55c43570937765ff22ccf327c31c4023106f5af196ae7",
    "INPUT_BINDING.json": "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e",
    "typed_binding.py": "7b7002acdb96b172acc5dfb4d1bfec33f3c8910deefdb7b5aae40511c6f09768",
    "1071.json": "f900ed78353182039c737adf4739359ba8f4915bf51472af50df9f7131b5de1e",
}


def need(condition, why):
    if not condition:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, digest, limit=2**20):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and re.fullmatch(r"[0-9a-f]{64}", digest) and sha(raw) == digest,
         "unexpected bytes: " + str(path))
    return raw


def verify_stage(stage, source_sha):
    manifest = json.loads(pinned(stage / "SOURCE_FILES.json", source_sha))
    files = {"contract.py", "node.py", "worker.py", "host.py", "container.sh", "check_local.py", "README.zh-CN.md"}
    need(set(manifest["files"]) == files, "unexpected launcher package member set")
    for name, digest in manifest["files"].items():
        pinned(stage / name, digest)
    for name, digest in STAGED.items():
        pinned(stage / name, digest)
    pair = stage / "pair"
    members = json.loads(pinned(pair / "SOURCE_FILES.json", PAIR_MANIFEST))["files"]
    need(set(members) == {"README.zh-CN.md", "SCOPE.json", "aggregate.py", "bindings.py", "check_local.py", "run_pair.py"},
         "unexpected caller package members")
    for name, pin in members.items():
        need(len(pinned(pair / name, pin["SHA256"])) == pin["bytes"], "caller exact member size differs")


def read_json(path, limit=2**20):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit, "invalid bounded JSON size")
    return json.loads(raw)


def put(path, value, *, replace=False):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    target = path.with_name(f"{path.name}.tmp-{os.getpid()}") if replace else path
    with target.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    if replace:
        os.replace(target, path)


def validate_claim(q, source_sha):
    expected = dict(request=ID, label=LABEL, parent=PARENT, nodes=NODES, source_SHA256=source_sha,
                    caller_SHA256=PAIR_MANIFEST, ports=PORTS, DCU=16, GPU_CPU_per_node=64,
                    GPU_memory_MiB_per_node=224000, keeper_CPU=2, keeper_memory_MiB=8192,
                    construct_seconds=CONSTRUCT, load_seconds=LOAD, work_seconds=WORK, inside_seconds=INSIDE,
                    GPU_seconds=GPU, keeper_seconds=KEEP, whole_seconds=WHOLE, max_GPU_steps=1,
                    GPU_GO=True, Model_GO=False)
    need(re.fullmatch(r"[0-9a-f]{32}", q["request"]) and all(type(q.get(k)) is type(v) and q[k] == v for k,v in expected.items()),
         "claim placement, resource, deadline or source differs")
    need(q["keeper_node"] == "f12r2n13", "keeper placement lacks this scope's prior static identity")
    need(len(q["data_IPs"]) == len(set(q["data_IPs"])) == 2
         and all(str(ipaddress.IPv4Address(ip)) == ip and not ipaddress.IPv4Address(ip).is_loopback
                 and not ipaddress.IPv4Address(ip).is_unspecified for ip in q["data_IPs"]),
         "expected two canonical observed non-loopback data IPv4 addresses")
    need(re.fullmatch(r"[0-9a-f]{64}", q["root_admission_SHA256"]), "missing upstream fresh admission binding")
    return q


def budget(epoch, cap, now=None):
    current = time.time() if now is None else now
    need(type(epoch) in (int,float) and math.isfinite(epoch) and math.isfinite(current)
         and 0 < epoch <= current < epoch + cap, "original absolute deadline exhausted or invalid")
    return epoch + cap - current


def step_identity():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    need(re.fullmatch(PARENT+r"\.\d+", step), "wrong actual parent/step")
    return step


def proc_start(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19])
