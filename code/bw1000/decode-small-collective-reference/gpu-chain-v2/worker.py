"""Run one node of a published raw AllReduce probe inside its bounded Slurm step."""
import argparse
import hashlib
import json
import os
import re
import socket
import types
from pathlib import Path

from driver import DEVICES, Native, run_node

CONTROL_SHA256 = "269cc1a5097e0264052d1c929a18ac0bbfe20ade0b44947455dd21bf9e6b804d"
CORES_PER_OWNER = 4
CHAIN_RUNTIME_SONAMES = {"libinfiniccl.so", "libinfinirt.so", "libcudart.so.12", "librccl.so.1"}


def read_build_receipt(raw, receipt_sha, native_sha):
    if hashlib.sha256(raw).hexdigest() != receipt_sha:
        raise ValueError("component CPU build receipt SHA256 mismatch")
    receipt = json.loads(raw)
    libs = receipt["required_runtime_libraries"]
    if (receipt["native_sha256"] != native_sha or receipt["gpu_executed"] is not False
            or receipt["python_dlopen"] is not False
            or not re.fullmatch(r"[0-9]+\.[0-9]+", receipt["actual_step"])
            or set(libs) != CHAIN_RUNTIME_SONAMES):
        raise ValueError("component CPU build does not bind this DSO and dependency set")
    return libs


from runtime_identity import verify_runtime_maps


def load_control(path):
    raw = path.read_bytes()
    if not __debug__ or hashlib.sha256(raw).hexdigest() != CONTROL_SHA256:
        raise ValueError("StageControl requires assertions and the pinned source")
    module = types.ModuleType("stage_control_r2386")
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module.StageControl


def run_worker(native, ranks, node, step, native_sha256, host, bind_cpu, timeout,
               control_type=None, head=None, port=None, nonce=None, *, deep_chain=False):
    nodes = ranks // DEVICES
    if ranks not in (8, 16) or node not in range(nodes) or (deep_chain and ranks != 8):
        raise ValueError("invalid TP geometry or node rank")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", step):
        raise ValueError("expected a numeric Slurm job.step")
    if not re.fullmatch(r"[a-f0-9]{64}", native_sha256):
        raise ValueError("expected the pinned Native SHA256")
    if not 1 <= timeout <= 240:
        raise ValueError("invalid synchronization timeout")
    control = None
    if nodes > 1:
        if control_type is None or not head or not nonce or not 1024 <= (port or 0) <= 65535:
            raise ValueError("TP16 requires a configured StageControl rendezvous")
        control = control_type(node, head, port, nonce, nodes, timeout)
    try:
        if control is None:
            identifier = native.bootstrap()
            def collect(_label, payload):
                return [payload]
        else:
            common = {"actual_step": step, "native_sha256": native_sha256, "TP": ranks}
            membership = control.collect("decode-probe-membership", dict(common, node=node, host=host))
            expected_nodes = list(range(nodes))
            if ([row["node"] for row in membership] != expected_nodes
                    or len({row["host"] for row in membership}) != nodes
                    or any(any(row[k] != v for k, v in common.items()) for row in membership)):
                raise ValueError("StageControl nodes do not describe one probe step")
            bootstrap = {"id": native.bootstrap().hex()} if node == 0 else {}
            leader, follower = control.collect("decode-probe-bootstrap", bootstrap)
            if set(leader) != {"id"} or follower:
                raise ValueError("unexpected communicator bootstrap sender")
            identifier = bytes.fromhex(leader["id"])
            collect = control.collect
        report = run_node(native, ranks, node, identifier, bind_cpu, collect, timeout, deep_chain=deep_chain)
    finally:
        if control is not None:
            control.close()

    return {"actual_step": step, "native_sha256": native_sha256, "host": host,
            "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", required=True, choices=(8, 16), type=int)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--stage-control", type=Path)
    parser.add_argument("--head")
    parser.add_argument("--port", type=int)
    parser.add_argument("--deep-chain", action="store_true")
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--build-receipt-sha256")
    parser.add_argument("--timeout", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    node = int(os.environ["SLURM_PROCID"])
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    if (node not in range(args.tp // DEVICES) or not re.fullmatch(r"[0-9]+\.[0-9]+", step)
            or not re.fullmatch(r"[a-f0-9]{64}", args.native_sha256)
            or not 1 <= args.timeout <= 240):
        parser.error("invalid Slurm step, Native SHA256 or timeout")
    if args.deep_chain and (args.tp != 8 or args.build_receipt is None
                            or not re.fullmatch(r"[a-f0-9]{64}", args.build_receipt_sha256 or "")):
        parser.error("--deep-chain requires TP8 and the pinned CPU build receipt")
    if args.tp == 16 and args.stage_control is None:
        parser.error("TP16 requires the pinned StageControl source")
    out = args.output_dir / f"decode-ccl-{step}-node{node}.json"
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < DEVICES * CORES_PER_OWNER:
        parser.error("expected at least 32 allowed logical CPUs")

    def bind_cpu(device):
        start = device * CORES_PER_OWNER
        os.sched_setaffinity(0, cpus[start:start + CORES_PER_OWNER])

    # The outer launcher owns deduplication and the hard deadline. Reserve this
    # path before loading the DSO so repeated invocations cannot overwrite it.
    with out.open("x", encoding="utf-8", newline="\n") as output:
        control_type = load_control(args.stage_control) if args.tp == 16 else None
        libraries = (read_build_receipt(args.build_receipt.read_bytes(), args.build_receipt_sha256,
                                        args.native_sha256) if args.deep_chain else None)
        native = Native(args.native, args.native_sha256)
        maps = Path("/proc/self/maps")
        before = verify_runtime_maps(libraries, maps.read_text()) if libraries is not None else None
        result = run_worker(
            native, args.tp, node, step,
            args.native_sha256, socket.gethostname(), bind_cpu, args.timeout, control_type,
            args.head, args.port, os.environ.get("DECODE_CCL_CONTROL_NONCE"),
            deep_chain=args.deep_chain,
        )
        if before is not None:
            after = verify_runtime_maps(libraries, maps.read_text())
            if after != before:
                raise ValueError("runtime map changed during the probe")
            result["CPU_build_receipt_SHA256"] = args.build_receipt_sha256
            result["runtime_maps_before_and_after"] = before
        result["NCCL_NET"] = os.environ.get("NCCL_NET")
        result["NCCL_NET_PLUGIN"] = os.environ.get("NCCL_NET_PLUGIN")
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if result["report"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
