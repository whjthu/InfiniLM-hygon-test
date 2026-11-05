"""Private two-stage GLM block comparison entry. Verification is the default."""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import socket
import time

from .checkpoint_loader import bind_library
from .cli import prepare_public, validate_deployment_paths, verify_package
from .diagnostic_worker import configure_process, process_owner
from .protocol import Run
from .freeze_continuation_inputs import freeze, PARENT
from .prefix_native_guard import NATIVE_SHA256, RECEIPT_SHA, verify, verify_config
from .resident_continuation import PLAN_SHA256
from .topology_r404 import validated_nodes_r444


def read_pinned(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("GLM comparison input hash differs")
    return json.loads(raw)


def validate_config(config):
    validate_deployment_paths(config)
    verify_config(config)
    validated_nodes_r444(config["nodes"])
    if (
        config["backend"] != "qualified-native-glm-nov16"
        or config["Native_SHA256"] != NATIVE_SHA256
        or config["model_type"] != "glm_moe_dsa"
        or tuple(config[key] for key in ("TP", "PP", "rows", "context", "vocab_size"))
        != (8, 2, 1, 32768, 154880)
    ):
        raise ValueError("GLM comparison requires the original Native and geometry")
    native, service = (config[key] for key in ("Native_seconds", "service_seconds"))
    if type(native) is not int or type(service) is not int or service <= 120 or native - service < 120:
        raise ValueError("comparison must reserve bounded Native cleanup time")
    plan = config["block_compare_private_plan"]
    if type(plan) is not str or "\\" in plan or not PurePosixPath(plan).is_absolute():
        raise ValueError("comparison plan must be an exact POSIX path")


def require_claim(claim, config):
    expected = {
        "GPU_GO": True, "scope": "GLM952 staged-prefix continuation", "Native_CPU_receipt_SHA256": RECEIPT_SHA,
        "parent_job": config["actual_step"].split(".")[0], "run": config["run"],
        "source_manifest_SHA256": config["source_manifest_SHA256"],
        "provider_config_SHA256": config["provider_config_SHA256"],
        "private_plan_SHA256": PLAN_SHA256, "Native_SHA256": NATIVE_SHA256,
        "nodes": config["nodes"], "cards": 16, "cases": 3, "forwards_per_owner": 55,
        "Native_seconds": config["Native_seconds"], "service_seconds": config["service_seconds"],
    }
    if any(claim.get(key) != value for key, value in expected.items()):
        raise ValueError("no matching published-scope comparison claim")
    if config.get("prefix_continuation_execution_authorized") is not True:
        raise ValueError("private source configuration cannot execute")


def _stop_owned(processes, grace):
    for process in processes:
        if process.is_alive():
            process.terminate()
    end = time.monotonic() + grace
    for process in processes:
        process.join(timeout=max(0, end - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    end = time.monotonic() + 1
    for process in processes:
        process.join(timeout=max(0, end - time.monotonic()))
    return [p.name for p in processes if p.is_alive()]


def run_stage(config):
    run = Run(config)
    stage = config["stage"]
    processes = []
    # An existing stage intent belongs to its original process group.
    run.write(f"block-stage-{stage}.json", dict(stage=stage, PID=os.getpid()))
    try:
        run.deadline_check("before private GLM stage", 120)
        configure_process(config)
        prepare_public(config)
        if stage == 0:
            binding = bind_library(config["Native_library"], config["Native_SHA256"])
            ids = dict(world=binding.unique(), tp=[binding.unique() for _ in range(2)],
                       control=binding.unique())
            run.write("COMMUNICATORS.json", dict(payload=ids))
        run.wait("COMMUNICATORS.json")
        spawn = multiprocessing.get_context("spawn")
        for rank in range(8):
            run.deadline_check("before private GLM owner spawn")
            child = spawn.Process(target=process_owner, args=(rank, config),
                                  name=f"GLM-block-{stage}-{rank}")
            try:
                child.start()
            finally:
                if child.pid is not None:
                    processes.append(child)
        run.write(f"block-actors-{stage}.json", dict(stage=stage, PIDs=[p.pid for p in processes]))
        while True:
            run.raise_if_failed()
            if run.remaining() <= 0:
                raise TimeoutError("original GLM Native deadline expired")
            if any(p.exitcode not in (0, None) for p in processes):
                raise RuntimeError("a private GLM owner exited unsuccessfully")
            if all(p.exitcode == 0 for p in processes):
                break
            time.sleep(.2)
        for child in processes:
            child.join(timeout=0)
        for node_stage in range(2):
            for rank in range(8):
                done = run.wait(f"closed-{node_stage}-{rank}.json", minimum=0)
                if (done["stage"], done["rank"]) != (node_stage, rank) or not (
                    done["normal"] and done["post_close_Torch_liveness"]
                ):
                    raise RuntimeError("missing normal original owner shutdown")
        run.raise_if_failed()
        run.write(f"block-stage-terminal-{stage}.json", dict(
            stage=stage, normal=True, local_processes_exited=8, all16_closed=True,
        ))
    except BaseException as error:
        try:
            run.fail(error, stage=stage)
        except Exception as publication_error:
            error.add_note(f"Private stage failure publication: {publication_error!r}")
        live = _stop_owned(processes, min(10, max(0, run.remaining())))
        if live:
            error.add_note("Owned private GLM processes not yet reaped: " + ", ".join(live))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--claim-sha")
    parser.add_argument("--stage", type=int, choices=(0, 1))
    parser.add_argument("--run")
    parser.add_argument("--actual-step")
    parser.add_argument("--gpu-start-epoch", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = verify_package(args.source_sha)
    config = read_pinned(args.config, args.config_sha)
    validate_config(config)
    if not args.execute:
        print(json.dumps(dict(source_verified=True, GPU=False, Slurm=False)))
        return
    if (args.stage is None or not args.run or not args.actual_step
            or args.gpu_start_epoch is None or not math.isfinite(args.gpu_start_epoch)
            or not args.claim or not args.claim_sha):
        parser.error("execution requires a separate pinned claim and original step identity")
    if not re.fullmatch(r"\d+\.\d+", args.actual_step):
        raise ValueError("invalid original Slurm step")
    config.update(stage=args.stage, run=args.run, actual_step=args.actual_step,
                  gpu_start_epoch=args.gpu_start_epoch, source_manifest_SHA256=args.source_sha,
                  provider_config_SHA256=args.config_sha, public_python_root=str(root / "public-python"))
    require_claim(read_pinned(args.claim, args.claim_sha), config)
    if (socket.gethostname().split(".")[0] != config["nodes"][args.stage]
            or os.environ.get("SLURM_PROCID") != str(args.stage)
            or ".".join(os.environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURM_STEP_ID")) != args.actual_step):
        raise ValueError("private comparison is outside the claimed stage and original step")
    now = time.time()
    if not args.gpu_start_epoch <= now < args.gpu_start_epoch + config["service_seconds"] - 120:
        raise TimeoutError("private comparison original epoch unavailable or expired")
    verify(config)
    original = Path(config["block_compare_private_plan"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != PARENT or (
            hashlib.sha256(freeze(original)).hexdigest() != PLAN_SHA256):
        raise ValueError("private GLM continuation plan differs")
    run_stage(config)


if __name__ == "__main__":
    main()
