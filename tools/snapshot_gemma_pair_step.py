"""Capture the original CPU preflight or independent post-step on the Gemma node."""

import argparse
import os
from pathlib import Path
import socket

from compare_gemma_domain_recovery import read_receipt, require
from read_gemma_domain_terminal import (
    LABEL, NODE, bind_original_steps, bound_step, numbered_step, token,
)
from run_gemma_domain_pair import DEVICE_HELPER, pinned_module, write_json


def own_process_still_present(owner):
    pid, start = owner["PID"], owner["startticks"]
    require(type(pid) is int and type(start) is int and pid > 0 and start > 0, "invalid rank process identity")
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1].split()
    except FileNotFoundError:
        return False
    return int(fields[19]) == start and fields[0] != "Z"


def remaining_owned_ranks(root, request, gpu_step):
    live = []
    evidence = {}
    identities = set()
    complete = True
    for arm, kind in (("continuous", "continuous-witness"), ("resumed", "resume-witness")):
        try:
            outcome = read_receipt(root, Path(arm) / "native-outcome.json", evidence)
        except FileNotFoundError:
            complete = False
            continue
        require(outcome["step"] == gpu_step and outcome["request_id"] == request,
                "mismatched original native owner registry")
        if (outcome["passed"] is not True or outcome["owned_processes_remaining"] != 0
                or len(outcome["ranks"]) != 8 or {row["rank"] for row in outcome["ranks"]} != set(range(8))):
            complete = False
        for rank in outcome["ranks"]:
            path = Path(arm) / "train" / kind / f"rank-{rank['rank']:02d}" / "owner.json"
            owner = read_receipt(root, path, evidence)
            identity = (owner["PID"], owner["startticks"])
            require(evidence[path.as_posix()] == rank["owner_SHA256"]
                    and owner["rank"] == owner["local_rank"] == rank["rank"] and identity not in identities,
                    "native owner bytes or fresh process identity differs")
            identities.add(identity)
            if own_process_still_present(owner):
                live.append(owner["PID"])
    return complete, live


def capture(repo, root, intent_sha, phase):
    require(phase in {"pre", "post"}, "invalid snapshot phase")
    root = root.resolve(strict=True)
    evidence = {}
    intent = read_receipt(root, Path("intent.json"), evidence)
    require(evidence["intent.json"] == intent_sha and intent["label"] == LABEL
            and token(intent["request_id"], 32) and "steps" not in intent, "original unnumbered intent differs")
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    require(numbered_step(step) and socket.gethostname().split(".")[0] == NODE,
            "snapshot is outside the original bound node and parent")
    source = pinned_module(repo, DEVICE_HELPER, "gemma_host_device_snapshot")
    source["NODE"] = NODE
    snapshot = source["snapshot"](False)
    require(snapshot["step"] == step and snapshot["full_fd_inventory_performed"] is True,
            "host snapshot is not independently attributable to the CPU step")
    port = intent["master_port"]
    require(type(port) is int and 1024 <= port <= 65535, "invalid original rendezvous port")
    with socket.socket() as available:
        try:
            available.bind(("127.0.0.1", port))
        except OSError:
            free = False
        else:
            free = True
    snapshot.update(request_id=intent["request_id"], intent_SHA256=intent_sha,
                    master_port=port, master_port_available=free)
    if phase == "post":
        before = read_receipt(root, Path("before.json"), evidence)
        gpu = read_receipt(root, Path("gpu-step-binding.json"), evidence)
        pre_step = bound_step(before, intent["request_id"], intent_sha)
        gpu_step = bound_step(gpu, intent["request_id"], intent_sha)
        require(len({pre_step, gpu_step, step}) == 3 and gpu["pre_step"] == pre_step
                and gpu["before_SHA256"] == evidence["before.json"], "cleanup is not bound to the original pre/GPU steps")
        complete, live = remaining_owned_ranks(root, intent["request_id"], gpu_step)
        snapshot.update(gpu_step=gpu_step, gpu_step_binding_SHA256=evidence["gpu-step-binding.json"],
                        owned_registry_present=complete, owned_processes_remaining=live if complete else None,
                        observed_owned_processes_remaining=live)
    write_json(root / ("before.json" if phase == "pre" else "after.json"), snapshot)
    return (snapshot["master_port_available"] and snapshot["kfd_holder_count"] == 0
            and (phase == "pre" or snapshot["owned_processes_remaining"] == []))


def seal_accounting(root, intent_sha, accounting):
    root = root.resolve(strict=True)
    evidence = {}
    intent = read_receipt(root, Path("intent.json"), evidence)
    require(evidence["intent.json"] == intent_sha and token(intent["request_id"], 32)
            and intent["label"] == LABEL and "steps" not in intent, "unknown original accounting intent")
    before, gpu, after = (read_receipt(root, Path(name), evidence)
                           for name in ("before.json", "gpu-step-binding.json", "after.json"))
    binding = bind_original_steps(intent, intent_sha, before, gpu, after, evidence)
    columns, rows = accounting["columns"], accounting["rows"]
    keys = [name.split("%", 1)[0] for name in columns]
    require(len(set(keys)) == len(keys) and {"JobIDRaw", "JobName", "State"} <= set(keys)
            and set(rows) == set(binding["steps"]), "accounting columns or original stages are incomplete")
    terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
                "DEADLINE", "REVOKED"}
    for phase, row in rows.items():
        value = dict(zip(keys, row, strict=True))
        require(value["JobIDRaw"] == binding["steps"][phase] and value["JobName"] == f"{LABEL}-{phase}"
                and value["State"].split()[0].rstrip("+") in terminal, "accounting row is unbound or still active")
    normalized = dict(request_id=intent["request_id"], intent_SHA256=intent_sha,
                      gpu_step_binding_SHA256=evidence["gpu-step-binding.json"],
                      after_SHA256=evidence["after.json"], columns=columns, rows=rows)
    write_json(root / "terminal-accounting.json", normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pre", "post"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--intent-sha", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(0 if capture(args.repo, args.root, args.intent_sha, args.phase) else 1)


if __name__ == "__main__":
    main()
