"""Render the adopted Gemma zero-GRES container invocation without submitting it."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex

import index_entry


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "code/bw1000/training-gemma-vizwiz-reviewed-train-native-CPU-v1/receive_once.py"
ORIGINAL_SHA = "3aac2d24674445c93939f87f6298d5915161da48d8c23b14c41fcb53a7afe9da"
PROJECT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
LABEL = "training-gemma-vizwiz-native-index-cpu-v1"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def source_path(value):
    path = PurePosixPath(value)
    require(str(path) == value and ".." not in path.parts and path.is_relative_to(PROJECT)
            and re.fullmatch(r"/[A-Za-z0-9._/-]+", value), "EXACT_TARGET_PATH")
    return path


def compute_command(parent, node, epoch, source, view, out):
    require(re.fullmatch(r"[0-9]+", parent) and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", node)
            and math.isfinite(epoch) and epoch > 0, "FRESH_CPU_PLACEMENT_REQUIRED")
    source, view, out = map(source_path, (source, view, out))
    require(len({source, view, out}) == 3 and all(
        left not in right.parents for left in (source, view, out) for right in (source, view, out) if left != right),
        "CPU_MOUNT_ISOLATION")
    old = ORIGINAL.read_bytes()
    require(sha256(old).hexdigest() == ORIGINAL_SHA, "FROZEN_CPU_RENDERER")
    functions = [stmt for stmt in ast.parse(old).body if isinstance(stmt, ast.FunctionDef) and stmt.name == "command"]
    require(len(functions) == 1, "FROZEN_CPU_RENDERER")
    prior_source = RESULTS / "training-gemma-vizwiz-reviewed-train-native-CPU-v1"
    reviewed = RESULTS / "training-gemma-vizwiz-reviewed-train-native-v1"
    env = {"shlex": shlex, "Path": PurePosixPath, "WORK_SECONDS": index_entry.LIMIT,
           "PARENT": parent, "NODE": node, "LABEL": LABEL,
           "HIDDEN": ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"),
           "DATA_STAGE": reviewed,
           "FRAMEWORK": RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework",
           "PROCESSOR": RESULTS / "training-gemma-processor-assets-v1/models/gemma-3-4b-it",
           "RUNTIME": RESULTS / "training-vlm-swift-offline-runtime-v1",
           "TRANSFORMERS": RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site",
           "CONTAINER": PROJECT / "containers/dtk26-sandbox-r1",
           "APPTAINER": "/public/software/apps/apptainer/1.3.4/bin/apptainer"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<original .2883 command>", "exec"), env)
    argv = env["command"](prior_source, epoch)
    replaced = str(prior_source / "evidence") + ":/audit-out:rw"
    require(argv.count(replaced) == 1, "ORIGINAL_RECEIPT_BIND")
    argv[argv.index(replaced)] = f"{out}:/index-out:rw"
    addition = [(source, "/candidate-source", "ro"), (view, "/vizwiz-view", "ro")]
    five = RESULTS / "training-gemma-vizwiz-five-native-input-audit-v1"
    addition.append((five, str(five), "ro"))
    for stage, _, _, _, _, _, _, _ in index_entry.ORIGINAL_WORKERS:
        origin = RESULTS / stage
        for name in ("outcome.json", "evidence/worker/result.json"):
            path = origin / name
            addition.append((path, str(path), "ro"))
    offset = argv.index(str(env["CONTAINER"]))
    new_binds = [part for host, target, mode in addition for part in ("--bind", f"{host}:{target}:{mode}")]
    argv[offset:offset] = new_binds
    offset = argv.index(env["APPTAINER"])
    argv[offset:offset] = ["--threads-per-core=1", "--cpu-bind=threads", "--kill-on-bad-exit=1"]
    old_worker = "python3 -B -u /audit-source/inside.py"
    new_worker = ["python3", "-B", "-u", "/candidate-source/index_entry.py", "--parent", parent,
                  "--node", node, "--epoch", repr(epoch), "--view", "/vizwiz-view/train800-swift.jsonl",
                  "--index", "/index-out/native-index.json"]
    require(argv[-1].count(old_worker) == 1, "ORIGINAL_WORKER_COMMAND")
    argv[-1] = argv[-1].replace(old_worker, shlex.join(new_worker), 1)
    require(argv.count("--gres=none") == 1 and sum(item.endswith(":rw") for item in argv) == 1
            and "--rocm" not in argv, "ZERO_GRES_CPU_COMMAND")
    return argv


def terminal(raw, parent, node, step, *, require_success=False):
    require(re.fullmatch(re.escape(parent) + r"\.[0-9]+", step), "ACTUAL_NUMERIC_STEP_REQUIRED")
    rows = [line.split("|") for line in raw.splitlines() if line.split("|", 1)[0] == step]
    require(len(rows) == 1 and len(rows[0]) == 7, "EXACT_INDEPENDENT_ACCOUNTING")
    row = rows[0]
    resources = row[6].split(",")
    require(row[1] == LABEL and row[2].split()[0].rstrip("+") in TERMINAL
            and re.fullmatch(r"[0-9]+:[0-9]+", row[3]) and row[4].isdecimal()
            and row[5] == node and len(resources) == 3 and set(resources) in (
                {"cpu=4", "mem=8G", "node=1"}, {"cpu=4", "mem=8192M", "node=1"}), "ACTUAL_ZERO_GRES_ACCOUNTING")
    require(not require_success or row[2:4] == ["COMPLETED", "0:0"], "NATIVE_INDEX_TERMINAL_FAILED")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--epoch", type=float, required=True)
    for name in ("source", "view", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    argv = compute_command(args.parent, args.node, args.epoch, args.source, args.view, args.output)
    print(json.dumps({"argv": argv, "native_calls": 0, "new_Slurm_calls": 0, "planned_train_rows": 12,
                      "terminal_query_argv": ["sacct", "-nP", "-j", "<ACTUAL_NUMERIC_STEP>", "--format=" + COLUMNS]}))


if __name__ == "__main__":
    main()
