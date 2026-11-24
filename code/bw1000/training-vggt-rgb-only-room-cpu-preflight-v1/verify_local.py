"""Produce the exact private input manifest and verify local preflight boundaries."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import preflight


OVERLAY = Path("code/bw1000/training-vggt-rgb-only-room-v1")
CANONICAL = Path("code/bw1000/training-qwen-A100-first-updates-v1/deltas/generation/src/gencuda/backends/vggt_training.py")
T12 = Path("code/bw1000/training-vggt-native-provider-HIP-CPU-v2")
RENDERER_SHA = "0c4dd386c848620d66673ab7f624957ba4ffb97fc6d4f9b645a82f5bc3a2ec88"


def checked_module(name, path, digest):
    raw = path.read_bytes()
    assert preflight.sha(raw) == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def must_fail(function, expected):
    try:
        function()
    except (ValueError, FileExistsError) as error:
        assert isinstance(error, expected)
    else:
        raise AssertionError("invalid boundary passed")


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode()


def _discover_inputs(root):
    read_order = []
    reader = checked_module("published_room_reader", root / OVERLAY / "vggt_rgb_only_room.py", preflight.READER_SHA)
    checked_read = reader._verified

    def observe(path, pin, limit):
        payload = checked_read(path, pin, limit)
        read_order.append({"path": path.relative_to(root).as_posix(), "bytes": len(payload), "SHA256": preflight.sha(payload)})
        return payload

    with mock.patch.object(reader, "_verified", side_effect=observe):
        batches = reader.read_rgb_only_room(root)
    assert tuple(row["path"] for row in read_order) == preflight.DATA_PATHS
    assert len(batches) == 7 and all(batch["pose_valid"].tolist() == [True, True] for batch in batches)
    return {"schema": preflight.DATA_SCHEMA, "frozen_window_denominator": 12, "windows": list(range(2, 9)),
            "files": read_order, "total_bytes": sum(row["bytes"] for row in read_order)}


def _local_checks(root, temp, candidate, encoded):
    candidate.write_bytes(encoded)
    digest = preflight.sha(encoded)
    assert preflight.read_data_manifest(root, candidate, digest)["windows"] == list(range(2, 9))
    must_fail(lambda: preflight.read_data_manifest(root, candidate, "0" * 64), ValueError)
    original = json.loads(encoded)
    for files in (original["files"][::-1], original["files"][:-1], [*original["files"], original["files"][0]]):
        mutated = _json_bytes(dict(original, files=files, total_bytes=sum(item["bytes"] for item in files)))
        candidate.write_bytes(mutated)
        must_fail(lambda: preflight.read_data_manifest(root, candidate, preflight.sha(mutated)), ValueError)
    first = dict(original["files"][0], SHA256="0" * 64)
    bad_pin = _json_bytes(dict(original, files=[first, *original["files"][1:]]))
    candidate.write_bytes(bad_pin)
    must_fail(lambda: preflight.read_data_manifest(root, candidate, preflight.sha(bad_pin)), ValueError)
    candidate.write_bytes(encoded)

    env = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "1", "SLURM_CPUS_PER_TASK": "4"}
    advisory = preflight.binding(100, 101, dict(env, SLURM_JOB_GPUS="0", SLURM_GPUS_ON_NODE="0",
                                                SLURM_GPUS_PER_TASK="0", SLURM_TRES_PER_TASK="cpu=4"),
                                 "123", "testnode", "testnode")
    assert advisory["deadline"] == 220
    assert advisory["advisory_parent_gpu_vars_present"] == ["SLURM_JOB_GPUS"]
    for now, supplied_env, job, node in (
        (220, env, "123", "testnode"), (99, env, "123", "testnode"), (101, env, "999", "testnode"),
        (101, env, "123", "wrongnode"), (101, dict(env, SLURM_STEP_GPUS="0,1"), "123", "testnode"),
        (101, dict(env, SLURM_STEP_GPUS="0"), "123", "testnode"),
        (101, dict(env, SLURM_GPUS_ON_NODE="1"), "123", "testnode"),
        (101, dict(env, SLURM_GPUS_PER_TASK="1"), "123", "testnode"),
        (101, dict(env, SLURM_TRES_PER_TASK="cpu=4,gres/gpu=1"), "123", "testnode"),
        (101, dict(env, SLURM_TRES_PER_TASK="cpu=4,gres/dcu=1"), "123", "testnode"),
        (101, dict(env, SLURM_TRES_PER_TASK="cpu=4,GRES/DCU:1"), "123", "testnode"),
    ):
        must_fail(lambda: preflight.binding(100, now, supplied_env, job, node, "testnode"), ValueError)

    receipt = temp / "blocked.json"
    command = [
        sys.executable, "-I", "-B", str(Path(preflight.__file__).resolve()),
        "--authority", str(root / T12), "--framework", str(temp),
        "--upstream", str(temp), "--runtime-site", str(temp), "--data-root", str(root), "--manifest", str(candidate),
        "--manifest-sha256", digest, "--receipt", str(receipt), "--epoch", str(time.time()),
        "--expected-job", "none", "--expected-node", "none",
    ]
    first_run = subprocess.run(command, cwd=temp, capture_output=True, check=False, timeout=10, text=True)
    assert first_run.returncode == 1 and not first_run.stderr, first_run
    blocked = receipt.read_bytes()
    result = json.loads(blocked)
    assert result["status"] == "BLOCKED" and result["phase"] == "binding"
    assert result["native_cpu_pairs"] == result["native_targets"] == result["optimizer_updates"] == 0
    assert json.loads(first_run.stdout)["receipt"] == {"bytes": len(blocked), "SHA256": preflight.sha(blocked)}
    duplicate = subprocess.run(command, cwd=temp, capture_output=True, check=False, timeout=10, text=True)
    assert duplicate.returncode == 2 and receipt.read_bytes() == blocked
    must_fail(lambda: preflight.write_receipt(temp / "large.json", {"blob": "x" * 4096}), ValueError)
    assert not (temp / "large.json").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("private_manifest", type=Path)
    parser.add_argument("private_receipt", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve(strict=True)
    assert not any(path.exists() or path.is_symlink() for path in (args.private_manifest, args.private_receipt))
    renderer = checked_module("published_room_renderer", root / OVERLAY / "render_backend.py", RENDERER_SHA)
    rendered = renderer.render((root / CANONICAL).read_bytes()).encode()
    assert len(rendered) == 62427 and preflight.sha(rendered) == preflight.RENDERED_SHA
    preflight_source = Path(preflight.__file__).read_bytes()
    compile(preflight_source, "<preflight>", "exec")
    attributes = {node.attr for node in ast.walk(ast.parse(preflight_source)) if isinstance(node, ast.Attribute)}
    assert {"_tensor_batch", "_vggt_batch_content_sha256", "extri_intri_to_pose_encoding", "tree_sha256"} <= attributes
    t12 = {name: json.loads(preflight._verified(root / T12, name, size, digest))
           for name, (size, digest) in preflight.T12_MANIFESTS.items()}
    assert len(t12["FRAMEWORK_FILES.json"]["files"]) == 323 and len(t12["UPSTREAM_FILES.json"]["files"]) == 69
    assert t12["FRAMEWORK_FILES.json"]["files"][preflight.BACKEND]["SHA256"] == preflight.BASE_SHA
    manifest = _discover_inputs(root)
    encoded = _json_bytes(manifest)
    with tempfile.TemporaryDirectory(prefix="vggt-room-rgb-audit-") as directory:
        temp = Path(directory)
        _local_checks(root, temp, temp / "minimum.json", encoded)
    with args.private_manifest.open("xb") as stream:
        stream.write(encoded)
    receipt = {
        "schema": preflight.SCHEMA + "-local-verification",
        "observed_at_China": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "overlay_publication": preflight.PUBLISHED_OVERLAY,
        "preflight_SHA256": preflight.sha(preflight_source),
        "base_backend_SHA256": preflight.BASE_SHA,
        "rendered_backend_SHA256": preflight.RENDERED_SHA,
        "reader_SHA256": preflight.READER_SHA,
        "renderer_SHA256": RENDERER_SHA,
        "private_manifest": {"SHA256": preflight.sha(encoded), "bytes": len(encoded)},
        "adapter_reads": {"public": 3, "private": 9, "total_bytes": manifest["total_bytes"]},
        "local_pairs": 7, "local_frames": 14, "frozen_window_denominator": 12,
        "local_checks": ["exact_adapter_observed_reads", "source_identity_and_syntax", "manifest_pin_and_scope_rejections",
                         "deadline_and_node_binding_step_gpu_dcu_rejections_and_parent_gpu_advisory", "entrypoint_reports_blocked_without_native_runtime",
                         "compact_receipt_size_and_no_overwrite"],
        "state": "LOCAL_VERIFIED_TARGET_NOT_STAGED",
        "native_cpu_pairs": 0, "native_targets": 0, "native_content_hashes": 0, "optimizer_updates": 0,
        "remote_actions": 0, "slurm_steps": 0, "GPU": 0,
    }
    pin = preflight.write_receipt(args.private_receipt, receipt)
    print(json.dumps({"receipt": pin, "manifest": receipt["private_manifest"], "adapter_reads": receipt["adapter_reads"]}))


if __name__ == "__main__":
    main()
