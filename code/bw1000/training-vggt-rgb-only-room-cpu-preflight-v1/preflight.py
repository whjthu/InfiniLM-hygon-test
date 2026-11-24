"""Native VGGT RGB-only CPU preflight for the exact seven Freiburg room pairs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
from pathlib import Path, PurePosixPath


SCHEMA = "vggt-room-rgb-only-cpu-preflight-v1"
DATA_SCHEMA = SCHEMA + "-inputs"
LIMIT_SECONDS = 120
BACKEND = "generation/src/gencuda/backends/vggt_training.py"
READER = "generation/src/gencuda/backends/vggt_rgb_only_room.py"
RENDERED_SHA = "73a2426d3f71c4b5ae6610fe1ca2b78355ed5859ea40be022ade942139ec2333"
READER_SHA = "d606640d98fde669052f9c08d020322c3b299f68f90f11fc6ba92435d4d3b517"
BASE_SHA = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
PUBLISHED_OVERLAY = "b32edb5228032f31d84ed8bbb275f6263753103c"
T12_MANIFESTS = {
    "FRAMEWORK_FILES.json": (59162, "cc8fe56800a27efd7876650df1bf2b0c77dee0a1764f9cadf8441ebfb15b663c"),
    "UPSTREAM_FILES.json": (11329, "64380b341ce377fb7e4b24c3f4f5986b6557b5339f4f7db57950dfa7e66b04c0"),
    "DEPENDENCIES.json": (955, "f505a7cc192583d1665a5565bd267729116c108d2ff475ae712cecc67e1fc81f"),
}
DATA_PATHS = tuple(
    relative
    for group, publication, bom, crop, pose in (
        ("window02", "-rgb224-local-camera-convention.json", "window02-private.json",
         "frames224-rgb8-private.npy", "poses-first-camera-private.json"),
        ("windows03to05", "-rgb224-replay.json", "windows03to05-private-bom.json",
         "frames224-private.npy", "camera-targets-private.json"),
        ("windows06to08", "-rgb224-replay.json", "windows06to08-private-bom.json",
         "frames224-private.npy", "poses-private.json"),
    )
    for stem in ("specialized48-tum-fr1-room-bag-" + group,)
    for relative in (
        f"results/bw1000/2026-11-25/{stem}{publication}",
        f"local-only/{stem}-v1/{bom}",
        f"local-only/{stem}-rgb224-v1/{crop}",
        f"local-only/{stem}-rgb224-v1/{pose}",
    )
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _source_member(root, relative):
    name = PurePosixPath(relative)
    require(str(name) == relative and not name.is_absolute() and ".." not in name.parts
            and "\\" not in relative and ":" not in relative, "unsafe manifest member")
    file = root.joinpath(*name.parts)
    require(file.resolve(strict=True) == root.resolve(strict=True).joinpath(*name.parts)
            and file.is_file(), "member is missing or symbolic")
    return file


def _verified(root, relative, size, digest):
    require(type(size) is int and 0 <= size <= 16_000_000, "invalid pin size")
    file = _source_member(root, relative)
    with file.open("rb") as stream:
        raw = stream.read(size + 1)
    require(len(raw) == size and sha(raw) == digest, "pin differs")
    return raw


def read_data_manifest(root, path, digest):
    require(len(digest) == 64 and path.is_file() and not path.is_symlink(), "invalid manifest")
    with path.open("rb") as stream:
        raw = stream.read(8193)
    require(len(raw) <= 8192 and sha(raw) == digest, "manifest pin differs")
    manifest = json.loads(raw)
    require(set(manifest) == {"schema", "frozen_window_denominator", "windows", "files", "total_bytes"}
            and manifest["schema"] == DATA_SCHEMA and manifest["frozen_window_denominator"] == 12
            and manifest["windows"] == list(range(2, 9)), "manifest scope differs")
    rows = manifest["files"]
    require(tuple(row["path"] for row in rows) == DATA_PATHS
            and all(set(row) == {"path", "bytes", "SHA256"} for row in rows)
            and sum(row["bytes"] for row in rows) == manifest["total_bytes"], "manifest members differ")
    for row in rows:
        _verified(root, row["path"], row["bytes"], row["SHA256"])
    return manifest


def verify_sources(authority, framework, upstream):
    pinned = {
        name: json.loads(_verified(authority, name, size, digest))
        for name, (size, digest) in T12_MANIFESTS.items()
    }
    framework_files = pinned["FRAMEWORK_FILES.json"]
    require(framework_files["framework_commit"] == "3b1d6909efe4b9d25d4a1562068e2006298f82ed"
            and len(framework_files["files"]) == 323
            and framework_files["files"][BACKEND]["SHA256"] == BASE_SHA, "framework source identity differs")
    for relative, pin in framework_files["files"].items():
        if relative == BACKEND:
            _verified(framework, relative, 62427, RENDERED_SHA)
        else:
            _verified(framework, relative, pin["bytes"], pin["SHA256"])
    _verified(framework, READER, 8201, READER_SHA)
    upstream_files = pinned["UPSTREAM_FILES.json"]["files"]
    require(len(upstream_files) == 69, "unexpected upstream closure")
    for pin in upstream_files:
        _verified(upstream, pin["name"], pin["bytes"], pin["SHA256"])
    for base, expected in (
        (framework / "generation/src/gencuda", {name for name in framework_files["files"]
                                                if name.startswith("generation/src/gencuda/") and name.endswith(".py")} | {READER}),
        (upstream / "vggt", {pin["name"] for pin in upstream_files
                               if pin["name"].startswith("vggt/") and pin["name"].endswith(".py")}),
    ):
        origin = framework if base == framework / "generation/src/gencuda" else upstream
        seen = {path.relative_to(origin).as_posix() for path in base.rglob("*.py")}
        require(seen == expected and not list(base.rglob("__pycache__")), "unexpected Python module or bytecode")
    return pinned["DEPENDENCIES.json"]


def binding(epoch, now, env, expected_job, expected_node, observed_node):
    job, step = env.get("SLURM_JOB_ID", ""), env.get("SLURM_STEP_ID", "")
    require(job.isascii() and job.isdecimal() and step.isascii() and step.isdecimal()
            and job == expected_job and observed_node == expected_node and env.get("SLURM_CPUS_PER_TASK") == "4",
            "Slurm step, node or CPU binding differs")
    step_counts = ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK")
    step_tres = env.get("SLURM_TRES_PER_TASK", "").lower()
    require(not env.get("SLURM_STEP_GPUS")
            and all(env.get(name, "") in ("", "0") for name in step_counts)
            and not any(device in step_tres for device in ("gpu", "dcu")),
            "step accelerator allocation indicators are present")
    # Parent allocation variables can be populated for a zero-GRES child step.
    parent_gpu_fields = ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_PER_NODE")
    require(math.isfinite(epoch) and epoch <= now < epoch + LIMIT_SECONDS, "CPU step epoch expired or invalid")
    return {"job_id": job, "step_id": step, "node": observed_node, "cpu_per_task": 4,
            "epoch": epoch, "deadline": epoch + LIMIT_SECONDS,
            "advisory_parent_gpu_vars_present": [key for key in parent_gpu_fields if env.get(key)]}


def _assert_cpu_batch(torch, batch):
    shape = {"images": (1, 2, 3, 224, 224), "extrinsics": (1, 2, 3, 4),
             "intrinsics": (1, 2, 3, 3), "pose_valid": (1, 2)}
    require(set(batch) == set(shape) | {"seq_name", "ids"}, "unexpected CPU batch fields")
    for key, dimensions in shape.items():
        tensor = batch[key]
        dtype = torch.bool if key == "pose_valid" else torch.float32
        require(tensor.device.type == "cpu" and tuple(tensor.shape) == dimensions and tensor.dtype == dtype,
                "CPU tensor shape, dtype or device differs")
        require(bool(torch.isfinite(tensor).all()), "nonfinite CPU tensor")
    require(batch["seq_name"] == "freiburg1_room" and bool(batch["pose_valid"].all())
            and batch["ids"][0] < batch["ids"][1], "native pair sequence, validity or order differs")


def _native(args, result):
    dependency = verify_sources(args.authority, args.framework, args.upstream)
    runtime_site = args.runtime_site.resolve(strict=True)
    require(runtime_site.is_dir() and runtime_site == args.runtime_site.absolute(), "runtime site is not unaliased")
    sys.path.insert(0, str(runtime_site))
    relevant = ("torch", "numpy")
    versions = {name: importlib.metadata.version(name) for name in relevant}
    require(versions == {name: dependency["versions"][name] for name in relevant}, "runtime versions differ")
    result["phase"] = "native_import"
    import torch
    require(torch.version.hip == dependency["HIP"] and not torch.cuda.is_initialized(), "CPU runtime identity differs")

    def forbid_device(*_args, **_kwargs):
        raise RuntimeError("accelerator initialization forbidden")

    torch.cuda._lazy_init = forbid_device
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(args.framework / "generation/src"), str(args.upstream)]
    from gencuda import hy_resume_checkpoint
    from gencuda.backends import vggt_rgb_only_room as room
    from gencuda.backends import vggt_training as backend
    from vggt.utils import pose_enc

    for module, path in (
        (backend, args.framework / BACKEND), (room, args.framework / READER),
        (hy_resume_checkpoint, args.framework / "generation/src/gencuda/hy_resume_checkpoint.py"),
        (pose_enc, args.upstream / "vggt/utils/pose_enc.py"),
    ):
        require(Path(module.__file__).resolve() == path.resolve(), "native import provenance differs")

    result["phase"] = "native_cpu_conversion_pose_and_hash"
    raw_batches = room.read_rgb_only_room(args.data_root)
    require(len(raw_batches) == 7, "incorrect candidate denominator")
    with torch.no_grad():
        batches = [backend._tensor_batch(torch, raw, rgb_only=True) for raw in raw_batches]
        for raw, batch in zip(raw_batches, batches):
            _assert_cpu_batch(torch, batch)
            require(batch["ids"] == raw["ids"], "native ordered IDs differ")
        require(len({item for batch in batches for item in batch["ids"]}) == 14, "repeated native frame")

        digest = backend._vggt_batch_content_sha256(batches)
        projected = [
            {key: batch[key] for key in ("seq_name", "ids", "images", "extrinsics", "intrinsics", "pose_valid")}
            for batch in batches
        ]
        recalculated = hy_resume_checkpoint.tree_sha256({"mode": "tum_room_rgb_only", "batches": projected})
        rebuilt = [backend._tensor_batch(torch, raw, rgb_only=True) for raw in raw_batches]
        require(digest == recalculated == backend._vggt_batch_content_sha256(rebuilt)
                and digest != backend._vggt_batch_content_sha256(list(reversed(batches))), "native hash check failed")

        targets = []
        for batch in batches:
            inputs = (batch["extrinsics"], batch["intrinsics"], batch["images"].shape[-2:])
            target = pose_enc.extri_intri_to_pose_encoding(*inputs, pose_encoding_type="absT_quaR_FoV")
            repeated = pose_enc.extri_intri_to_pose_encoding(*inputs, pose_encoding_type="absT_quaR_FoV")
            require(target.device.type == "cpu" and target.dtype == torch.float32 and tuple(target.shape) == (1, 2, 9)
                    and bool(torch.isfinite(target).all()) and torch.equal(target, repeated), "native target check failed")
            targets.append(target)
        targets_SHA = hy_resume_checkpoint.tree_sha256(targets)
    require(not torch.cuda.is_initialized(), "accelerator initialized during CPU preflight")
    result.update(status="PASS", phase="complete", native_cpu_pairs=len(batches), native_cpu_frames=14,
                  pose_valid_frames=14, native_targets=len(targets), native_content_SHA256=digest,
                  independently_assembled_content_SHA256=recalculated, native_targets_SHA256=targets_SHA,
                  runtime=versions, python=sys.version.split()[0], HIP=torch.version.hip, CUDA_initialized=False)


def write_receipt(path, receipt):
    raw = (json.dumps(receipt, ensure_ascii=True, sort_keys=True) + "\n").encode()
    require(len(raw) <= 4096, "receipt exceeds limit")
    with path.open("xb") as stream:
        stream.write(raw)
    return {"bytes": len(raw), "SHA256": sha(raw)}


def _expired(*_args):
    raise TimeoutError("CPU preflight deadline")


def _digest_argument(value):
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authority", "framework", "upstream", "runtime-site", "data-root", "manifest", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True, type=_digest_argument)
    parser.add_argument("--epoch", required=True, type=float)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--expected-node", required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        parser.error("receipt already exists")
    result = dict(schema=SCHEMA, phase="binding", status="BLOCKED", native_cpu_pairs=0, native_cpu_frames=0,
                  native_targets=0, optimizer_updates=0, model_weight_bytes=0, GPU_cards_requested=0,
                  heldout_sites=0, frozen_window_denominator=12, overlay_publication=PUBLISHED_OVERLAY,
                  rendered_backend_SHA256=RENDERED_SHA, dataset_SHA256=args.manifest_sha256)
    start = time.monotonic()
    try:
        require(sys.platform == "linux" and sys.flags.isolated and sys.flags.dont_write_bytecode, "isolated Linux Python required")
        result["binding"] = binding(args.epoch, time.time(), os.environ, args.expected_job, args.expected_node,
                                    os.uname().nodename.split(".")[0])
        signal.signal(signal.SIGALRM, _expired)
        remaining = result["binding"]["deadline"] - time.time()
        require(remaining > 0, "CPU step epoch expired before alarm")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[key] = ""
        os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        result["phase"] = "data_manifest"
        manifest = read_data_manifest(args.data_root, args.manifest, args.manifest_sha256)
        result.update(input_members=len(manifest["files"]), input_bytes=manifest["total_bytes"])
        result["phase"] = "source_and_runtime"
        _native(args, result)
    except (ValueError, OSError, ImportError, RuntimeError, TimeoutError, KeyError, TypeError) as error:
        result["error_type"] = type(error).__name__
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
    result["wall_seconds"] = round(time.monotonic() - start, 4)
    receipt = write_receipt(args.receipt, result)
    print(json.dumps({"status": result["status"], "phase": result["phase"], "receipt": receipt}))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
