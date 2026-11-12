"""Verify already-staged VGGT inputs on CPU; never copy assets or start training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "config", "input-root", "upstream-root", "upstream-files", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reference-sha256", "config-sha256", "upstream-files-sha256", "backend-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    anchor = time.time()
    if not (0 < args.deadline - anchor <= args.deadline - args.epoch <= 240):
        raise ValueError("invalid or expired original CPU epoch")
    remaining = args.deadline - anchor

    def deadline():
        if time.time() >= args.deadline or time.monotonic() - start >= remaining:
            raise TimeoutError("original CPU input epoch exhausted")

    if any(os.environ.get(key) != "" for key in
           ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("CPU input checker requires explicit empty device visibility")
    if args.output.exists() or args.output.with_suffix(".progress.jsonl").exists():
        raise ValueError("input-check output already exists")

    def pinned(path, expected):
        deadline()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("input-check dependency identity mismatch")
        return raw

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation/src"))
    from gencuda.vggt_inputs import FIELDS, checked_files, check_batches

    result = {"passed": False, "GPU_cards": 0, "model_instantiated": False,
              "weight_bytes_read": 0, "CPU_preprocessing_qualified": False}
    with args.output.with_suffix(".progress.jsonl").open("x", encoding="utf-8") as log:
        def progress(value):
            log.write(json.dumps(value, allow_nan=False) + "\n")
            log.flush()

        try:
            reference = json.loads(pinned(args.reference, args.reference_sha256))
            config = json.loads(pinned(args.config, args.config_sha256))
            upstream = json.loads(pinned(args.upstream_files, args.upstream_files_sha256))
            if config != reference["source_config"]:
                raise ValueError("original recipe mismatch")
            progress({"event": "reference", "planned_attempts": len(reference["attempts"]),
                      "planned_files": len(reference["files"]), "reference_SHA256": args.reference_sha256})
            result["input_files"] = checked_files(args.input_root, reference["files"], deadline)
            result["upstream_files"] = checked_files(args.upstream_root, upstream["files"], deadline)
            backend = Path(__file__).resolve().parents[1] / "generation/src/gencuda/backends/vggt_training.py"
            pinned(backend, args.backend_sha256)

            import numpy as np
            import torch
            import cv2

            if torch.cuda.is_initialized():
                raise RuntimeError("unexpected GPU initialization")

            def forbid(*_args, **_kwargs):
                raise RuntimeError("GPU initialization forbidden during CPU input check")

            torch.cuda._lazy_init = forbid
            torch.set_num_threads(2)
            cv2.setNumThreads(2)
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            sys.path.insert(0, str(args.upstream_root))
            sys.path.insert(0, str(args.upstream_root / "training"))
            from gencuda.backends import vggt_training as native
            from gencuda.hy_resume_checkpoint import tree_sha256
            from data.datasets.co3d import Co3dDataset

            dataset = Co3dDataset(
                native._common_config(config["image_size"]), split="train",
                CO3D_DIR=str(args.input_root / "co3d-single"),
                CO3D_ANNOTATION_DIR=str(args.input_root / "co3d-vggt-annotations"),
                min_num_images=config["minimum_sequence_frames"], len_train=1000,
            )

            def describe(batch):
                found = {}
                for name in FIELDS:
                    value = batch[name]
                    if value.device.type != "cpu" or (value.is_floating_point()
                            and not bool(torch.isfinite(value).all())):
                        raise ValueError("invalid CPU input tensor")
                    found[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "SHA256": tree_sha256(value)}
                return found

            result.update(check_batches(
                dataset, config, reference, make_ids=lambda ids: np.asarray(ids, dtype=np.int64),
                make_batch=lambda raw: native._tensor_batch(torch, raw),
                valid_depth=native.valid_depth_batch, describe=describe,
                batch_hash=tree_sha256, selected_hash=native._vggt_batch_content_sha256,
                deadline=deadline, progress=progress,
            ))
            if torch.cuda.is_initialized():
                raise RuntimeError("unexpected GPU initialization")
            result["CPU_preprocessing_qualified"] = result["passed"]
            result["versions"] = {name: importlib.metadata.version(name)
                                  for name in ("torch", "numpy", "torchvision", "Pillow")}
            result["versions"]["opencv_module"] = cv2.__version__
        except BaseException as error:
            result.update(passed=False, CPU_preprocessing_qualified=False,
                          error_type=type(error).__name__)
            progress({"event": "failed_terminal", "error_type": type(error).__name__})
        result.update(epoch=args.epoch, deadline=args.deadline, wall_seconds=time.monotonic() - start,
                      reference_SHA256=args.reference_sha256, config_SHA256=args.config_sha256,
                      upstream_files_SHA256=args.upstream_files_sha256,
                      backend_SHA256=args.backend_sha256, Model_GO=False,
                      qualification="CPU input preprocessing identities only; no model, feature cache, training or task effects.")
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({key: result.get(key) for key in
                      ("passed", "candidate_batches", "valid_batches", "rejected_batches", "tensor_fields_checked", "error_type")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
