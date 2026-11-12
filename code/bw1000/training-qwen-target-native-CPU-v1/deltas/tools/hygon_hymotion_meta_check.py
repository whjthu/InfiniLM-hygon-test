"""Check native HY-Motion meta structure through the unified training backend."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.abc
import io
import json
import math
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def deadline_check(epoch: float, deadline: float, now: float | None = None) -> None:
    current = time.time() if now is None else now
    if not all(math.isfinite(x) for x in (epoch, deadline, current)):
        raise ValueError("CPU epoch and deadline must be finite")
    if not 0 < deadline - epoch <= 220 or not epoch <= current < deadline:
        raise TimeoutError("native meta CPU epoch is invalid or exhausted")


def pinned_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if len(expected) != 64 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("input metadata SHA256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input metadata must be an object")
    return value


def verify_sources(root: Path, pins: dict) -> dict:
    resolved = root.resolve(strict=True)
    actual = {}
    for name, pin in pins.items():
        part = Path(name)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("upstream source path escapes its root")
        path = resolved / part
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved):
            raise ValueError("upstream source must remain inside its pinned root")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != pin["bytes"] or digest != pin["SHA256"]:
            raise ValueError("upstream source bytes differ from the reference")
        actual[name] = digest
    if len(actual) != 16:
        raise ValueError("native source reference requires all sixteen Python files")
    return actual



@contextmanager
def without_optional_gpu_extensions():
    """Keep optional device-only imports outside this CPU meta qualification."""
    names = ("flash_attn", "flash_attn_2_cuda")
    if any(n == prefix or n.startswith(prefix + ".")
           for n in sys.modules for prefix in names):
        raise RuntimeError("optional GPU extensions were already imported")

    class BlockOptionalGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in names):
                raise ModuleNotFoundError("optional GPU extension excluded from CPU meta scope", name=fullname)
            return None

    finder = BlockOptionalGPU()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    deadline_check(args.epoch, args.deadline)
    if any(os.environ.get(n) not in ("", "-1") for n in
           ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("explicit empty device visibility is required")
    spec = pinned_json(args.spec, args.spec_sha256)
    upstream = verify_sources(args.source_root, spec["upstream_files"])
    backend = ROOT / "generation/src/gencuda/backends/hy_motion_training.py"
    if hashlib.sha256(backend.read_bytes()).hexdigest() != spec["backend_SHA256"]:
        raise ValueError("unified native training backend identity differs")
    if args.output.exists():
        raise FileExistsError("meta result already exists; reconcile the original run")
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "numpy", "einops", "accelerate", "PyYAML")}
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("meta checker requires a fresh CPU-only process")
    torch.set_num_threads(2)

    def forbid(*_args, **_kwargs):
        raise RuntimeError("device context is forbidden during native meta checks")

    torch.cuda._lazy_init = forbid
    from accelerate import init_empty_weights
    sys.path.insert(0, str(ROOT / "generation/src"))
    sys.path.insert(0, str(args.source_root.resolve()))
    from gencuda.backends import hy_motion_training as native
    from hymotion.utils.loaders import load_object
    deadline_check(args.epoch, args.deadline)
    config = spec["model_config"]
    recipe = spec["recipe"]
    transcript = io.StringIO()
    with redirect_stdout(transcript), redirect_stderr(transcript), init_empty_weights(include_buffers=False), without_optional_gpu_extensions():
        model = load_object(config["classes"]["network_module"], dict(config["network_module_args"]))
        if (model.motion_input_dim, model.output_dim, model.num_layers, model.feat_dim,
            len(model.double_blocks), len(model.single_blocks)) != (201, 201, 27, 1280, 9, 18):
            raise ValueError("native Transformer dimensions differ from the original")
        if not all(p.is_meta for p in model.parameters()):
            raise ValueError("native model materialized parameter values")
        base_count = sum(p.numel() for p in model.parameters())
        model.to(dtype=torch.bfloat16)
        layers = native._apply_lora(model, torch, rank=recipe["lora_rank"],
                                    alpha=recipe["lora_alpha"], target_fragments=recipe["lora_target_modules"])

        def schema():
            return {n: {"shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel(), "meta": p.is_meta}
                    for n, p in native._canonical_adapter_parameters(model).items()}

        before = schema()
        if before != spec["reference_trainable_manifest"] or sorted(layers) != spec["reference_lora_layers"]:
            raise ValueError("native trainable structure differs from the original A100")
        if base_count != spec["reference_base_parameters_meta"]:
            raise ValueError("native base parameter count differs")
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.endswith((".lora_A", ".lora_B"))):
            raise ValueError("native base parameters were not frozen")
        native._apply_gradient_checkpointing(model)
        if schema() != before:
            raise ValueError("checkpoint wrapping changed the canonical trainable manifest")
        params = list(native._canonical_adapter_parameters(model).values())
        optimizer = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        if optimizer.state or len(optimizer.param_groups) != 1 or [id(p) for p in params] != [
                id(p) for p in optimizer.param_groups[0]["params"]]:
            raise ValueError("native optimizer parameter identity differs")
    deadline_check(args.epoch, args.deadline)
    module_files = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("hymotion.") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            rel = path.relative_to(args.source_root.resolve()).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if upstream.get(rel) != digest:
                raise ValueError("runtime imported an unpinned upstream module")
            module_files[name] = {"path": rel, "SHA256": digest}
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss >= 4 * 1024**3 or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only native meta scope exceeded its constraints")
    value = {"passed": True, "qualification": "native-meta-structure-without-optional-GPU-extensions", "versions": versions,
             "torch_version": torch.__version__, "HIP": torch.version.hip,
             "CUDA_initialized": False, "GPU_cards": 0, "weights_or_cache_tensor_bytes_read": 0,
             "base_parameters_meta": base_count, "trainable_tensors": len(before),
             "trainable_parameters": sum(x["numel"] for x in before.values()),
             "trainable_manifest": before, "lora_layers": sorted(layers),
             "GC_canonical_manifest_unchanged": True, "base_frozen": True,
             "optimizer_empty_and_exact_binding": True, "runtime_project_modules": module_files,
             "full_pipeline_constructed": False, "frame_length_qualified": False,
             "optional_GPU_extensions_excluded": ["flash_attn", "flash_attn_2_cuda"],
             "production_attention_qualified": False,
             "numerical_alignment": False, "forward_backward_updates": 0,
             "formal_performance": False, "heldout_effect": False, "Model_GO": False,
             "epoch": args.epoch, "deadline": args.deadline,
             "wall_seconds": time.monotonic() - started, "peak_RSS_bytes": rss,
             "construction_log_SHA256": hashlib.sha256(transcript.getvalue().encode()).hexdigest(),
             "owner": {"PID": os.getpid(), "PGID": os.getpgrp(),
                       "startticks": int(Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19]),
                       "node": os.uname().nodename.split(".")[0],
                       "Slurm_job": os.environ.get("SLURM_JOB_ID"),
                       "Slurm_step": os.environ.get("SLURM_STEP_ID")}}
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        value = run(args)
    except Exception as error:
        # Preserve a failed attempt without overwriting any original result.
        failure = {"passed": False, "qualification": "native-meta-check-failed",
                   "error_type": type(error).__name__,
                   "epoch": args.epoch if math.isfinite(args.epoch) else None,
                   "deadline": args.deadline if math.isfinite(args.deadline) else None, "CUDA_initialized": None,
                   "frame_length_qualified": False, "numerical_alignment": False,
                   "formal_performance": False, "heldout_effect": False, "Model_GO": False}
        if not args.output.exists():
            with args.output.open("x", encoding="utf8") as stream:
                json.dump(failure, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps({k: failure[k] for k in ("passed", "qualification", "error_type", "Model_GO")}))
        return 1
    print(json.dumps({k: value[k] for k in ("passed", "qualification", "trainable_tensors",
                                           "trainable_parameters", "Model_GO")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
