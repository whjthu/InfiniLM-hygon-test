from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import yaml

from .manifest import mark_failed, merge_backend_result, new_manifest, sha256_path, write_manifest
from .lease import acquire, lease_location, parse_cards
from .models import ModelSpec, get_spec, resolve_weights
from .measurement import consume_backend_window, stamp_completed_measurement, stamp_failed_measurement
from .runtime import inspect_runtime, validate_accelerator
from .recovery_validation import validate_recovery_enabled
from .trellis_output_contract import validate_output_contract


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKTREE_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODELS = Path(str(files("gencuda").joinpath("data/models")))
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
EXTERNAL_RESUME_ROUTES = frozenset({
    ("train", "sdxl-1.0"),
    ("train", "vggt-1b"),
    ("train", "hunyuanvideo-1.5"),
    ("train", "hy-motion-1.0"),
    ("train", "z-image-turbo"),
    ("train", "stable-fast-3d"),
    ("train", "trellis-shape-flow-1.3b"),
    ("rl", "sdxl-1.0"),
    ("rl", "hunyuanvideo-1.5"),
    ("rl", "hy-motion-1.0"),
    ("rl", "z-image-turbo"),
})


def validate_resume_request(command: str, model: str, config: dict[str, Any]) -> None:
    checkpoint = config.get("resume_from_checkpoint")
    if checkpoint is None:
        return
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("resume_from_checkpoint must be a non-empty path string")
    # Never turn an unsupported continuation request into a fresh training run.
    if (command, model) not in EXTERNAL_RESUME_ROUTES:
        raise ValueError(f"external checkpoint resume is not supported for {model} {command}")
    path = Path(checkpoint.strip()).expanduser()
    if (command, model) == ("train", "hunyuanvideo-1.5") and config.get("parallel_mode") == "data_parallel":
        if not path.is_dir() or not (path / "COMMITTED.json").is_file():
            raise ValueError("DP2 resume requires a committed v2 checkpoint directory")
        return
    if (command, model) == ("train", "sdxl-1.0"):
        if not re.fullmatch(r"checkpoint-\d+", path.name) or not (
            path / "pytorch_lora_weights.safetensors"
        ).is_file():
            raise ValueError("SDXL resume_from_checkpoint must name a checkpoint-N directory with an adapter")
        return
    checkpoint_path = path / "state.pt" if path.is_dir() else path
    if not checkpoint_path.is_file():
        raise ValueError("resume_from_checkpoint must name an existing checkpoint file")


def preflight_external_resume(command: str, model: str, config: dict[str, Any]) -> None:
    """Reject malformed full-state checkpoints before creating a run."""
    if (command, model) == ("train", "hunyuanvideo-1.5") and config.get("parallel_mode") == "data_parallel":
        if config.get("resume_from_checkpoint") is not None:
            from .hunyuanvideo_distributed_checkpoint import load_committed
            load_committed(Path(config["resume_from_checkpoint"]).expanduser().resolve())
        return
    if (command, model) == ("train", "trellis-shape-flow-1.3b"):
        if config.get("resume_from_checkpoint"):
            from .trellis_training_resume_checkpoint import load_checkpoint
            load_checkpoint(Path(config["resume_from_checkpoint"]), requested_config=config)
        return
    if (command, model) != ("train", "z-image-turbo"):
        return
    value = config.get("resume_from_checkpoint")
    if value is None:
        return
    path = Path(value).expanduser()
    checkpoint = path / "state.pt" if path.is_dir() else path
    from . import zimage_training_resume_checkpoint as checkpoint_format

    try:
        payload = checkpoint_format.load_checkpoint(checkpoint)
    except Exception as error:
        raise checkpoint_format.CheckpointValidationError(
            f"cannot load Z-Image training resume checkpoint: {checkpoint}"
        ) from error
    checkpoint_format.preflight_checkpoint(payload, requested_config=config)


def load_run_config(path: Path, model: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"configuration is missing: {path}")
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    value = value or {}
    if not isinstance(value, dict):
        raise RuntimeError("configuration root must be a mapping")
    declared = value.get("model")
    if declared is not None and declared != model:
        raise RuntimeError(f"configuration model {declared!r} does not match {model!r}")
    validate_recovery_enabled(value)
    if model == "trellis-2-4b":
        validate_output_contract(value)
    if model == "trellis-shape-flow-1.3b":
        from .trellis_training_contract import resolve_config
        value = resolve_config(value)
    return value


def create_run_dir(output_root: Path, model: str, command: str, run_name: str) -> Path:
    if not RUN_NAME.fullmatch(run_name):
        raise ValueError("run name must contain only letters, digits, '.', '_' or '-'")
    resolved_root = output_root.expanduser().resolve()
    try:
        resolved_root.relative_to(WORKTREE_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output root must be external to the delivery worktree")
    resolved_root.mkdir(parents=True, exist_ok=True)
    for index, part in enumerate((model, command, run_name)):
        candidate = resolved_root / part if index == 0 else candidate / part
        if candidate.exists():
            if candidate.is_symlink():
                raise ValueError(f"run path component must not be a symlink: {candidate}")
            if index == 2:
                raise FileExistsError(f"refusing to overwrite existing run: {candidate}")
            if not candidate.is_dir():
                raise ValueError(f"run path component is not a directory: {candidate}")
        else:
            candidate.mkdir()
        final_path = candidate.resolve()
        try:
            final_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"run path escapes the output root: {candidate}") from exc
        try:
            final_path.relative_to(WORKTREE_ROOT.resolve())
        except ValueError:
            continue
        raise ValueError(f"run path resolves into the delivery worktree: {candidate}")
    return candidate


def _source_dir(root: Path | None, name: str) -> Path:
    if root is None:
        raise RuntimeError("--source-root is required for this command")
    path = root / name
    if not path.is_dir():
        raise RuntimeError(f"required official source is missing: {path}")
    return path


def _call(module: str, name: str, *args: Any, **kwargs: Any) -> Any:
    # Import only the selected backend: optional model stacks must not affect --help.
    function: Callable[..., Any] = getattr(importlib.import_module(module), name)
    return function(*args, **kwargs)


def configure_visible_cards(cards: tuple[int, ...]) -> None:
    requested = ",".join(str(card) for card in cards)
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
        return
    if configured != requested:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES is already configured as {configured!r}; "
            f"refusing to switch it to {requested!r} in this process"
        )


def run_backend(
    command: str, spec: ModelSpec, weights: Path, config: dict[str, Any], directory: Path,
    source_root: Path | None,
) -> dict[str, Any]:
    validate_resume_request(command, spec.name, config)
    if command == "eval":
        raise RuntimeError("full evaluation is not implemented; no evaluation claim is produced")
    if command == "infer":
        if spec.name in {"sdxl-1.0", "z-image-turbo"}:
            return _call("gencuda.backends.diffusers_image", "run_image_inference", spec.name, weights, config, directory)
        routes = {
            "hunyuanvideo-1.5": ("hunyuanvideo_inference", "run_hunyuanvideo_inference", "hunyuanvideo-1.5"),
            "hy-motion-1.0": ("hy_motion_inference", "run_hy_motion_inference", "hy-motion"),
            "trellis-2-4b": ("trellis2_generation", "run_trellis2_inference", "trellis2"),
            "vggt-1b": ("vggt_geometry", "run_vggt_inference", "vggt"),
            "stable-fast-3d": ("sf3d_reconstruction", "run_sf3d_inference", "stable-fast-3d"),
        }
    elif command == "train":
        if spec.name == "hunyuanvideo-1.5" and config.get("parallel_mode") == "data_parallel":
            from .backends.hunyuanvideo_data_parallel import validate_source_root
            if source_root is None:
                raise RuntimeError("--source-root is required for DP2")
            selected = validate_source_root(source_root / "hunyuanvideo-1.5", approved_parent=source_root)
            return _call("gencuda.backends.hunyuanvideo_data_parallel", "run_hunyuanvideo_data_parallel",
                         weights, config, directory, selected)
        if spec.name == "hunyuanvideo-1.5" and config.get("parallel_mode") not in (None, "single_rank"):
            raise ValueError("unsupported HunyuanVideo parallel_mode")
        routes = {
            "sdxl-1.0": ("sdxl_training", "run_sdxl_training", "diffusers-v039"),
            "z-image-turbo": ("zimage_training", "run_zimage_training", "diffsynth-studio-grpo"),
            "hunyuanvideo-1.5": ("hunyuanvideo_training_walltime", "run_hunyuanvideo_training", "hunyuanvideo-1.5"),
            "hy-motion-1.0": ("hy_motion_training", "run_hy_motion_training", "hy-motion"),
            "vggt-1b": ("vggt_training", "run_vggt_training", "vggt"),
            "stable-fast-3d": ("sf3d_training", "run_sf3d_training", "stable-fast-3d"),
            "trellis-shape-flow-1.3b": ("trellis2_training", "run_trellis2_training", "trellis2"),
        }
    elif command == "rl":
        routes = {
            "sdxl-1.0": ("sdxl_ddpo", "run_sdxl_ddpo", None),
            "z-image-turbo": ("zimage_flow_grpo", "run_zimage_flow_grpo", "diffsynth-studio-grpo"),
            "hunyuanvideo-1.5": ("hunyuanvideo_sage_grpo", "run_hunyuanvideo_sage_grpo", "hunyuanvideo-1.5"),
            "hy-motion-1.0": ("hy_motion_motionrft", "run_hy_motion_motionrft", "hy-motion"),
        }
    else:
        raise ValueError(f"unknown command: {command}")
    if spec.name not in routes:
        raise RuntimeError(f"{command} is not implemented for {spec.name}")
    module, function, source = routes[spec.name]
    if command == "infer" and spec.name == "vggt-1b":
        return _call(
            f"gencuda.backends.{module}", function, weights, config, directory,
            source_root=_source_dir(source_root, source),
        )
    if command == "rl" and spec.name == "hunyuanvideo-1.5":
        scheduler_override = config.get("sage_scheduler_root")
        if scheduler_override is None:
            scheduler_root = _source_dir(source_root, "sage-grpo")
        else:
            scheduler_root = Path(scheduler_override)
            if not scheduler_root.is_absolute():
                raise ValueError("separate SAGE scheduler requires an absolute path")
            if not (scheduler_root / "hyvideo/schedulers/scheduling_flow_match_discrete.py").is_file():
                raise ValueError("separately staged SAGE scheduler file is missing")
        return _call(f"gencuda.backends.{module}", function, weights, config, directory,
                     _source_dir(source_root, source), scheduler_root)
    if command == "rl" and spec.name == "hy-motion-1.0":
        return _call(f"gencuda.backends.{module}", function, weights, config, directory, _source_dir(source_root, source), _source_dir(source_root, "motionrft"))
    if source is None:
        return _call(f"gencuda.backends.{module}", function, weights, config, directory)
    return _call(f"gencuda.backends.{module}", function, weights, config, directory, _source_dir(source_root, source))


def normalize_backend_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(f"backend returned unsupported result type: {type(result).__name__}")
    return result


def failed_backend_checks(model: str, checks: dict[str, Any]) -> list[str]:
    capability_limits = {"formal_output_contract"} if model == "trellis-2-4b" else set()
    return sorted(
        name for name, value in checks.items()
        if value is False and name not in capability_limits
    )


def cmd_run(args: argparse.Namespace) -> int:
    if args.command == "eval":
        raise RuntimeError("full evaluation is not implemented; no evaluation claim is produced")
    spec = get_spec(Path(args.models_config), args.model)
    if spec.name == "trellis-shape-flow-1.3b" and args.command != "train":
        raise ValueError("trellis-shape-flow-1.3b supports component training only")
    config = load_run_config(Path(args.config), spec.name)
    if args.command == "train" and spec.name == "hunyuanvideo-1.5":
        config["cards"] = list(parse_cards(args.cards))
        if config.get("parallel_mode") == "data_parallel":
            from .backends.hunyuanvideo_data_parallel import bind_visible_cards, validate_parent_launch, validate_source_root
            validate_parent_launch(config)
            if not args.source_root:
                raise RuntimeError("--source-root is required for DP2")
            approved_parent = Path(args.source_root)
            validate_source_root(approved_parent / "hunyuanvideo-1.5", approved_parent=approved_parent)
            bind_visible_cards(config)
        elif config.get("parallel_mode") not in (None, "single_rank"):
            raise ValueError("unsupported HunyuanVideo parallel_mode")
    validate_resume_request(args.command, spec.name, config)
    preflight_external_resume(args.command, spec.name, config)
    if args.command == "train" and spec.name in {"hunyuanvideo-1.5", "hy-motion-1.0", "z-image-turbo", "trellis-shape-flow-1.3b"}:
        required_cards = 2 if spec.name == "hunyuanvideo-1.5" and config.get("parallel_mode") == "data_parallel" else 1
        if len(parse_cards(args.cards)) != required_cards:
            count_word = "one" if required_cards == 1 else "two"
            raise ValueError(f"native {spec.name} training requires exactly {count_word} card(s)")
    config["model"] = spec.name
    config["run_name"] = args.run_name
    config["output_root"] = str(Path(args.output_root))
    expected_accelerator = getattr(args, "accelerator", None)
    runtime_plan = {"inspection": "deferred"}
    if expected_accelerator is not None:
        runtime_plan["expected_accelerator"] = expected_accelerator
    if args.dry_run:
        directory = create_run_dir(Path(args.output_root), spec.name, args.command, args.run_name)
        manifest_path = directory / "manifest.json"
        manifest = new_manifest(args.command, spec.name, config, runtime_plan)
        manifest["status"] = "planned"
        manifest["execution"]["status"] = "planned"
        write_manifest(manifest_path, manifest)
        print(manifest_path)
        return 0
    cards = parse_cards(args.cards)
    config["cards"] = list(cards)
    directory = create_run_dir(Path(args.output_root), spec.name, args.command, args.run_name)
    manifest_path = directory / "manifest.json"
    manifest = new_manifest(args.command, spec.name, config, runtime_plan)
    manifest["status"] = "running"
    manifest["execution"]["status"] = "running"
    manifest["execution"]["stage"] = "initializing"
    write_manifest(manifest_path, manifest)
    stage = "initializing"
    t0: float | None = None
    try:
        stage = "resolve_weights"
        weights = resolve_weights(
            spec,
            Path(args.model_root) if args.model_root else None,
            Path(args.weights) if args.weights else None,
        )
        stage = "configure_lease"
        lock_root, lock_host = lease_location(args.lock_root, args.lock_host)
        configure_visible_cards(cards)
        with acquire(cards, lock_root, host=lock_host):
            if args.command == "infer" and spec.name == "trellis-2-4b" and config.get("output_contract") == "full_mesh":
                stage = "cpu_texture_preflight"
                from .backends.trellis2_generation import preflight_cpu_texture_export
                preflight_cpu_texture_export(config, directory)
            stage = "inspect_runtime"
            runtime = inspect_runtime()
            manifest["runtime"] = runtime.__dict__.copy()
            if expected_accelerator is not None:
                manifest["runtime"]["expected_accelerator"] = expected_accelerator
                validate_accelerator(runtime, expected_accelerator, len(cards))
            manifest["execution"]["stage"] = "running"
            write_manifest(manifest_path, manifest)
            stage = "run_backend"
            t0 = perf_counter()
            result = normalize_backend_result(run_backend(
                args.command, spec, weights, config, directory,
                Path(args.source_root) if args.source_root else None,
            ))
            t1 = perf_counter()
            window = consume_backend_window(
                result,
                recovery_enabled=validate_recovery_enabled(config),
                t0=t0,
                t1=t1,
            )
            stage = "validate_result"
            if args.command == "train" and spec.name == "hunyuanvideo-1.5" and config.get("parallel_mode") == "data_parallel":
                from .backends.hunyuanvideo_data_parallel import validate_dp_result
                validate_dp_result(result, resumed=config.get("resume_from_checkpoint") is not None)
            merge_backend_result(manifest, result)
            if args.command == "train" and spec.name == "hunyuanvideo-1.5" and config.get("parallel_mode") == "data_parallel":
                # DP checks have a closed schema; do not retain legacy nullable defaults.
                manifest["checks"] = dict(result["checks"])
            failed_checks = failed_backend_checks(spec.name, result["checks"])
            if failed_checks:
                manifest["execution"]["failed_checks"] = failed_checks
                raise RuntimeError(f"backend checks failed: {', '.join(failed_checks)}")
            stage = "write_completed_manifest"
            write_manifest(manifest_path, manifest)
            t3 = perf_counter()
            stamp_completed_measurement(
                manifest,
                window,
                recovery_enabled=validate_recovery_enabled(config),
                t0=t0,
                t1=t1,
                t3=t3,
            )
            stage = "write_measurement_metadata"
            write_manifest(manifest_path, manifest)
    except BaseException as exc:
        mark_failed(manifest, stage, exc)
        stamp_failed_measurement(manifest)
        write_manifest(manifest_path, manifest)
        raise
    print(manifest_path)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .models import load_specs

    print(json.dumps(sorted(load_specs(Path(args.models_config)),), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gencuda")
    parser.add_argument("--models-config", default=DEFAULT_MODELS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    listed = subparsers.add_parser("models")
    listed.set_defaults(handler=cmd_models)
    for command in ("infer", "train", "rl", "eval"):
        item = subparsers.add_parser(command)
        item.add_argument("--model", required=True)
        item.add_argument("--config", required=True)
        item.add_argument("--output-root", required=True)
        item.add_argument("--run-name", required=True)
        item.add_argument("--cards", required=True, help="comma-separated device IDs")
        item.add_argument("--accelerator", choices=("cuda", "hip"), help="require the selected PyTorch runtime and exact visible device count")
        item.add_argument("--lock-root")
        item.add_argument("--lock-host")
        item.add_argument("--model-root")
        item.add_argument("--weights")
        item.add_argument("--source-root")
        item.add_argument("--dry-run", action="store_true")
        item.set_defaults(handler=cmd_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
