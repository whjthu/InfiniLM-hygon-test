"""Translate frozen A100 VLM launch recipes to a Hygon runtime path layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


A100_LAUNCH_SHA256 = {
    "T01.sft": "28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400",
    "T02.sft": "6ff6fecf17d423e672c2c660a9f774d67e03a8a79ca17f388853d34ca42a15fc",
    "T03.sft": "7fefdc16f5519a998a94fdef703c2dd326b3c15e2e31d5a3735f6c4558269c67",
    "T04.sft": "849e72c9dde3d3e4a7bd40a4c1ebc700854af944fa6fbf86317f8447f55f7c9b",
    "T05.sft": "015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789",
}

GENERATION_SOURCE = Path(__file__).resolve().parents[1] / "generation" / "src"
RUNTIME_PROBE = """
import json
import sys
from dataclasses import asdict
sys.path.insert(0, sys.argv[1])
from gencuda.runtime import inspect_runtime
print(json.dumps(asdict(inspect_runtime())))
"""
RANK_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "NNODES", "NODE_RANK", "MASTER_ADDR")


def absolute_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
        raise argparse.ArgumentTypeError("expected an absolute path on the target Linux host")
    return path


def replace_option(argv: list[str], option: str, value: PurePosixPath) -> None:
    if argv.count(option) != 1:
        raise ValueError(f"expected exactly one {option} in the frozen recipe")
    argv[argv.index(option) + 1] = str(value)


def build_launch(
    case: str,
    original: bytes,
    swift: PurePosixPath,
    models: PurePosixPath,
    dataset: PurePosixPath,
    plugins: PurePosixPath,
    output: PurePosixPath,
    master_port: int,
    *,
    diagnostic_plugin: PurePosixPath | list[PurePosixPath] | None = None,
    diagnostic_callback: str | list[str] | None = None,
    diagnostic_callback_position: str = "after",
    resume_from_checkpoint: PurePosixPath | None = None,
) -> dict:
    identity = hashlib.sha256(original).hexdigest()
    if identity != A100_LAUNCH_SHA256[case]:
        raise ValueError(f"{case}: launch SHA256 differs from the frozen A100 original")
    source = json.loads(original)
    argv = source["command"].copy()
    env = source["environment"].copy()
    if argv[1] != "sft" or PurePosixPath(argv[0]).name != "swift":
        raise ValueError("expected a Swift SFT recipe")
    if not 1024 <= master_port <= 65535:
        raise ValueError("master port must be between 1024 and 65535")
    if (diagnostic_plugin is None) != (diagnostic_callback is None):
        raise ValueError("diagnostic plugin and callback must be supplied together")
    if diagnostic_callback_position not in {"before", "after"}:
        raise ValueError("diagnostic callback position must be before or after")
    if diagnostic_plugin is None and diagnostic_callback_position != "after":
        raise ValueError("callback placement requires diagnostic observers")
    observers = []
    if diagnostic_plugin is not None:
        diagnostic_plugins = diagnostic_plugin if isinstance(diagnostic_plugin, list) else [diagnostic_plugin]
        diagnostic_callbacks = diagnostic_callback if isinstance(diagnostic_callback, list) else [diagnostic_callback]
        if not diagnostic_plugins or len(diagnostic_plugins) != len(diagnostic_callbacks):
            raise ValueError("diagnostic plugins and callbacks must have matching nonempty ordered lists")
        for plugin, callback in zip(diagnostic_plugins, diagnostic_callbacks):
            plugin = absolute_path(str(plugin))
            if not isinstance(callback, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", callback) is None:
                raise ValueError("diagnostic callback must be a registry identifier")
            observers.append({"plugin": str(plugin), "callback": callback})
        if (len({x["plugin"] for x in observers}) != len(observers)
                or len({x["callback"] for x in observers}) != len(observers)):
            raise ValueError("duplicate diagnostic plugin or callback")

    original_model = PurePosixPath(argv[argv.index("--model") + 1])
    model = models / original_model.name
    argv[0] = str(swift)
    replace_option(argv, "--model", model)
    replace_option(argv, "--dataset", dataset)
    replace_option(argv, "--output_dir", output)

    plugin_index = argv.index("--external_plugins") + 1
    if plugin_index >= len(argv) or argv[plugin_index].startswith("--"):
        raise ValueError("no plugin in the frozen recipe")
    while plugin_index < len(argv) and not argv[plugin_index].startswith("--"):
        argv[plugin_index] = str(plugins / PurePosixPath(argv[plugin_index]).name)
        plugin_index += 1
    if observers:
        original_plugins = argv[argv.index("--external_plugins") + 1:plugin_index]
        if any(x["plugin"] in original_plugins for x in observers):
            raise ValueError("diagnostic plugin duplicates an existing plugin")
        argv[plugin_index:plugin_index] = [x["plugin"] for x in observers]
        plugin_index += len(observers)
        if argv.count("--callbacks") != 1:
            raise ValueError("expected exactly one callback group in the frozen recipe")
        first_callback = callback_index = argv.index("--callbacks") + 1
        while callback_index < len(argv) and not argv[callback_index].startswith("--"):
            callback_index += 1
        if any(x["callback"] in argv[first_callback:callback_index] for x in observers):
            raise ValueError("diagnostic callback duplicates an existing callback")
        insertion = first_callback if diagnostic_callback_position == "before" else callback_index
        argv[insertion:insertion] = [x["callback"] for x in observers]

    if resume_from_checkpoint is not None:
        checkpoint = absolute_path(str(resume_from_checkpoint))
        if (case != "T05.sft" or len(observers) != 1
                or observers[0]["callback"] != "native_resume_witness"
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("MiniCPM full resume requires its single dedicated native witness")
        if (checkpoint == output or checkpoint in output.parents or output in checkpoint.parents
                or any(option in argv for option in (
                    "--resume_from_checkpoint", "--resume_only_model", "--ignore_data_skip", "--load_args"))):
            raise ValueError("resume output overlaps checkpoint or frozen resume flags already exist")
        argv.remove("nvidia_baseline")
        argv += ["--resume_from_checkpoint", str(checkpoint), "--resume_only_model", "false",
                 "--ignore_data_skip", "false", "--load_args", "false"]

    # Inherit the selected Hygon environment's PATH and loader configuration.
    del env["PATH"]
    env.update(
        MODELS_DIR=str(models), DATASET=str(dataset), OUTPUT_DIR=str(output),
        MASTER_PORT=str(master_port),
    )
    return {
        "case": case,
        "a100_launch_sha256": identity,
        "command": argv,
        "environment_overrides": env,
        "required_assets": {
            "model_directory": str(model), "dataset": str(dataset),
            "plugins": argv[argv.index("--external_plugins") + 1:plugin_index],
        },
        "qualification": ("diagnostic_recipe_rebound; no formal performance qualification"
                          if diagnostic_plugin is not None else
                          "recipe_rebound; target assets and runtime execution unverified"),
        "diagnostic_callback_position": diagnostic_callback_position,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "diagnostic_observer": (None if not observers else
            {**observers[0], "formal_performance": False} if len(observers) == 1 else
            {"observers": observers, "formal_performance": False}),
    }


def validate_diagnostic_outputs(launch: dict, environment: dict) -> None:
    """Keep optional diagnostic writers inside this launch's output scope."""
    observer = launch.get("diagnostic_observer") or {}
    callbacks = {item.get("callback") for item in observer.get("observers", [observer])}
    scope = PurePosixPath(launch["environment_overrides"]["OUTPUT_DIR"]).parent

    def require_path(value: str | None, expected: PurePosixPath) -> None:
        path = PurePosixPath(value or "")
        if not path.is_absolute() or ".." in path.parts or path != expected:
            raise ValueError("diagnostic output does not belong to this launch scope")

    if "native_lora_dropout_alignment" in callbacks:
        config_path = environment.get("NATIVE_LORA_DROPOUT_CONFIG")
        if not config_path:
            raise ValueError("missing native dropout diagnostic configuration")
        raw = Path(config_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != environment.get("NATIVE_LORA_DROPOUT_CONFIG_SHA256"):
            raise ValueError("native dropout diagnostic configuration identity differs")
        config = json.loads(raw)
        require_path(config.get("output"), scope / "masks")
    if "original_first_updates" in callbacks:
        require_path(environment.get("FIRST_UPDATE_CAPTURE"), scope / "capture")
    if launch.get("resume_from_checkpoint"):
        config_path = environment.get("NATIVE_RESUME_CONFIG")
        if not config_path:
            raise ValueError("missing native resume witness configuration")
        raw = Path(config_path).read_bytes()
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != environment.get("NATIVE_RESUME_CONFIG_SHA256"):
            raise ValueError("native resume witness configuration identity differs")
        config = json.loads(raw)
        if config["checkpoint"] != launch["resume_from_checkpoint"]:
            raise ValueError("resume witness checkpoint differs from actual command")
        checkpoint = Path(config["checkpoint"]).resolve(strict=True)
        output = Path(launch["environment_overrides"]["OUTPUT_DIR"]).resolve()
        if checkpoint == output or checkpoint in output.parents or output in checkpoint.parents:
            raise ValueError("resolved resume output overlaps original checkpoint")
        pins = config["checkpoint_files"]
        expected = {"trainer_state.json", "args.json", "adapter_model.safetensors",
                    "adapter_config.json", "optimizer.pt", "scheduler.pt"}
        expected.update("rng_state_%d.pth" % rank for rank in range(8))
        if set(pins) != expected:
            raise ValueError("resume manifest is missing full-state or rank RNG files")
        for name in sorted(expected):
            path, pin = checkpoint / name, pins[name]
            if (path.is_symlink() or type(pin["bytes"]) is not int
                    or not 0 < pin["bytes"] <= 256 * 1024**2
                    or path.stat().st_size != pin["bytes"]):
                raise ValueError("resume checkpoint file identity differs")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != pin["SHA256"]:
                raise ValueError("resume checkpoint SHA256 differs before runtime admission")


def run_swift(launch: dict, receipt_path: Path, *, dry_run: bool) -> int:
    def record(receipt, stage: str, **fields) -> None:
        receipt.write(json.dumps({"stage": stage, **fields}) + "\n")
        receipt.flush()

    with receipt_path.open("x", encoding="utf-8") as receipt:
        record(receipt, "planned", launch=launch, expected_accelerator="hip")
        if dry_run:
            return 0
        try:
            overrides = launch["environment_overrides"]
            environment = os.environ.copy()
            validate_diagnostic_outputs(launch, environment)
            inherited_rank = sorted(set(RANK_ENV).intersection(environment))
            if inherited_rank:
                raise ValueError(f"launch Swift from an unranked single-node parent; found {inherited_rank}")
            visible = overrides["CUDA_VISIBLE_DEVICES"]
            if environment.get("CUDA_VISIBLE_DEVICES", visible) != visible:
                raise ValueError("inherited CUDA_VISIBLE_DEVICES conflicts with the frozen eight-card mask")
            environment.update(overrides)
            swift, *_ = launch["command"]
            assets = launch["required_assets"]
            if not Path(swift).is_file():
                raise ValueError("missing Swift entry script")
            if not Path(assets["model_directory"]).is_dir():
                raise ValueError("missing model directory")
            for path in (assets["dataset"], *assets["plugins"]):
                if not Path(path).is_file():
                    raise ValueError(f"missing training input {path}")

            sys.path.insert(0, str(GENERATION_SOURCE))
            from gencuda.runtime import RuntimeInfo, validate_accelerator

            probe = subprocess.run(
                [sys.executable, "-c", RUNTIME_PROBE, str(GENERATION_SOURCE)],
                env=environment, capture_output=True, text=True, check=True, timeout=45,
            )
            observed = json.loads(probe.stdout)
            record(receipt, "runtime", observed=observed)
            validate_accelerator(RuntimeInfo(**observed), "hip", int(overrides["NPROC_PER_NODE"]))
            output = Path(overrides["OUTPUT_DIR"])
            output.mkdir(exist_ok=False)
            record(receipt, "running")
            result = subprocess.run([sys.executable, *launch["command"]], env=environment)
            record(receipt, "exited", returncode=result.returncode)
            return result.returncode
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            record(receipt, "failed", error=str(error))
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=A100_LAUNCH_SHA256)
    parser.add_argument("--a100-launch", type=Path, required=True)
    for name in ("swift", "models", "dataset", "plugins", "output"):
        parser.add_argument(f"--{name}", type=absolute_path, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--diagnostic-plugin", type=absolute_path, action="append")
    parser.add_argument("--diagnostic-callback", action="append")
    parser.add_argument("--diagnostic-callback-position", choices=("before", "after"), default="after",
                        help="place the ordered diagnostic callbacks before or after the original callbacks")
    parser.add_argument("--resume-from-checkpoint", type=absolute_path,
                        help="MiniCPM full-state diagnostic resume; requires native_resume_witness")
    parser.add_argument("--run", action="store_true", help="invoke the frozen Swift command")
    parser.add_argument("--dry-run", action="store_true", help="record the command without starting training")
    parser.add_argument("--receipt", type=Path, help="new JSONL receipt path in an existing directory")
    args = parser.parse_args(argv)
    if args.run and args.receipt is None:
        parser.error("--run requires --receipt")
    if not args.run and (args.dry_run or args.receipt is not None):
        parser.error("--dry-run and --receipt require --run")
    try:
        launch = build_launch(
            args.case, args.a100_launch.read_bytes(), args.swift, args.models,
            args.dataset, args.plugins, args.output, args.master_port,
            diagnostic_plugin=args.diagnostic_plugin,
            diagnostic_callback=args.diagnostic_callback,
            diagnostic_callback_position=args.diagnostic_callback_position,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        if args.run:
            return run_swift(launch, args.receipt, dry_run=args.dry_run)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(launch, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
