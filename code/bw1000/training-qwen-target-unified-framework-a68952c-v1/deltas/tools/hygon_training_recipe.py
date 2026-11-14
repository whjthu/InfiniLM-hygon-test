"""Build Hygon training launch plans from the frozen A100 originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import hygon_vlm_recipe as vlm


# (launch SHA256, gencuda model, action, original card count)
GENERATION = {
    "T06.sft": ("514ab79beec4cdd16ba5035f97086e4afc254ab67df35022c1d5c9eff1b66f02", "sdxl-1.0", "train", 1),
    "T06.ddpo": ("a34a845ac9d76aaa6183dba700bf6ad8bb94d667eb8344425d648c5626ef956d", "sdxl-1.0", "rl", 1),
    "T07.sft": ("f0bc54e478ad0e8b5d9e0b7fe632a172e5aa6d2e9d6fa166b6d09d280ecdc89d", "z-image-turbo", "train", 1),
    "T07.flow-grpo": ("d02a7e8c2ca1a2081d6a4408df3cc0b2e7f069ee0a9386e476c7b3d38de7471c", "z-image-turbo", "rl", 1),
    "T08.flow-matching-lora": ("bbbbb046a8ba2ed39dc00d54ae9254122345fe5f3dd97ea47b825a5f2b8631af", "hy-motion-1.0", "train", 1),
    "T08.motion-rft": ("de543cd5c66e3822b97c6b8ae6d4f2901dc26bb3b0f7bc93b9423c06ae96cb2f", "hy-motion-1.0", "rl", 1),
    "T09.sft": ("3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f", "hunyuanvideo-1.5", "train", 1),
    "T09.dp2": ("181b1ca706a7d62741298192c3391a0336acf0c60624aa9f2645e90a55934a6f", "hunyuanvideo-1.5", "train", 2),
    "T09.sage-grpo": ("6b230bb6ac909a4599faa37fc9a20e85d425c97e8c643ca111b3f124ef16c5a3", "hunyuanvideo-1.5", "rl", 2),
    "T10.shape-flow-lora": ("d5b23154d64af63332e54e079d6b0fda0a2904f53bdd0e6a0541fa265fad29fa", "trellis-shape-flow-1.3b", "train", 1),
    "T11.material-head": ("a1a2c1ec43825856065600c313e148fffc23c8ff6ef97534aef8a31113529287", "stable-fast-3d", "train", 1),
    "T12.camera-head": ("206dfea1f2d30487f9466af40b8d6d71e64162640b4b900254e6138b1962bbfa", "vggt-1b", "train", 1),
}

# Checkout names consumed by gencuda's normal training and RL routes.
SOURCE_SUBDIR = {
    "sdxl-1.0": "diffusers-v039",
    "z-image-turbo": "diffsynth-studio-grpo",
    "hy-motion-1.0": "hy-motion",
    "hunyuanvideo-1.5": "hunyuanvideo-1.5",
    "trellis-shape-flow-1.3b": "trellis2",
    "stable-fast-3d": "stable-fast-3d",
    "vggt-1b": "vggt",
}


def is_asset(value: object) -> bool:
    return isinstance(value, str) and value.startswith("/")


def asset_paths(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from asset_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from asset_paths(item)
    elif is_asset(value):
        yield value


def bind_paths(value: object, bindings: dict[str, str]):
    if isinstance(value, dict):
        return {key: bind_paths(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [bind_paths(item, bindings) for item in value]
    return bindings[value] if is_asset(value) else value


def build_generation_plan(case: str, launch: bytes, bindings: dict | None) -> dict:
    identity, model, action, cards = GENERATION[case]
    if hashlib.sha256(launch).hexdigest() != identity:
        raise ValueError(f"{case}: launch SHA256 differs from the frozen A100 original")
    original = json.loads(launch)
    if case == "T12.camera-head":
        # Its original artifact is the config itself, not a launch wrapper.
        recipe = {"config": original}
    else:
        recipe = {"config": original["config"], "weights": original["weights"]}
        if is_asset(original.get("source")):
            recipe["source"] = original["source"]

    assets = set(asset_paths(recipe))
    result = {
        "case": case, "a100_launch_sha256": identity,
        "model": model, "action": action, "cards": cards,
        "backend_entrypoint": "gencuda", "backend_options": ["--accelerator", "hip"],
        "original_assets": sorted(assets),
    }
    if case == "T08.flow-matching-lora":
        result["required_training_frames"] = 128
        result["required_feature_dim"] = 201
    if bindings is None:
        result["qualification"] = "assets listed; target assets and runtime unverified"
        return result
    if not isinstance(bindings, dict) or set(bindings) != assets:
        raise ValueError("asset map must contain exactly the original absolute paths")
    if not all(isinstance(target, str) for target in bindings.values()):
        raise ValueError("asset targets must be absolute Linux paths")
    bindings = {key: str(vlm.absolute_path(target)) for key, target in bindings.items()}
    if len(set(bindings.values())) != len(bindings):
        raise ValueError("distinct original assets cannot share a target path")
    result.update(bind_paths(recipe, bindings))
    result["qualification"] = "recipe rebound; target bytes and runtime unverified"
    return result


def verify_motion_manifest(plan: dict) -> dict | None:
    """Reject a different temporal workload before importing the device backend."""
    if plan["case"] != "T08.flow-matching-lora":
        return None
    config = plan["config"]
    path = Path(config["training_data_manifest"])
    if not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise ValueError("HY-Motion requires its pinned training data manifest")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config["training_data_manifest_sha256"]:
        raise ValueError("HY-Motion training data manifest SHA256 differs from the frozen recipe")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 2
            or manifest.get("format") != "hy-motion-training-shards-v1"):
        raise ValueError("HY-Motion training data manifest schema differs")
    frames, features = manifest.get("frames"), manifest.get("feature_dim")
    if type(frames) is not int or frames != 128:
        raise ValueError(
            f"HY-Motion requires 128 frames; the pinned manifest declares {frames}. "
            "Sample count does not establish frame count; resolve the recipe and A100 reference first."
        )
    if type(features) is not int or features != 201:
        raise ValueError("HY-Motion requires the complete 201-dimensional motion target")
    return {"SHA256": digest, "frames": frames, "feature_dim": features,
            "qualification": "Manifest metadata only; tensor values and model execution unverified"}


def run_generation(plan: dict, options: argparse.Namespace) -> int:
    if options.asset_map is None:
        raise ValueError("--run requires --asset-map")
    for name in ("cards", "output_root", "run_name", "lock_root", "lock_host"):
        if not getattr(options, name):
            raise ValueError(f"--run requires --{name.replace('_', '-')}")
    if not options.dry_run:
        verify_motion_manifest(plan)

    # Select the backend from this checkout, even when another gencuda is installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation" / "src"))
    from gencuda import cli

    if len(cli.parse_cards(options.cards)) != plan["cards"]:
        raise ValueError(f"{plan['case']} requires exactly {plan['cards']} card(s)")
    if options.vggt_weights is not None and plan["case"] != "T12.camera-head":
        raise ValueError("--vggt-weights is only valid for T12.camera-head")
    weights = plan.get("weights") or options.vggt_weights
    if weights is None:
        raise ValueError("T12.camera-head requires --vggt-weights")

    source_args = []
    if plan["case"] != "T06.ddpo":
        if options.source_root is None:
            raise ValueError(f"{plan['case']} requires --source-root")
        expected = options.source_root / SOURCE_SUBDIR[plan["model"]]
        if "source" in plan and expected != vlm.absolute_path(plan["source"]):
            raise ValueError(f"bound source must match the backend checkout {expected}")
        source_args = ["--source-root", str(options.source_root)]

    with tempfile.TemporaryDirectory(prefix="hygon-training-") as work:
        config = Path(work) / "recipe.json"
        config.write_text(json.dumps(plan["config"]), encoding="utf-8")
        args = cli.build_parser().parse_args([
            plan["action"], "--model", plan["model"], "--config", str(config),
            "--weights", str(weights), "--cards", options.cards,
            "--output-root", str(options.output_root), "--run-name", options.run_name,
            "--lock-root", str(options.lock_root), "--lock-host", options.lock_host,
            *plan["backend_options"], *source_args, *(["--dry-run"] if options.dry_run else []),
        ])
        return int(args.handler(args))


def main(argv: list[str] | None = None, *, generation_accelerator: str = "hip") -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in vlm.A100_LAUNCH_SHA256:
        return vlm.main(args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["list", *GENERATION])
    parser.add_argument("--a100-launch", type=Path)
    parser.add_argument("--asset-map", type=Path)
    parser.add_argument("--run", action="store_true", help="invoke the registered generation backend")
    parser.add_argument("--dry-run", action="store_true", help="record a backend plan without starting training")
    parser.add_argument("--cards", help="comma-separated device IDs")
    parser.add_argument("--run-name")
    parser.add_argument("--lock-host")
    for name in ("output-root", "lock-root", "source-root", "vggt-weights"):
        parser.add_argument(f"--{name}", type=vlm.absolute_path)
    parser.add_argument("--resume-from-checkpoint", type=vlm.absolute_path,
                        help="resume HunyuanVideo SFT from a native checkpoint file or DP2 from a committed checkpoint directory")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--capture-first-policy", action="store_true",
                        help="capture the first SAGE-GRPO rollout and policy update for numerical diagnostics")
    capture.add_argument("--capture-first-policy-only", action="store_true",
                         help="stop after recording the first SAGE-GRPO policy update")
    parser.add_argument("--replay-first-trajectory", type=vlm.absolute_path)
    parser.add_argument("--replay-first-trajectory-sha256")
    parser.add_argument("--sage-vae-dir", type=vlm.absolute_path,
                        help="use a separately staged VAE directory for T09.sage-grpo")
    parser.add_argument("--sage-scheduler-root", type=vlm.absolute_path,
                        help="use a separately staged SAGE checkout for T09.sage-grpo")
    parser.add_argument("--trace-first-actor-blocks", action="store_true",
                        help="fingerprint checkpointed actor block outputs and their gradients on CUDA")
    parser.add_argument("--trace-first-actor-detail-block", type=int,
                        help="also fingerprint attention and MLP boundaries in one zero-based double-stream block")
    options = parser.parse_args(args)
    if options.capture_first_policy_only:
        options.capture_first_policy = True
    if options.capture_first_policy and (options.case != "T09.sage-grpo" or not options.run):
        parser.error("--capture-first-policy requires T09.sage-grpo --run")
    if (options.replay_first_trajectory is None) != (options.replay_first_trajectory_sha256 is None):
        parser.error("fixed trajectory replay requires both the path and its SHA256")
    if options.replay_first_trajectory is not None:
        if options.case != "T09.sage-grpo" or not options.run or options.capture_first_policy:
            parser.error("fixed trajectory replay requires T09.sage-grpo --run without capture")
        digest = options.replay_first_trajectory_sha256
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            parser.error("fixed trajectory replay requires a lowercase SHA256")
    if (options.sage_vae_dir is not None or options.sage_scheduler_root is not None) and (
        options.case != "T09.sage-grpo" or not options.run or generation_accelerator != "hip"
    ):
        parser.error("separate SAGE assets require T09.sage-grpo --run on HIP")
    if options.trace_first_actor_blocks and (
        generation_accelerator != "cuda" or not (
            options.capture_first_policy_only or options.replay_first_trajectory is not None
        )
    ):
        parser.error("--trace-first-actor-blocks requires CUDA and capture-only or fixed replay")
    if options.trace_first_actor_detail_block is not None and (
        options.trace_first_actor_detail_block < 0 or not options.trace_first_actor_blocks
        or options.replay_first_trajectory is None
    ):
        parser.error("--trace-first-actor-detail-block requires block tracing and fixed replay")
    if options.resume_from_checkpoint is not None:
        if options.case not in ("T09.sft", "T09.dp2", "T09.sage-grpo") or not options.run:
            parser.error("--resume-from-checkpoint requires --run with a T09 use case")
        if options.capture_first_policy or options.replay_first_trajectory is not None:
            parser.error("--resume-from-checkpoint cannot be combined with first-policy diagnostics")
    if options.dry_run and not options.run:
        parser.error("--dry-run requires --run")
    if options.case == "list":
        if options.run:
            parser.error("list does not accept --run")
        print(json.dumps([*vlm.A100_LAUNCH_SHA256, *GENERATION], indent=2))
        return 0
    if options.a100_launch is None:
        parser.error("--a100-launch is required for generation cases")
    try:
        mapping = json.loads(options.asset_map.read_bytes()) if options.asset_map else None
        plan = build_generation_plan(options.case, options.a100_launch.read_bytes(), mapping)
        if generation_accelerator not in ("cuda", "hip"):
            raise ValueError("unsupported generation accelerator")
        plan["backend_options"] = ["--accelerator", generation_accelerator]
        if options.capture_first_policy:
            if options.asset_map is None:
                raise ValueError("--capture-first-policy requires --asset-map")
            plan["config"]["capture_first_policy"] = True
            if options.capture_first_policy_only:
                plan["config"]["capture_first_policy_only"] = True
        if options.replay_first_trajectory is not None:
            if options.asset_map is None:
                raise ValueError("fixed trajectory replay requires --asset-map")
            plan["config"].update(replay_first_trajectory=str(options.replay_first_trajectory),
                                  replay_first_trajectory_sha256=options.replay_first_trajectory_sha256)
        if options.sage_vae_dir is not None:
            plan["config"]["sage_vae_dir"] = str(options.sage_vae_dir)
        if options.sage_scheduler_root is not None:
            plan["config"]["sage_scheduler_root"] = str(options.sage_scheduler_root)
        if options.trace_first_actor_blocks:
            plan["config"]["trace_first_actor_blocks"] = True
            if options.trace_first_actor_detail_block is not None:
                plan["config"]["trace_first_actor_detail_block"] = options.trace_first_actor_detail_block
        if options.resume_from_checkpoint is not None:
            if options.asset_map is None:
                raise ValueError("--resume-from-checkpoint requires --asset-map")
            if plan["config"].get("resume_from_checkpoint") is not None:
                raise ValueError("the frozen launch already has a resume checkpoint")
            plan["config"]["resume_from_checkpoint"] = str(options.resume_from_checkpoint)
        if options.run:
            return run_generation(plan, options)
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
