"""Build both Gemma domain commands through the same pinned VLM entry."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from build_vlm_shared_resume_overlay import build_sources, pinned


ORIGINAL = (
    "code/bw1000/training-gemma-A100-native-contract-CPU-v2/original-launch.json",
    "6ff6fecf17d423e672c2c660a9f774d67e03a8a79ca17f388853d34ca42a15fc",
)
EXTRAS = {
    "vlm/plugins/gemma_domain_continuous.py": (
        "code/bw1000/training-gemma-usgs-shared-resume-v1/gemma_domain_continuous.py",
        "bdb42a51e576b52f90ce67c6d378fc4c73f8ce2c092b3599f259b1d21f9e5b4d",
    ),
    "vlm/runtime/vlm_update_contract.py": (
        "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/update_contract.py",
        "e973c0e48f9538996953922cfaf7e5c1f7c3a2aed81fb3f8be71c95d0afa6f95",
    ),
}


def within_own_root(path):
    return path.anchor == "/" and len(path.parts) > 1 and ".." not in path.parts


def candidate(repo, target, models, plugins, port):
    if not all(within_own_root(path) for path in (target, models, plugins)):
        raise ValueError("candidate requires concrete absolute paths without traversal")
    sources = build_sources(repo)
    for name, pin in EXTRAS.items():
        raw = pinned(repo, *pin)
        compile(raw, name, "exec")
        sources[name] = raw
    translator = "tools/hygon_vlm_recipe.py"
    scope = {"__name__": "gemma_domain_common_entry", "__file__": translator}
    exec(compile(sources[translator], translator, "exec"), scope)
    original = pinned(repo, *ORIGINAL)
    framework = target / "framework"
    dataset = target / "view/train800-swift.jsonl"
    checkpoint = target / "continuous/train/checkpoint-20"
    original_plugin = str(plugins / "swift_callback_nvidia_baseline_v2.py")

    plans = {}
    for arm, resume_path in (("continuous", None), ("resumed", checkpoint)):
        callback = "gemma_domain_resume_witness" if resume_path else "gemma_domain_continuous_witness"
        filename = "gemma_domain_resume.py" if resume_path else "gemma_domain_continuous.py"
        plan = scope["build_launch"](
            "T02.sft", original, framework / "vlm/runtime/swift_entry.py", models, dataset, plugins,
            target / arm / "train", port, diagnostic_plugin=framework / "vlm/plugins" / filename,
            diagnostic_callback=callback, diagnostic_callback_position="before", resume_from_checkpoint=resume_path,
        )
        argv = plan["command"]
        if not resume_path:
            argv.remove("nvidia_baseline")
        argv.remove(original_plugin)
        plan["required_assets"]["plugins"].remove(original_plugin)
        plan.update(domain_arm=arm, planned_optimizer_updates=5 if resume_path else 25,
                    expected_checkpoint_steps=[25] if resume_path else [10, 20, 25],
                    live_admission_verified=False, training_ready=False)
        plans[arm] = plan

    return sources, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", type=PurePosixPath, required=True)
    parser.add_argument("--models", type=PurePosixPath, required=True)
    parser.add_argument("--plugins", type=PurePosixPath, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources, plans = candidate(args.repo, args.target, args.models, args.plugins, args.port)
    args.output.mkdir(mode=0o700)
    for name, raw in sources.items():
        path = args.output / "overlay" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for arm, plan in plans.items():
        (args.output / f"{arm}-private.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"overlay_SHA256": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                      "arms": {arm: plan["planned_optimizer_updates"] for arm, plan in plans.items()},
                      "training_ready": False}))


if __name__ == "__main__":
    main()
