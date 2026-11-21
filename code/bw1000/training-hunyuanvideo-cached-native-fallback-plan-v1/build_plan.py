"""Reconstruct the frozen common T09 plan locally; never dispatch or load tensors."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ARCHIVE = "code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz"
ARCHIVE_SHA = "85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f"
MANIFEST = "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json"
MANIFEST_SHA = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
RECIPE = "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/deltas/tools/hygon_training_recipe.py"
VLM = "code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py"
LAUNCH = "code/bw1000/training-hunyuanvideo-sft-native-replay-v2/ORIGINAL_LAUNCH.json"
R = "/public/home/aclsylqidf/hygon-inference-refactor-20261014"
P = R + "/results/"


def need(value, message):
    if not value:
        raise ValueError(message)


def pinned(path, digest, size=None):
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest and (size is None or len(raw) == size),
         "Pinned local source differs: " + path.name)
    return raw


def build(root):
    inventory = json.loads(pinned(root / MANIFEST, MANIFEST_SHA, 112315))["files"]
    module_sources = {}
    for filename, relative in [("hygon_vlm_recipe", VLM), ("hygon_training_recipe", RECIPE)]:
        pin = inventory["tools/" + filename + ".py"]
        module_sources[filename] = pinned(root / relative, pin["SHA256"], pin["bytes"])
    launch = pinned(root / LAUNCH,
                    "3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f")
    original = json.loads(launch)
    bindings = {
        original["weights"]: P + "training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5",
        original["source"]: P + "training-hunyuanvideo-original-source-v1/source/hunyuanvideo-1.5",
        original["config"]["training_cache"]: P + "training-hunyuanvideo-original-inputs-v1/sample.pt",
    }
    # Execute only these two pinned stdlib recipe modules, not a model/provider.
    previous = sys.modules.get("hygon_vlm_recipe")
    try:
        with tempfile.TemporaryDirectory(prefix="t09-pinned-plan-") as folder:
            loaded = {}
            for name, body in module_sources.items():
                path = Path(folder) / (name + ".py")
                path.write_bytes(body)
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                if name == "hygon_vlm_recipe":
                    sys.modules[name] = module
                spec.loader.exec_module(module)
                loaded[name] = module
            compiler = loaded["hygon_training_recipe"]
            plan = compiler.build_generation_plan("T09.sft", launch, bindings)
            rejected_incomplete = False
            try:
                compiler.build_generation_plan("T09.sft", launch, {})
            except ValueError:
                rejected_incomplete = True
            need(rejected_incomplete, "Native compiler accepted an incomplete binding")
    finally:
        if previous is None:
            sys.modules.pop("hygon_vlm_recipe", None)
        else:
            sys.modules["hygon_vlm_recipe"] = previous
    pinned(root / ARCHIVE, ARCHIVE_SHA, 738802)
    paths = ["generation/src/gencuda/backends/hunyuanvideo_training.py",
             "generation/src/gencuda/hunyuanvideo_resume_checkpoint.py"]
    native_pins = {}
    with tarfile.open(root / ARCHIVE) as archive:
        for name in paths:
            raw = archive.extractfile("framework/" + name).read()
            pin = inventory[name]
            need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["SHA256"],
                 "Archived native body is not the exact deployed-framework identity")
            native_pins[name] = pin
            if name.endswith("backends/hunyuanvideo_training.py"):
                tree = ast.parse(raw)
                run = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name == "run_hunyuanvideo_training")
                calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)]
                need(any(isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                         and c.func.value.id == "optimizer" and c.func.attr == "step" for c in calls),
                     "Original native optimizer update is absent")
                need(b'weights_only=False' in raw,
                     "Re-audit the native cache loader before changing its trust boundary")
    cfg = plan["config"]
    need(cfg == {**original["config"], "training_cache": bindings[original["config"]["training_cache"]]},
         "Common compiler changed original model mathematics or schedule")
    need(cfg["warmup_steps"] == 5 and cfg["max_steps"] == 20 and cfg["validate_recovery"] is False,
         "Unexpected original update/recovery denominator")
    return {
        "state": "LOCAL_COMMON_NATIVE_PLAN_VERIFIED_TARGET_NOT_READY",
        "passed": True, "READY": False, "GPU_GO": False, "training_GO": False,
        "planned_optimizer_updates": 25, "planned_new_process_resume_updates": 0,
        "planned_unique_cache_files": 1, "new_samples": 0,
        "cache_provenance": "Existing authorized original cached tensors; no public video-license inference",
        "framework_manifest_SHA256": MANIFEST_SHA, "native_source_pins": native_pins,
        "bindings": bindings, "common_plan": plan,
        "local_checks": {"exact_four_393_member_sources": True, "actual_common_compiler": True,
                         "unchanged_config": True, "incomplete_binding_rejected": True,
                         "native_optimizer_step_present": True},
        "remaining": [
            "Root policy decision: independent cached-input functional regression, not original numerical qualification",
            "Fresh target native source/runtime closure and accelerator admission; historical model/cache receipts are not fresh readiness",
            "Exact pinned cache is loaded by existing weights_only=False contract: no new or substituted tensor file is permitted",
            "Original before.pt/inputs.pt transfer UNKNOWN remains unresolved; required for original A100 numerical comparison",
            "Root-owned bounded one-DCU controller, unique request/claim, source publication and terminal consumer are not supplied here",
        ],
        "actual_GPU_calls": 0, "actual_optimizer_updates": 0,
        "formal17_qualified": False, "domain_effect_qualified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build(args.repo)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "READY": False,
                      "planned_optimizer_updates": 25, "actual_optimizer_updates": 0}))


if __name__ == "__main__":
    main()
