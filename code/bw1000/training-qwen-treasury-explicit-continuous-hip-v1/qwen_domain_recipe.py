"""Derive a separate five-step Treasury SFT plan from the common Qwen launcher."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


BASE = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
MODEL = BASE / "training-qwen-original-public-model-v1/model"
ORIGINAL_PLUGINS = BASE / "training-vlm-original-assets-v1/plugins"
DATASET = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-sft-v1/inputs.jsonl")
CASE = "qwen3vl.treasury-table-qa"
TRANSLATOR_SHA256 = "71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2"
ORIGINAL_SHA256 = "28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400"
DATASET_SHA256 = "5a6a8f02dbfc77aa46487a0180000275771aefd56e3cc933ffdcf7c97496162e"
PREFLIGHT_SOURCE_SHA256 = "a2a96db5955052fe93db34acb11fff219dec5813ea8e37b6838787d297fe3f79"
PREFLIGHT_RESULT_SHA256 = "bf3915f0fe9a85bd47e278f92193d02cf30c25bc76b8a690c3f0106aec4feda9"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def native_translator(framework):
    path = framework / "tools/hygon_vlm_recipe.py"
    raw = path.read_bytes()
    require(sha(raw) == TRANSLATOR_SHA256, "common framework translator changed")
    original = b'''        if (case != "T05.sft" or len(observers) != 1
                or observers[0]["callback"] != "native_resume_witness"
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("MiniCPM full resume requires its single dedicated native witness")'''
    replacement = b'''        resume_witness = {"T05.sft": "native_resume_witness",
                          "T01.sft": "qwen_treasury_resume_witness"}.get(case)
        if (resume_witness is None or len(observers) != 1
                or observers[0]["callback"] != resume_witness
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("full resume requires the case-specific native witness")'''
    require(raw.count(original) == 1, "common framework resume guard changed")
    candidate = raw.replace(original, replacement)
    module = {"__name__": "treasury_common_vlm_recipe", "__file__": str(path)}
    exec(compile(candidate, str(path), "exec"), module)
    return module["build_launch"], sha(candidate)


def check_preflight(raw):
    require(sha(raw) == PREFLIGHT_RESULT_SHA256, "explicit input CPU326 receipt identity differs")
    receipt = json.loads(raw)
    accounting = receipt.get("accounting", {})
    row = accounting.get("row", [])
    metrics = receipt.get("metrics", {})
    require(receipt.get("passed") is True and receipt.get("current_step_outcome") == "TERMINAL"
            and receipt.get("rc") == 0 and receipt.get("client_timeout") is False
            and receipt.get("GPU_cards") == 0 and accounting.get("terminal_visible") is True
            and len(row) == 7 and row[:4] == ["1087656.326", "training-qwen-treasury-explicit-native-CPU-v2", "COMPLETED", "0:0"]
            and row[5] == "a01r4n14" and "gres/dcu=" not in row[6]
            and metrics.get("rows") == 160 and metrics.get("dataset_SHA256") == DATASET_SHA256
            and metrics.get("actual_GPU_cards") == 0 and metrics.get("actual_updates") == 0
            and metrics.get("max_pixels") == 262144 and metrics.get("rope_model_parameters_on_meta") is True,
            "explicit input CPU326 is not qualified")
    return row[0]


def make_plan(framework, original, cpu_receipt, root, plugin, master_port, arm):
    require(sha(original) == ORIGINAL_SHA256, "frozen Qwen T01 recipe changed")
    cpu_step = check_preflight(cpu_receipt)
    require(arm == "continuous", "this independent scope only permits continuous training")
    for path in (root, plugin):
        require(path.is_absolute() and path.anchor == "/" and len(path.parts) > 1 and ".." not in path.parts,
                "expected an absolute container path without traversal")
    output = root / arm / "train"
    checkpoint = root / "continuous/train/checkpoint-2" if arm == "resumed" else None
    callback = "qwen_treasury_resume_witness" if checkpoint else "qwen_treasury_continuous_witness"
    build, translator_sha = native_translator(framework)
    launch = build("T01.sft", original, PurePosixPath(framework.as_posix()) / "vlm/runtime/swift_entry.py",
                   MODEL.parent, DATASET, ORIGINAL_PLUGINS, output, master_port,
                   diagnostic_plugin=plugin, diagnostic_callback=callback, diagnostic_callback_position="before",
                   resume_from_checkpoint=checkpoint)
    argv = launch["command"]
    if checkpoint is None:
        require(argv.count("nvidia_baseline") == 1, "expected the original source callback before domain binding")
        argv.remove("nvidia_baseline")
    baseline_plugin = str(ORIGINAL_PLUGINS / "swift_callback_nvidia_baseline_v2.py")
    require(argv.count(baseline_plugin) == 1, "expected the original source plugin before domain binding")
    argv.remove(baseline_plugin)
    launch["required_assets"]["plugins"].remove(baseline_plugin)
    changes = {"--model": str(MODEL), "--max_steps": "5", "--save_steps": "2"}
    frozen = {"--model_type": "qwen3_vl", "--template": "qwen3_vl", "--max_pixels": "262144",
              "--max_length": "4096", "--torch_dtype": "bfloat16", "--dataloader_drop_last": "false",
              "--per_device_train_batch_size": "1", "--gradient_accumulation_steps": "4",
              "--gradient_checkpointing": "true", "--attn_impl": "sdpa", "--freeze_vit": "true",
              "--optim": "adamw_torch", "--lr_scheduler_type": "cosine", "--target_modules": "all-linear",
              "--lora_rank": "16", "--lora_alpha": "32", "--lora_dropout": "0.05"}
    for option, value in frozen.items():
        require(argv.count(option) == 1 and argv[argv.index(option) + 1] == value,
                f"frozen option differs: {option}")
    for option, value in changes.items():
        require(argv.count(option) == 1, f"ambiguous option: {option}")
        argv[argv.index(option) + 1] = value
    env = launch["environment_overrides"]
    require(env["NPROC_PER_NODE"] == "8" and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7",
            "single-node native eight-rank topology changed")
    env.update(MAX_STEPS="5", SAVE_STEPS="2")
    launch["required_assets"]["model_directory"] = str(MODEL)
    launch.update(case=CASE, source_recipe_case="T01.sft", domain_arm=arm, native_preflight_step=cpu_step,
                  dataset_SHA256=DATASET_SHA256, planned_updates=5, planned_checkpoint_step=2,
                  candidate_framework_translator_SHA256=translator_sha,
                  formal_performance=False, qualification="command plan; actual training requires installed witness")
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=PurePosixPath, required=True)
    parser.add_argument("--plugin", type=PurePosixPath, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--arm", choices=("continuous", "resumed"), required=True)
    options = parser.parse_args()
    launch = make_plan(options.framework, options.original.read_bytes(), options.cpu_receipt.read_bytes(),
                       options.run_root, options.plugin, options.master_port, options.arm)
    print(json.dumps(launch, sort_keys=True))


if __name__ == "__main__":
    main()
