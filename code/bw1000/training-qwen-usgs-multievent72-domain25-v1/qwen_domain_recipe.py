"""Build the new multievent72 continuous25 plan using the exact common Qwen launcher."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


BASE = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
MODEL = BASE / "training-qwen-original-public-model-v1/model"
ORIGINAL_PLUGINS = BASE / "training-vlm-original-assets-v1/plugins"
DATASET = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-native-CPU-v1/data/train-swift.jsonl")
CASE = "qwen3vl.usgs-multievent72"
TRANSLATOR_SHA256 = "71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2"
ORIGINAL_SHA256 = "28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400"
DATASET_SHA256 = "a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def native_translator(framework):
    path = framework / "tools/hygon_vlm_recipe.py"
    raw = path.read_bytes()
    require(sha(raw) == TRANSLATOR_SHA256, "common framework translator changed")
    candidate = raw
    module = {"__name__": "treasury_common_vlm_recipe", "__file__": str(path)}
    exec(compile(candidate, str(path), "exec"), module)
    return module["build_launch"], sha(candidate)


def check_preflight(raw):
    receipt=json.loads(raw)
    require(receipt['state']=='NATIVE_QWEN_MULTIEVENT72_PASS' and receipt['rows']==72
            and receipt['dataset_SHA256']==DATASET_SHA256 and receipt['labels_and_EOS_valid']==72
            and receipt['full_sequence_retained']==72 and receipt['actual_GPU_allocated']==0
            and receipt['collator_rows']==72 and receipt['max_length']==4096 and receipt['max_pixels']==262144
            and receipt['explicit_GPU_queries']==0 and receipt['actual_updates']==0,
            'new Qwen native72 must actually pass before training')
    return receipt['step']


def make_plan(framework, original, cpu_receipt, root, plugin, master_port, arm, dataset, dataset_sha):
    require(sha(original) == ORIGINAL_SHA256, "frozen Qwen T01 recipe changed")
    cpu_step = check_preflight(cpu_receipt)
    require(arm == "continuous", "unknown paired training arm")
    require(dataset == root / "view/training800.jsonl" and len(dataset_sha)==64, "unexpected exposure view")
    for path in (root, plugin):
        require(path.is_absolute() and path.anchor == "/" and len(path.parts) > 1 and ".." not in path.parts,
                "expected an absolute container path without traversal")
    output = root / arm / "train"
    checkpoint = root / "continuous/train/checkpoint-20" if arm == "resumed" else None
    callback = "qwen_treasury_resume_witness" if checkpoint else "qwen_treasury_continuous_witness"
    build, translator_sha = native_translator(framework)
    launch = build("T01.sft", original, PurePosixPath(framework.as_posix()) / "vlm/runtime/swift_entry.py",
                   MODEL.parent, dataset, ORIGINAL_PLUGINS, output, master_port,
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
    changes = {"--model": str(MODEL), "--max_steps": "25", "--save_steps": "10"}
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
    env.update(MAX_STEPS="25", SAVE_STEPS="10")
    launch["required_assets"]["model_directory"] = str(MODEL)
    launch.update(case=CASE, source_recipe_case="T01.sft", domain_arm=arm, native_preflight_step=cpu_step,
                  dataset_SHA256=dataset_sha, source_dataset_SHA256=DATASET_SHA256,
                  planned_updates=25 if arm=="continuous" else 5, checkpoint_steps=[10,20,25] if arm=="continuous" else [25],
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
    parser.add_argument("--arm", choices=("continuous",), required=True)
    parser.add_argument("--dataset",type=PurePosixPath,required=True)
    parser.add_argument("--dataset-sha256",required=True)
    options = parser.parse_args()
    launch = make_plan(options.framework, options.original.read_bytes(), options.cpu_receipt.read_bytes(),
                       options.run_root, options.plugin, options.master_port, options.arm, options.dataset,options.dataset_sha256)
    print(json.dumps(launch, sort_keys=True))


if __name__ == "__main__":
    main()
