"""Verify the actual Qwen recovery source, checkpoint and parser on the target."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import time

stage = Path("/stage")
src = Path("/fix")
framework = Path("/framework")
original = Path("/original")
sys.path.insert(0, str(framework))
epoch = float(os.environ["QWEN_CPU_EPOCH"])
deadline = float(os.environ["QWEN_CPU_DEADLINE"])
assert deadline == epoch + 230 and epoch <= time.time() < deadline
signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
assert os.environ["SLURM_JOB_ID"] == "1087656"
assert all(os.environ[k] == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))

from tools.hygon_vlm_recipe import build_launch
from vlm.runtime.resume_witness import pinned_checkpoint
from vlm.runtime.resume_cursor import plan_epoch_boundary_resume
from swift.pipelines.train.sft import SwiftSft
from transformers import HfArgumentParser
import torch

assert not torch.cuda.is_initialized()
def no_device(*args, **kwargs):
    raise RuntimeError("Device initialization forbidden in CPU recovery preflight")
torch.cuda._lazy_init = no_device
from vlm.plugins.qwen_resume_training import QwenResume

data = json.loads((src/"RESUME_INPUTS.json").read_bytes())
pins = {n: v for n, v in data["checkpoint20"].items() if n != "training_args.bin"}
ckpt = original/"train/checkpoint-20"
checkpoint = pinned_checkpoint(ckpt, pins, 8)
plan = plan_epoch_boundary_resume(checkpoint["state"], checkpoint["args"], checkpoint_step=20,
    max_steps=25, world_size=8, per_device_batch=1, gradient_accumulation_steps=4,
    dataloader_batches_per_rank=8)
P = PurePosixPath
asset = P("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
model = asset/"training-qwen-original-public-model-v1/model"
launch = build_launch("T01.sft", (stage/"original-launch.json").read_bytes(),
    P("/framework/vlm/runtime/swift_entry.py"), model.parent,
    asset/"training-vlm-original-assets-v1/vlm_report_coco64.hip.jsonl",
    asset/"training-vlm-original-assets-v1/plugins", P("/evidence/unused"), 29895,
    diagnostic_plugin=P("/framework/vlm/plugins/qwen_resume_training.py"),
    diagnostic_callback="qwen_native_resume_witness",
    resume_from_checkpoint=P(ckpt))
argv = launch["command"]
argv[argv.index("--model")+1] = str(model)
parsed, unknown = HfArgumentParser([SwiftSft.args_class]).parse_known_args(argv[2:])
assert not unknown and parsed.callbacks == ["qwen_native_resume_witness"]
assert not parsed.resume_only_model and not parsed.ignore_data_skip and not parsed.load_args
assert int(parsed.max_steps) == 25 and parsed.resume_from_checkpoint == str(ckpt)
assert len(checkpoint["RNG"]) == 8 and not torch.cuda.is_initialized()
proof = dict(passed=True, step="1087656."+os.environ["SLURM_STEP_ID"], GPU_cards=0,
    checkpoint_files_verified=len(pins), rank_rng_records=len(checkpoint["RNG"]),
    optimizer=checkpoint["optimizer"], scheduler=checkpoint["scheduler"],
    adapter=checkpoint["adapter"], plan=plan, actual_native_resume=False,
    module=QwenResume.__module__, input_scope_SHA256=hashlib.sha256((src/"RESUME_INPUTS.json").read_bytes()).hexdigest())
with Path("/evidence/compute.json").open("x") as f:
    json.dump(proof, f, indent=2)
