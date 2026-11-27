"""Load the recovered OneVision model and verify its real CPU LoRA structure."""

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time


BINDING_SHA = "0e8bc1decb2f98037cd99499432146beebf6a5ac75c1466835e7a089b8603586"
MODEL_SOURCE = "bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198"
MODEL_RECEIPT_SHA = "3710c08069b372f95a4870beb9126d658d86a3fc050ec4fdf029bfe639fa592c"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned_document(path, pin):
    raw = path.read_bytes()
    need(0 < len(raw) <= 1 << 20 and sha256(raw).hexdigest() == pin, "Frozen document differs")
    return json.loads(raw)


def main(root, binding_path, receipt_path):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Zero-GRES CPU scope required")
    receipt = pinned_document(receipt_path, MODEL_RECEIPT_SHA)
    binding = pinned_document(binding_path, BINDING_SHA)
    need({path.name for path in root.iterdir()} == set(receipt["files"]) and receipt["verified_complete_shards"] == 4,
         "Recovered model members differ")
    for name, pin in receipt["files"].items():
        path = root / name
        st = path.stat()
        need(st.st_size == pin["bytes"] and (st.st_ino, str(st.st_mtime_ns)) == (pin["inode"], pin["mtime_ns"]),
             "Recovered model inode or extent differs")

    import numpy as np
    import torch

    need(not torch.cuda.is_initialized(), "The CPU probe initialized a GPU")
    torch.set_num_threads(6)
    random.seed(42)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)

    def no_device(*_args, **_kwargs):
        raise RuntimeError("Explicit GPU initialization is outside the CPU scope")

    torch.cuda._lazy_init = no_device
    from vlm.plugins import llava_onevision  # Registers the original local Swift compatibility adapter.
    from swift.model import get_model_processor
    from swift.arguments.tuner_args import TunerArguments
    from swift.pipelines.train.tuner import get_target_modules, prepare_adapter
    from vlm.runtime.native_recipe_contract import parameter_manifest

    started = time.monotonic()
    core, processor = get_model_processor(str(root), model_type="llava_onevision1_5", load_model=True,
                                          download_model=False, device_map="cpu", attn_impl="eager",
                                          torch_dtype=torch.bfloat16)
    load_seconds = time.monotonic() - started
    model_source = Path(inspect.getsourcefile(type(core))).read_bytes()
    need(sha256(model_source).hexdigest() == MODEL_SOURCE == binding["model_source_SHA256"]
         and processor.model_meta.template == "llava_onevision1_5"
         and all(p.device.type == "cpu" for p in core.parameters()),
         "Actual loaded class, template or parameter placement differs")
    need(core.config._attn_implementation == "eager", "Original eager attention path differs")

    @dataclass
    class NativeTunerArguments(TunerArguments):
        model_meta: object = None
        tuner_type: str = "lora"

    core.requires_grad_(False)
    tuner = NativeTunerArguments(lora_rank=16, lora_alpha=32, lora_dropout=.05,
                                 freeze_vit=True, model_meta=processor.model_meta)
    need(tuner.freeze_aligner is True and tuner.freeze_llm is False, "Frozen vision or aligner boundary differs")
    tuner.task_type = "causal_lm"
    tuner.tuner_type = "lora"
    tuner.tuner_backend = "peft"
    tuner.use_swift_lora = False
    modules = get_target_modules(tuner, core)
    model = prepare_adapter(tuner, core, task_type="CAUSAL_LM")
    trainable = sorted((name, p) for name, p in model.named_parameters() if p.requires_grad)
    need(parameter_manifest(trainable) == binding["trainable_manifest"]
         and modules == binding["target_modules"] and len(trainable) == 504,
         "Original exact LoRA parameter contract differs")
    need(all(p.device.type == "cpu" and bool(torch.isfinite(p).all()) for _, p in trainable)
         and not torch.cuda.is_initialized(), "Adapter contains invalid values or escaped CPU scope")
    return dict(schema="LLAVA_RECOVERED_ORIGINAL_ACTUAL_NATIVE_CPU_LOAD_V1", model_source_SHA256=MODEL_SOURCE,
                model_receipt_SHA256=MODEL_RECEIPT_SHA, bound_template=processor.model_meta.template,
                dtype="torch.bfloat16", attention="eager", load_seconds=load_seconds,
                base_model_loaded=True, actual_tensors=sum(1 for _ in core.parameters()),
                trainable_tensors=len(trainable), trainable_parameters=sum(p.numel() for _, p in trainable),
                trainable_names_shapes_dtypes_exact=True, initial_numeric_reference_verified=False,
                optimizer_updates=0, forward_passes=0, gpu_initialized=False, formal_training_GO=False)


if __name__ == "__main__":
    print(json.dumps(main(*(Path(x) for x in sys.argv[1:])), sort_keys=True, allow_nan=False))
