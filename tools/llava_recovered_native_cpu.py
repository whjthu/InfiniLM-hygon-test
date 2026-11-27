"""Load the recovered OneVision model and verify its real CPU LoRA structure."""

from dataclasses import dataclass
import faulthandler
from hashlib import sha256
import inspect
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import time


BINDING_SHA = "0e8bc1decb2f98037cd99499432146beebf6a5ac75c1466835e7a089b8603586"
MODEL_SOURCE = "bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198"
MODEL_RECEIPT_SHA = "3710c08069b372f95a4870beb9126d658d86a3fc050ec4fdf029bfe639fa592c"
VISION_MANIFEST_SHA = "75ee6a875b49a53c749cf7091de9c070c3276ef488a1af6ba3d2dab0571d551b"
CACHE_PROVIDER = {"bytes": 73550, "SHA256": "7935a009f4131f1c21b53fb731f06813bcbaf8a34fe216cb9e83707aaafcf949"}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned_document(path, pin):
    raw = path.read_bytes()
    need(0 < len(raw) <= 1 << 20 and sha256(raw).hexdigest() == pin, "Frozen document differs")
    return json.loads(raw)


def phase(name):
    print(json.dumps(dict(llava_cpu_phase=name, monotonic_seconds=time.monotonic()), sort_keys=True), file=sys.stderr, flush=True)


def preimport_verified_class(root, compatibility):
    vision_root = Path("/vision")
    expected = pinned_document(Path(__file__).with_name("VISION_FILES.json"), VISION_MANIFEST_SHA)
    need(len(expected) == 7, "Original vision provider manifest differs")
    for name, pin in expected.items():
        source = vision_root / name
        raw = source.read_bytes()
        need({"bytes": len(raw), "SHA256": sha256(raw).hexdigest()} == pin, "Original vision provider differs")
    import qwen_vl_utils
    import transformers
    from transformers import AutoConfig, cache_utils
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from cache_alias import scoped_sliding_cache_alias

    need(Path(qwen_vl_utils.__file__).resolve() == vision_root / "qwen_vl_utils/__init__.py"
         and metadata.version("qwen-vl-utils") == "0.0.14", "Loaded vision provider differs")
    need(transformers.__version__ == "5.5.0", "Target Transformers provider differs")
    native_source = root / "modeling_llavaonevision1_5.py"
    need(sha256(native_source.read_bytes()).hexdigest() == MODEL_SOURCE, "Pinned OneVision module differs")
    provider = Path(cache_utils.__file__).read_bytes()
    provider_pin = {"bytes": len(provider), "SHA256": sha256(provider).hexdigest()}
    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    for part in (config, config.text_config, config.vision_config):
        part._attn_implementation = "eager"
    compatibility._prepare_load_config(config)
    with scoped_sliding_cache_alias(cache_utils, provider_pin=provider_pin, expected_provider=CACHE_PROVIDER) as static:
        compatibility._preimport_verified_eager_class(str(root), config, {"attn_implementation": "eager"})
        native_class = get_class_from_dynamic_module(
            "modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration", str(root), local_files_only=True
        )
        need(vars(sys.modules[native_class.__module__]).get("SlidingWindowCache") is static,
             "Native eager import did not bind the qualified cache type")
    need("SlidingWindowCache" not in vars(cache_utils), "Scoped cache export leaked")
    return native_class


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
    phase("receipt_members_bound")

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
    from vlm.plugins import llava_onevision
    from swift.model import get_model_processor
    from swift.arguments.tuner_args import TunerArguments
    from swift.pipelines.train.tuner import get_target_modules, prepare_adapter
    from vlm.runtime.native_recipe_contract import parameter_manifest

    native_class = preimport_verified_class(root, llava_onevision)
    phase("verified_eager_import_completed")
    started = time.monotonic()
    core, processor = get_model_processor(str(root), model_type="llava_onevision1_5", load_model=True,
                                          download_model=False, device_map="cpu", attn_impl="eager",
                                          torch_dtype=torch.bfloat16)
    load_seconds = time.monotonic() - started
    phase("swift_weight_and_rope_loader_returned")
    model_source = Path(inspect.getsourcefile(type(core))).read_bytes()
    need(type(core) is native_class and sha256(model_source).hexdigest() == MODEL_SOURCE == binding["model_source_SHA256"]
         and processor.model_meta.template == "llava_onevision1_5"
         and all(p.device.type == "cpu" for p in core.parameters()),
         "Actual loaded class, template or parameter placement differs")
    need(core.config._attn_implementation == "eager", "Original eager attention path differs")
    llava_onevision._verify_loaded_native_eager_classes(core, native_class.__module__)
    need(getattr(core, "_vlm_onevision_rope_initialized", False), "Materialized RoPE repair did not run")

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
    phase("adapter_prepare_started")
    model = prepare_adapter(tuner, core, task_type="CAUSAL_LM")
    phase("adapter_prepare_returned")
    trainable = sorted((name, p) for name, p in model.named_parameters() if p.requires_grad)
    need(parameter_manifest(trainable) == binding["trainable_manifest"]
         and modules == binding["target_modules"] and len(trainable) == 504,
         "Original exact LoRA parameter contract differs")
    need(all(p.device.type == "cpu" and bool(torch.isfinite(p).all()) for _, p in trainable)
         and not torch.cuda.is_initialized(), "Adapter contains invalid values or escaped CPU scope")
    phase("adapter_parameters_verified")
    return dict(schema="LLAVA_RECOVERED_ORIGINAL_ACTUAL_NATIVE_CPU_LOAD_V1", model_source_SHA256=MODEL_SOURCE,
                model_receipt_SHA256=MODEL_RECEIPT_SHA, bound_template=processor.model_meta.template,
                dtype="torch.bfloat16", attention="eager", load_seconds=load_seconds,
                base_model_loaded=True, actual_tensors=sum(1 for _ in core.parameters()),
                trainable_tensors=len(trainable), trainable_parameters=sum(p.numel() for _, p in trainable),
                prior_meta_import_compatibility_reused=True, materialized_rope_initialized=True,
                trainable_names_shapes_dtypes_exact=True, initial_numeric_reference_verified=False,
                optimizer_updates=0, forward_passes=0, gpu_initialized=False, formal_training_GO=False)


if __name__ == "__main__":
    # Swift's dynamic module loader places its file locks in the ModelScope cache.
    with TemporaryDirectory(dir="/tmp", prefix="llava-modelscope-") as cache:
        os.environ["MODELSCOPE_CACHE"] = cache
        faulthandler.dump_traceback_later(45, repeat=True)
        try:
            result = main(*(Path(x) for x in sys.argv[1:]))
        finally:
            faulthandler.cancel_dump_traceback_later()
    print(json.dumps(dict(result, modelscope_cache_private_and_removed=True), sort_keys=True, allow_nan=False))
