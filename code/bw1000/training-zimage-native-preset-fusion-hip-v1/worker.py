"""Observe the full native Z-Image Turbo preset fusion on one Hygon DCU."""

import gc
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import time


ROOT = Path("/fusion-run")
UPSTREAM = Path("/native-source")
RUNTIME = Path("/sealed-runtime")


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(path, limit=None):
    size = path.stat().st_size
    need(limit is None or size == limit, "Frozen size differs: " + path.name)
    h = sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
    with (ROOT / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def frozen_file(root, name, pin):
    path = root / name
    need(path.is_file() and path.resolve(strict=True) == path and root in path.parents,
         "Missing or aliased frozen file: " + name)
    need(digest(path, pin["bytes"]) == pin["SHA256"], "Frozen digest differs: " + name)


def original_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trainable_boundary(torch, model, expected):
    components = ("dit", "text_encoder", "vae_encoder", "vae_decoder", "tokenizer")
    need(all(getattr(model.pipe, name, None) is not None for name in components), "Incomplete native pipeline")
    trained = {name: value for name, value in model.pipe.dit.named_parameters() if value.requires_grad}
    frozen = [value for value in model.pipe.dit.parameters() if not value.requires_grad]
    need(set(trained) == set(expected) and len(trained) == 476
         and sum(value.numel() for value in trained.values()) == 9922560, "Native rank-4 trainable set differs")
    need(len(frozen) == 521 and sum(value.numel() for value in frozen) == 6154908736
         and all(value.device.type == "cuda" and value.dtype == torch.bfloat16 for value in frozen),
         "Frozen native DiT boundary differs")
    need(sum(value.requires_grad for value in model.parameters()) == 476, "A non-LoRA parameter was trainable")
    for name in {name.split(".lora_")[0] for name in trained}:
        block = model.pipe.dit.get_submodule(name)
        need(block.r["default"] == block.lora_alpha["default"] == 4 and block.scaling["default"] == 1
             and getattr(block.lora_dropout["default"], "p", 0) == 0, "Active PEFT hyperparameters differ")
    adapters = {}
    for name, value in trained.items():
        need(list(value.shape) == expected[name]["shape"] and value.device.type == "cuda"
             and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), "Trainable tensor differs: "+name)
        adapters[name] = sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return dict(components=components, trainable_tensors=len(trained), trainable_elements=9922560,
                frozen_DiT_tensors=len(frozen), frozen_DiT_elements=6154908736, initial_adapter_SHA256=adapters)


def main():
    claim = json.loads((ROOT / "CLAIM.json").read_bytes())
    intent = json.loads((ROOT / "intent.json").read_bytes())
    deadline = intent["epoch"] + claim["worker_seconds"]
    need(intent["request"] == claim["request"] and time.time() < deadline, "Original worker deadline differs")
    need(socket.gethostname().split(".")[0] == claim["node"] and os.environ["SLURM_JOB_ID"] == claim["parent"]
         and os.environ["SLURM_STEP_ID"].isdigit()
         and int(os.environ["SLURM_CPUS_PER_TASK"]) == claim["cpus"], "Original Slurm worker scope differs")
    step = claim["parent"]+"."+os.environ["SLURM_STEP_ID"]
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Fusion deadline")))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())

    def check():
        need(time.time() < deadline, "Fusion deadline reached")

    started = time.monotonic()
    result = dict(request=claim["request"], step=step, passed=False,
                  actual_optimizer_updates=0, actual_forward_calls=0, actual_backward_calls=0,
                  matched_A100_tensor_values=False, training_ready=False, optimizer_started=False)
    phase = "source"
    model = torch = None
    try:
        source_spec = json.loads((ROOT / "META_SPEC.json").read_bytes())["upstream_files"]
        need(len(source_spec) == 40, "Frozen native source count differs")
        for name, pin in sorted(source_spec.items()):
            check()
            frozen_file(UPSTREAM, name, pin)
        phase = "weight_identity_before_load"
        runtime_proof = json.loads((ROOT / "RUNTIME_EVIDENCE.json").read_bytes())["independent_runtime"]
        members = {int(item["asset_id"]): item for item in runtime_proof["files"]}
        need(len(members) == 20 and sum(item["bytes"] for item in members.values()) == 33018433789,
             "Verified publisher file set differs")
        beginning = time.monotonic()
        for pin in members.values():
            check()
            frozen_file(RUNTIME, pin["path"], pin)
        result.update(runtime_verified_files=len(members), runtime_rehash_seconds=time.monotonic()-beginning)
        phase = "actual_HIP_and_native_imports"
        import numpy
        import torch
        import flash_attn
        from diffsynth.utils.lora.general import GeneralLoRALoader
        phase = "device_admission"
        need(torch.version.hip == "6.3.26113" and torch.cuda.device_count() == 1,
             "Unexpected HIP runtime or visibility")
        props = torch.cuda.get_device_properties(0)
        arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
        need(arch == "gfx936" and callable(getattr(flash_attn, "flash_attn_func", None)), "Native HIP device or Flash2 export differs")
        free_before, total = torch.cuda.mem_get_info()
        need(free_before >= 36 << 30, "Insufficient free memory for the full native pipeline")
        torch.set_num_threads(2)
        torch.manual_seed(20260911)
        random.seed(20260911)
        numpy.random.seed(20260911)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        result.update(torch=str(torch.__version__), hip=torch.version.hip, architecture=arch,
                      flash_attn_provider=digest(Path(flash_attn.__file__)),
                      versions={name: importlib.metadata.version(name) for name in
                                ("transformers", "accelerate", "peft", "safetensors", "flash-attn")},
                      total_device_memory=total, free_before=free_before)
        phase = "native_full_pipeline_fusion"
        native = original_module("frozen_zimage_train", UPSTREAM/"examples/z_image/model_training/train.py")
        helper = original_module("published_fusion_observer", ROOT/"fusion_observer.py")
        shapes = json.loads((ROOT/"PRESET_SHAPES.json").read_bytes())
        before_load = time.monotonic()
        with (ROOT/"parameter-progress.jsonl").open("x", encoding="utf8") as progress:
            def sink(row):
                progress.write(json.dumps(row, allow_nan=False) + "\n")
                progress.flush()

            audit = helper.PresetFusionAudit(GeneralLoRALoader, shapes, deadline=check, sink=sink)
            def asset(index):
                return str(RUNTIME/members[index]["path"])

            model_paths = [[asset(i) for i in (14, 15, 16)], [asset(i) for i in (5, 6, 7)], asset(19)]
            with audit.observe():
                model = native.ZImageTrainingModule(model_paths=json.dumps(model_paths),
                    tokenizer_path=str(RUNTIME/"base/tokenizer"), lora_base_model="dit",
                    lora_target_modules="to_q,to_k,to_v,to_out.0,w1,w2,w3", lora_rank=4,
                    preset_lora_path=asset(20), preset_lora_model="dit",
                    use_gradient_checkpointing=True, device="cuda", task="sft")
        torch.cuda.synchronize()
        need(audit.result and audit.result["passed"], "Native fusion observer did not pass")
        result.update(preset_fusion_executed=True, fusion_identity_SHA256=save("FUSION_IDENTITY.private.json",dict(audit.result, reference="Same HIP BF16 base + (B @ A); full-byte SHA per parameter")),
                      changed_base_tensors=audit.result["changed_target_tensors"], base_tensors=audit.result["base_tensors"],
                      preset_factors=audit.result["preset_factors"], unchanged_base_tensors=audit.result["unmodified_other_tensors"],
                      preset_input_factors_unchanged=audit.result["preset_input_factors_unchanged"],
                      RNG_unchanged_during_fusion=audit.result["RNG_unchanged"],
                      native_load_and_audit_seconds=time.monotonic()-before_load,
                      native_fusion_host_seconds=audit.result["native_host_call_seconds"])
        phase = "native_LoRA_trainable_boundary"
        expected = json.loads((ROOT/"TRAINABLE_METADATA.json").read_bytes())
        boundary = trainable_boundary(torch, model, expected)
        result["initial_adapter_hashes_SHA256"] = save("INITIAL_ADAPTER_HASHES.private.json", boundary.pop("initial_adapter_SHA256"))
        result.update(boundary)
        result["passed"] = True
    except Exception as error:
        import traceback
        save("failure.private.json", dict(phase=phase, detail=traceback.format_exc()[-8000:]))
        result.update(error_type=type(error).__name__, error=str(error)[:240])
    finally:
        if torch is not None and torch.cuda.is_initialized():
            try:
                result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
                model = None
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                result.update(allocator_after_destructor_bytes=torch.cuda.memory_allocated(),
                              allocator_after_empty_cache_bytes=torch.cuda.memory_reserved())
            except Exception as error:
                result.update(device_release_error=type(error).__name__, passed=False)
        signal.setitimer(signal.ITIMER_REAL, 0)
    result.update(phase=phase, wall_seconds=time.monotonic()-started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save("worker-result.json", result)
    print(json.dumps({name:result.get(name) for name in ("request","step","passed","phase","error_type","error")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
