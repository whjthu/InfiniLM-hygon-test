"""Run one loaded OneVision model through base and original .5000 CP25 on native Swift."""

import argparse
from collections import Counter
import ctypes
import importlib.util
from importlib import metadata
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import time

from contract import ARMS, CAP, HEX, generation, identity, inputs, need, parse, read, record, seal, sha


SWIFT = Path("/swift-runtime/site")
FRAMEWORK = Path("/llava-framework")
HELPERS = Path("/llava-helpers")
ADAPTER = "llava_original5000_cp25"
CP25_SHA = "b8b8a8b9dfe5012c4ce5f80499a1b3fc5e7b8b140a7a734cada07c4a39b83719"
CONFIG_SHA = "5b3bfd425c25480a7a59b8acb420d01ebd5533898b60b4255fca0b74ef092c3e"
RELOAD_TOOL = "1f5f85c267c47d8a48c569262c3eddff2b0c47962850887abf326b92ac33063b"
HELPER_PINS = {
    "probe.py": "446d4d0b27d9eff6488a6ebe07bce959fdf6e81b4f708e805c7c6cd237183aaf",
    "llava_recovered_native_cpu.py": "d94ddf5a68f9eeb2f3fe19396989fd87a2bd5df3bbfe0ab16ceb525f84e782ad",
    "cache_alias.py": "f843bf60d441c82813680f323796aaf9177df8256c8f25fada42fe5164ec9c94",
    "llava_target_eager_import.py": "fd998f6c192cf6cf8021ba37d34397894852d4fb505759b6e9b46441d4d12ba1",
}
ENGINE = {
    "__init__.py": "cea0885e9a4b3329f8cb861a665c24c6476028d1ea3c316f3bc41b59e81a44fa",
    "transformers_engine.py": "b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb",
    "infer_engine.py": "857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969",
    "utils.py": "5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644",
    "protocol.py": "f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e",
}
VERSIONS = {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "ms-swift": "4.4.2", "transformers": "5.5.0"}


def readonly(path):
    need(path.is_absolute() and path.resolve(strict=True) == path
         and os.statvfs(path).f_flag & os.ST_RDONLY, "EXACT_READONLY_MOUNT")


def authority(args):
    owner = parse(read(args.owner, 16 << 10, args.owner_sha))
    need(type(owner) is dict and set(owner) == {"request", "step", "node", "epoch", "deadline", "gpu_count",
         "GPU_GO", "boot", "pci", "requests_SHA256", "images_SHA256", "output", "checkpoint_SHA256"},
         "EXACT_OWNER_FIELDS")
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    parent = args.output.parent
    need(re.fullmatch(r"[0-9a-f]{32}", args.request) and owner["request"] == args.request
         and job.isdecimal() and step.isdecimal() and owner["step"] == job + "." + step
         and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_STEP_GPUS", ""))
         and owner["node"] == socket.gethostname().split(".")[0]
         and owner["boot"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
         and re.search(rf"/job_{job}(?:\.scope)?/step_{step}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE)
         and type(owner["gpu_count"]) is int and owner["gpu_count"] == 1 and owner["GPU_GO"] is True
         and type(owner["pci"]) is list and len(owner["pci"]) == len(set(owner["pci"])) == 8
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", val) for val in owner["pci"])
         and owner["checkpoint_SHA256"] == CP25_SHA and owner["output"] == str(args.output)
         and args.output.is_absolute() and parent.resolve(strict=True) == parent
         and parent.stat().st_uid == os.geteuid() and stat.S_IMODE(parent.stat().st_mode) == 0o700
         and owner["requests_SHA256"] == args.requests_sha and owner["images_SHA256"] == args.images_sha
         and all(type(owner[key]) in (int, float) for key in ("epoch", "deadline"))
         and owner["epoch"] <= time.time() < owner["deadline"] and 0 < owner["deadline"] - owner["epoch"] <= 2400,
         "BOUND_SINGLE_DCU_OWNER")
    readonly(args.owner)
    return owner


def pci_id():
    hip = ctypes.CDLL("libamdhip64.so")
    count, device = ctypes.c_int(), ctypes.create_string_buffer(32)
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
         and hip.hipDeviceGetPCIBusId(device, len(device), 0) == 0, "EXACT_VISIBLE_HIP_PCI")
    value = device.value.decode("ascii").lower()
    return "0000:" + value if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value) else value


def base_files(model, model_receipt, recovered):
    receipt = recovered.pinned_document(model_receipt, recovered.MODEL_RECEIPT_SHA)
    need(receipt["verified_complete_shards"] == 4 and len(receipt["files"]) == 18
         and {item.name for item in model.iterdir()} == set(receipt["files"]), "ORIGINAL_BASE_CLOSURE")
    observed = {}
    for name, pin in receipt["files"].items():
        path = model / name
        readonly(path)
        current = path.lstat()
        need(stat.S_ISREG(current.st_mode) and (current.st_size, current.st_ino, str(current.st_mtime_ns)) ==
             (pin["bytes"], pin["inode"], pin["mtime_ns"]), "ORIGINAL_BASE_FILE_IDENTITY")
        observed[name] = identity(current)
    return observed


def tensor_identity(value, torch):
    if type(value) is int:
        return ["int", value]
    if type(value) in (list, tuple):
        need(bool(value) and len(value) <= 32, "BOUNDED_NATIVE_PIXEL_GROUP")
        return [type(value).__name__, [tensor_identity(item, torch) for item in value]]
    need(torch.is_tensor(value) and value.numel() > 0, "EXACT_NATIVE_TENSOR")
    host = value.detach().cpu().contiguous()
    need(not host.is_floating_point() or bool(torch.isfinite(host).all()), "NONFINITE_NATIVE_INPUT")
    return [str(host.dtype), list(host.shape), sha(host.reshape(-1).view(torch.uint8).numpy().tobytes())]


def safe_type_label(value):
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_.:+-]{1,64}", text) else "redacted"


def original_base_gate(model, processor, native_class, torch, current):
    # Project structure only. Parameter values and configuration keys stay private.
    diagnostic = dict(predicates={}, parameter_tensors=None, floating_parameter_tensors=None,
                      dtype_device_counts=[], omitted_dtype_device_tensors=None,
                      lora_parameter_tensors=None, peft_config=None)
    current["base_diagnostic"] = diagnostic
    config_missing = object()
    peft = getattr(model, "peft_config", config_missing)
    present = peft is not config_missing
    diagnostic["peft_config"] = dict(
        present=present, type=safe_type_label(type(peft).__name__) if present else None,
        count=dict.__len__(peft) if isinstance(peft, dict)
        else len(peft) if type(peft) in (list, tuple, set, frozenset) else None)
    histogram = Counter()
    devices, float_dtypes = set(), set()
    total = float_count = 0
    for param in model.parameters():
        device = str(param.device)
        floating = bool(param.is_floating_point())
        devices.add(device)
        if floating:
            float_dtypes.add(param.dtype)
        histogram[(safe_type_label(param.dtype), safe_type_label(device), floating)] += 1
        total += 1
        float_count += floating
    groups = sorted(histogram.items())
    diagnostic.update(parameter_tensors=total, floating_parameter_tensors=float_count,
                      dtype_device_counts=[dict(dtype=dtype, device=device, floating=floating, tensors=count)
                                           for (dtype, device, floating), count in groups[:32]],
                      omitted_dtype_device_tensors=sum(count for _, count in groups[32:]),
                      lora_parameter_tensors=sum(bool(re.search(r"(?:^|\.)lora_", name))
                                                 for name, _ in model.named_parameters()))
    predicates = diagnostic["predicates"]
    predicates.update(native_class_identity=type(model) is native_class, peft_config_absent=not present,
                      parameter_tensors_1202=total == 1202, only_cuda0=devices == {"cuda:0"},
                      only_floating_bfloat16=float_dtypes == {torch.bfloat16})
    for key, observe in (
        ("original_template", lambda: processor.model_meta.template == "llava_onevision1_5"),
        ("eager_attention", lambda: model.config._attn_implementation == "eager"),
        ("rope_initialized", lambda: bool(getattr(model, "_vlm_onevision_rope_initialized", False))),
    ):
        try:
            predicates[key] = bool(observe())
        except Exception:
            predicates[key] = None
    need(all(ok is True for ok in predicates.values()), "ACTUAL_ORIGINAL_EAGER_BASE")


def native(args, owner, facts, current):
    need(all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
         "MODELSCOPE_OFFLINE", "PYTHONDONTWRITEBYTECODE")), "EXACT_OFFLINE_RUNTIME")
    cache = Path("/tmp") / ("llava-modules-" + args.request)
    cache.mkdir(mode=0o700)
    os.environ["HF_MODULES_CACHE"] = str(cache)
    sys.path[:0] = [str(HELPERS), str(FRAMEWORK), str(SWIFT), "/vision"]
    for path in (args.input, args.input / "images", args.model, args.model_receipt, args.resume_receipt,
                 args.checkpoint, args.reload_tool, HELPERS, FRAMEWORK, SWIFT, Path("/vision")):
        readonly(path)
    for name in ("requests.jsonl", "IMAGES.json", *(f"images/{i:02d}.png" for i in range(16))):
        readonly(args.input / name)
    prompts = inputs(args.input, args.requests_sha, args.images_sha)
    current["phase"] = "provider"
    need({name: metadata.version(name) for name in VERSIONS} == VERSIONS, "ACTUAL_RUNTIME_VERSIONS")
    for name, pin in HELPER_PINS.items():
        read(HELPERS / name, 64 << 10, pin)
    for name, pin in ENGINE.items():
        read(SWIFT / "swift/infer_engine" / name, 2 << 20, pin)
    read(args.reload_tool, 48 << 10, RELOAD_TOOL)
    loader = importlib.util.spec_from_file_location("llava_clevr_cp25_reload", args.reload_tool)
    reload_contract = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(reload_contract)
    import probe
    import llava_recovered_native_cpu as recovered
    import llava_target_eager_import as eager
    need(all(Path(module.__file__) == HELPERS / (module.__name__ + ".py")
             for module in (probe, recovered, eager)), "ACTUAL_LLaVA_HELPER_PATH")
    for _ in probe.verified_files(HELPERS / "FRAMEWORK_FILES.json", probe.FRAMEWORK, FRAMEWORK, 393, nested=True):
        pass
    original = base_files(args.model, args.model_receipt, recovered)
    pins = reload_contract.checkpoint_pins(args.resume_receipt, args.checkpoint)
    need(pins["SHA256"] == CONFIG_SHA, "EXACT_ORIGINAL_CP25_CONFIG")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        readonly(args.checkpoint / name)
    raw = read(args.checkpoint / "adapter_model.safetensors", 174663096, CP25_SHA, 174663096)
    adapter_stat = identity((args.checkpoint / "adapter_model.safetensors").lstat())

    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load as load_safetensors
    from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template
    from transformers.utils import hub
    from vlm.plugins import llava_onevision

    for filename in ENGINE:
        suffix = "" if filename == "__init__.py" else "." + filename[:-3]
        module = importlib.import_module("swift.infer_engine" + suffix)
        need(Path(module.__file__) == SWIFT / "swift/infer_engine" / filename, "RESOLVED_INFERENCE_PROVIDER")
    need(Path(hub.HF_MODULES_CACHE) == cache and Path(llava_onevision.__file__) ==
         FRAMEWORK / "vlm/plugins/llava_onevision.py", "ACTUAL_PRIVATE_CACHE_AND_ONEVISION_PLUGIN")
    need(torch.version.hip == "6.3.26113" and torch.version.cuda is None
         and torch.cuda.device_count() == 1, "ACTUAL_SINGLE_HIP_RUNTIME")
    torch.cuda.set_device(0)
    facts["PCI"] = pci_id()
    need(facts["PCI"] in owner["pci"], "PCI_OUTSIDE_OWNER_ROSTER")
    saved = load_safetensors(raw)
    del raw
    need(len(saved) == 504 and sum(tensor.numel() for tensor in saved.values()) == 43646976
         and all(name.endswith((".lora_A.weight", ".lora_B.weight")) and bool(torch.isfinite(value).all())
                 for name, value in saved.items()), "ORIGINAL_CP25_TENSOR_LAYOUT")

    current["phase"] = "load_base"
    native_class = eager.preimport(args.model, llava_onevision, recovered)
    model, processor = get_model_processor(str(args.model), model_type="llava_onevision1_5", load_model=True,
                                             download_model=False, device_map={"": 0}, attn_impl="eager",
                                             torch_dtype=torch.bfloat16)
    facts["model_loads"] += 1
    original_base_gate(model, processor, native_class, torch, current)
    llava_onevision._verify_loaded_native_eager_classes(model, native_class.__module__)
    model.eval().requires_grad_(False)
    template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
                            padding_side="right", padding_free=False, truncation_strategy="raise")
    need(template.template_meta.template_type == "llava_onevision1_5", "ACTUAL_ONEVISION_TEMPLATE")
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    cfg = RequestConfig(max_tokens=CAP, temperature=0, num_beams=1, n=1, stream=False, return_details=True, stop=[])
    policy = engine._prepare_generation_config(cfg)
    need(not policy.do_sample and policy.num_beams == 1 and policy.max_new_tokens == CAP, "FIXED_GREEDY_POLICY")
    collate, generate, get_ids = template.data_collator, template.generate, template.get_generate_ids
    input_pairs = {}

    def observe_collate(*args, **kwargs):
        batch = collate(*args, **kwargs)
        ids = batch["input_ids"]
        need(tuple(ids.shape[:1]) == (1,) and ids.ndim == 2 and 1 < ids.shape[-1] <= 4096
             and not ids.is_floating_point() and batch.get("labels") is None
             and batch.get("pixel_values") is not None and current["input"] is None, "ACTUAL_NATIVE_IMAGE_BATCH")
        fields = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "image_sizes",
                  "pixel_attention_mask")
        current["input"] = {name: tensor_identity(batch[name], torch) for name in fields if batch.get(name) is not None}
        if current["arm"] == "base":
            input_pairs[current["slot"]] = current["input"]
        else:
            need(input_pairs.get(current["slot"]) == current["input"], "PAIRED_IMAGE_OR_PROMPT_CHANGED")
            facts["paired_equal_native_inputs"] += 1
        return batch

    def observe_generate(*args, **kwargs):
        policy = kwargs["generation_config"]
        need(current["input"] is not None and current["eos"] is None and not policy.do_sample
             and policy.num_beams == 1 and policy.max_new_tokens == CAP and not engine.model.training
             and kwargs.get("past_key_values") is None, "ACTUAL_GREEDY_ONEVISION_GENERATE")
        if current["arm"] == "base":
            need(engine.model is model and not hasattr(model, "peft_config")
                 and not kwargs.get("adapter_names"), "BASE_ALREADY_ADAPTED")
        else:
            need(kwargs.get("adapter_names") == [ADAPTER] and engine.model.get_base_model() is model
                 and set(engine.model.peft_config) == {ADAPTER}, "ACTUAL_CP25_ARM_WRONG_ADAPTER")
            if facts["adapter_tensors_verified"] == 0:
                live = get_peft_model_state_dict(engine.model, adapter_name=ADAPTER)
                need(set(live) == set(saved) and all(live[name].dtype == tensor.dtype
                     and live[name].shape == tensor.shape and torch.equal(live[name].detach().cpu(), tensor)
                     for name, tensor in saved.items()), "LIVE_CP25_NAMED_VALUES_DIFFER")
                facts["adapter_tensors_verified"] = len(live)
        eos = policy.eos_token_id
        current["eos"] = eos if isinstance(eos, list) else [eos]
        facts["actual_generate_calls"] += 1
        return generate(*args, **kwargs)

    def observe_ids(*args, **kwargs):
        ids = get_ids(*args, **kwargs)
        need(current["ids"] is None and ids.ndim == 2 and ids.shape[0] == 1, "SINGLE_ACTUAL_CONTINUATION")
        current["ids"] = ids[0].detach().cpu().tolist()
        return ids

    template.data_collator, template.generate, template.get_generate_ids = observe_collate, observe_generate, observe_ids
    for arm in ARMS:
        adapter = None if arm == "base" else AdapterRequest(name=ADAPTER, path=str(args.checkpoint))
        for prompt in prompts:
            slot = prompt["slot"]
            current.update(phase="generation", arm=arm, slot=slot, input=None, eos=None, ids=None)
            row = dict(arm=arm, slot=slot, outcome="error", text=None)
            try:
                response, = engine.infer([InferRequest(messages=prompt["messages"],
                    images=[str(args.input / prompt["images"][0])])], cfg, use_tqdm=False, adapter_request=adapter)
                if isinstance(response, BaseException):
                    raise response
                choice, = response.choices
                outcome = generation(current["ids"], current["eos"], engine.tokenizer.pad_token_id,
                                     choice.finish_reason, choice.token_ids)
                need(type(choice.message.content) is str and len(choice.message.content) <= 8192,
                     "ACTUAL_NATIVE_DECODED_TEXT")
                row.update(outcome=outcome, text=choice.message.content)
            except Exception:
                record(args.output, row)
                raise
            record(args.output, row)
    current["phase"] = "asset_continuity"
    need(base_files(args.model, args.model_receipt, recovered) == original
         and identity((args.checkpoint / "adapter_model.safetensors").lstat()) == adapter_stat
         and inputs(args.input, args.requests_sha, args.images_sha) == prompts,
         "INPUT_WEIGHT_FILE_CHANGED_DURING_PAIRED_RUN")


def run(args):
    owner = authority(args)
    args.output.mkdir(mode=0o700)
    facts = dict(step=owner["step"], model_loads=0, actual_generate_calls=0, adapter_tensors_verified=0,
                 paired_equal_native_inputs=0, requests_SHA256=args.requests_sha, images_SHA256=args.images_sha,
                 adapter_SHA256=CP25_SHA, greedy_max_new_tokens=CAP)
    current = dict(phase="inputs")
    failure = None

    def timed_out(*_):
        raise TimeoutError("NATIVE_SHARED_DEADLINE")

    signal.signal(signal.SIGALRM, timed_out)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, owner["deadline"] - time.time() - 10))
        native(args, owner, facts, current)
    except BaseException as error:
        failure = dict(phase=current["phase"], arm=current.get("arm"), slot=current.get("slot"),
                       type=type(error).__name__, detail_SHA256=sha(str(error).encode()))
        if current["phase"] == "load_base" and "base_diagnostic" in current:
            failure["base_diagnostic"] = current["base_diagnostic"]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    report = seal(args.output, args.request, facts, failure)
    print(f"{report['passed']} {report['predictions_SHA256']} planned={report['planned']}", flush=True)
    return 0 if report["passed"] else 2


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="mode", required=True)
    for mode in ("verify-input", "run"):
        command = children.add_parser(mode)
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--requests-sha", required=True)
        command.add_argument("--images-sha", required=True)
        if mode == "run":
            for name in ("owner", "output", "model", "model-receipt", "resume-receipt", "checkpoint", "reload-tool"):
                command.add_argument("--" + name, required=True, type=Path)
            for name in ("request", "owner-sha"):
                command.add_argument("--" + name, required=True)
    args = parser.parse_args()
    need(HEX.fullmatch(args.requests_sha) and HEX.fullmatch(args.images_sha), "INPUT_DIGEST_AUTHORITY")
    if args.mode == "verify-input":
        print(f"answerless_rows={len(inputs(args.input, args.requests_sha, args.images_sha))} native_calls=0")
        return 0
    need(HEX.fullmatch(args.owner_sha), "OWNER_DIGEST_AUTHORITY")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(cli())
