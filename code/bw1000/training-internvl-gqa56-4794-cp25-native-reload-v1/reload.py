"""Load the original InternVL base and exact .4794 CP25 on a fresh CPU process."""

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import stat


BASE = Path("/InternVL3_5-8B-HF")
ADAPTER = Path("/gqa-cp25")
INVENTORY = Path("/gqa-lineage/checkpoint-25-files.json")
ARCHIVE = Path("/gqa-lineage/original-4794.json")
AUDIT = Path("/gqa-framework/vlm/runtime/first_update_audit.py")
ARCHIVE_SHA = "6b5c2264e3db2afdf675ed6dd1efe37114346952996122500e48b2e44445d599"
INVENTORY_SHA = "74d8681a3131065d398282e657f571d6ada19a13b8e69e3f210ab247b9f5937c"
NATIVE_SHA = "13ef509956869b3addf62dfa9fb404d4276198f237251928292fdf589b7d8c29"
EXPORT_SHA = "0006bf1e58e7e84e4a13a6266762f51f603fd0814bd7a880f71402d14882a603"
AUDIT_SHA = "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49"
MEMBERS = ("adapter_config.json", "adapter_model.safetensors")
TENSORS, ELEMENTS = 504, 43_646_976
VERSIONS = {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "transformers": "5.5.0",
            "ms-swift": "4.4.2", "peft": "0.19.1", "accelerate": "1.14.0", "tokenizers": "0.22.2"}
DEVICES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def readonly(path):
    need(path.resolve(strict=True) == path and bool(os.statvfs(path).f_flag & os.ST_RDONLY),
         "EXACT_READONLY_MOUNT")


def pinned(path, digest, limit, size=None, capture=False):
    need(path.resolve(strict=True) == path, "EXACT_CANONICAL_FILE")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit
             and (size is None or before.st_size == size), "EXACT_REGULAR_FILE_EXTENT")
        hasher = sha256()
        retained = bytearray()
        count = 0
        while chunk := stream.read(4 * 1024**2):
            count += len(chunk)
            need(count <= before.st_size, "PINNED_READ_GREW")
            hasher.update(chunk)
            if capture:
                retained.extend(chunk)
        need(count == before.st_size and stamp(before) == stamp(os.fstat(stream.fileno()))
             == stamp(path.lstat()) and hasher.hexdigest() == digest, "PHYSICAL_FILE_SHA_OR_IDENTITY")
    return (bytes(retained) if capture else None), stamp(before)


def check_archive(archive):
    source = archive["actual_eight_DCU"]
    final = source["verified_checkpoints"]["25"]
    need(source["actual_step"] == "1103606.4794" and source["global_updates"] == 22
         and source["resource_clean"] is True
         and (final["inventory_SHA256"], final["parameter_SHA256"], final["files"],
              final["named_tensor_count"], final["weight_elements"], final["epoch"])
         == (INVENTORY_SHA, NATIVE_SHA, 17, TENSORS, ELEMENTS, 12.5), "ORIGINAL_CP25_PROVENANCE")


def adapter_pins(inventory):
    files = inventory["files"]
    need(inventory["step"] == 25 and inventory["native_schedule_steps"] == 25
         and type(files) is dict and len(files) == 17 and set(MEMBERS) <= files.keys(),
         "EXACT_CP25_NATIVE_INVENTORY")
    pins = {name: files[name] for name in MEMBERS}
    need(all(type(row["bytes"]) is int and row["bytes"] > 0
             and re.fullmatch(r"[0-9a-f]{64}", row["SHA256"]) for row in pins.values()),
         "EXACT_CP25_ADAPTER_PAIR_SCHEMA")
    return pins


def check_values(torch, fingerprint, wrapped, original, saved, expected_sha,
                 tensor_count=TENSORS, elements=ELEMENTS):
    need(wrapped is not original and wrapped.get_base_model() is original
         and set(wrapped.peft_config) == {"default"} and list(wrapped.active_adapters) == ["default"],
         "EXACT_ACTIVE_NATIVE_CP25")
    need(len(saved) == tensor_count and sum(value.numel() for value in saved.values()) == elements,
         "EXACT_SAVED_ADAPTER_DENOMINATOR")
    live = dict(wrapped.named_parameters())
    selected = {}
    for name, tensor in saved.items():
        need(re.search(r".lora_[AB].weight$", name) is not None
             and tensor.dtype == torch.float32 and tensor.device.type == "cpu", "NATIVE_LORA_STATE_SCHEMA")
        live_name = name.removesuffix(".weight") + ".default.weight"
        value = live.get(live_name)
        need(value is not None and value.device.type == "cpu" and value.dtype == tensor.dtype
             and value.shape == tensor.shape and torch.equal(value.detach(), tensor), "NATIVE_LOADED_VALUE_MISMATCH")
        selected[live_name] = value.detach()
    live_lora = {name for name in live if re.search(r".lora_[AB]..*.weight$", name)}
    need(set(selected) == live_lora, "EXACT_NATIVE_LORA_NAMES")
    result = fingerprint(selected)
    need((result["SHA256"], result["tensors"], result["floating_elements"], result["finite"])
         == (expected_sha, tensor_count, elements, True), "ORIGINAL_4794_RANK25_NAMED_FINGERPRINT")
    return result


def run(request, step, result):
    need(all(os.environ.get(key) == value for key, value in {
        **dict.fromkeys(DEVICES, "-1"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1"}.items()), "OFFLINE_ZERO_DEVICE_PROCESS_ENVIRONMENT")
    need(os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "") == step,
         "EXACT_INDEPENDENT_NUMERIC_STEP")
    for path in (BASE, ADAPTER, INVENTORY, ARCHIVE, AUDIT):
        readonly(path)
    check_archive(json.loads(pinned(ARCHIVE, ARCHIVE_SHA, 64 * 1024, capture=True)[0]))
    pins = adapter_pins(json.loads(pinned(INVENTORY, INVENTORY_SHA, 16 * 1024, capture=True)[0]))
    need(set(p.name for p in ADAPTER.iterdir()) == set(MEMBERS), "PAIR_ONLY_ADAPTER_MOUNT")
    witnesses = {}
    for name, pin in pins.items():
        readonly(ADAPTER / name)
        limit = 16 * 1024 if name == MEMBERS[0] else 192 * 1024**2
        witnesses[name] = pinned(ADAPTER / name, pin["SHA256"], limit, size=pin["bytes"])[1]
    pinned(AUDIT, AUDIT_SHA, 64 * 1024)
    result["adapter_members_verified"] = 2
    result["phase"] = "PROVIDERS"
    need({name: metadata.version(name) for name in VERSIONS} == VERSIONS, "EXACT_ORIGINAL_PACKAGE_VERSIONS")

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from swift.model import get_model_processor
    from vlm.runtime import first_update_audit

    need(Path(first_update_audit.__file__) == AUDIT, "EXACT_NATIVE_FINGERPRINT_PROVIDER")
    need(not torch.cuda.is_initialized(), "UNINITIALIZED_DEVICE_RUNTIME")

    def reject_device_init(*args, **kwargs):
        raise RuntimeError("ZERO_GRES_LAZY_DEVICE_INIT_FORBIDDEN")

    torch.cuda._lazy_init = reject_device_init
    torch.set_num_threads(4)
    result["phase"] = "BASE_LOAD"
    with torch.no_grad():
        model, processor = get_model_processor(str(BASE), model_type="internvl", load_model=True,
                                               download_model=False, device_map={"": "cpu"},
                                               torch_dtype=torch.bfloat16, attn_impl="sdpa")
        need(processor.model_meta.template == "internvl_hf" and not hasattr(model, "peft_config")
             and {param.device.type for param in model.parameters()} == {"cpu"}
             and model.config._attn_implementation == "sdpa", "ORIGINAL_UNADAPTED_NATIVE_CPU_BASE")
        model.eval().requires_grad_(False)
        result["base_loads"] = 1
        result["phase"] = "PEFT_LOAD"
        wrapped = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False, local_files_only=True)
        need({param.device.type for param in wrapped.parameters()} == {"cpu"} and not torch.cuda.is_initialized()
             and not any(param.requires_grad for param in wrapped.parameters()), "FROZEN_CPU_MODEL_AND_ADAPTER")
        result["phase"] = "LIVE_VALUES"
        saved = load_file(str(ADAPTER / MEMBERS[1]), device="cpu")
        value = check_values(torch, first_update_audit.fingerprint, wrapped, model, saved, NATIVE_SHA)
        need(all(stamp((ADAPTER / name).lstat()) == witness for name, witness in witnesses.items()),
             "UNCHANGED_VERIFIED_ADAPTER_FILES")
    result.update(phase="COMPLETE", native_cp25_parameter_SHA256=value["SHA256"],
                  named_tensors_verified=value["tensors"], floating_elements_verified=value["floating_elements"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--step", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.request) or not re.fullmatch(r"[1-9][0-9]*[.](0|[1-9][0-9]*)", args.step):
        parser.error("unique 32-hex reload request and numeric step required")
    need(args.request != "83d7a712b3174ffe9893bb5a1cf8784b" and args.step != "1103606.4794",
         "FRESH_RELOAD_SCOPE")
    result = dict(schema="INTERNVL_GQA56_CP25_FRESH_CPU_RELOAD_V1", request=args.request, step=args.step,
                  original_step="1103606.4794", phase="INPUT_BINDINGS", effective_device="cpu",
                  original_inventory_SHA256=INVENTORY_SHA, original_archive_SHA256=ARCHIVE_SHA,
                  base_loads=0, named_tensors_planned=TENSORS, named_tensors_verified=0, model_forwards=0,
                  predictions=0, adapter_members_verified=0, optimizer_updates=0)
    code = 0
    try:
        run(args.request, args.step, result)
    except Exception as error:
        result["error_class"] = type(error).__name__
        result["error_message_SHA256"] = sha256(str(error).encode()).hexdigest()
        if type(error) is ValueError and re.fullmatch(r"[A-Z0-9_]+", str(error)):
            result["guard"] = str(error)
        code = 2
    print("INTERNVL_CP25_RELOAD_V1 " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
