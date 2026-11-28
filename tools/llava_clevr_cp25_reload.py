"""Load the exact .5000 adapter into the previously qualified OneVision CPU path."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


NATIVE_SHA = "d94ddf5a68f9eeb2f3fe19396989fd87a2bd5df3bbfe0ab16ceb525f84e782ad"
TRAINED_RECEIPT_SHA = "06507397652e12b4e6e5773810f664d7305041f2668ae523ee390fd149c78b04"
TRAINED_REQUEST = "0e4a27af103c45e7b3bddcaeeb7d7c67"
WEIGHT_PIN = {"bytes": 174663096, "SHA256": "b8b8a8b9dfe5012c4ce5f80499a1b3fc5e7b8b140a7a734cada07c4a39b83719"}
INSERTION = '    return dict(schema="LLAVA_RECOVERED_ORIGINAL_ACTUAL_NATIVE_CPU_LOAD_V1",'


def need(condition, evidence):
    if not condition:
        raise ValueError(evidence)


def safe_member(path, limit):
    info = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit,
         "A checkpoint member is not a bounded canonical file")
    return info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size


def checked_bytes(path, digest, limit):
    stamp = safe_member(path, limit)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(len(data) == stamp[-1] and safe_member(path, limit) == stamp and sha256(data).hexdigest() == digest,
         "A pinned member has different bytes")
    return data


def checkpoint_pins(receipt, checkpoint):
    need(checkpoint.name == "checkpoint-25" and checkpoint.resolve(strict=True) == checkpoint,
         "Expected the exact original CP25 directory")
    outcome = json.loads(checked_bytes(receipt, TRAINED_RECEIPT_SHA, 16 << 10))
    need(outcome["request"] == TRAINED_REQUEST and outcome["passed"] is True
         and (outcome["witnessed_forwards"], outcome["rank_optimizer_returns"]) == (736, 184),
         "Original checkpoint provenance differs")
    original = outcome["checkpoints"]["25"]
    config = original["adapter_config.json"]
    need(len(original) == 17 and original["adapter_model.safetensors"] == WEIGHT_PIN,
         "Original checkpoint inventory differs")
    data = checked_bytes(checkpoint / "adapter_config.json", config["SHA256"], 32 << 10)
    need(len(data) == config["bytes"], "Adapter config extent differs")
    original_config = json.loads(data)
    need((original_config["r"], original_config["lora_alpha"], original_config["lora_dropout"],
          original_config["bias"], original_config["peft_type"]) == (16, 32, 0.05, "none", "LORA"),
         "Original LoRA config differs")
    return config


def read_weights(torch, model, trainable, path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load

    data = checked_bytes(path, WEIGHT_PIN["SHA256"], WEIGHT_PIN["bytes"])
    need(len(data) == WEIGHT_PIN["bytes"], "Adapter weight extent differs")
    saved = load(data)
    del data
    native = dict(trainable)
    need(len(saved) == len(native) == 504, "Expected 504 checkpoint and runtime tensors")
    names = {name.removesuffix(".weight") + ".default.weight": name for name in saved}
    need(set(names) == set(native) and len(names) == len(saved)
         and sum(value.numel() for value in saved.values()) == 43646976, "Saved LoRA names or elements differ")
    changed = 0
    for param_name, saved_name in names.items():
        expected = saved[saved_name]
        value = native[param_name].detach()
        need(saved_name.endswith((".lora_A.weight", ".lora_B.weight")) and value.shape == expected.shape
             and value.dtype == expected.dtype and bool(torch.isfinite(expected).all()),
             "Saved adapter tensor contract differs")
        changed += not torch.equal(value, expected)
    need(changed > 0, "Checkpoint must differ from the freshly initialized adapter")
    loaded = set_peft_model_state_dict(model, saved, adapter_name="default")
    need(not loaded.unexpected_keys and not set(loaded.missing_keys) & set(names)
         and set(model.peft_config) == {"default"} and list(model.active_adapters) == ["default"],
         "Native adapter assignment or activation differs")
    for param_name, saved_name in names.items():
        value = native[param_name].detach()
        need(value.device.type == "cpu" and torch.equal(value, saved[saved_name]),
             "Native adapter tensor differs after actual assignment")
    need(not torch.cuda.is_initialized(), "The fresh CPU process initialized a GPU")
    return {"loaded_tensors": len(saved), "loaded_elements": 43646976, "different_from_initial_tensors": changed}


def derive(source):
    need(sha256(source).hexdigest() == NATIVE_SHA, "The qualified native loader SHA differs")
    text = source.decode("utf-8")
    need(text.count(INSERTION) == 1, "Qualified native loader return boundary changed")
    text = text.replace(INSERTION, "    trained = read_trained(torch, model, trainable)\n"
                        "    phase('original_CP25_values_verified')\n" + INSERTION + " trained_adapter=trained,")
    return compile(text, "qualified_llava_native_with_cp25_readback.py", "exec")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("native-loader", "vision-manifest", "model", "native-binding", "model-receipt", "resume-receipt",
                 "checkpoint", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    need(all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                                     "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID", "").isdigit(),
         "An actual zero-GRES Slurm step with GPU visibility disabled is required")
    config = checkpoint_pins(args.resume_receipt, args.checkpoint)
    source = checked_bytes(args.native_loader, NATIVE_SHA, 48 << 10)
    module = ModuleType("llava_cp25_verified_derivative")
    need(args.vision_manifest.name == "VISION_FILES.json", "Unexpected vision manifest basename")
    module.__file__ = str(args.vision_manifest.with_name(args.native_loader.name))
    module.read_trained = lambda torch, model, trainable: read_weights(
        torch, model, trainable, args.checkpoint / "adapter_model.safetensors")
    sys.modules[module.__name__] = module
    with TemporaryDirectory(dir="/tmp", prefix="llava-cp25-modelscope-") as cache:
        os.environ["MODELSCOPE_CACHE"] = cache
        exec(derive(source), module.__dict__)
        report = module.main(args.model, args.native_binding, args.model_receipt)
    report.update(schema="LLAVA_CLEVR_ORIGINAL5000_CP25_INDEPENDENT_CPU_RELOAD_V1",
                  source_step="1103606.5000", resume_receipt_SHA256=TRAINED_RECEIPT_SHA,
                  checkpoint_adapter_SHA256=WEIGHT_PIN["SHA256"], checkpoint_adapter_config_SHA256=config["SHA256"],
                  step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  independent_blind_effect=False, optimizer_updates=0, device_cards=0)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
