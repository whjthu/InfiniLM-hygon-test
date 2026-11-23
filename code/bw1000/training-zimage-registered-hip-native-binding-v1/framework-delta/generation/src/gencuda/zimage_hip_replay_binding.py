"""The original single-device Hygon COCO20 replay contract."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re


PROFILE = "hygon-coco20-first-fixed-v1"
MODEL_BYTES = 33018433789
PROGRESS_SHA256 = "a1694d2ffc3491d5c89d896cd73260f9ded215bd83e6ea5eba78e5b14403d2d0"
FUSION_SHA256 = "24b683a6a3fabc5b8ca0400c25c6b701f05aea43b54bf62d1158c64e4c123dc8"
SOURCE_TREE_SHA256 = "22f7084b30a5d84588e8496ba972f97a9405088d19ce2a85f899d4f02a6fed14"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(path, pin, limit):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and path.resolve() == path,
            "Replay archive is missing or aliased")
    require(0 < pin["bytes"] <= limit and path.stat().st_size == pin["bytes"],
            "Replay archive size differs")
    with path.open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"],
            "Replay archive bytes differ")
    return raw


def validate_hip_binding(binding):
    require(binding["platform"] == "hip" and binding["model"]["model_files"] == 20,
            "The retained Hygon runtime requires its own 20-member binding")
    root = Path(binding["model_root"])
    files = binding["model"]["files"]
    require(root.is_absolute() and len(files) == 20 and sum(p["bytes"] for p in files.values()) == MODEL_BYTES,
            "Hygon model archive has the wrong extent")
    for name, pin in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and re.fullmatch("[0-9a-f]{64}", pin["SHA256"]) is not None,
                "Unexpected Hygon runtime file identity")
    groups = [
        [str(root / f"base/transformer/diffusion_pytorch_model-{i:05}-of-00003.safetensors") for i in (1, 2, 3)],
        [str(root / f"base/text_encoder/model-{i:05}-of-00003.safetensors") for i in (1, 2, 3)],
        str(root / "base/vae/diffusion_pytorch_model.safetensors"),
    ]
    preset = root / "preset/zimage_turbo_training_adapter_v1.safetensors"
    require(binding["model_paths"] == groups and binding["preset_path"] == str(preset)
            and binding["tokenizer_path"] == str(root / "base/tokenizer")
            and all(Path(path).relative_to(root).as_posix() in files for path in [*groups[0], *groups[1], groups[2], str(preset)]),
            "Hygon model paths differ from the established native layout")
    paths = binding["initial"]["artifacts"]
    require(set(paths) == {"trainable", "rng"}
            and Path(paths["trainable"]["path"]).name == "initial.pt" and Path(paths["rng"]["path"]).name == "input.pt",
            "Hygon replay must use the original initial and pre-forward capture")
    require(binding["fused_base"]["SHA256"] == FUSION_SHA256
            and binding["source_tree_SHA256"] == SOURCE_TREE_SHA256 and len(binding["source_files"]) == 651,
            "Unexpected native source or fused base identity")
    require(binding["HIP_version"] == "6.3.26113" and binding["GPU_arch"] == "gfx936"
            and binding["GPU_total_bytes"] == 68702699520 and binding["initial_minimum_free_bytes"] == 36 << 30,
            "Unexpected Hygon device contract")


def initial_state(torch, numpy, common, binding):
    artifacts = binding["initial"]["artifacts"]
    def load(key, limit):
        pin = artifacts[key]
        return torch.load(BytesIO(read_pin(pin["path"], pin, limit)), map_location="cpu", weights_only=True)

    initial = load("trainable", 32 << 20)
    observation = load("rng", 64 << 10)
    require(type(initial) is dict and len(initial) == 476
            and all(name.startswith("pipe.dit.") and ".lora_" in name and value.dtype == torch.bfloat16
                    and value.device.type == "cpu" and bool(torch.isfinite(value).all())
                    for name, value in initial.items())
            and sum(value.numel() for value in initial.values()) == 9922560, "Unexpected original LoRA state")
    require(type(observation) is dict and set(observation) == {"batch", "RNG", "input_SHA256"}
            and common.tree_sha256(observation["batch"]) == observation["input_SHA256"]
            and observation["input_SHA256"] == binding["original_input_SHA256"], "Original input capture differs")
    native = observation["RNG"]
    require(type(native) is dict and set(native) == {"python", "numpy", "torch_CPU", "device"},
            "Incomplete original RNG capture")
    state = native["numpy"]
    require(type(state) is dict and set(state) == {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
            and state["algorithm"] == "MT19937" and len(state["keys"]) == 624
            and all(type(i) is int and 0 <= i < 2**32 for i in state["keys"]), "Invalid original NumPy state")
    rng = {"python": native["python"], "torch_cpu": native["torch_CPU"], "cuda": native["device"],
           "numpy": (state["algorithm"], numpy.asarray(state["keys"], dtype=numpy.uint32),
                     state["position"], state["has_gauss"], state["cached_gaussian"])}
    common.validate_rng(rng, [0], native=False)
    return initial, rng, common.tree_sha256(native)


def verify_fused_base(torch, model, binding, tensor_sha, check):
    pin = binding["fused_base"]
    receipt = json.loads(read_pin(pin["path"], pin, 1 << 20))
    require(receipt["passed"] is True and receipt["base_tensors"] == 521
            and receipt["changed_target_tensors"] == 240, "Original native fusion was not qualified")
    params = {}
    for name, value in model.pipe.dit.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name:
            continue
        key = name.replace(".base_layer.", ".")
        require(key not in params, "Duplicate PEFT-wrapped base tensor")
        params[key] = value
    expected = receipt["parameters"]
    require(set(params) == set(expected) and len(expected) == 521, "Fused base inventory differs")
    for name, value in params.items():
        check()
        saved = expected[name]
        require(list(value.shape) == saved["shape"] and value.dtype == torch.bfloat16
                and value.numel() == saved["elements"] and saved["expected_SHA256"] == saved["after_SHA256"]
                and tensor_sha(value) == saved["after_SHA256"], "Fused base tensor differs: " + name)
    return {"matched_tensors": len(params), "original_changed_targets": receipt["changed_target_tensors"],
            "original_fusion_receipt_SHA256": pin["SHA256"]}
