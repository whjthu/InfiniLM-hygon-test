"""Exercise the original DiffSynth VAE encoder with the sealed target weights."""

import argparse
import ast
import copy
import csv
import ctypes
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
TRAIN = Path("/zimage-input/train")
NATIVE = Path("/native-source")
INPUT_SOURCE = Path("/original-input-stage/source/tools/preflight_zimage_style18_train_input.py")
WEIGHT = Path("/sealed-vae/diffusion_pytorch_model.safetensors")
CONFIG = WEIGHT.with_name("config.json")
WEIGHT_PIN = {"bytes": 167666902,
              "SHA256": "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(path, pin):
    require(path.resolve(strict=True) == path and path.is_file(), "Expected canonical file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size == pin["bytes"], "Unexpected file extent")
        data = stream.read(pin["bytes"] + 1)
        after = os.fstat(stream.fileno())
    stamp = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    require(stamp(before) == stamp(after) == stamp(path.stat()) and len(data) == pin["bytes"]
            and sha256(data).hexdigest() == pin["SHA256"], "Pinned bytes changed")
    return data


def module(path, name, pin):
    source = exact(path, pin)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), result.__dict__)
    return result


def loaded_pci(pre):
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    require(hip.hipDeviceGetPCIBusId(buf, len(buf), 0) == 0, "HIP PCI identity unavailable")
    pci = buf.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        pci = "0000:" + pci
    require(pci in pre["PCIs"], "Allocated device differs from node inventory")
    return pci


def get_preprocess(torch, np, repeat):
    path = NATIVE / "diffsynth/diffusion/base_pipeline.py"
    raw = exact(path, {"bytes": 21062, "SHA256": "13d916193a7838de41ca60af36dbddd5813075dff78846f5ef8e8ac1f2d1f43b"})
    tree = ast.parse(raw, filename=str(path))
    [owner] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BasePipeline"]
    [entry] = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "preprocess_image"]
    runtime = {"torch": torch, "np": np, "repeat": repeat}
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), "exec"), runtime)
    return runtime["preprocess_image"]


def tensor_hash(torch, tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha256(data).hexdigest()


def run(args, result):
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    require(args.deadline == args.epoch + claim["whole_seconds"]
            and args.epoch <= time.time() < args.epoch + claim["work_seconds"], "Original epoch differs")
    require(socket.gethostname().split(".")[0] == claim["node"]
            and os.environ["SLURM_JOB_ID"] == claim["parent"]
            and os.environ["SLURM_STEP_ID"].isdigit()
            and int(os.environ["SLURM_CPUS_PER_TASK"]) == claim["cpus"], "Actual Slurm worker identity differs")
    result["step"] = claim["parent"] + "." + os.environ["SLURM_STEP_ID"]
    result["phase"] = "SOURCE_AND_INPUT_BINDING"
    import torch
    import numpy as np
    from einops import repeat
    from safetensors.torch import load

    require(torch.version.hip and not torch.cuda.is_initialized(), "Expected fresh HIP Torch")
    torch.set_num_threads(4)
    source = json.loads((HERE / "SOURCE_FILES.json").read_bytes())
    entry = module(INPUT_SOURCE, "frozen_native12_dataset", claim["dataset_entry"])
    encoder = module(HERE / "upstream/flux_vae.py", "frozen_flux_vae", source["files"]["upstream/flux_vae.py"])
    converter = module(HERE / "upstream/flux_vae_converter.py", "frozen_flux_vae_converter",
                       source["files"]["upstream/flux_vae_converter.py"])
    binding = entry.load_binding()
    csv_path = entry.inspect_inputs(TRAIN, binding)
    dataset_type, providers = entry.check_native_entry(NATIVE)
    dataset = dataset_type(base_path=str(TRAIN), metadata_path=str(csv_path), data_file_keys=("image",),
                           main_data_operator=dataset_type.default_image_operator(
                               base_path=str(TRAIN), height=256, width=256, height_division_factor=16,
                               width_division_factor=16))
    require(len(dataset) == 12, "Frozen sample roster differs")
    preprocess = get_preprocess(torch, np, repeat)
    result["phase"] = "WEIGHT_HASH_STRICT_CONVERSION"
    started = time.monotonic()
    config = json.loads(exact(CONFIG, claim["sealed_config"]))
    require(config["_class_name"] == "AutoencoderKL" and config["latent_channels"] == 16
            and config["scaling_factor"] == .3611 and config["shift_factor"] == .1159, "Original VAE config differs")
    checkpoint = load(exact(WEIGHT, WEIGHT_PIN))
    converted = converter.FluxVAEEncoderStateDictConverterDiffusers(checkpoint)
    with torch.device("meta"):
        reference = encoder.FluxVAEEncoder(use_conv_attention=False)
    require(set(reference.state_dict()) == set(converted), "VAE converter key closure differs")
    reference.load_state_dict(converted, strict=True, assign=True)
    result["checkpoint_tensor_count"] = len(checkpoint)
    del checkpoint, converted
    reference.to(device="cpu", dtype=torch.float32).requires_grad_(False).eval()
    result["encoder_tensor_count"] = len(reference.state_dict())
    result["encoder_parameter_count"] = sum(p.numel() for p in reference.parameters())
    result["load_hash_and_convert_host_s"] = time.monotonic() - started

    result["phase"] = "ONE_HIP_DEVICE"
    pre = json.loads(args.pre.read_bytes())
    require(len(pre["PCIs"]) == 8 and not pre["kfd_process_count"] and not pre["own_device_fd_process_count"]
            and pre["node"] == claim["node"], "Node preflight was not clean")
    require(torch.cuda.device_count() == 1, "Expected exactly one allocated device")
    torch.cuda.set_device(0)
    result["actual_PCI"] = loaded_pci(pre)
    model = copy.deepcopy(reference).to(device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    result["phase"] = "NATIVE12_VAE_LATENTS"
    cpu_options = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32)
    gpu_options = SimpleNamespace(device=torch.device("cuda:0"), torch_dtype=torch.bfloat16)
    result["torch"] = str(torch.__version__)
    result["HIP"] = torch.version.hip
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for ordinal, sample_pin in enumerate(binding["rows"]):
            row = dataset[ordinal]
            image = row["image"]
            require(isinstance(image, providers.Image.Image) and image.size == (256, 256) and image.mode == "RGB"
                    and sha256(image.tobytes()).hexdigest() == sample_pin["RGB256_SHA256"]
                    and sha256(row["prompt"].encode()).hexdigest() == sample_pin["prompt_SHA256"],
                    "Native image or prompt changed")
            cpu_input = preprocess(cpu_options, image)
            gpu_input = preprocess(gpu_options, image)
            reference_latents = reference(cpu_input)
            torch.cuda.synchronize()
            started = time.monotonic()
            gpu_latents = model(gpu_input)
            torch.cuda.synchronize()
            encode_s = time.monotonic() - started
            require(list(gpu_latents.shape) == list(reference_latents.shape) == [1, 16, 32, 32]
                    and gpu_latents.dtype == torch.bfloat16 and torch.isfinite(gpu_latents).all().item()
                    and torch.isfinite(reference_latents).all().item(), "Nonfinite or malformed VAE latents")
            delta = gpu_latents.float().cpu() - reference_latents
            record = {"ordinal": ordinal, "shape": list(gpu_latents.shape), "dtype": str(gpu_latents.dtype),
                      "latent_SHA256": tensor_hash(torch, gpu_latents), "encode_host_sync_s": encode_s,
                      "max_abs_vs_cpu_fp32": delta.abs().max().item(),
                      "relative_l2_vs_cpu_fp32": (delta.norm()/reference_latents.norm()).item()}
            if ordinal == 0:
                record["same_device_repeat_bitwise"] = tensor_hash(torch, model(gpu_input)) == record["latent_SHA256"]
            result["rows"].append(record)
    entry.inspect_inputs(TRAIN, binding)
    result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"rows": [], "planned_rows": 12, "optimizer_updates": 0,
              "whole_model_loaded": False, "A100_latent_parity_qualified": False,
              "training_ready": False, "passed": False, "phase": "ASSIGNED_STEP"}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("VAE worker deadline")))
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time() - 50))
    try:
        run(args, result)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:220])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: result.get(key) for key in ("passed", "phase", "step", "error_type")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
