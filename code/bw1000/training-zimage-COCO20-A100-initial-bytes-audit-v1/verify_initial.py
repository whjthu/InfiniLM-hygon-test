"""Check original LoRA bytes against the A100 artifact pin without transferring parameters."""

import argparse
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import tempfile


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--claim", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    claim = json.loads(args.claim.read_bytes())
    actual = claim["HIP_initial"]
    original = claim["A100_initial"]
    need(all(os.environ.get(k) == "" for k in
             ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")), "Expected CPU only")
    need(args.source.is_file() and not args.source.is_symlink() and args.source.resolve() == args.source
         and args.source.stat().st_size == actual["bytes"], "Original HIP initial tensor path or extent differs")
    with args.source.open("rb") as stream:
        raw = stream.read(actual["bytes"] + 1)
    need(len(raw) == actual["bytes"] and sha256(raw).hexdigest() == actual["SHA256"], "Original HIP tensor bytes differ")
    import torch
    import safetensors.torch

    need(not torch.cuda.is_initialized(), "Unexpected device initialization")
    torch.set_num_threads(4)
    version = importlib.metadata.version("safetensors")
    need(version == claim["A100_safetensors_version"] == "0.8.0", "The exact A100 serializer is unavailable")
    fixture = {"b": torch.tensor([2], dtype=torch.bfloat16), "a": torch.tensor([1], dtype=torch.bfloat16)}
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        path = Path(tmp) / "serializer-check.safetensors"
        safetensors.torch.save_file(fixture, str(path))
        need(path.read_bytes() == safetensors.torch.save(fixture), "In-memory and file serializers differ")
    values = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    need(type(values) is dict and len(values) == 476, "Original trainable tensor set differs")
    prefix = "pipe.dit."
    cores = {}
    for name, tensor in values.items():
        need(type(name) is str and name.startswith(prefix) and ".lora_" in name and
             isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and
             tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor).all()), "Unexpected trainable tensor")
        cores[name.removeprefix(prefix)] = tensor.contiguous()
    need(len(cores) == 476 and sum(v.numel() for v in cores.values()) == 9_922_560, "Original trainable extent differs")
    encoded = safetensors.torch.save(cores)
    reversed_dict = dict(reversed(tuple(cores.items())))
    need(encoded == safetensors.torch.save(reversed_dict), "Serializer bytes depend on dictionary order")
    result = {"case": claim["case"], "request": claim["request"], "safetensors": version, "Torch": str(torch.__version__),
              "native_trainable_tensors": len(cores), "native_trainable_elements": sum(t.numel() for t in cores.values()),
              "HIP_initial": actual, "A100_initial": original,
              "HIP_initial_reencoded": {"bytes": len(encoded), "SHA256": sha256(encoded).hexdigest()},
              "in_memory_serializer_matches_file_fixture": True, "reverse_input_order_matches": True,
              "raw_tensor_bytes_written": 0, "raw_tensor_bytes_transferred": 0, "actual_DCU": 0,
              "actual_optimizer_updates": 0, "formal_same_realized_inputs": False, "formal_numerics": False}
    result["initial_parameter_bytes_exact"] = result["HIP_initial_reencoded"] == original
    need(not torch.cuda.is_initialized(), "Unexpected accelerator use")
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
