"""Target-only tiny CPU probe of one independently pinned bert_padding source.

No model/native package/weights are loaded. Root owns any actual execution.
"""
import argparse
import hashlib
import inspect
from importlib.machinery import PathFinder
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import time

LABEL = "training-hunyuanvideo-unpad-provider-CPU-v1"
REQUEST = "942d670a2941466b93b9e686f66ec5ef"
PREFIX = "T09_UNPAD_PROVIDER_CPU_RESULT_V1:"
OUTPUT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / LABEL / "provider-result-private.json"
ACTIVE_PHASE = "entry"
CURRENT_SOURCE = None


def read_source(path, expected):
    if path.name != "bert_padding.py" or path.parent.name != "flash_attn" or path.is_symlink():
        raise ValueError("Expected the exact independently identified flash_attn/bert_padding.py")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if len(raw) > 256 * 1024 or (expected is not None and observed != expected):
        raise ValueError("Independent current provider pin differs")
    return {"path": str(path.resolve()), "bytes": len(raw), "SHA256": observed,
            "expected_independent_pin_provided": expected is not None}


def resolve_provider():
    # Resolve with the active DTK sys.path without importing flash_attn.__init__.
    package = PathFinder.find_spec("flash_attn", sys.path)
    if package is None or not package.submodule_search_locations:
        raise ImportError("Active DTK flash_attn package not found")
    module = PathFinder.find_spec("flash_attn.bert_padding", list(package.submodule_search_locations))
    if module is None or not module.origin:
        raise ImportError("Active DTK flash_attn.bert_padding provider not found")
    return Path(module.origin)


def timeout(_sig, _frame):
    raise TimeoutError("Original 40-second CPU probe budget exhausted")


def probe(path=None, expected=None):
    global ACTIVE_PHASE, CURRENT_SOURCE
    ACTIVE_PHASE = "visibility_and_active_module_resolution"
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if path is None:
        path = resolve_provider()
    source = read_source(path, expected)  # Reject before any Torch/provider import.
    CURRENT_SOURCE = source
    ACTIVE_PHASE = "Torch_CPU_import"
    import torch
    # Direct exact file execution avoids importing flash_attn.__init__ or its CUDA extension.
    # Its own imports still require source review; implicit loader GPU use remains UNKNOWN.
    ACTIVE_PHASE = "exact_provider_file_import"
    namespace = runpy.run_path(str(path))
    function = namespace["unpad_input"]
    if Path(inspect.getsourcefile(function)).resolve() != path.resolve():
        raise ValueError("Actual function implementation is not the approved provider file")
    if read_source(path, expected) != source:
        raise ValueError("Provider source changed during import")
    hidden = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64, device="cpu")
    ACTIVE_PHASE = "tiny_CPU_unpad_call"
    actual = function(hidden, mask)
    ACTIVE_PHASE = "actual_return_semantics"
    if not isinstance(actual, (tuple, list)):
        raise ValueError("Actual unpad_input result is not a sequence")
    rows = []
    finite = True
    for value in actual:
        row = {"type": type(value).__name__}
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Tiny provider probe unexpectedly returned a non-CPU tensor")
            row.update(shape=list(value.shape), dtype=str(value.dtype), device="cpu")
            finite = finite and bool(torch.isfinite(value).all().item())
        elif isinstance(value, (int, float, bool)) or value is None:
            row["scalar_type_only"] = True
            if isinstance(value, float): finite = finite and math.isfinite(value)
        else:
            raise ValueError("Unexpected provider result member type")
        rows.append(row)
    if len(actual) < 4: raise ValueError("Provider result has fewer than four protocol fields")
    identity = {"unpadded_values": torch.equal(actual[0], hidden[mask.bool()]),
                "indices": torch.equal(actual[1], torch.tensor([0, 1, 3], device="cpu")),
                "cu_seqlens": torch.equal(actual[2], torch.tensor([0, 2, 3], device="cpu")),
                "max_seqlen": int(actual[3]) == 2}
    pad_function = namespace["pad_input"]
    if Path(inspect.getsourcefile(pad_function)).resolve() != path.resolve():
        raise ValueError("Actual padding inverse is outside the approved current provider")
    ACTIVE_PHASE = "tiny_CPU_padding_roundtrip"
    padded = pad_function(actual[0], actual[1], 2, 3)
    if padded.device.type != "cpu": raise ValueError("Padding inverse unexpectedly returned non-CPU data")
    identity["padded_roundtrip"] = (tuple(padded.shape) == (2, 3, 4)
        and torch.equal(padded, hidden * mask.unsqueeze(-1)))
    finite = finite and bool(torch.isfinite(padded).all().item())
    if not finite or not all(identity.values()): raise ValueError("Tiny CPU unpad semantic identity failed")
    return {"passed": True, "actual_current_provider": source, "actual_return_arity": len(actual),
            "actual_return_metadata": rows, "source_API_five_assignment_compatible": len(actual) == 5,
            "finite": finite, "first_four_semantic_identities": identity,
            "padded_roundtrip_shape": list(padded.shape),
            "torch_version_observed": str(torch.__version__),
            "inputs": {"hidden_shape": [2, 3, 4], "mask_shape": [2, 3], "device": "cpu"},
            "model_loads": 0, "optimizer_updates": 0, "explicit_GPU_tensor_calls": 0,
            "implicit_loader_GPU_use": "UNKNOWN", "actual_original_GPU_failure_site": "STILL_UNPROVEN"}


if __name__ == "__main__":
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(40)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file", type=Path, help="Optional independently resolved exact provider")
    p.add_argument("--source-sha256", help="Optional independent pin; a supplied pin is enforced")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output != OUTPUT or not OUTPUT.parent.is_dir():
        raise ValueError("Expected the root-staged unique provider CPU namespace")
    if args.source_sha256 and args.source_file is None:
        raise ValueError("An explicit independent SHA requires its exact source file")
    if args.output.exists(): raise ValueError("Original CPU probe output already exists")
    with (OUTPUT.parent / "intent-private.json").open("x", encoding="utf-8") as stream:
        json.dump({"request_id": REQUEST, "label": LABEL, "PID": os.getpid(),
                   "epoch": time.time(), "max_inside_seconds": 40,
                   "explicit_model_loads": 0}, stream)
        stream.flush(); os.fsync(stream.fileno())
    try:
        result = probe(args.source_file, args.source_sha256)
    except Exception as error:
        result = {"passed": False, "error_type": type(error).__name__, "message": str(error)[:512],
                  "phase": ACTIVE_PHASE, "actual_current_provider": CURRENT_SOURCE,
                  "model_loads": 0, "optimizer_updates": 0, "implicit_loader_GPU_use": "UNKNOWN"}
    result["wall_seconds"] = time.monotonic() - started
    result["label"], result["request_id"] = LABEL, REQUEST
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    receipt_bytes = args.output.read_bytes()
    print(PREFIX + json.dumps({"passed": result["passed"], "request_id": REQUEST,
          "result_path": str(args.output), "result_bytes": len(receipt_bytes),
          "result_SHA256": hashlib.sha256(receipt_bytes).hexdigest(),
          "return_arity": result.get("actual_return_arity")}))
    raise SystemExit(0 if result["passed"] else 1)
