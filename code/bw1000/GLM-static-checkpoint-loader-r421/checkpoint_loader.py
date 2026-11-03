"""Static extraction of the qualified GLM raw checkpoint loader.

Import performs no Torch import, device selection, environment mutation, CDLL
load, model construction, or filesystem access. Context lifetime stays with
the resident owner; exceptions propagate to its existing cleanup path.
"""
import base64
import collections
import ctypes
import hashlib
import json
from pathlib import Path
import time


class Binding:
    def __init__(self, lib):
        self.lib = lib

    @staticmethod
    def ptr(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.modern_glm_model_error().decode())

    def unique(self):
        raw = ctypes.create_string_buffer(self.lib.modern_glm_model_id_size())
        self.check(self.lib.modern_glm_model_id(raw))
        return base64.b64encode(raw.raw).decode()


def declare(lib):
    """The original BASE prefix's CAPI declarations, without opening a DSO."""
    lib.modern_glm_model_error.restype = ctypes.c_char_p
    lib.modern_glm_model_id_size.restype = ctypes.c_size_t
    lib.modern_glm_model_id.argtypes = [ctypes.c_void_p]
    lib.modern_glm_model_create.argtypes = ([ctypes.c_int] * 2
        + [ctypes.c_size_t] * 2 + [ctypes.c_char_p] + [ctypes.c_void_p] * 3)
    lib.modern_glm_model_create.restype = ctypes.c_void_p
    lib.modern_glm_model_keys.argtypes = [ctypes.c_void_p]
    lib.modern_glm_model_keys.restype = ctypes.c_char_p
    lib.modern_glm_model_load_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.modern_glm_model_finish.argtypes = [ctypes.c_void_p]
    lib.modern_glm_model_forward.argtypes = ([ctypes.c_void_p]
        + [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 4
        + [ctypes.c_int, ctypes.c_void_p])
    lib.modern_glm_model_control.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_void_p]
    lib.modern_glm_model_reset.argtypes = [ctypes.c_void_p]
    lib.modern_glm_model_destroy.argtypes = [ctypes.c_void_p]
    return Binding(lib)


def bind_library(library_path, expected_SHA256):
    """Explicit caller-owned load; no hard-coded BASE/library/model path."""
    path = Path(library_path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_SHA256:
        raise ValueError("GLM Native library identity differs")
    return declare(ctypes.CDLL(str(path)))


def load_checkpoint(binding, ctx, *, model_path, torch, safe_open, stage, rank,
                    report, deadline_check, phase_marker,
                    expected_sdk_directory="/install/hygon-glm-5f91ad7-r3672/lib/",
                    maps_text=None, clock=time.perf_counter, progress=print):
    """Load every actual Native key, preserving original raw byte encodings.

    Torch/safetensors are real production dependencies supplied by the owner;
    only CPU checks substitute these leaves. The function does not construct,
    reset, destroy, select a device, start an engine, or claim model readiness.
    Existing owner finally/reset/destroy must surround it, including failures.
    """
    lib, check = binding.lib, binding.check
    model = Path(model_path)
    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text()
    sdk = sorted({line.split()[-1] for line in maps_text.splitlines()
                  if "libinfiniops.so" in line})
    assert len(sdk) == 1 and expected_sdk_directory in sdk[0]
    report["sdk_sha256"] = hashlib.sha256(Path(sdk[0]).read_bytes()).hexdigest()
    keys = set(json.loads(lib.modern_glm_model_keys(ctx)))
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    assert keys <= set(mapping), sorted(keys - set(mapping))[:5]
    grouped = collections.defaultdict(list)
    for key in keys:
        grouped[mapping[key]].append(key)
    pending = []
    loaded = 0
    source_bytes = 0
    begin = clock()

    def flush():
        nonlocal loaded
        if not pending:
            return
        desc = [{"name": name, "pointer": w.data_ptr(), "shape": list(w.shape),
                 "dtype": {torch.float32: 0, torch.bfloat16: 1, torch.uint8: 2}[w.dtype]}
                for name, w in pending]
        deadline_check("modern_glm_model_load_batch")
        check(lib.modern_glm_model_load_batch(ctx, json.dumps(desc).encode()))
        loaded += len(pending)
        pending.clear()
        progress("GLM78_LOAD_PROGRESS", stage, rank, loaded, len(keys),
                 round(clock() - begin, 3), flush=True)

    for filename, names in sorted(grouped.items()):
        deadline_check("safe_open")
        with safe_open(model / filename, framework="pt", device="cpu") as shard:
            for name in sorted(names):
                w = shard.get_tensor(name).contiguous()
                if w.dtype == torch.float8_e4m3fn:
                    w = w.view(torch.uint8)
                assert w.dtype in [torch.float32, torch.bfloat16, torch.uint8], (name, w.dtype)
                source_bytes += w.numel() * w.element_size()
                pending.append((name, w))
                if len(pending) >= 2048:
                    flush()
            flush()
    phase_marker("raw_weight_load.return.before_finish", rank)
    phase_marker("finish.enter", rank)
    deadline_check("modern_glm_model_finish")
    check(lib.modern_glm_model_finish(ctx))
    phase_marker("finish.return", rank)
    report.update(loaded_parameters=loaded, source_parameter_bytes=source_bytes,
                  loading_seconds=clock() - begin,
                  free_total_gpu_bytes_after_loading=torch.cuda.mem_get_info())
    progress("GLM78_WEIGHTS_READY", stage, rank, report["loading_seconds"],
             report["free_total_gpu_bytes_after_loading"], flush=True)
    return report
