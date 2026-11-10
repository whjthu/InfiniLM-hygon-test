"""Construct the fixed I03 TP8 model, audit keys, and destroy before weight loading."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


SOURCE = Path("/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1")
MODEL = Path("/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8")
MANIFEST = "bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a"
NATIVE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
EXPECTED_ABSENT = {f"layers.{i}.attn.wo_a.scale" for i in range(43)} | {"mtp.0.attn.wo_a.scale"}
NODES = {f"f12r3n{i}" for i in range(12, 16)}
FLOOR = 16 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_manifest(path, expected):
    manifest_path = path / "SOURCE_MANIFEST.json"
    require(manifest_path.stat().st_size <= 2**20, "source manifest exceeds cap")
    raw = manifest_path.read_bytes()
    require(digest(raw) == expected, "source manifest identity differs")
    files = json.loads(raw)["files_SHA256"]
    for name, expected_sha in files.items():
        item = (path / name).resolve(strict=True)
        require(item.is_relative_to(path.resolve()) and item.stat().st_size <= 2**20,
                "source file outside frozen inventory")
        require(digest(item.read_bytes()) == expected_sha, f"source file digest differs: {name}")


def check_keys(raw, mapping):
    require(isinstance(raw, bytes) and len(raw) <= 16 * 1024 * 1024, "invalid native key reply")
    keys = json.loads(raw)
    require(isinstance(keys, list) and len(keys) == 67569
            and all(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,180}", key) for key in keys),
            "invalid native key list")
    selected = set(keys)
    require(len(selected) == len(keys), "duplicate native keys")
    outside = selected - mapping.keys()
    missing = selected & EXPECTED_ABSENT
    return {
        "selected_count": len(selected),
        "selected_sha256": digest(("\n".join(sorted(selected)) + "\n").encode("ascii")),
        "unindexed_count": len(outside),
        "physically_missing_known_count": len(missing),
    }


def sample_memory(torch):
    require(torch.version.hip and torch.cuda.is_available() and torch.cuda.device_count() == 8,
            "fixed engine requires eight visible HIP devices")
    values = [tuple(torch.cuda.mem_get_info(i)) for i in range(8)]
    require(all(type(free) is int and type(total) is int and 0 <= free <= total for free, total in values),
            "invalid memory sample")
    return values


def native_probe(binding, cfg, mapping, emit, deadline, after_create, memory, observe, alive):
    lib = binding.lib
    ctx = None
    owner_threads = []
    primary = None
    try:
        before = memory()
        emit("before_create", memory_bytes=before)
        require(all(free >= FLOOR for free, _ in before), "insufficient pre-create headroom")
        deadline("create", 120)
        ctx = lib.modern_native_engine_create(json.dumps(cfg).encode(), 1, 32768)
        require(ctx, "native constructor returned null")
        after = memory()
        emit("created", memory_bytes=after)
        require(all(free >= FLOOR for free, _ in after), "insufficient constructor headroom")
        after_create()
        owner_threads = observe()
        require(len(owner_threads) == 8 and len(set(owner_threads)) == 8, "eight owners not observed")
        deadline("query keys", 30)
        result = check_keys(lib.modern_native_engine_keys(ctx), mapping)
        emit("required_keys", **result)
        require(not result["unindexed_count"] and not result["physically_missing_known_count"],
                "required keys include absent checkpoint entries")
    except BaseException as error:
        primary = error
        emit("failed", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        if ctx is not None:
            try:
                binding.check(lib.modern_native_engine_destroy(ctx))
                live = sum(alive(tid) for tid in owner_threads)
                emit("destroyed", memory_bytes=memory(), observed_owner_count=len(owner_threads),
                     observed_owners_remaining=live)
                require(not live, "native owner threads survived destroy")
            except BaseException as error:
                emit("cleanup_failed", error_type=type(error).__name__, reason=str(error))
                if primary is None:
                    raise
                if hasattr(primary, "add_note"):
                    primary.add_note(f"Native destroy: {error!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    options = parser.parse_args()
    node = socket.gethostname().split(".")[0]
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    require(node in NODES and re.fullmatch(r"1087656\.\d+", step), "wrong Slurm ownership")
    remaining = options.deadline_epoch - time.time()
    require(120 < remaining <= 270, "deadline must come from this bounded dispatch")
    end = time.monotonic() + remaining

    def deadline(stage, minimum):
        require(end - time.monotonic() >= minimum, f"budget below required minimum at {stage}")

    def emit(stage, **fields):
        print(json.dumps({"kind": "i03_native_key_inventory", "stage": stage, "node": node,
                          "step": step, **fields}, allow_nan=False), flush=True)

    checked_manifest(SOURCE, MANIFEST)
    sys.path[:0] = [
        str(SOURCE), "/refactor/incoming/public-Deep-native-provider-r427/source-v3",
        "/refactor/incoming/public-GLM-native-provider-r425",
    ]
    from public_deep_model import checkpoint_loader, native_binding, native_context, worker

    require(Path(native_context.__file__).resolve().is_relative_to(SOURCE.resolve()),
            "unexpected native context package")
    config = {"TP": 8, "PP": 1, "rows": 1, "context": 32768, "HC_mode": 0, "actual_step": step}
    worker.configure_process(config)
    _, receipt, _ = native_binding.binding()
    require(receipt["library_SHA256"] == NATIVE, "unexpected Native library")
    cfg, mapping = checkpoint_loader.read_model_metadata(MODEL)
    require(len(mapping) == 69187 and EXPECTED_ABSENT <= mapping.keys(), "unexpected checkpoint index")
    torch = worker.torch_setup()
    sample_memory(torch)
    binding = checkpoint_loader.bind_library(receipt["library"], NATIVE)
    lib = binding.lib
    require(all(getter() == 0 for getter in (lib.deep_hc_canonical_get_r7130,
            lib.modern_deep_c4_tile_get_r5920, lib.modern_deep_layout_get_r7052,
            lib.deep_moe_tail_get_r7211)), "expected frozen Native policy")
    binding.check(lib.modern_deep_allselected_set_r813(1))

    def after_create():
        native_context.providers(binding, receipt)
        class Info(ctypes.Structure):
            _fields_ = [(name, typ) for name, typ in (("fname", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p))]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
        dladdr.restype = ctypes.c_int
        info = Info()
        require(dladdr(ctypes.cast(lib.modern_native_engine_keys, ctypes.c_void_p), ctypes.byref(info))
                and Path(os.fsdecode(info.fname)).resolve() == Path(receipt["library"]).resolve(),
                "unexpected Native key provider")

    def observe():
        report = native_context.observe_threads(lib)
        require(report["status"] == "OBSERVED", "could not identify all native owners")
        return [row["TID"] for row in report["owners"]]

    emit("admitted", rows=1, capacity=32768, tp=8, pp=1, native_sha256=NATIVE)
    native_probe(binding, cfg, mapping, emit, deadline, after_create,
                 lambda: sample_memory(torch), observe, lambda tid: Path(f"/proc/self/task/{tid}").exists())
    emit("complete", weights_loaded=False, forward_calls=0, model_ready=False)


if __name__ == "__main__":
    main()
