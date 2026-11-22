"""Run the fixed TP8 DeepSeek caller through the real R454 checkpoint and native decode."""

import argparse
import ctypes
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
MODEL = Path("/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8")
NATIVE = Path("/refactor/results/deep-rope-width-caller-tu15-native-cpu-v1/artifacts/libDeep_inverse_rope_width_caller.so")
LOADER = Path("/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1")
DEEP_PROVIDER = Path("/refactor/incoming/public-Deep-native-provider-r427/source-v3")
GLM_PROVIDER = Path("/refactor/incoming/public-GLM-native-provider-r425")
INPUT = Path("/refactor/candidates/Deep-r467-current-profile-CPU-v1/private-input-IDs.json")
BLAS = Path("/refactor/candidates/blas-policy-r271/libblas_policy_r271.so")
CANDIDATE_ARMS = {"legacy", "h1-pack32", "h1-pack64"}
ENGINE_NAMES = ("create", "keys", "load", "finish", "forward", "rank_logits_equal", "reset", "destroy", "error")
CABI = tuple("modern_native_engine_" + name for name in ENGINE_NAMES)
ROPE = ("deep_inverse_rope_h1_pack32_v1", "deep_inverse_rope_h1_pack64_v1")
FLOOR = 16 * 1024**3


def need(valid, detail):
    if not valid:
        raise ValueError(detail)


def file_pin(path, maximum):
    need(path.is_file(), "Original input is missing: " + path.name)
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    need(0 < len(raw) <= maximum, "Unexpected file extent: " + path.name)
    return raw, {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def checked(path, expected, maximum=8 * 1024**2):
    raw, observed = file_pin(path, maximum)
    need(observed == expected, "Original bytes differ: " + path.name)
    return raw


def manifest(root, bound):
    raw = checked(root / "SOURCE_MANIFEST.json", bound)
    doc = json.loads(raw)
    for name, digest in doc["files_SHA256"].items():
        file = root / name
        need(file.resolve(strict=True).is_relative_to(root.resolve(strict=True)), "Source escaped package")
        need(file_pin(file, 2 * 1024**2)[1]["SHA256"] == digest, "Original package member differs: " + name)
    return len(doc["files_SHA256"])


def preflight(claim):
    assets = claim["assets"]
    counts = {
        "r454": manifest(LOADER, assets["loader_manifest"]),
        "r427": manifest(DEEP_PROVIDER, assets["Deep_provider_manifest"]),
        "r425": manifest(GLM_PROVIDER, assets["GLM_provider_manifest"]),
    }
    checked(NATIVE, assets["challenger"])
    checked(BLAS, assets["blas_policy"])
    checked(HERE / "load_probe.py", claim["load_probe"])
    cfg = json.loads(checked(MODEL / "config.json", assets["model_config"]))
    mapping = json.loads(checked(MODEL / "model.safetensors.index.json", assets["model_index"]))["weight_map"]
    ids = json.loads(checked(INPUT, assets["public_model_inputs"], 64 * 1024))["prompt_token_ids"]
    need(type(ids) is list and len(ids) == 8489 and all(type(token) is int and 0 <= token < cfg["vocab_size"] for token in ids),
         "Original full input differs")
    need(sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == claim["prompt_ids_SHA256"],
         "Original token sequence differs")
    need(len(mapping) == 69187 and all(Path(name).name == name and name.endswith(".safetensors")
                                      for name in mapping.values()), "Original checkpoint index differs")
    shards = sorted(set(mapping.values()))
    need(len(shards) == 46 and all((MODEL / name).stat().st_size > 8 for name in shards),
         "Original checkpoint shard roster is incomplete")
    exports = subprocess.check_output(["/usr/bin/nm", "-D", "--defined-only", str(NATIVE)], text=True, timeout=10)
    names = {line.split()[-1] for line in exports.splitlines() if line.strip()}
    need(set(CABI + ROPE) <= names, "Candidate lacks original CABI or width providers")
    return cfg, mapping, ids, {"source_members": counts, "checkpoint_index_entries": len(mapping),
                               "shards": len(shards), "input_tokens": len(ids), "CABI_symbols": len(CABI + ROPE)}


def visible_provider(lib):
    class DlInfo(ctypes.Structure):
        _fields_ = [("fname", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]

    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    for name in CABI + ROPE:
        info = DlInfo()
        need(dladdr(ctypes.cast(getattr(lib, name), ctypes.c_void_p), ctypes.byref(info))
             and Path(os.fsdecode(info.fname)).resolve() == NATIVE.resolve(),
             "Native symbol resolved to another image: " + name)


class Reason(Enum):
    EOS_TOKEN = "EOS"
    LENGTH = "cap"
    CANCELED = "canceled"


class Request:
    def __init__(self, ids):
        self.prompt_token_ids = list(ids)
        self.max_new_tokens = 4096
        self.generated_token_ids = []
        self.emitted_at = []
        self.cancel_event = threading.Event()
        self.abort_reason = None

    def emit_token(self, token):
        self.generated_token_ids.append(token)
        self.emitted_at.append(time.monotonic())


def execute(config, cfg, mapping, ids, emit):
    from load_probe import load_once, native_probe, sample_memory
    from public_deep_model import checkpoint_loader, native_context, native_decode, worker
    from public_deep_model.forward import make_forward
    import numpy as np

    worker.configure_process({"TP": 8, "PP": 1, "rows": 1, "context": 32768, "HC_mode": 0,
                              "actual_step": config["step"]})
    need(os.environ.get("INFINILM_DEEP_INVERSE_OUTPUT_ROPE") == config["arm"], "Width arm differs")
    torch = worker.torch_setup()
    binding = checkpoint_loader.declare(ctypes.CDLL(str(NATIVE), mode=ctypes.RTLD_LOCAL))
    lib = binding.lib
    visible_provider(lib)
    need(all(fn() == 0 for fn in (lib.deep_hc_canonical_get_r7130, lib.modern_deep_c4_tile_get_r5920,
                                 lib.modern_deep_layout_get_r7052, lib.deep_moe_tail_get_r7211)),
         "Frozen native policies differ")
    binding.check(lib.modern_deep_allselected_set_r813(1))

    def deadline(stage, minimum=0):
        need(config["deadline_epoch"] - time.time() >= minimum, "Original epoch exhausted: " + stage)

    def memory():
        return sample_memory(torch)

    def observe_owners():
        evidence = native_context.observe_threads(lib)
        need(evidence["status"] == "OBSERVED", "All eight native owner threads are required")
        return [row["TID"] for row in evidence["owners"]]

    def load_and_request(ctx):
        load_once(checkpoint_loader.load_checkpoint, binding, ctx, torch=torch,
                  safe_open=worker.safe_open_dependency(), mapping=mapping, memory=memory, emit=emit, deadline=deadline)
        report = {}
        forward = make_forward(lib, ctx, cfg, report, binding.check)
        req = Request(ids)
        binding.check(lib.modern_native_engine_reset(ctx))
        emit("request_started", input_tokens=len(ids), budget=4096, arm=config["arm"])
        start = time.monotonic()

        def finite(logits):
            need(logits.dtype == np.uint16 and logits.shape == (1, cfg["vocab_size"])
                 and not np.any((logits & 0x7f80) == 0x7f80), "Real BF16 logits are not finite")

        try:
            reason, fragments = native_decode.generate(
                req, forward, binding.check, finite, Reason, context=32768, chunk_size=1024,
                eos=1, vocab=cfg["vocab_size"], absolute_deadline=config["deadline_epoch"] - 12,
            )
            binding.check(lib.modern_native_engine_rank_logits_equal(ctx))
            binding.check(lib.modern_native_engine_reset(ctx))
        except BaseException:
            emit("request_partial", generated_ids=req.generated_token_ids,
                 token_timestamps=req.emitted_at, arm=config["arm"])
            raise
        count = len(req.generated_token_ids)
        natural = reason is Reason.EOS_TOKEN
        need(count >= 1 and count <= 4096 and req.generated_token_ids.count(1) == int(natural)
             and (natural or (reason is Reason.LENGTH and count == 4096)), "Inconsistent EOS or cap")
        visible = req.emitted_at[:-1] if natural else req.emitted_at
        rate = (len(visible) - 1) / (visible[-1] - visible[0]) if len(visible) > 1 else None
        after = memory()
        need(all(free >= FLOOR for free, _ in after), "Post-request memory floor violated")
        emit("request_finished", arm=config["arm"], generated_ids=req.generated_token_ids,
             token_timestamps=req.emitted_at, fragments=fragments, native_forward_counters=report,
             EOS=natural, cap=not natural, rows=1, context=32768, owner_decode_tokens_s=rate,
             owner_first_emit_s=visible[0] - start if visible else None, post_request_memory=after)

    native_probe(binding, cfg, mapping, emit, deadline, lambda: visible_provider(lib), memory,
                 observe_owners, lambda tid: Path(f"/proc/self/task/{tid}").exists(), load_and_request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(CANDIDATE_ARMS))
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    parent = os.environ["SLURM_JOB_ID"]
    step = os.environ["SLURM_STEP_ID"]
    need(parent == claim["parent"] and node in claim["nodes"] and step.isdecimal(), "Unexpected Slurm placement")
    out = Path("/audit-out") / "phase.jsonl"
    with out.open("x", encoding="utf-8") as stream:
        def emit(stage, **data):
            row = {"stage": stage, "node": node, "step": parent + "." + step, **data}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        try:
            cfg, mapping, ids, facts = preflight(claim)
            emit("assets_verified", **facts)
            if args.arm is None:
                need(not os.environ.get("SLURM_STEP_GPUS") and
                     all(os.environ.get(k) == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
                     "Expected a zero-GRES metadata preflight")
                emit("preflight_pass", explicit_GPU_steps=0, model_loads=0, checkpoint_weight_payload_bytes=0)
                return 0
            # A separate published and admitted GPU scope is necessary before invoking this branch.
            need(claim["model_GO"] and claim["arm_by_node"][node] == args.arm and args.deadline_epoch is not None
                 and 120 < args.deadline_epoch - time.time() <= claim["wall_seconds"], "GPU claim or epoch differs")
            sys.path[:0] = [str(LOADER), str(DEEP_PROVIDER), str(GLM_PROVIDER), str(HERE)]
            execute({"arm": args.arm, "step": parent + "." + step, "deadline_epoch": args.deadline_epoch},
                    cfg, mapping, ids, emit)
            emit("normal_terminal", arm=args.arm, actual_GPU_steps=1)
        except BaseException as error:
            emit("failed", error=type(error).__name__, detail=str(error)[:220])
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
