"""Gate the .918 GLM fresh K/V pack exports on one physical DCU."""

import argparse
from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
from time import monotonic_ns, time
from types import ModuleType


LABEL = "GLM474-fresh-kv-pack-numeric-GPU-v1"
REQUEST = "5667064d92534d079d0edb1956750456"
PARENT, NODE = "1103606", "f12r2n13"
WHOLE_SECONDS, WORK_SECONDS = 300, 270
SOURCE = Path("/source")
AREA = Path("code/bw1000/GLM-C1-layer-observer-r474")
PLAN = AREA / "depth-stratified-observer-CPU-v1/PLAN.json"
PLAN_SHA = "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6"
HELPER = AREA / "routed-i128-one-gpu-caller-v1/hip_runtime.py"
HELPER_SHA = "74a50bc161afa681043f498977d7c0e79e9de1defb3352905013645bcda16c89"
KFD_HELPER = AREA / "TP16-FFN-f32-transport-pair-v1/assigned_kfd.py"
KFD_HELPER_SHA = "270da1e9fbf1da0cb73759afaa582b1f30f355b20ab8535cf06bb97b168d21df"
ARCHIVE = Path("results/bw1000/2026-11-22/GLM474-explicit-AR-Native-v2-original918-terminal-and-independent-ABI.json")
ARCHIVE_SHA = "6809da6c2106c7e33886d600983c1b18dc6906478a7ab650c2b05458c61f679b"
NATIVE = Path("/refactor/candidates/GLM474-fresh-kv-explicit-ar-Native-CPU-v2/libGLM_fresh_kv_pack.so")
NATIVE_SHA = "08cf47237d030dbf928b6b7a405556da26dd9630858d2a1cd506b61bd94a07d7"
NATIVE_BYTES = 1_961_800
HIP_LINK = "/opt/dtk/lib/libgalaxyhip.so.5"
HIP_NAME = "libgalaxyhip.so.5.2.26113.2065-6699a1f0"
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
SEEDS = (17, 83)
TILES = (64, 128)
SIZES = (2048, 2048, 3584, 128)
BAD_FIELDS = ("key", "value", "projected", "rotary", "stream")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_pin(path, expected, limit):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, f"artifact pin differs: {path.name}")
    return raw


def remaining(scope):
    now = time()
    left = scope["work_deadline"] - now
    if not math.isfinite(now) or not scope["epoch"] <= now or left <= 0:
        raise TimeoutError("original GPU work deadline exhausted or clock moved before origin")
    return left


def verify_intent(scope, node, step, affinity):
    expected = {"label": LABEL, "request": REQUEST, "parent": PARENT, "node": NODE,
                "archive_SHA256": ARCHIVE_SHA, "replay_allowed": False}
    need(all(type(scope.get(key)) is type(value) and scope[key] == value for key, value in expected.items()),
         "wrong single-run GPU intent")
    epoch = scope.get("epoch")
    need(type(epoch) in {float, int} and math.isfinite(epoch) and epoch > 0
         and scope.get("deadline") == epoch + WHOLE_SECONDS
         and scope.get("work_deadline") == epoch + WORK_SECONDS, "wrong GPU epoch or deadline")
    need(node == NODE and re.fullmatch(PARENT + r"\.\d+", step) and affinity == 4,
         "wrong original GPU worker placement")
    need(re.fullmatch(r"[0-9a-f]{64}", scope.get("pre_admission_SHA256", "")),
         "missing fresh pre-admission evidence pin")
    remaining(scope)


def verify_admission(snapshot, worker_step, boot_id, now):
    pci = snapshot.get("pci")
    observed = snapshot.get("monotonic_ns")
    step = snapshot.get("step", "")
    need(snapshot.get("kind") == "training_qwen_device_snapshot"
         and snapshot.get("mode") == "host_complete" and snapshot.get("node") == NODE
         and re.fullmatch(PARENT + r"\.\d+", step) and step != worker_step
         and snapshot.get("full_fd_inventory_performed") is True
         and type(snapshot.get("kfd_holder_count")) is int and snapshot["kfd_holder_count"] == 0
         and snapshot.get("boot_id") == boot_id and type(observed) is int
         and 0 <= now - observed <= 45_000_000_000, "GPU pre-admission is stale, occupied or incomplete")
    need(type(pci) is list and len(pci) == 8 and all(type(x) is str and PCI.fullmatch(x) for x in pci)
         and len(set(pci)) == 8, "fresh whole-node PCI inventory differs")
    return pci


def provider_paths(raw):
    need("not found" not in raw and "undefined symbol" not in raw, "Native provider resolution failed")
    return set(re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)", raw, re.M))


def verify_providers(scope, expected):
    result = subprocess.run(
        ["/usr/bin/ldd", "-r", str(NATIVE)], check=True, capture_output=True,
        timeout=min(30, remaining(scope)), env={**os.environ, "LC_ALL": "C"},
    )
    need(0 < len(result.stdout) <= 1024**2 and not result.stderr, "unexpected passive loader diagnostics")
    need(len(expected) == 71 and provider_paths(result.stdout.decode("utf-8")) == set(expected),
         "Native resolved provider paths differ from .918 recipe")
    total = 0
    for name, expected_sha in sorted(expected.items()):
        actual = sha256()
        with Path(name).open("rb") as handle:
            while block := handle.read(4 * 1024**2):
                remaining(scope)
                total += len(block)
                actual.update(block)
        need(actual.hexdigest() == expected_sha, f"Native provider bytes differ: {name}")
    return {"count": len(expected), "bytes": total, "passive_ldd_SHA256": sha256(result.stdout).hexdigest()}


def load_helper(path, pin, module_name):
    raw = read_pin(SOURCE / path, pin, 32_768)
    module = ModuleType(module_name)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def bindings(scope, expected, admitted_pci):
    native_raw = read_pin(NATIVE, NATIVE_SHA, NATIVE_BYTES)
    need(len(native_raw) == NATIVE_BYTES and NATIVE.resolve(strict=True) == NATIVE, "wrong .918 DSO identity")
    hip_path = Path(HIP_LINK).resolve(strict=True)
    need(hip_path == Path(HIP_LINK).with_name(HIP_NAME) and expected.get(HIP_LINK) == HIP_SHA,
         "unexpected canonical HIP provider")
    check = verify_providers(scope, expected)
    helper = load_helper(HELPER, HELPER_SHA, "verified_glm_r474_hip_runtime")
    kfd_helper = load_helper(KFD_HELPER, KFD_HELPER_SHA, "verified_glm_r474_assigned_kfd")
    lookup = C.CDLL("libdl.so.2").dladdr
    lookup.argtypes, lookup.restype = [C.c_void_p, C.POINTER(helper.DlInfo)], C.c_int
    mode = os.RTLD_NOW | os.RTLD_LOCAL
    hip_lib = C.CDLL(str(hip_path), mode=mode)
    hip = helper.Hip(hip_lib, lookup, hip_path)
    need(hip.scalar("GetDeviceCount") == 1, "one assigned DCU required")
    address = C.create_string_buffer(32)
    hip.call("DeviceGetPCIBusId", address, len(address), 0)
    pci = helper.pci_identity(address.value)
    need(pci in admitted_pci, "assigned DCU is outside fresh whole-node inventory")
    device = hip.device(pci)
    device["kfd"] = kfd_helper.assigned_kfd(pci)
    remaining(scope)
    native = C.CDLL(str(NATIVE), mode=mode)
    signature = [C.c_void_p] * 4 + [C.c_size_t, C.c_size_t, C.c_void_p]
    exports = {tile: helper.symbol(native, lookup, f"glm_r474_fresh_kv_pack_t{tile}", signature, NATIVE)
               for tile in TILES}
    need(helper.digest(NATIVE) == NATIVE_SHA and helper.digest(hip_path) == HIP_SHA,
         "active Native or HIP bytes changed")
    return hip, helper.Guarded, native, exports, check, device


def fixture(seed):
    def encode(words, salt):
        # All words are finite BF16, with both signs and distinct rows and columns.
        return struct.pack(f"<{words}H", *((0x3D00 + ((n * 193 + salt) % 1531))
                                           | (((n + salt) % 2) << 15) for n in range(words)))

    return encode(4 * 448, seed), encode(64, seed + 223)


def reference(projected, rotary):
    """Reproduce the frozen fallback's narrow/copy/contiguous contract in bytes."""
    need(len(projected) == SIZES[2] and len(rotary) == SIZES[3], "wrong fixture shape")
    rows = [projected[start:start + 896] for start in range(0, len(projected), 896)]
    return b"".join(row[:384] + rotary for row in rows), b"".join(row[384:] for row in rows)


def bad_arguments(args):
    for index, role in zip((0, 1, 2, 3, 6), BAD_FIELDS):
        yield f"null_{role}", args[:index] + (None,) + args[index + 1:]
    yield "zero_tokens", args[:4] + (0,) + args[5:]
    yield "over_max_tokens", args[:4] + (513,) + args[5:]
    yield "wrong_heads", args[:5] + (3,) + args[6:]
    for right in range(4):
        yield f"misaligned_{BAD_FIELDS[right]}", args[:right] + (C.c_void_p(args[right].value + 1),) + args[right + 1:]
        for left in range(right):
            yield f"overlap_{BAD_FIELDS[left]}_{BAD_FIELDS[right]}", args[:right] + (args[left],) + args[right + 1:]


PLANNED = {"valid": [[seed, tile] for seed in SEEDS for tile in TILES],
           "rejected": [[tile, name] for tile in TILES for name, _ in
                        bad_arguments((C.c_void_p(256),) * 4 + (1, 4, None))]}


def run_matrix(hip, buffers, stream, exports, tick, progress):
    key, value, projected, rotary = buffers
    args = tuple(buf.ptr for buf in buffers) + (1, 4, stream)
    for seed in SEEDS:
        inputs = fixture(seed)
        for buf, raw in zip((projected, rotary), inputs):
            buf.upload(raw)
        expected = reference(*inputs)
        for tile in TILES:
            tick()
            progress("valid", [seed, tile], "STARTED")
            key.sentinel(stream)
            value.sentinel(stream)
            result = exports[tile](*args)
            hip.call("StreamSynchronize", stream)
            need(result == 0, f"valid launch returned {result}")
            actual = key.readback(), value.readback()
            for label, got, want in zip(("key", "value"), actual, expected):
                if got != want:
                    index = next((i // 2 for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
                    raise ValueError(f"BF16 {label} differs at word {index}")
            for buf in (projected, rotary):
                buf.input_audit(tick)
            progress("valid", [seed, tile], "PASS", tuple(sha256(x).hexdigest() for x in actual))
    sentinel = b"\xff" * SIZES[0]
    for tile in TILES:
        for name, altered in bad_arguments(args):
            tick()
            progress("rejected", [tile, name], "STARTED")
            key.sentinel(stream)
            value.sentinel(stream)
            result = exports[tile](*altered)
            hip.call("StreamSynchronize", stream)
            need(result == 1, f"expected exact argument rejection, got {result}")
            need(key.readback() == value.readback() == sentinel, "rejected argument changed output")
            for buf in (projected, rotary):
                buf.input_audit(tick)
            progress("rejected", [tile, name], "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--pre-admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.intent.open("rb") as handle:
        raw = handle.read(16_385)
    need(0 < len(raw) <= 16_384, "unexpected intent size")
    scope = json.loads(raw)
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    verify_intent(scope, socket.gethostname().split(".")[0], step, len(os.sched_getaffinity(0)))
    # Single writer. The complete denominator precedes every attempt and survives termination.
    with args.out.open("x", encoding="utf-8") as out:
        def write(record):
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())

        def progress(group, identity, state, hashes=None):
            write(dict(group=group, id=identity, state=state, output_SHA256=hashes))

        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        startticks = int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])
        write(dict(state="STARTED", label=LABEL, request=REQUEST, step=step, node=NODE,
                   pid=os.getpid(), boot_id=boot_id, startticks=startticks,
                   intent_SHA256=sha256(raw).hexdigest(), native_SHA256=NATIVE_SHA, planned=PLANNED,
                   numerical_component_only=True, performance_qualified=False, Model_GO=False))
        try:
            pre_raw = read_pin(args.pre_admission, scope["pre_admission_SHA256"], 16_384)
            admitted_pci = verify_admission(json.loads(pre_raw), step, boot_id, monotonic_ns())
            archive = json.loads(read_pin(SOURCE / ARCHIVE, ARCHIVE_SHA, 16_384))
            need(archive["audit"]["library_SHA256"] == NATIVE_SHA and archive["independent_post"]["clean"] is True,
                 "original terminal archive differs")
            plan = json.loads(read_pin(SOURCE / PLAN, PLAN_SHA, 1_439_584))
            hip, guarded, native_owner, exports, providers, device = bindings(
                scope, plan["original169"]["providers"], admitted_pci)
            write(dict(state="BOUND", providers=providers, physical_device=device,
                       before=hip.mem(), hip_SHA256=HIP_SHA))
            with ExitStack() as cleanup:
                buffers = [guarded(hip, cleanup, size) for size in SIZES]
                with hip.stream() as (stream, *_):
                    run_matrix(hip, buffers, stream, exports, lambda: remaining(scope), progress)
                    during = hip.mem()
            remaining(scope)
            write(dict(state="PASS", during=during, after=hip.mem(), guards="bitwise", teardown="normal"))
        except Exception as error:
            write(dict(state="FAIL", type=type(error).__name__, error=str(error)[:1024]))
            raise


if __name__ == "__main__":
    main()
