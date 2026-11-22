"""Gate the .918 GLM fresh K/V pack exports on one physical DCU."""

import argparse
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
from time import monotonic_ns, time
from types import ModuleType


LABEL = "GLM474-fresh-kv-pack-event-paired-GPU-v1"
REQUEST = "7c6ec3205bdf41578b0a6d1698787f13"
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
NUMERIC = Path("results/bw1000/2026-11-22/GLM474-fresh-kv-pack-v2-original941-device-numerics-terminal.json")
NUMERIC_SHA = "240dcb8986383f3d786264a96828f008062f977ecc72feba1343a0508c726628"
TIMING = Path("code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-event-paired-v1/timing.py")
TIMING_SHA = "6962475a0135702b647a2237eec7fc328676e5e8b277f05bcf6c0c9514f260bf"
ARCHIVE_SHA = "6809da6c2106c7e33886d600983c1b18dc6906478a7ab650c2b05458c61f679b"
NATIVE = Path("/refactor/candidates/GLM474-fresh-kv-explicit-ar-Native-CPU-v2/libGLM_fresh_kv_pack.so")
NATIVE_SHA = "08cf47237d030dbf928b6b7a405556da26dd9630858d2a1cd506b61bd94a07d7"
NATIVE_BYTES = 1_961_800
HIP_LINK = "/opt/dtk/lib/libgalaxyhip.so.5"
HIP_CANONICAL = Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_BYTES = 12349016
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
TILES = (64, 128)
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


def hip_identity(expected):
    need(expected.get(HIP_LINK) == HIP_SHA, "unexpected original HIP provider SHA pin")
    hip_path = Path(HIP_LINK).resolve(strict=True)
    need(hip_path == HIP_CANONICAL, "unexpected canonical HIP provider path")
    need(hip_path.is_file() and hip_path.stat().st_size == HIP_BYTES, "unexpected canonical HIP provider size")
    return hip_path


def bindings(scope, expected, admitted_pci):
    native_raw = read_pin(NATIVE, NATIVE_SHA, NATIVE_BYTES)
    need(len(native_raw) == NATIVE_BYTES and NATIVE.resolve(strict=True) == NATIVE, "wrong .918 DSO identity")
    hip_path = hip_identity(expected)
    check = verify_providers(scope, expected)
    check.update(canonical_HIP=str(hip_path), canonical_HIP_bytes=HIP_BYTES)
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


PLANNED = {'arms': [[1, 0, 64], [1, 0, 128], [512, 0, 64], [512, 0, 128], [1, 1, 128], [1, 1, 64], [512, 1, 128], [512, 1, 64], [1, 2, 64], [1, 2, 128], [512, 2, 64], [512, 2, 128]]}


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
            numeric = json.loads(read_pin(SOURCE / NUMERIC, NUMERIC_SHA, 16_384))
            need(numeric["state"] == "GPU_NUMERICAL_PASS_POST_CLEAN"
                 and numeric["accounting"]["gpu"][:4] == ["1103606.941", "GLM474-fresh-kv-pack-numeric-GPU-v2", "COMPLETED", "0:0"]
                 and numeric["independent_post_accounting"][0] == "1103606.943"
                 and numeric["passed_valid"] == 4 and numeric["passed_argument_rejections"] == 36
                 and numeric["provider_paths_and_SHA256_checked"] == 71, "qualified original device precedent differs")
            timing = load_helper(TIMING, TIMING_SHA, "fresh_kv_timing")
            need(timing.PLAN == PLANNED, "fixed timing denominator differs")
            plan = json.loads(read_pin(SOURCE / PLAN, PLAN_SHA, 1_439_584))
            hip, guarded, native_owner, exports, providers, device = bindings(
                scope, plan["original169"]["providers"], admitted_pci)
            write(dict(state="BOUND", providers=providers, physical_device=device,
                       before=hip.mem(), hip_SHA256=HIP_SHA))
            during = timing.run(hip, guarded, exports, lambda: remaining(scope), write)
            remaining(scope)
            write(dict(state="PASS", during=during, after=hip.mem(), guards="bitwise", teardown="normal"))
        except Exception as error:
            write(dict(state="FAIL", type=type(error).__name__, error=str(error)[:1024]))
            raise


if __name__ == "__main__":
    main()
