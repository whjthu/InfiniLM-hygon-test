"""Run paired selected-value comparisons inside the rank owner process."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import threading

from checkpoint_ranges import load_owner

CAPACITY = 32768
MEASURED_RANKS = (0, 8)
# 1313 is a historical upper bound, used here solely as a controlled shape.
CASES = ((65, 64), (1313, 1312), (2048, 32767))
ORDERS = (0, 1, 1, 0) * 4
OWNER = "/refactor/results/GLM474-selected-device-owner-CPU-v2/libGLM_selected_device_owner.so"
OWNER_SHA = "8a57a1bc9d1938255c1a68aa2d2c8906af1a3e5099f249488bae7f73456ebcf4"
NATIVE = "/refactor/results/GLM474-ffn-window-observer-CPU-v1/libGLM_ffn_window_observer.so"
NATIVE_SHA = "017e981ace530ec762bf56bc981a73fd8bcb5c00afc4f5e8062403ac652a424e"
BINDING_SHA = "f356bdc3e47c92bce3f666c6ec4e80664681e0083e818f2a9acaa169826acfa7"
FIELDS = ("original_ns", "packed_ns", "elements", "different_bits",
          "nonfinite_original", "nonfinite_packed", "packed_cached_dispatch")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs():
    # Signed, finite, nonzero BF16, varying across both rows and columns.
    def cache(width, seed):
        period = 97
        words = [((0x3d00 + ((row*31 + col*17 + seed) % 383)) |
                  (0x8000 if (row*13 + col*7 + seed) % 11 < 4 else 0))
                 for row in range(period) for col in range(width)]
        tile = struct.pack(f"<{len(words)}H", *words)
        repeat, tail = divmod(CAPACITY, period)
        return tile*repeat + tile[:tail*width*2]
    return cache(512, 3), cache(64, 59)


def causal_ids(rows, last):
    if (rows, last) not in CASES:
        raise ValueError("unknown frozen workload shape")
    ids = [i*last//(rows-1) for i in range(rows)]
    return struct.pack(f"<{rows}q", *ids)


def measured_pairs(compare, checkpoint, numerical_gate):
    """Record all failures and enforce the numerical gate before timing."""
    plan = [("numbers", rows, last, 0) for rows, last in CASES]
    plan += [("timing", rows, last, order)
             for rows, last in CASES for order in ORDERS]
    records = []
    checkpoint(plan, records)
    for phase, rows, last, order in plan:
        if phase == "timing" and len(records) == len(CASES):
            numerical_gate()
        record = dict(phase=phase, rows=rows, last_visible=last, packed_first=bool(order))
        records.append(record)
        try:
            rc, values = compare(rows, last, order)
            record.update(rc=rc, **dict(zip(FIELDS, values)))
            if (rc != 0 or len(values) != len(FIELDS) or values[2] != rows*4*256
                    or any(values[3:6]) or values[6] not in (0, 1)
                    or min(values[:2]) <= 0):
                raise ValueError("selected-value outputs or timing invalid")
        except BaseException as exc:
            record["error"] = str(exc)[:240]
            raise
        finally:
            checkpoint(plan, records)
    return records


class DlInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("base", C.c_void_p),
                ("symbol", C.c_char_p), ("address", C.c_void_p)]


def bind_library(path, pin, prefix, signatures):
    if sha(Path(path).read_bytes()) != pin:
        raise ValueError("DSO identity changed")
    lib = C.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOW)
    address = C.CDLL("libdl.so.2").dladdr
    address.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    address.restype = C.c_int
    api = {}
    for key, (args, result) in signatures.items():
        f = getattr(lib, prefix + key)
        info = DlInfo()
        if (not address(C.cast(f, C.c_void_p), C.byref(info)) or not info.name
                or Path(os.fsdecode(info.name)).resolve() != Path(path).resolve()):
            raise ValueError(f"unexpected ELF provider for {key}")
        f.argtypes, f.restype = args, result
        api[key] = f
    return lib, api


def new_uids():
    """Called once by the owner that publishes the two private CCL IDs."""
    V, Z, I = C.c_void_p, C.c_size_t, C.c_int
    _, api = bind_library(NATIVE, NATIVE_SHA, "modern_glm_global_",
                          {"id_size_r470": ([], Z), "id_r470": ([V, Z], I)})
    size = api["id_size_r470"]()
    if not 0 < size <= 4096:
        raise ValueError("unexpected CCL ID size")
    ids = []
    for _ in range(2):
        buf = C.create_string_buffer(size)
        if api["id_r470"](buf, size):
            raise RuntimeError("CCL ID generation failed")
        ids.append(buf.raw)
    if ids[0] == ids[1]:
        raise ValueError("duplicate CCL IDs")
    return tuple(ids)


def run_owner(node, device, world, tp, model, binding_path, gate, save):
    """Execute on the originating rank thread with cross-node gate callbacks."""
    if (type(node) is not int or node not in (0, 1) or type(device) is not int
            or not 0 <= device < 8 or os.environ.get("SLURM_JOB_ID") != "1103606"):
        raise ValueError("unexpected component allocation or owner")
    if not isinstance(world, bytes) or not isinstance(tp, bytes) or not world or (
            len(world) != len(tp) or len(world) > 4096 or world == tp):
        raise ValueError("invalid CCL IDs")
    rank = node*8 + device
    measured = rank in MEASURED_RANKS
    raw = binding_path.read_bytes()
    if sha(raw) != BINDING_SHA:
        raise ValueError("checkpoint binding changed")
    binding = json.loads(raw)
    raw_config = (model / "config.json").read_bytes()
    if sha(raw_config) != binding["config_SHA256"]:
        raise ValueError("checkpoint config changed")
    config = json.loads(raw_config)
    config["glm_global_tp16_r470"] = True
    V, P, Z, I = C.c_void_p, C.c_char_p, C.c_size_t, C.c_int
    lib, api = bind_library(OWNER, OWNER_SHA, "glm_selected_device_", {
        "create": ([I,I,P,P,P,Z], V), "error": ([],P), "close": ([V],I),
        "load": ([V,P,Z,P,Z],I),
        "measure": ([V,P,Z,P,Z,P,Z,C.c_int64,I,C.POINTER(C.c_uint64),Z],I),
    })
    cuda = C.CDLL("/opt/dtk/cuda/cuda/lib64/libcudart.so.12")
    pci_lookup = cuda.cudaDeviceGetPCIBusId
    pci_lookup.argtypes, pci_lookup.restype = [P, I, I], I
    pci = C.create_string_buffer(64)
    if pci_lookup(pci, len(pci), device):
        raise RuntimeError("device PCI identity unavailable")
    report = dict(rank=rank, device=device, measured=measured, TP=16, context=CAPACITY,
                  physical_PCI=pci.value.decode("ascii").lower(), UID_SHA256=[sha(world),sha(tp)],
                  step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  owner_PID=os.getpid(), owner_TID=threading.get_native_id(),
                  startticks=Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
                  owner_library_SHA256=OWNER_SHA, checkpoint_binding_SHA256=BINDING_SHA,
                  inputs="controlled causal component caches",
                  passed=False, normal_close=False)
    save(report)
    context = None
    try:
        gate("before-create", report)
        context = api["create"](node,device,json.dumps(config).encode(),world,tp,len(world))
        if not context:
            raise RuntimeError(api["error"]())
        report["both_ccl_initialized"] = True
        gate("after-create", report)
        if measured:
            load_owner(lib, context, model, binding, rank)
            latent, rotary = inputs()
            index = {(rows,last):causal_ids(rows,last) for rows,last in CASES}
            report.update(latent_SHA256=sha(latent), rotary_SHA256=sha(rotary),
                          index_SHA256=[sha(index[key]) for key in CASES])
            def checkpoint(plan, records):
                report.update(plan=plan, records=records)
                save(report)
            def compare(rows, last, order):
                data = (C.c_uint64*len(FIELDS))()
                rc = api["measure"](context,latent,len(latent),rotary,len(rotary),
                                    index[rows,last],rows,last,order,data,len(data))
                return rc, list(data)
            measured_pairs(compare, checkpoint, lambda: gate("numerics-passed", report))
        else:
            gate("numerics-passed", report)
        gate("before-close", report)
        pointer, context = context, None
        if api["close"](pointer):
            raise RuntimeError(api["error"]())
        report["normal_close"] = True
        gate("after-close", report)
        report["passed"] = True
    except BaseException as exc:
        report["error"] = str(exc)[:240]
        raise
    finally:
        if context:
            report["error_cleanup_rc"] = api["close"](context)
        save(report)
    return report
