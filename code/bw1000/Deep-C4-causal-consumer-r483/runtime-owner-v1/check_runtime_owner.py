"""Verify active C ABI symbol ownership without calling a probe entry point."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import stat


ROOT = Path("/refactor")
LABEL = "Deep-C4-causal-consumer-r483-runtime-owner-CPU-v1"
PROBE = ROOT / "candidates/Deep-C4-causal-consumer-r483-CPU-v2/libDeep_c4_causal_probe_r483.so"
BUILD_AUDIT = ROOT / "results/Deep-C4-causal-consumer-r483-CPU-v2-independent-1103606.156.json"
BINDING = ROOT / "incoming/Deep-C4-causal-consumer-r483/CPU-v2/native-src/binding.py"


class DlInfo(C.Structure):
    _fields_ = [
        ("file", C.c_char_p), ("base", C.c_void_p),
        ("symbol", C.c_char_p), ("address", C.c_void_p),
    ]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path, sha):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "provider is not a regular file")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        current = path.stat()
    key = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    require(key(before) == key(current) and digest.hexdigest() == sha, "artifact identity differs: " + path.name)
    return path, key(before)


def loaded_project_libraries(root):
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith(str(root) + "/"):
            continue
        path = Path(fields[5])
        if ".so" not in path.name:
            continue
        require(not fields[5].endswith(" (deleted)"), "mapped project library was unlinked")
        paths.add(path.resolve(strict=True))
    return paths


def load_and_verify(probe, audit, binding, root):
    """Return the verified live handle and its binding for a later device caller."""
    probe_expected = identity(probe, audit["probe_SHA256"])
    providers = {name: identity(name, sha) for name, sha in audit["project_provider_SHA256"].items()}
    require(len(providers) == 6 and len({value[0] for value in providers.values()}) == 6,
            "project provider denominator differs")
    required = {binding["PREFIX"] + name + "_r483" for name in binding["SIGNATURES"]}
    require(len(required) == 13 and required == set(audit["exports"]), "binding ABI differs from the build")
    expected_paths = {probe_expected[0], *(value[0] for value in providers.values())}
    require(not expected_paths.intersection(loaded_project_libraries(root)), "candidate/provider already mapped")

    dladdr = C.CDLL(None).dladdr
    dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dladdr.restype = C.c_int
    handle = C.CDLL(str(probe_expected[0]), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    api = binding["bind"](handle)
    for name, function in api.items():
        expected_symbol = binding["PREFIX"] + name + "_r483"
        address = C.cast(function, C.c_void_p)
        info = DlInfo()
        require(dladdr(address, C.byref(info)) != 0 and info.file and info.symbol
                and info.address == address.value
                and info.symbol.decode() == expected_symbol
                and Path(os.fsdecode(info.file)).resolve(strict=True) == probe_expected[0],
                "live entry owner differs: " + expected_symbol)
    require(loaded_project_libraries(root) == expected_paths, "actual mapped project DSO set differs")
    require(identity(probe, audit["probe_SHA256"]) == probe_expected, "probe changed across dlopen")
    for name, sha in audit["project_provider_SHA256"].items():
        require(identity(name, sha) == providers[name], "provider changed across dlopen")
    return handle, api, {
        "resolved_entry_points": sorted(required),
        "loaded_project_DSO_SHA256": {str(probe): audit["probe_SHA256"], **audit["project_provider_SHA256"]},
        "active_dlopen": True, "entry_calls": 0,
        "runtime_created": False, "checkpoint_weight_bytes_read": 0,
        "library_constructor_device_side_effects": "unobserved",
    }


def main():
    require(os.environ.get("SLURM_JOB_ID") == "1103606"
            and socket.gethostname().split(".")[0] == "f13r2n01"
            and all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "not the CPU-only resource scope")
    step = int(os.environ["SLURM_STEP_ID"])
    require(step > 0, "original Slurm step identity missing")
    claim = json.loads(Path(__file__).with_name("CLAIM.json").read_bytes())
    identity(BUILD_AUDIT, claim["build_audit_SHA256"])
    audit = json.loads(BUILD_AUDIT.read_bytes())
    require(audit["accounting"]["job"] == "1103606.156" and audit["passive_ELF_verified"], "wrong build audit")
    identity(BINDING, claim["binding_SHA256"])
    binding = runpy.run_path(str(BINDING))
    handle, api, result = load_and_verify(PROBE, audit, binding, ROOT)
    require(handle and len(api) == 13, "verified handle unavailable")
    result.update(actual_CPU=f"1103606.{step}", qualified_for_gpu=False)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
