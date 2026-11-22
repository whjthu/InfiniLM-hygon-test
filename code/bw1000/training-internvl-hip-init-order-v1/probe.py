"""Report the first HIP enumeration boundary in a separate container process."""

import ctypes
import json
import os
from pathlib import Path
import re
import sys


def emit(phase, **fields):
    print(json.dumps(dict(phase=phase, **fields), sort_keys=True), flush=True)


def runtime():
    emit("runtime_enter")
    import native

    report = native.runtime_identity()
    emit("runtime_return", torch_device_initialized=report["torch_device_initialized"],
         versions=report["versions"])


def hip():
    emit("hip_dlopen_enter", torch_loaded="torch" in sys.modules)
    lib = ctypes.CDLL("libamdhip64.so")
    providers = sorted({row.rsplit(None, 1)[-1] for row in Path("/proc/self/maps").read_text().splitlines()
                        if "/libamdhip64.so" in row})
    emit("hip_dlopen_return", providers=providers)
    lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.hipGetDeviceCount.restype = ctypes.c_int
    count = ctypes.c_int(-1)
    emit("hip_count_enter")
    rc = lib.hipGetDeviceCount(ctypes.byref(count))
    emit("hip_count_return", rc=rc, count=count.value)
    if rc != 0 or count.value != 1:
        raise ValueError("HIP did not enumerate the exact one-card Slurm step")
    lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    rc = lib.hipDeviceGetPCIBusId(buf, len(buf), 0)
    address = buf.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", address):
        address = "0000:" + address
    if rc != 0 or not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", address):
        raise ValueError("HIP PCI lookup failed")
    emit("hip_pci_return", PCI=address)


def main(mode, stage):
    sys.path.insert(0, stage)
    filesystems = {}
    for name in ("/tmp", "/dev/shm", "."):
        value = os.statvfs(name)
        filesystems[name] = dict(bytes_available=value.f_bavail * value.f_frsize,
                                 inodes_available=value.f_favail)
    emit("start", mode=mode, statvfs=filesystems)
    arms = {"hip_first": (hip, runtime), "runtime_first": (runtime, hip)}
    for operation in arms[mode]:
        operation()
    emit("complete", mode=mode)


if __name__ == "__main__":
    main(*sys.argv[1:])
