"""Exact runtime_pci function from audited T09 replay-v2 device_entry.py."""
import ctypes
import re

def runtime_pci():
    hip = ctypes.CDLL("libamdhip64.so")
    get_count = hip.hipGetDeviceCount
    get_count.argtypes = (ctypes.POINTER(ctypes.c_int),)
    get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if get_count(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError("HIP did not expose exactly one Slurm-assigned DCU")
    get_bus = hip.hipDeviceGetPCIBusId
    get_bus.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int)
    get_bus.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(32)
    if get_bus(buffer, len(buffer), 0) != 0:
        raise RuntimeError("Could not resolve the assigned DCU PCI address")
    pci = buffer.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        pci = "0000:" + pci
    if not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
        raise RuntimeError("HIP reported an invalid PCI address")
    return pci
