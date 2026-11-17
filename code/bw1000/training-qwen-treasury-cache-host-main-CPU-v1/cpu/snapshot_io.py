"""Separate immutable process streams and strictly decode one complete HOST snapshot."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import stat


@contextmanager
def separate_logs(stdout_path, stderr_path):
    if stdout_path == stderr_path:
        raise ValueError("snapshot streams must have distinct paths")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        yield stdout, stderr


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate snapshot JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-finite snapshot JSON value")


def parse_host_snapshot(raw, expected_node, expected_step):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 1024**2:
        raise ValueError("missing/oversized snapshot")
    value = json.loads(raw.decode("utf8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    keys = {"kind", "mode", "node", "step", "pci", "pci_to_kfd", "kfd_holder_count",
            "full_fd_inventory_performed", "holder_inventory_basis", "boot_id", "monotonic_ns"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("snapshot exact schema differs")
    if not (value["kind"] == "training_qwen_device_snapshot" and value["mode"] == "host_complete"
            and value["node"] == expected_node and value["step"] == expected_step
            and re.fullmatch(r"[0-9]+\.[0-9]+", expected_step)):
        raise ValueError("snapshot identity differs")
    if not (type(value["kfd_holder_count"]) is int and value["kfd_holder_count"] == 0
            and value["full_fd_inventory_performed"] is True
            and value["holder_inventory_basis"] == "kernel_KFD_plus_sameUID_FD_complete"
            and value["pci_to_kfd"] is None):
        raise ValueError("HOST holder inventory is incomplete or occupied")
    pci = value["pci"]
    if not (isinstance(pci, list) and len(pci) == 8 and all(isinstance(x, str) for x in pci)
            and len(set(pci)) == 8 and pci == sorted(pci)
            and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in pci)):
        raise ValueError("HOST PCI denominator differs")
    if not (isinstance(value["boot_id"], str) and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value["boot_id"])
            and type(value["monotonic_ns"]) is int and value["monotonic_ns"] > 0):
        raise ValueError("HOST boot/monotonic anchor differs")
    return value


def read_host_snapshot(path, expected_node, expected_step):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024**2:
        raise ValueError("unsafe snapshot stdout file")
    raw = path.read_bytes()
    after = path.lstat()
    if any(getattr(before, field) != getattr(after, field)
           for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("snapshot stdout changed during read")
    return parse_host_snapshot(raw, expected_node, expected_step)
