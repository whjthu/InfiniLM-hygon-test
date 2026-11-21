"""Typed HIP runtime and separately owned routed-I128 C ABI providers."""
import ctypes as C
from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import re
from types import SimpleNamespace


V, I, U, S = C.c_void_p, C.c_int, C.c_uint, C.c_size_t
GUARD_BYTES = 256
GUARD_START, GUARD_END = b"\xa5" * GUARD_BYTES, b"\x5a" * GUARD_BYTES
CHUNK_BYTES = 4 * 1024 * 1024
MEMORY_FLOOR = 8 * 1024**3


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            sha.update(block)
    return sha.hexdigest()


def file_pin(item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("missing artifact path")
    path = Path(item["path"])
    sha = item.get("sha256")
    if (not path.is_absolute() or not re.fullmatch(r".+\.so(?:\.\d+)*", path.name)
            or not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha)):
        raise ValueError("unbound shared object")
    if path.resolve(strict=True) != path or not path.is_file() or not path.stat().st_size:
        raise ValueError("absent, empty, or noncanonical shared object")
    if digest(path) != sha:
        raise ValueError("shared object SHA256 mismatch")
    return path, sha


class DlInfo(C.Structure):
    _fields_ = [("path", C.c_char_p), ("base", V), ("symbol", C.c_char_p), ("address", V)]


def symbol(lib, lookup, name, signature, path):
    fn = getattr(lib, name)
    fn.argtypes = signature
    fn.restype = I
    info = DlInfo()
    if (not lookup(C.cast(fn, V), C.byref(info)) or not info.path
            or Path(os.fsdecode(info.path)).resolve(strict=True) != path):
        raise ValueError(f"wrong shared-object provider: {name}")
    return fn


def own_symbols(pins, hip_pin):
    """Resolve each handle locally and keep all handles alive until teardown."""
    dl = C.CDLL("libdl.so.2")
    lookup = dl.dladdr
    lookup.argtypes, lookup.restype = [V, C.POINTER(DlInfo)], I
    mode = os.RTLD_NOW | os.RTLD_LOCAL
    hip_path, hip_sha = hip_pin
    hip_lib = C.CDLL(str(hip_path), mode=mode)
    hip = Hip(hip_lib, lookup, hip_path)
    providers = {}
    names = {
        "validate": ("glm_r470_i128_validate", [V, S, S, V, V]),
        "gate": ("glm_r470_i128_gate", [V] * 9 + [S, S, V]),
        "down": ("glm_r470_i128_down", [V] * 6 + [S, S, V]),
    }
    for role, (path, sha) in pins.items():
        lib = C.CDLL(str(path), mode=mode)
        api = {key: symbol(lib, lookup, name, types, path) for key, (name, types) in names.items()}
        providers[role] = SimpleNamespace(library=lib, **api)
        if digest(path) != sha:
            raise ValueError(f"mapped provider changed on disk: {role}")
    if digest(hip_path) != hip_sha:
        raise ValueError("mapped HIP provider changed on disk")
    return hip, providers


def checked(code, op):
    if code != 0:
        raise RuntimeError(f"{op} failed with status {code}")


def pci_identity(raw):
    value = raw.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", value):
        return "0000:" + value
    return value


class Hip:
    def __init__(self, lib, lookup, path):
        self.lib = lib
        signatures = {
            "GetDeviceCount": [C.POINTER(I)], "GetDevice": [C.POINTER(I)], "SetDevice": [I],
            "DeviceGetPCIBusId": [V, I, I], "RuntimeGetVersion": [C.POINTER(I)],
            "DriverGetVersion": [C.POINTER(I)], "MemGetInfo": [C.POINTER(S), C.POINTER(S)],
            "StreamCreateWithFlags": [C.POINTER(V), U], "StreamSynchronize": [V], "StreamDestroy": [V],
            "EventCreateWithFlags": [C.POINTER(V), U], "EventDestroy": [V], "EventRecord": [V, V],
            "EventSynchronize": [V], "EventElapsedTime": [C.POINTER(C.c_float), V, V],
            "Malloc": [C.POINTER(V), S], "Free": [V], "Memcpy": [V, V, S, I],
            "MemsetAsync": [V, I, S, V],
        }
        self.functions = {name: symbol(lib, lookup, "hip" + name, types, path)
                          for name, types in signatures.items()}

    def call(self, op, *args):
        checked(self.functions[op](*args), "hip" + op)

    def scalar(self, op):
        value = I()
        self.call(op, C.byref(value))
        return value.value

    def device(self, expected):
        if self.scalar("GetDeviceCount") != 1:
            raise ValueError("one visible HIP GPU required")
        self.call("SetDevice", 0)
        if self.scalar("GetDevice") != 0:
            raise ValueError("wrong HIP device")
        pci = C.create_string_buffer(32)
        self.call("DeviceGetPCIBusId", pci, len(pci), 0)
        identity = pci_identity(pci.value)
        if identity != expected:
            raise ValueError("HIP PCI does not match admission")
        return dict(pci=identity, visible_devices=1, runtime_version=self.scalar("RuntimeGetVersion"),
                    driver_version=self.scalar("DriverGetVersion"))

    def mem(self):
        free, total = S(), S()
        self.call("MemGetInfo", C.byref(free), C.byref(total))
        if not MEMORY_FLOOR <= free.value <= total.value:
            raise RuntimeError("HIP free memory fell below 8 GiB")
        return dict(free_bytes=free.value, total_bytes=total.value)

    @contextmanager
    def stream(self):
        with ExitStack() as cleanup:
            owner = V()
            self.call("StreamCreateWithFlags", C.byref(owner), 1)
            cleanup.callback(self.call, "StreamDestroy", owner)
            events = []
            for _ in range(2):
                event = V()
                self.call("EventCreateWithFlags", C.byref(event), 0)
                cleanup.callback(self.call, "EventDestroy", event)
                events.append(event)
            cleanup.callback(self.call, "StreamSynchronize", owner)
            yield (owner, *events)


def add(ptr, offset):
    return V(ptr.value + offset)


class Guarded:
    def __init__(self, hip, cleanup, size):
        self.hip = hip
        self.size = size
        self.base = V()
        self.pin = None
        hip.call("Malloc", C.byref(self.base), size + 2 * GUARD_BYTES)
        cleanup.callback(hip.call, "Free", self.base)
        self.ptr = add(self.base, GUARD_BYTES)
        self.suffix = add(self.ptr, size)
        self.copy_in(self.base, GUARD_START)
        self.copy_in(self.suffix, GUARD_END)

    def copy_in(self, destination, content):
        if isinstance(content, bytearray):
            owner = (C.c_char * len(content)).from_buffer(content)
        else:
            owner = C.c_char_p(content)
        self.hip.call("Memcpy", destination, owner, len(content), 1)

    def copy_out(self, ptr, size):
        host = C.create_string_buffer(size)
        self.hip.call("Memcpy", host, ptr, size, 2)
        return host.raw

    def upload(self, content):
        if len(content) != self.size:
            raise ValueError("HIP input byte count")
        self.copy_in(self.ptr, content)
        self.pin = hashlib.sha256(content).hexdigest()

    def sentinel(self, stream, byte=255):
        self.hip.call("MemsetAsync", self.ptr, byte, self.size, stream)

    def guards(self):
        if (self.copy_out(self.base, GUARD_BYTES) != GUARD_START
                or self.copy_out(self.suffix, GUARD_BYTES) != GUARD_END):
            raise RuntimeError("HIP input/output guard changed")

    def readback(self):
        self.guards()
        return self.copy_out(self.ptr, self.size)

    def input_audit(self, check_deadline):
        self.guards()
        sha = hashlib.sha256()
        for start in range(0, self.size, CHUNK_BYTES):
            check_deadline()
            sha.update(self.copy_out(add(self.ptr, start), min(CHUNK_BYTES, self.size - start)))
        if sha.hexdigest() != self.pin:
            raise RuntimeError("read-only input payload changed")
