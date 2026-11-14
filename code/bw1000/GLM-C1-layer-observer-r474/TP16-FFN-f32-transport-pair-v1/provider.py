"""Checked CUDA and InfiniCCL bindings for the guarded GLM replay."""
import ctypes as C
import hashlib
from pathlib import Path

from collective import replay


CUDA_PATH = "/opt/dtk/cuda/cuda/lib64/libcudart.so.12"
CCL_PATH = "/refactor/install/hygon-glm-5f91ad7-r3672/lib/libinfiniccl.so"
CCL_SHA256 = "76eb65adacc8a6832970991ca276b32562543b7fb7e8dd09cda9f4c269f24809"
F32 = 10
SUM = 0
UID_BYTES = 136


def bind(library, name, *argtypes):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = C.c_int

    def checked(*args):
        status = function(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with status {status}")

    return checked


class Provider:
    def __init__(self):
        hasher = hashlib.sha256()
        with Path(CCL_PATH).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != CCL_SHA256:
            raise ValueError("GLM InfiniCCL provider SHA256 differs")
        self.cuda = C.CDLL(CUDA_PATH)
        self.ccl = C.CDLL(CCL_PATH)
        pointer = C.c_void_p
        output = C.POINTER(pointer)
        size = C.c_size_t
        integer = C.c_int
        self.alloc = bind(self.cuda, "cudaMalloc", output, size)
        self.free = bind(self.cuda, "cudaFree", pointer)
        self.copy_async = bind(self.cuda, "cudaMemcpyAsync", pointer, pointer, size, integer, pointer)
        self.sync_stream = bind(self.cuda, "cudaStreamSynchronize", pointer)
        self.event_create = bind(self.cuda, "cudaEventCreate", output)
        self.event_destroy = bind(self.cuda, "cudaEventDestroy", pointer)
        self.event_record = bind(self.cuda, "cudaEventRecord", pointer, pointer)
        self.event_elapsed = bind(self.cuda, "cudaEventElapsedTime", C.POINTER(C.c_float), pointer, pointer)
        self.reduce = bind(self.ccl, "infinicclAllReduce", pointer, pointer, size, integer, integer,
                           pointer, pointer)

    def run(self, comm, stream, rank, rows, deadline):
        if not comm or not stream:
            raise ValueError("expected admitted communicator and explicit owner stream")
        return replay(self, comm, stream, rank, rows, F32, SUM, deadline)
