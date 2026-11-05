"""Bind the exact two-object diagnostic DSO over the verified a314 parent chain."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path

BASELINE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
NATIVE = "1b729a2d29d39f084addc4978fbe3db7bd4b685360c384f7dfe2ae8908fc9e7f"
RECEIPT_SHA = "775a26846037e5925cf3e3343569c8c39140cd78d65fbee44bf8c72118d9ea28"
LIBRARY = Path("/refactor/candidates/Deep-C4-current-host-split-CPU-v1/libDeep_c4_host_split_v1.so")
OBJECTS = {
    "/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_csa_attention_batch_r784.cpp.o":
        "f9b8e946e503cece699c82df6aad93ff69de7643d2afa8cda8ea1b2171b50419",
    "/refactor/candidates/deep-c4-smallt-r9252/deep_c4_smallt_r9252.cpp.o":
        "efebf0b88153a68a126b863937e68eadbd483152de42234785f5097865c4b27d",
}


def receipt_from_worker(worker):
    _, parent, normalized = worker.verify_native_chain()
    assert parent["library_SHA256"] == BASELINE and len(normalized["link"]["full"]["objects"]) == 50
    path = LIBRARY.with_name("CPU-build.json")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == RECEIPT_SHA
    proof = json.loads(raw)
    assert proof["completed"] and proof["actual_step"] == "910374.1458"
    assert not proof["GPU_initialized"] and not proof["Native_dlopen"]
    assert proof["reused"] == 48 and proof["dynamic_NEEDED_unchanged"]
    assert proof["old_strong_exports"] == 337
    assert proof["new_strong_exports"] == ["deep_c4_host_split_read_v1"]
    assert len(proof["replaced"]) == 2
    assert {x["parent"]: x["SHA256"] for x in proof["replaced"]} == OBJECTS
    assert set(OBJECTS) <= normalized["link"]["full"]["objects"].keys()
    assert proof["library"] == str(LIBRARY) and proof["library_SHA256"] == NATIVE
    return dict(parent, library=str(LIBRARY), library_SHA256=NATIVE,
                baseline_library_SHA256=BASELINE, diagnostic_CPU_receipt_SHA256=RECEIPT_SHA)


def bind_library(path, expected):
    from public_deep_model.checkpoint_loader import declare

    assert Path(path) == LIBRARY and expected == NATIVE
    assert sha256(LIBRARY.read_bytes()).hexdigest() == NATIVE
    return declare(ctypes.CDLL(str(LIBRARY)))
