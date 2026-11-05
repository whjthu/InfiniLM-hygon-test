"""Bind the staged-prefix DSO to the original successful CPU build receipt."""
from hashlib import sha256
import json
from pathlib import Path

NATIVE_SHA256 = '35c08ce45814178e3692c55eb13dd5f074643993be1fbe911ecf91a2c200a0cb'
NATIVE_PATH = '/refactor/candidates/GLM952-block-prefix-native-v1/libGLM952_block_prefix_v1.so'
RECEIPT_PATH = '/refactor/candidates/GLM952-block-prefix-native-v1/CPU-build.json'
RECEIPT_SHA = '6f6ff56db562391fc8003eb9fe0faab945cbded4eeff22def31c53c519be62dd'
ADDED_CAPI = {
    "modern_glm_block_stage_v1", "modern_glm_block_prepare_v1", "modern_glm_block_publish_v1",
}


def verify_config(config):
    if (config["Native_SHA256"], config["Native_library"]) != (NATIVE_SHA256, NATIVE_PATH):
        raise ValueError("GLM continuation Native identity differs")


def verify(config):
    verify_config(config)
    raw = Path(RECEIPT_PATH).read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("GLM continuation CPU build receipt differs")
    receipt = json.loads(raw)
    if (
        receipt["actual_CPU"] != "910371.835"
        or receipt["completed"] is not True
        or receipt["GPU"] is not False
        or receipt["Model"] is not False
        or receipt["dlopen"] is not False
        or receipt["compile_count"] != 8
        or receipt["retained_objects"] != 21
        or receipt["dynamic_metadata_equal"] is not True
        or set(receipt["added_CAPI"]) != ADDED_CAPI
        or (receipt["library_SHA256"], receipt["library"]) != (NATIVE_SHA256, NATIVE_PATH)
    ):
        raise ValueError("GLM continuation CPU build receipt scope differs")
    if sha256(Path(NATIVE_PATH).read_bytes()).hexdigest() != NATIVE_SHA256:
        raise ValueError("GLM continuation Native bytes differ")
    return dict(library_SHA256=NATIVE_SHA256, library_actual_path=NATIVE_PATH,
                cpu_receipt_SHA256=RECEIPT_SHA, cpu_step=receipt["actual_CPU"])
