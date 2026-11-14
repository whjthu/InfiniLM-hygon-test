"""Bind the exact linked library from the completed 1103606.139 CPU step."""
import ctypes
import hashlib
import json
from pathlib import Path

from native_arm import require, sha256, verify_symbols

BUILD = Path("/refactor/results/GLM474-selected-layer0-explicit-decode-CPU-v1")
LIBRARY = BUILD / "libGLM_selected_layer0_decode.so"
DIGEST = "bc1513a10a21860db41fab192093acb9850f4fb4412838c9940f6dfac130fa4e"
CLAIM = "3a6bff6cfa7fb7742a5bea940327dca46dc81b17"
RECEIPT = "1d91dcb6a59a2d8e7d431d407ccf8663351e4a271f35342594a0376f27a4bec2"
BINDING = "2beba1694691f721ee12c2e845ccb019192e2f89e87ebcef28be545894d448b8"
ARMS = {"socket": (LIBRARY, DIGEST)}


def verify_build(outcome, build):
    expected = [["1103606.139", BUILD.name, "COMPLETED", "0:0", "34", "f13r1n05"]]
    require(outcome["claim"] == CLAIM and outcome["passed"] is True
            and outcome["accounting"] == expected and outcome["owned_srun_reaped"] is True
            and outcome["terminal"]["terminal_visible"] is True
            and outcome["terminal"]["row"] == expected[0]
            and outcome["build_SHA256"] == RECEIPT and outcome["build"] == build,
            "selected-value CPU step provenance differs")
    require(build["passed"] is True and build["tls_storage_verified"] is True
            and build["step"] == expected[0][0] and build["library_SHA256"] == DIGEST
            and build["new_HOST_TUs"] == 2 and build["new_GPU_TUs"] == 0
            and build["providers"] == 71 and build["strong_exports_preserved"] == 279
            and build["diagnostic_exports"] == ["modern_glm_global_selected_value_layer0_forward"],
            "selected-value linked library receipt differs")


def load_owner(arm, typed_binding, attach_profile):
    require(arm in ARMS, "selected value requires the fixed Socket transport")
    raw = (BUILD / "build-result.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RECEIPT, "selected-value build receipt changed")
    verify_build(json.loads((BUILD / "outcome.json").read_bytes()), json.loads(raw))
    require(sha256(LIBRARY) == DIGEST and sha256(Path(typed_binding.__file__)) == BINDING,
            "selected-value library or typed binding changed")
    for row in Path("/proc/self/maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) == 6:
            require(Path(fields[-1].removesuffix(" (deleted)")).name != LIBRARY.name,
                    "selected-value library was already mapped")
    lib = ctypes.CDLL(str(LIBRARY))
    api = typed_binding.attach(lib)
    selected = typed_binding.attach_selected_value(lib)
    profile = attach_profile(lib)
    verify_symbols(LIBRARY, {**api, **selected, "pipeline_last_profile": profile})
    return lib, api, selected, profile, dict(arm=arm, library=str(LIBRARY),
                                           library_SHA256=DIGEST, build_SHA256=RECEIPT)
