"""Select and verify one qualified Native library in a fresh GLM owner process."""

import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath


BUILD = Path("/refactor/results/GLM474-ffn-window-observer-CPU-v1")
OUTCOME_SHA256 = "e61341a25dc5845ab548c9c93f8c9885d5e3f42b0505275e2286a81d3571f12d"
BUILD_CLAIM = "cad7809c62327072c1b472288e4d8709b1009ee1"
BUILD_REQUEST = "d8541bb2f8cc4ffead761e6b2a4cee68"
BUILD_SHA256 = "d300246ef893c1c3714490a86fd3c8bcbd06802b401487fb4ffcc9e984b81d35"
LIBRARY = (PurePosixPath("/refactor/results/GLM474-ffn-window-observer-CPU-v1/libGLM_ffn_window_observer.so"), "017e981ace530ec762bf56bc981a73fd8bcb5c00afc4f5e8062403ac652a424e")
ARMS = {"socket": LIBRARY}
TYPED_BINDING_SHA256 = "ea014b57f1ac1b450ca5acc8cb28e41283d041b80f82e88141377fb005b86fb7"
OBSERVER_BINDING_SHA256 = "4d00164482c7daa532f05cc15d5c89b3e0df98d1d8e166b30f8acc3cf1fcac56"


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_outcome(outcome):
    require(outcome["claim"] == BUILD_CLAIM and outcome["request_id"] == BUILD_REQUEST
            and outcome["passed"] is True and outcome["owned_srun_reaped"] is True
            and outcome["forced_cleanup"] is False and outcome["returncode"] == 0
            and outcome["terminal"]["terminal_visible"] is True and outcome["GPU_cards"] == 0
            and outcome["build_SHA256"] == BUILD_SHA256, "FFN observer build receipt differs")
    require(outcome["accounting"] == [["1103606.107", BUILD.name, "COMPLETED", "0:0", "33", "f13r1n05"]],
            "FFN observer CPU accounting differs")
    build = outcome["build"]
    require(build["passed"] is True and build["step"] == "1103606.107"
            and build["new_HOST_TUs"] == 1 and build["new_GPU_TUs"] == 0
            and build["MD"]["equal_fresh_M"] is True and build["MD"]["count"] == 994
            and build["MD"]["SHA256"] == build["fresh_M"]["SHA256"]
            and build["linked_objects"] == 32 and build["reused_objects"] == 31
            and build["strong_exports_preserved"] == 277 and build["providers"] == 71
            and build["library_SHA256"] == LIBRARY[1], "FFN observer link evidence differs")


def verified_library(arm):
    require(arm in ARMS, "unknown GLM Native arm")
    raw = (BUILD / "outcome.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OUTCOME_SHA256, "paired link receipt hash changed")
    verify_outcome(json.loads(raw))
    library, digest = ARMS[arm]
    path = Path(library)
    require(sha256(path) == digest, f"{arm}: Native library hash changed")
    return path, digest


def verify_symbols(library_path, functions):
    expected = library_path.resolve(strict=True)
    dladdr = ctypes.CDLL("libdl.so.2").dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    for name, function in functions.items():
        info = DlInfo()
        address = ctypes.cast(function, ctypes.c_void_p)
        require(dladdr(address, ctypes.byref(info)) and info.name, f"{name}: no DSO provider")
        actual = Path(info.name.decode()).resolve(strict=True)
        require(actual == expected, f"{name}: unexpected DSO provider")


def load_owner(arm, typed_binding, observer_binding, attach_profile):
    """Call once per owner, before any Native context or communicator is created."""
    path, digest = verified_library(arm)
    require(
        sha256(Path(typed_binding.__file__)) == TYPED_BINDING_SHA256
        and sha256(Path(observer_binding.__file__)) == OBSERVER_BINDING_SHA256,
        "frozen typed owner binding changed",
    )
    names = {library.name for library, _ in ARMS.values()}
    for row in Path("/proc/self/maps").read_text().splitlines():
        columns = row.split(maxsplit=5)
        if len(columns) == 6:
            mapped = columns[-1].removesuffix(" (deleted)")
            require(Path(mapped).name not in names, "a GLM Native arm is already mapped")
    library = ctypes.CDLL(str(path))
    typed = typed_binding.attach(library)
    profile = attach_profile(library)
    observer = {symbol: getattr(library, symbol) for symbol, _, _ in observer_binding.NAMES.values()}
    verify_symbols(path, {**typed, **observer, "pipeline_last_profile": profile})
    return library, typed, profile, {"arm": arm, "library": str(path), "library_SHA256": digest}
