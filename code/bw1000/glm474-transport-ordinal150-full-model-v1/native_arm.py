"""Select and verify one qualified Native library in a fresh GLM owner process."""

import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath


BUILD = Path("/refactor/results/GLM474-projection-single-drain-pair-CPU-v1")
OUTCOME_SHA256 = "791de113db9a310cd566d8547b1008fbc437062c8c5e28e898a486376d4db8c0"
BUILD_CLAIM = "992a508fbc75676ab2d5af6f7ca279015c8e3c82"
BUILD_ARMS = {
    "baseline": (
        PurePosixPath("/refactor/results/GLM474-projection-host-timing-CPU-v3/libGLM_projection_host_timing.so"),
        "ad4186d84766a034e7ef4085539cf7c338454e68d629586649cbde235bbede77",
    ),
    "single-drain": (
        PurePosixPath("/refactor/results/GLM474-projection-single-drain-CPU-v1/libGLM_projection_host_timing.so"),
        "abdf704dcf37898b3477ec2df2b1af6f8c456cb534edfa24d110666e17669b07",
    ),
}
ARMS = dict.fromkeys(("socket", "shca"), BUILD_ARMS["baseline"])
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
    require(
        outcome["claim"] == BUILD_CLAIM
        and outcome["passed"] is True
        and outcome["owned_srun_reaped"] is True
        and outcome["terminal"]["terminal_visible"] is True
        and outcome["GPU_cards"] == 0,
        "paired link outcome did not pass",
    )
    require(
        outcome["accounting"] == [[
            "1103606.59", BUILD.name, "COMPLETED", "0:0", "19", "f13r1n05"
        ]],
        "paired link Slurm accounting changed",
    )
    build = outcome["build"]
    require(
        build["passed"] is True
        and build["MD"]["equal_fresh_M"] is True
        and build["MD"]["count"] == 429
        and build["runtime_created"] is False
        and build["component_executed"] is False,
        "paired link build provenance changed",
    )
    require(build["binaries"].keys() == BUILD_ARMS.keys(), "paired link arms changed")
    for arm, (library, digest) in BUILD_ARMS.items():
        binary = build["binaries"][arm]
        require(
            binary["library"] == str(library)
            and binary["library_SHA256"] == digest
            and binary["providers"] == 71,
            f"{arm}: paired link provider evidence changed",
        )


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
