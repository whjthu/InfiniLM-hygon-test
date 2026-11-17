"""Select and verify one qualified Native library in a fresh GLM owner process."""

import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath


BUILD = Path('/refactor/results/GLM474-attention-detail-CPU-v1')
OUTCOME_SHA256 = 'f8ba2cb054fedff4948ed5140050d959d05a36273d972748428c0276d039c401'
BUILD_CLAIM = 'd7fe8c2672a53ad56288cc6f0aa38dd90ad4fa4b'
BUILD_SHA256 = 'dda690c7ae07180f3b58ae4039c73b9f53d13c6726d96f53c7fab07ef6f64cec'
LIBRARY = (PurePosixPath('/refactor/results/GLM474-attention-detail-CPU-v1/libGLM_attention_detail.so'), '184e60047ab0c1cd56e92509cce324348dee5d241ab6ad3e945f62fe0de8d0fa')
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
    require(outcome['passed'] is True and outcome['rc'] == 0 and outcome['client_timeout'] is False
            and outcome['current_step_outcome'] == 'TERMINAL' and outcome['GPU_cards'] == 0
            and outcome['deadline'] == outcome['epoch'] + 600
            and outcome['build_SHA256'] == BUILD_SHA256, 'actual169 outcome differs')
    row = outcome['accounting']['row']
    require(outcome['accounting']['terminal_visible'] is True
            and row == ['1103606.169', BUILD.name, 'COMPLETED', '0:0', '63', 'f13r1n05', 'cpu=4,mem=16G,node=1'],
            'actual169 CPU terminal accounting differs')
    build = outcome['build']
    require(build['passed'] is True and build['step'] == row[0]
            and build['new_HOST_TUs'] == 3 and build['new_GPU_TUs'] == build['GPU_cards'] == 0
            and build['linked_objects'] == 32 and build['reused_objects'] == 29
            and build['all3_fresh_M_before_compile'] is True and build['all3_MD_equal_fresh_M'] is True
            and build['strong_exports_preserved'] == 280 and build['providers'] == 71
            and build['tls_storage_verified'] is True and build['Native_dlopen'] is False
            and build['library_bytes'] == 1930288 and build['library_SHA256'] == LIBRARY[1],
            'actual169 library/dependency qualification differs')
    for name, count in [('nov', 1019), ('prepare', 993), ('output', 988)]:
        require(build['MD'][name]['equal_fresh_M'] is True and build['MD'][name]['count'] == count
                and build['fresh_M'][name]['count'] == count
                and len(build['fresh_M'][name]['dependencies_SHA256']) == count,
                'actual169 fresh-M/MD closure differs')


def verified_library(arm):
    require(arm in ARMS, "unknown GLM Native arm")
    raw = (BUILD / "outcome.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OUTCOME_SHA256, "paired link receipt hash changed")
    outcome = json.loads(raw)
    verify_outcome(outcome)
    intent_raw = (BUILD / 'intent.json').read_bytes()
    require(hashlib.sha256(intent_raw).hexdigest() == '66e17cb140a9c19715e233714f294fa36df872f293418a98ebaac784d3cd85fc'
            and json.loads(intent_raw)['publication'] == BUILD_CLAIM, 'actual169 original intent differs')
    build_raw = (BUILD / 'build-result.json').read_bytes()
    require(hashlib.sha256(build_raw).hexdigest() == BUILD_SHA256
            and json.loads(build_raw) == outcome['build'], 'actual169 original build differs')
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
