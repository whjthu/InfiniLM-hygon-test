"""Select and verify one qualified Native library in a fresh GLM owner process."""

import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath


BUILD = Path('/refactor/results/GLM474-depth-stratified-observer-CPU-v1')
OUTCOME_SHA256 = 'e37368cc62f1b883b53cbc0d72f9b82f08972263adf1697a6689b694a1d4f452'
BUILD_SHA256 = 'aa669c0ebc18072b0085c1f98fe34bee898e50af28f6e0542bb8c485ac200b6f'
LIBRARY = (PurePosixPath('/refactor/results/GLM474-depth-stratified-observer-CPU-v1/B/libGLM_depth_observer_B.so'), 'ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5')
ARMS = {'socket': LIBRARY}
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
    row=outcome['accounting']['row'];build=outcome['build']
    require(outcome['passed'] is True and outcome['rc']==0 and outcome['current_step_outcome']=='TERMINAL'
            and row[0]=='1103606.308' and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n05'
            and 'gres/dcu=' not in row[6] and outcome['GPU_cards']==0, 'actual308 original CPU terminal differs')
    require(outcome['build_SHA256']==BUILD_SHA256 and build['passed'] and build['step']==row[0]
            and build['actual']==dict(freshM=32,HOST_compile=8,GPU_compile=0,reused_objects=24,objects=32,links=1)
            and build['all32_M_before_compile'] and build['modified_MD_equal_freshM']
            and build['reused_B_dependencies_exact'] and build['Native_dlopen'] is False,
            'actual308 build closure differs')
    library=build['library']
    require(library['path']==str(LIBRARY[0]) and library['SHA256']==LIBRARY[1]
            and library['bytes']==1939136 and library['strong_exports']==281 and library['providers']==71
            and 'glm_sparse_r131::depth_pair' in library['TLS_single_definition'], 'actual308 depth library differs')


def verified_library(arm):
    require(arm in ARMS, 'unknown Native arm')
    raw=(BUILD/'outcome.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==OUTCOME_SHA256, '308 outcome pin')
    outcome=json.loads(raw);verify_outcome(outcome)
    raw=(BUILD/'build-result.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==BUILD_SHA256 and json.loads(raw)==outcome['build'], '308 build pin')
    path,digest=ARMS[arm]
    path=Path(path)
    require(sha256(path)==digest, '308 actual depth library pin')
    return path,digest


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
    names = {library.name for library, _ in ARMS.values()} | {"libGLM_history_value_B.so"}
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
