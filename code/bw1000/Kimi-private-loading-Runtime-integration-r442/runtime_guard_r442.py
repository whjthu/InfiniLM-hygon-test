"""Device-side strict post-import guard; CPU tests inject only the symbol leaf."""
import ctypes
import hashlib
from pathlib import Path

RUNTIME_SHA = '5b8df2e6dd334b88e3a1493b928dce1a25fe2a7878c2eccdf141a74aaa3ad36b'
SYMBOLS = (
    '_ZN8infinilm7runtime7Runtime6createEN6infini2rt6DeviceE',
    '_ZN8infinilm7runtime10TensorImpl9copy_fromERNS0_7RuntimeENS0_6TensorE',
    'modern_cpu_copy_row_hits_r436',
)


def symbol_providers(path):
    # Called only after maps proves this exact Runtime is ALREADY loaded.
    class DlInfo(ctypes.Structure):
        _fields_ = [('fname', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('sname', ctypes.c_char_p), ('saddr', ctypes.c_void_p)]
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    handle = ctypes.CDLL(str(path))
    result = {}
    for name in SYMBOLS:
        info = DlInfo()
        assert dladdr(ctypes.cast(getattr(handle, name), ctypes.c_void_p), ctypes.byref(info)) == 1
        result[name] = str(Path(info.fname.decode()).resolve())
    return result


def copy_hits(manifest):
    path = Path(manifest['private_Runtime_r442']['path']).resolve()
    # Refuse a first load here. The original post-import map guard must pass.
    verify_runtime(manifest, Path('/proc/self/maps').read_text())
    fn = getattr(ctypes.CDLL(str(path)), 'modern_cpu_copy_row_hits_r436')
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 2)()
    assert fn(values, 2) == 0
    return list(values)


def checked_load_delta(before, after):
    assert len(before) == len(after) == 2
    assert all(type(x) is int and x >= 0 for x in before + after)
    delta = [b-a for a, b in zip(before, after)]
    assert min(delta) >= 0 and delta[0] > 0, ('candidate did not choose any load copy', delta)
    return dict(chosen=delta[0], fallback=delta[1], scope='per leader process aggregate over eight Native owners; other copy geometries may fall back')


def verify_runtime(manifest, maps_text, probe=symbol_providers):
    expected = Path(manifest['private_Runtime_r442']['path']).resolve()
    assert expected.name == 'libinfinilm_runtime.so'
    assert manifest['private_Runtime_r442']['SHA256'] == RUNTIME_SHA
    assert manifest['loaded_library_SHA256']['libinfinilm_runtime.so'] == RUNTIME_SHA
    paths = set()
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and 'libinfinilm_runtime' in parts[5]:
            assert not parts[5].endswith(' (deleted)'), 'deleted Runtime mapping'
            paths.add(str(Path(parts[5]).resolve()))
    assert paths == {str(expected)}, ('single exact Runtime required', sorted(paths), str(expected))
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == RUNTIME_SHA
    providers = probe(expected)
    assert set(providers) == set(SYMBOLS)
    assert set(providers.values()) == {str(expected)}, providers
    return dict(path=str(expected), SHA256=RUNTIME_SHA, providers=providers,
                single_Runtime_mapping=True, scope='actual post-import; not a CPU qualification')
