"""Generate an immutable one-variable window around the exact TP16 gate/up TU."""
from hashlib import sha256


BASE_SHA256 = '4adb7a1c1b11b6a4434c93b8e9787966bec6a71787fa4837a5ed13e3afc128b6'
ROWS = (1, 2)
MARKERS = {
    b'  constexpr int W=64,V=2;\n': b'  constexpr int W=64,V=2;\n',
    b'  size_t row=size_t(blockIdx.x)*4+threadIdx.x/W,route=blockIdx.y;': (
        b'  size_t row=size_t(blockIdx.x)*gate_rows+threadIdx.x/W,route=blockIdx.y;'),
    b'  gate_up<<<dim3(32,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(\n': (
        b'  gate_up<<<dim3(128/gate_rows,unsigned(routes)),64*gate_rows,0,static_cast<cudaStream_t>(stream)>>>(\n'),
}


def variant(baseline, rows):
    if sha256(baseline).hexdigest() != BASE_SHA256 or type(rows) is not int or rows not in ROWS:
        raise ValueError('EXACT_BASE_OR_CANDIDATE_ROWS')
    insert = b'namespace {\n'
    if baseline.count(insert) != 1:
        raise ValueError('EXACT_TRANSLATION_UNIT_NAMESPACE')
    source = baseline.replace(insert, insert + f'constexpr unsigned gate_rows={rows};\n'.encode(), 1)
    for old, new in MARKERS.items():
        if source.count(old) != 1:
            raise ValueError('EXACT_SOURCE_BOUNDARY')
        source = source.replace(old, new, 1)
    return source
