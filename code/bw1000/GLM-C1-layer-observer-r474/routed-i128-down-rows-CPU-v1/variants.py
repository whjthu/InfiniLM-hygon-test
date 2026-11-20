"""Freeze two CTA packing changes in the TP16 routed I128 down source."""
from hashlib import sha256


BASE_SHA256 = '4adb7a1c1b11b6a4434c93b8e9787966bec6a71787fa4837a5ed13e3afc128b6'
ROWS = (2, 4)
MARKERS = (
    (b'  size_t row=size_t(blockIdx.x)*8+threadIdx.x/W,route=blockIdx.y;',
     b'  size_t row=size_t(blockIdx.x)*down_rows+threadIdx.x/W,route=blockIdx.y;'),
    (b'  down<<<dim3(768,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(\n',
     b'  down<<<dim3(6144/down_rows,unsigned(routes)),32*down_rows,0,static_cast<cudaStream_t>(stream)>>>(\n'),
)
ANCHOR = b'namespace {\n'


def variant(baseline, rows):
    if sha256(baseline).hexdigest() != BASE_SHA256 or type(rows) is not int or rows not in ROWS:
        raise ValueError('EXACT_BASE_OR_CANDIDATE_ROWS')
    if baseline.count(ANCHOR) != 1:
        raise ValueError('EXACT_TRANSLATION_UNIT_NAMESPACE')
    source = baseline.replace(ANCHOR, ANCHOR+f'constexpr unsigned down_rows={rows};\n'.encode(),1)
    for old,new in MARKERS:
        if source.count(old) != 1:
            raise ValueError('EXACT_DOWN_SOURCE_BOUNDARY')
        source = source.replace(old,new,1)
    return source
