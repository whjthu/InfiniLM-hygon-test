"""Render exact T07 operands, stock trace, and the new fixed-reuse worker."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
G = HERE.parents[2]
PREVIOUS = HERE.with_name("zimage-hipprof-route-witness-v2")
ORIGINAL = "results/bw1000/2026-11-23/"
OPERANDS = ORIGINAL + "training-zimage-registered-1787-actual-gemm-operands-terminal.json"
ROUTES = ORIGINAL + "O01-zimage-hipprof-route-original1877-terminal.json"
OPERAND_PRODUCER = "code/bw1000/training-zimage-hip-mm-operands-v1/zimage_mm_operands.py"
PINS = {
    "worker.py": "9977cd104fbc786d3f60ec7c0e7ee1a3643aa54ce7a0581b0453905395a6df0a",
    "contract.py": "956c0892690a554fb31549051797f9dff82d851f1c4c79e7a3e7a2e07059f148",
    "hipprof_trace.py": "fa4d17d4b3b56316e31f117a88fd7ad7351cb0b06ed1016848052eac265398c6",
    OPERANDS: "a3565707cdaffc4249b98aad784a9225da556a33ad56b776c0018c7012821087",
    ROUTES: "a721540a162c222e651b4d6a5c23b74d9e0be8ced93e456b9d3e301b405251e6",
    OPERAND_PRODUCER: "662041471b030f8068fd4c41c8ffe791e9c89328f87c99f9bcee2212adc6321c",
}
REQUEST = "a31421758fc74d319923dbce7045cc12"
LABEL = "O01-zimage-square-rhs-pack16-v1"


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def read(path, wanted):
    raw = path.read_bytes()
    if path.is_symlink() or pin(raw)["SHA256"] != wanted:
        raise ValueError("exact predecessor differs: " + path.name)
    return raw


def replace(text, anchor, value):
    if text.count(anchor) != 1:
        raise ValueError("frozen source anchor differs")
    return text.replace(anchor, value, 1)


def prove_real_routes():
    producer = read(G / OPERAND_PRODUCER, PINS[OPERAND_PRODUCER])
    if b"return sizes, strides, dtype, device, layout" not in producer:
        raise ValueError("frozen operand metadata contract differs")
    actual = json.loads(read(G / OPERANDS, PINS[OPERANDS]))["actual"]["operand_capture"]
    observed = json.loads(read(G / ROUTES, PINS[ROUTES]))["independent_audit"]
    expected = {}
    for name, stride in (("transposed", [1, 3840]), ("contiguous", [3840, 1])):
        [row] = [row for row in actual["examples"] if row["MNK"] == [288, 3840, 3840]
                 and row["rhs"]["stride"] == stride]
        [(symbol, multiplicity)] = row["linked_kernel_multiplicity"]
        witness = observed["observed"][name]["arms"]["cublas"]
        if (row["host_mm_calls"] != 120 or row["lhs"]["stride"] != [3840, 1] or multiplicity != 1
                or row["lhs"]["shape"] != [288, 3840] or row["rhs"]["shape"] != [3840, 3840]
                or row["lhs"]["dtype"] != row["rhs"]["dtype"] or row["rhs"]["dtype"] != "torch.bfloat16"
                or witness["symbol"] != symbol or witness["traced_MM_calls"] != 1
                or witness["BF16_bitwise_CPU_reference"] is not True):
            raise ValueError("actual operand or route identity changed")
        expected[name] = symbol
    if expected["transposed"] == expected["contiguous"]:
        raise ValueError("layout routes did not differ")
    return {"transposed": expected["transposed"], "packed": expected["contiguous"]}


def build():
    native = read(PREVIOUS / "worker.py", PINS["worker.py"]).decode()
    native = replace(native, 'import hipprof_trace as trace\n', 'import hipprof_trace as trace\nimport candidate\nimport stat\n')
    old = '    a, b, reference = open_layout(torch, fixed_inputs(torch)[:1], spec.LAYOUTS[layout])[0]'
    native = replace(native, old, '    a, b, reference = candidate.witness_operand(torch, guarded, layout)')
    begin = native.index("def size(directory):")
    end = native.index("\n\ndef trace_cell(", begin)
    # Late disappearing files may belong to the live tracer. The terminal scan stays strict.
    native = native[:begin] + """def size(directory, *, live=False):
    total = 0
    with os.scandir(directory) as entries:
        for ordinal, entry in enumerate(entries, 1):
            spec.need(ordinal <= 16, "UNEXPECTED_TRACER_OUTPUT_SCOPE")
            try:
                observed = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                if not live:
                    raise
                continue
            spec.need(stat.S_ISREG(observed.st_mode), "UNEXPECTED_TRACER_OUTPUT_SCOPE")
            total += observed.st_size
    return total
""" + native[end:]
    native = replace(native, '                spec.need(size(cell) <= trace.MAX_TRACE_BYTES, "TRACE_DIRECTORY_BUDGET")',
                     '                spec.need(size(cell, live=True) <= trace.MAX_TRACE_BYTES, "TRACE_DIRECTORY_BUDGET")')
    code = {"baseline.py": native.encode()}
    code.update({name: read(PREVIOUS / name, PINS[name]) for name in ("contract.py", "hipprof_trace.py")})
    code.update({name: (HERE / name).read_bytes() for name in ("candidate.py", "worker.py")})
    for name, blob in code.items():
        compile(blob, name, "exec")
    plan = {"request": REQUEST, "label": LABEL,
            "source_1787_and_1877": {key: PINS[key] for key in (OPERANDS, ROUTES, OPERAND_PRODUCER)},
            "provider_preference": "cublas", "stock_kernel_by_arm": prove_real_routes(),
            "training_RHS_reuse_identity_verified": False,
            "files": {name: pin(blob) for name, blob in code.items()}}
    return code, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{32}", REQUEST):
        raise ValueError("canonical request identity")
    files, plan = build()
    args.out.mkdir()
    for name, data in files.items():
        (args.out / name).write_bytes(data)
    manifest = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    (args.out / "SCOPED_FILES.json").write_bytes(manifest)
    print(json.dumps(pin(manifest), sort_keys=True))


if __name__ == "__main__":
    main()
