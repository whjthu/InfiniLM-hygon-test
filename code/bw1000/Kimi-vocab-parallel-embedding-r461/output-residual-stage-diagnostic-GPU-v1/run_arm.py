"""Run one isolated residual arm from exact BF16 files in a dedicated process."""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys

from stages import mark


HIDDEN = 7168
OUTPUT = 512 * HIDDEN
WORDS = {"residual": OUTPUT * 9, "norm": HIDDEN, "projection": HIDDEN}
ARG_TYPES = [C.c_int, C.c_int, C.c_double] + [
    field for typ in (C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16, C.c_double, C.c_uint64)
    for field in (C.POINTER(typ), C.c_size_t)
]


def load_exact(root, binding, name):
    words = WORDS[name]
    with (root / (name + ".bf16")).open("rb") as f:
        raw = f.read(2 * words + 1)
    if binding["bytes"] != len(raw) or len(raw) != 2 * words or hashlib.sha256(raw).hexdigest() != binding["SHA256"]:
        raise ValueError(f"fixed BF16 operand differs: {name}")
    return (C.c_uint16 * words).from_buffer_copy(raw)


def native_arm(lib, mode, pre_transpose, eps, operands):
    native = lib.kimi_output_residual_component_r461
    native.argtypes, native.restype = ARG_TYPES, C.c_int
    error = lib.kimi_output_residual_error_r461
    error.argtypes, error.restype = [], C.c_char_p
    output, metrics, memory = (C.c_uint16 * OUTPUT)(), (C.c_double * 3)(), (C.c_uint64 * 6)()
    arrays = operands + [output, metrics, memory]
    arguments = [part for owner in arrays for part in (owner, len(owner))]
    if native(mode, pre_transpose, eps, *arguments) != 0:
        raise RuntimeError((error() or b"native failure").decode("utf-8", "replace")[:200])
    seconds = list(metrics)
    driver = list(zip(memory[::2], memory[1::2], strict=True))
    if not all(math.isfinite(t) and t >= 0 for t in seconds) or seconds[2] < sum(seconds[:2]):
        raise ValueError("invalid native duration counters")
    if not all((4 << 30) <= free <= total for free, total in driver):
        raise ValueError("invalid native memory observations")
    return output, seconds, driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--mode", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise ValueError("little-endian BF16 ABI required")
    binding = json.loads(args.binding.read_bytes())
    if set(binding["BF16_files"]) != set(WORDS):
        raise ValueError("exactly three BF16 operands required")
    checkpoint = json.loads(Path(__file__).with_name("CHECKPOINT_WEIGHTS.json").read_bytes())
    if any(binding[key] != checkpoint[key] for key in ("config_SHA256", "index_SHA256", "rms_norm_eps")):
        raise ValueError("different frozen Kimi checkpoint")
    for name, weight in checkpoint["bindings"].items():
        if any(binding["BF16_files"][name][field] != weight[field] for field in ("bytes", "SHA256")):
            raise ValueError(f"not the original Kimi checkpoint weight: {name}")
    eps = binding["rms_norm_eps"]
    pre_transpose = binding["pre_transpose"]
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0 or type(pre_transpose) is not bool:
        raise ValueError("invalid frozen original postload configuration")
    operands = [load_exact(args.input_directory, binding["BF16_files"][name], name) for name in WORDS]
    library_path = args.library.resolve(strict=True)
    if hashlib.sha256(library_path.read_bytes()).hexdigest() != binding["component_SHA256"]:
        raise ValueError("exact component library differs")
    args.output_directory.mkdir(mode=0o700)
    result = {"mode": args.mode, "complete": False, "binding": binding}
    try:
        mark("load_library.enter")
        library = C.CDLL(str(library_path), mode=C.RTLD_LOCAL)
        mark("load_library.return")
        mark("native_cabi.enter")
        output, seconds, driver = native_arm(library, args.mode, int(pre_transpose), eps, operands)
        mark("native_cabi.return")
        raw = bytes(output)
        mark("output_write.enter")
        with (args.output_directory / "output.bf16").open("xb") as f:
            f.write(raw)
        mark("output_write.return")
        result.update(complete=True, output_bytes=len(raw), output_SHA256=hashlib.sha256(raw).hexdigest(),
                      sampled_driver_free_total=driver, seconds={
                          "dispatch_through_sync": seconds[0], "post_dispatch_audit": seconds[1],
                          "full_CABI_before_destroy": seconds[2]})
    finally:
        with (args.output_directory / "result.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
