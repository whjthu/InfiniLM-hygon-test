"""Read two complete little-endian BF16 residual outputs without device imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


WORDS = 512 * 7168


def compare(baseline, tiled, expected_words=WORDS):
    if len(baseline) != 2 * expected_words or len(tiled) != len(baseline) or expected_words <= 0:
        raise ValueError("incomplete BF16 residual output")
    def decode(word):
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]

    max_abs, reference_peak, error_square, reference_square, changed = 0.0, 0.0, 0.0, 0.0, 0
    for (before,), (after,) in zip(struct.iter_unpack("<H", baseline), struct.iter_unpack("<H", tiled), strict=True):
        if (before & 0x7f80) == 0x7f80 or (after & 0x7f80) == 0x7f80:
            raise ValueError("nonfinite BF16 residual output")
        changed += before != after
        a, b = decode(before), decode(after)
        delta = abs(a - b)
        max_abs, reference_peak = max(max_abs, delta), max(reference_peak, abs(a))
        error_square += delta * delta
        reference_square += a * a
    return {
        "words": expected_words, "changed_words": changed, "max_abs": max_abs,
        "relative_l2": math.sqrt(error_square / reference_square) if reference_square else None,
        "max_abs_over_reference_peak": max_abs / reference_peak if reference_peak else None,
        "baseline_SHA256": hashlib.sha256(baseline).hexdigest(),
        "tiled_SHA256": hashlib.sha256(tiled).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("tiled", type=Path)
    args = parser.parse_args()
    with args.baseline.open("rb") as first, args.tiled.open("rb") as second:
        result = compare(first.read(2 * WORDS + 1), second.read(2 * WORDS + 1))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
