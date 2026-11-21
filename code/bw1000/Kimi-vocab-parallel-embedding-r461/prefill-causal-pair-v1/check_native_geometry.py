"""Compile the archived native guards and check the paired call plan against them."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from causal_pair import WIDTHS, digest, schedule


REPO = Path(__file__).resolve().parent.parents[3]
NATIVE = REPO / "code/bw1000/Kimi-AR-persistent-storage-optin-r456/native"
PINS = {
    "long_geometry_r453.hpp": "920ce94861e42715472f0b06fe69b8d10b41155eb0270946770c53aab8d9c44a",
    "ar_storage_r456.hpp": "9e726d671b4615d058442dd7e8210fea22c7d8aa07a56bba7363f7837c0103e2",
}

CPP = r"""
#include "long_geometry_r453.hpp"
#include "ar_storage_r456.hpp"
#include <iostream>

int main() {
    int kind, row, past, flags;
    std::size_t count, width, write, pages;
    while (std::cin >> kind >> count >> row >> past >> width >> flags >> write >> pages) {
        try {
            kimi_long_geometry_r453::call_scope(kind, count, row, past, width, flags);
            kimi_ar_storage_r456::request(row, write, 2, pages, false);
            std::cout << "1\n";
        } catch (const std::invalid_argument&) {
            std::cout << "0\n";
        }
    }
}
"""


def main():
    for name, expected in PINS.items():
        if digest((NATIVE / name).read_bytes()) != expected:
            raise ValueError(f"Archived native guard changed: {name}")
    valid = []
    for length in (1, 255, 256, 257, 511, 512, 513, 1025, 28672):
        for width in WIDTHS:
            for kind, past, ids, _ in schedule(list(range(length)), [777, 942], width):
                valid.append((kind, 1, 1, past, len(ids), 3, 1, 128))
    invalid = [
        (2, 1, 1, 0, 512, 3, 1, 128),
        (0, 2, 1, 0, 512, 3, 1, 128),
        (0, 1, 0, 0, 512, 3, 1, 128),
        (0, 1, 1, -1, 512, 3, 1, 128),
        (0, 1, 1, 0, 513, 3, 1, 128),
        (1, 1, 1, 77, 2, 3, 1, 128),
        (1, 1, 1, 32768, 1, 3, 1, 128),
        (0, 1, 1, 0, 512, 7, 1, 128),
        (0, 1, 1, 0, 512, 3, 0, 128),
        (0, 1, 1, 0, 512, 3, 1, 127),
    ]
    payload = "\n".join(" ".join(map(str, row)) for row in valid + invalid) + "\n"
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Local g++ required for archived guard check")
    with TemporaryDirectory(prefix="kimi-prefill-guard-") as raw:
        path = Path(raw)
        source, binary = path / "guard.cpp", path / "guard.exe"
        source.write_text(CPP, encoding="utf-8")
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-Wno-misleading-indentation", "-I", str(NATIVE), str(source), "-o", str(binary)],
                       timeout=30, check=True)
        actual = subprocess.run([str(binary)], input=payload, text=True, capture_output=True,
                                check=True, timeout=8).stdout.splitlines()
    if actual != ["1"] * len(valid) + ["0"] * len(invalid):
        raise ValueError("Archived AR guard acceptance differs from the original call plan")
    print(f"Archived local C++ guards: {len(valid)} accepted, {len(invalid)} rejected")


if __name__ == "__main__":
    main()
