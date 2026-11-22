"""Exercise the rendered method with recording providers on the local CPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from render import ARCHIVE, DECLARATIONS, ENTRY, FUSED, LEGACY, TABLE, render


def main():
    here = Path(__file__).resolve().parent
    origin = ARCHIVE / "deepseek_v4_attention_prepare.cpp"
    raw, header, retained = origin.read_bytes(), origin.with_suffix(".hpp").read_bytes(), TABLE.read_bytes()
    caller, policy = render(raw, header, retained)
    reverted = caller.decode().replace("\r\n", "\n").replace(
        '#include "inverse_output_width_policy.hpp"\n#include <cstdlib>\n', ""
    ).replace(DECLARATIONS, "").replace(FUSED + LEGACY, LEGACY)
    assert reverted == raw.decode().replace("\r\n", "\n")
    for index in range(3):
        inputs = [raw, header, retained]
        inputs[index] += b"\n"
        try:
            render(*inputs)
        except ValueError:
            continue
        raise AssertionError("Mutated source or header accepted")
    start = caller.index(ENTRY.encode())
    end = caller.index(b"\n}\n}", start) + len(b"\n}")
    compiler = shutil.which("g++")
    assert compiler, "Local C++ compiler unavailable"
    cases = [
        (None, "normal", "ACLYY", ""),
        ("legacy", "normal", "ACLYY", ""),
        ("h1-pack32", "normal", "ASRUPYY", ""),
        ("h1-pack64", "normal", "ASRUQYY", ""),
        ("h1-pack64", "multi", "ACLYY", ""),
        ("h1-pack64", "other-heads", "ACLYY", ""),
        ("h1-pack64", "noncontiguous", "ACLYY", ""),
        ("h1-pack64", "no-stream", "ASCLYY", ""),
        ("h1-pack64", "graph", "A", "DeepSeek inverse rotary output geometry differs"),
        ("h1-pack64", "bad-dtype", "A", "DeepSeek inverse rotary output geometry differs"),
        ("h1-pack64", "bad-position", "ASR", "DeepSeek rotary position out of bounds"),
        ("h1-pack64", "bad-position-shape", "AS", "DeepSeek rotary position owner/shape differs"),
        ("h1-pack64", "no-position", "AS", "DeepSeek rotary position owner/shape differs"),
        ("h1-pack64", "provider-failure", "ASRUQY", "DeepSeek fused inverse RoPE launch failed: 31"),
        ("unrecognized", "normal", "A", "Invalid DeepSeek inverse RoPE width for this candidate"),
    ]
    with TemporaryDirectory(prefix="deep-rope-original-caller-") as scratch:
        path = Path(scratch)
        (path / "rendered_method.inc").write_bytes(caller[start:end])
        (path / "inverse_output_width_policy.hpp").write_bytes(policy)
        executable = path / "routes.exe"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2", "-I", str(path),
                        str(here / "check_caller.cpp"), "-o", str(executable)], timeout=30, check=True)
        for arm, scenario, expected_trace, error in cases:
            env = os.environ.copy()
            key = "INFINILM_DEEP_INVERSE_OUTPUT_ROPE"
            if arm is None:
                env.pop(key, None)
            else:
                env[key] = arm
            subprocess.run([str(executable), scenario, expected_trace, error], timeout=5, env=env, check=True)
    print(json.dumps({"route_cases": len(cases), "bad_source_rejections": 3, "host_compiles": 1,
                      "legacy_and_remaining_source_exact": True, "actual_GPU_steps": 0}))


if __name__ == "__main__":
    main()
