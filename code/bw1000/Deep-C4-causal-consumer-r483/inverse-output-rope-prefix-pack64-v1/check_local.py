"""Run the exact generated algorithm in the pinned scalar host launch fixture."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from render import BASE, HERE, LABEL, LAUNCH, SOURCE, SYMBOL, checked, render, replace_once


PARENT = HERE.parent / "inverse-output-rope-CPU-v1"
PAIR_CHECK = HERE.parent / "inverse-output-rope-prefix-pack32-v1" / "check_local.cpp"
PAIR_SHA = "da7b3276808e71960921338cb409e7e8db6836bce0359d7bdd3c7822eb22c722"
SHIM_SHA = "c5141170b13aa247fae064fc58610477585164d6ea434f39116b5c4357141a48"
REFERENCE_SHA = "bb5c067cf58adcbb7275dc00e6a6a2b42bb5d6a7cdf0d9621100d51d6736960c"
FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off"]


def host_source(source):
    text = replace_once(source, "#include <cuda_runtime.h>", '#include "local_runtime.hpp"')
    text = replace_once(text, LAUNCH, "cpu_launch((inverse_output<HeadsPerBlock, PrefixBits>), grid, threads, 0, stream,")
    return replace_once(text, "    cpu_launch(", "    last_prefix_bits = PrefixBits;\n    cpu_launch(")


def check_source():
    text = checked(PAIR_CHECK, PAIR_SHA).decode().replace("deep_inverse_rope_h1_pack32_v1", SYMBOL)
    text = replace_once(text, " % 4 == 2 * input_offset", " % 8 == 2 * input_offset")
    text = replace_once(text, " % 4 == 2 * output_offset", " % 8 == 2 * output_offset")
    text = replace_once(text, "    assert(last_packed_prefix == (input_offset == 0 && output_offset == 0));",
                        """    const unsigned offset_bits = input_offset | output_offset;
    assert(last_prefix_bits == (offset_bits % 2 ? 16u : offset_bits ? 32u : 64u));""")
    start = text.index("    for (const auto& offsets :")
    end = text.index("    alignas(4)", start)
    text = text[:start] + """    for (unsigned x = 0; x != 4; ++x) {
        for (unsigned y = 0; y != 4; ++y) {
            if (x == 0 && y == 0) {
                continue;
            }
            run_case(1, 8, 8496, x, y, prefix, tail);
            ++fallback;
        }
    }
""" + text[end:]
    text = text.replace("alignas(4)", "alignas(8)").replace("&& !last_packed_prefix", "&& last_prefix_bits == 16")
    return replace_once(text, "&& last_packed_prefix", "&& last_prefix_bits == 64")


def instructions(assembly, label):
    lines = assembly[assembly.index(label + ":"):].splitlines()[1:]
    opcodes = []
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "ret":
            return opcodes
        if parts and not parts[0].startswith("."):
            opcodes.append(parts[0])
    raise ValueError("host probe never returned")


def check(compiler):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        targets = root / "target"
        pins = render(targets)
        assert render(root / "repeat") == pins
        with patch.dict(os.environ, TUNING_CPU_MANIFEST_SHA=pins["manifest_SHA256"]):
            worker = runpy.run_path(str(targets / "compile_once.py"))
            _, plan = worker["checked_source"]()
        assert plan["label"] == LABEL and worker["EXPORTS"] == {SYMBOL}
        prior = json.loads((BASE / "PLAN.json").read_bytes())
        assert all(plan[key] == prior[key] for key in ("primary", "controls", "compiler_prefix", "tools", "producer_SHA256"))
        source = (targets / SOURCE).read_bytes().decode()
        anchor = "    if (lane < tail_pairs)"
        old_tail = (BASE / SOURCE).read_bytes().decode().split(anchor, 1)[1].split("bool overlaps", 1)[0]
        assert source.split(anchor, 1)[1].split("bool overlaps", 1)[0] == old_tail

        shim = checked(PARENT / "local_runtime.hpp", SHIM_SHA)
        (root / "local_runtime.hpp").write_bytes(shim + b"\ninline unsigned last_prefix_bits = 0;\n")
        old = checked(PARENT / "check_local.cpp", REFERENCE_SHA).decode()
        reference = old[old.index("float from_bits("):old.index("int main()")]
        (root / "independent_reference.inc").write_bytes(b"#include <cmath>\n#include <vector>\n" + reference.encode())
        (root / "converted.cpp").write_bytes(host_source(source).encode())
        (root / "check.cpp").write_bytes(check_source().encode())
        paths = [str(root / "converted.cpp"), str(root / "check.cpp")]
        executable = root / "check.exe"
        subprocess.run([compiler, *FLAGS, *paths, "-o", str(executable)], timeout=45, check=True)
        counts = json.loads(subprocess.check_output([str(executable)], timeout=25))
        assert (counts["aligned_cases"], counts["unaligned_fallback_cases"], counts["invalid_no_launch"]) == (18, 15, 11)
        assert counts["prefix_values_exact"] == 7 * counts["rotated_values_exact"]

        probes = host_source(source) + """
extern "C" void copy32_probe(uint16_t* out, const uint16_t* in) { copy_prefix_pair32(out, in); }
extern "C" void copy64_probe(uint16_t* out, const uint16_t* in) { copy_prefix_group64(out, in); }
"""
        assembly_path = root / "probes.s"
        (root / "probes.cpp").write_bytes(probes.encode())
        subprocess.run([compiler, *FLAGS, "-S", str(root / "probes.cpp"), "-o", str(assembly_path)], timeout=25, check=True)
        assembly = assembly_path.read_text()
        result32, result64 = (instructions(assembly, tag) for tag in ("copy32_probe", "copy64_probe"))
        assert result32 == ["movl", "movl"] and result64 == ["movq", "movq"]
        compiler_id = subprocess.check_output([compiler, "--version"], text=True, timeout=5).splitlines()[0]
        return dict(counts=counts, target_package=pins, host_compiler=compiler_id,
                    host_probe_copy_instructions={"pack32": result32, "pack64": result64},
                    logical_aligned_groups_per_head={"pack32": 224, "pack64": 112},
                    host_compiler_invocations=2, actual_gfx936_compiles=0, actual_GPU_steps=0,
                    Model_GO=False, GPU_GO=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    print(json.dumps(check(parser.parse_args().compiler), sort_keys=True, indent=2))
