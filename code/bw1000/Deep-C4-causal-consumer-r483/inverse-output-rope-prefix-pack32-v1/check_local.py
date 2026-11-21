"""Compile the rendered kernel via its pinned CPU shim and distinguish host copy widths."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / "render.py"))
FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off"]


def instructions(assembly, label):
    start = assembly.index(label + ":")
    body = assembly[start:assembly.index("ret", start)].splitlines()[1:]
    return [parts[0] for line in body if (parts := line.split()) and not parts[0].startswith(".")]


def run():
    index = json.loads(R["checked"](R["BASE"] / "SOURCE_FILES.json", R["BASE_MANIFEST"]))["files_SHA256"]
    old_check = R["checked"](R["BASE"] / "check_local.cpp", index["check_local.cpp"]).decode()
    reference = old_check[old_check.index("float from_bits("):old_check.index("int main()")]
    old_kernel = R["checked"](R["BASE"] / "inverse_output_rope.cu", index["inverse_output_rope.cu"]).decode()
    old_copy = old_kernel[old_kernel.index("        out[2 * pair] ="):old_kernel.index("    if (lane < tail_pairs)")]
    scalar = old_copy.rsplit("    }\n", 1)[0].replace("2 * pair", "0").replace("0 + 1", "1")
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        target = temp / "rendered"
        target_pins = R["materialize"](target)
        repeated = R["materialize"](temp / "repeat")
        assert target_pins == repeated
        with patch.dict(os.environ, TUNING_CPU_MANIFEST_SHA=target_pins["target_manifest_SHA256"]):
            worker = runpy.run_path(str(target / "compile_once.py"))
            _, plan = worker["checked_source"]()
        assert worker["EXPORTS"] == {R["SYMBOL"]} and worker["LABEL"] == R["LABEL"]
        assert plan["planned_device_TUs"] == 1 and len(plan["variants"]) == 1
        old_plan = json.loads((R["BASE"] / "PLAN.json").read_bytes())
        assert all(plan[key] == old_plan[key] for key in ("tools", "compiler_prefix", "producer_SHA256",
                                                            "primary", "controls", "cuda_runtime_header_SHA256"))
        try:
            R["materialize"](target)
        except FileExistsError:
            pass
        else:
            raise AssertionError("rendered directory replacement accepted")
        source = (target / "inverse_output_rope.cu").read_bytes().decode()
        assert source.count(R["COPY_PAIR"]) == 1 and "reinterpret_cast<uint32_t*>" not in source
        converted = R["once"](source, "#include <cuda_runtime.h>", '#include "local_runtime.hpp"')
        converted = R["once"](converted, R["LAUNCH"],
                              "cpu_launch((inverse_output<HeadsPerBlock, PackedPrefix>), grid, threads, 0, stream,")
        converted = R["once"](converted, "    cpu_launch(", "    last_packed_prefix = PackedPrefix;\n    cpu_launch(")
        shim = R["checked"](R["BASE"] / "local_runtime.hpp", index["local_runtime.hpp"])
        (temp / "local_runtime.hpp").write_bytes(shim + b"\ninline bool last_packed_prefix = false;\n")
        (temp / "independent_reference.inc").write_bytes(
            b"#include <cmath>\n#include <vector>\n" + reference.encode())
        host_source = temp / "candidate.cpp"
        host_source.write_bytes(converted.encode())
        exe = temp / "check.exe"
        subprocess.run(["g++", *FLAGS, "-I" + str(temp), str(host_source), str(HERE / "check_local.cpp"),
                        "-o", str(exe)], timeout=30, check=True)
        counts = json.loads(subprocess.check_output([str(exe)], timeout=20))
        widths = temp / "copy-widths.cpp"
        widths.write_bytes((converted + '\nextern "C" void probe_scalar_pair(uint16_t* out, const uint16_t* in) {\n'
                            + scalar + '}\nextern "C" void probe_pack32_pair(uint16_t* out, const uint16_t* in) {\n'
                            '    copy_prefix_pair32(out, in);\n}\n').encode())
        asm = temp / "copy-widths.s"
        subprocess.run(["g++", *FLAGS, "-S", str(widths), "-o", str(asm)], timeout=30, check=True)
        assembly = asm.read_text()
        scalar_ops = instructions(assembly, "probe_scalar_pair")
        packed_ops = instructions(assembly, "probe_pack32_pair")
        assert scalar_ops.count("movzwl") == scalar_ops.count("movw") == 2, scalar_ops
        assert packed_ops == ["movl", "movl"], packed_ops
        assert (counts["aligned_cases"], counts["unaligned_fallback_cases"], counts["invalid_no_launch"]) == (18, 3, 11)
        assert counts["prefix_values_exact"] == 7 * counts["rotated_values_exact"]
        return dict(counts, target_package=target_pins, host_pair_copy_instructions=dict(scalar=scalar_ops,
                    packed=packed_ops), actual_host_compile_invocations=2, host_executable_SHA256=sha256(
                        exe.read_bytes()).hexdigest(),
                    compiler=subprocess.check_output(["g++", "--version"], timeout=5, text=True).splitlines()[0],
                    actual_gfx936_compiles=0, actual_Slurm_submissions=0, actual_GPU_steps=0)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
