"""Compile the candidate translation unit locally and check arithmetic plus launcher boundaries."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / "render.py"))
LAUNCH = ("scores_kernel<<<kimi_output_residual_scores_v1::rows, kimi_output_residual_scores_v1::lanes,\n"
          "                    0, static_cast<cudaStream_t>(stream)>>>(")


def old_fixture():
    pins = json.loads(R["checked"](R["BASE"] / "SOURCE_FILES.json", R["BASE_PIN"]))["files_SHA256"]
    document = ast.parse(R["checked"](R["BASE"] / "check_local.py", pins["check_local.py"]))
    pairs = {key.id: ast.literal_eval(item.value) for item in document.body if isinstance(item, ast.Assign)
             for key in item.targets if isinstance(key, ast.Name) and key.id in ("SHIM", "CHECK")}
    return pairs["SHIM"], pairs["CHECK"]


def run():
    shim, c_abi = old_fixture()
    with TemporaryDirectory() as folder:
        temp = Path(folder)
        pack = temp / "rendered"
        result = R["materialize"](pack)
        assert result == R["materialize"](temp / "second")
        compiler = runpy.run_path(str(pack / "compile_once.py"))
        plan = compiler["admitted_sources"](pack.resolve(), result["target_manifest_SHA256"])
        assert compiler["ARMS"] == (R["ARM"],) and plan["label"] == R["LABEL"]
        assert plan["frozen_shape"]["input"] == [512, 9, 7168] and result["target_files"] == 7
        baseline_plan = json.loads((R["BASE"] / "BUILD_PLAN.json").read_bytes())
        assert plan["compiler"] == baseline_plan["compiler"]
        assert plan["original_gfx936_compiler_retains_across_barrier"] is None
        target_gate = plan["target_validation"]
        assert target_gate["require_eliminated_actual_input_reload"] and target_gate["require_no_added_scratch_or_spill"]
        assert target_gate["warmed_timing"]["order"] == ["AB", "BA", "AB"]
        arm = pack / R["ARM"]
        source = (arm / "fused_scores.cu").read_bytes().decode()
        translated = R["once"](source, '__attribute__((visibility("hidden"))) ', "")
        translated = R["once"](translated, "#include <cuda_runtime.h>", shim)
        translated = R["once"](translated, LAUNCH, "record(scores_kernel, 4608, 256, 0, stream,")
        fixture = R["once"](c_abi, "int main() {", "int main() {\n    check_numerical();")
        src, binary = temp / "candidate.cpp", temp / "candidate.exe"
        src.write_bytes((translated + "\n" + (HERE / "check_numeric.inc").read_text(encoding="utf-8") + fixture).encode())
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off",
                        "-I" + str(arm), str(src), "-o", str(binary)], timeout=30, check=True)
        exact = json.loads(subprocess.check_output([str(binary)], timeout=30))
        assert exact == dict(exact_bf16_scores=4608, exact_mapped_bf16_words=4608 * 7168,
                             square_and_dot_lane_pairs=4608 * 256)
        assert "retained28/fused_scores.o" == plan["artifact_expected"][R["ARM"]]
        return dict(exact, target_package=result, actual_host_compiles=1, actual_host_runs=1,
                    host_CABI_calls=10, invalid_no_launch=8, host_executable_SHA256=R["sha"](binary.read_bytes()),
                    compiler=subprocess.check_output(["g++", "--version"], timeout=5, text=True).splitlines()[0],
                    measured_HBM_bytes=None, measured_gfx936_spill_bytes=None, actual_gfx936_compiles=0,
                    actual_GPU_steps=0, actual_Slurm_submissions=0)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
