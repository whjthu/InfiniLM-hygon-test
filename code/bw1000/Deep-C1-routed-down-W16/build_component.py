"""Build only the private W16 component and bind its unchanged W32 provider."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from time import monotonic

SOURCE = "e64eb976cd44760b3a01af5655e51e601dabf743fd78bf6a1c07cafcc1931c9a"
PARENT_RECEIPT = "837b0e824e79a33a460754eedc943a8c11786acfa856dde5ca2c438351ab5389"
NATIVE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
BASE_OBJECT = "/refactor/candidates/models-5f91ad7-r3675/Deep/helpers/routed.o"
OBJECT_SHA = "08ed7c8bfb7fe56cee66a6d6bdc5a5d569b05574ae68b2e7741d755a563ea756"
TOOLS = {
    "/opt/dtk/cuda/cuda/bin/nvcc": "a8a0244ff9e8b323fe9836122bc8cb318dd2443a20f1daa9eac739756879f584",
    "/opt/dtk/cuda/cuda/include/sm_61_intrinsics.hpp": "04b9f47a4cbe6f446d6de688596af9c75802093f7e84d157a806da119701628e",
}


def digest(path):
    with Path(path).open("rb") as handle:
        h = sha256()
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
        return h.hexdigest()


def expect(path, expected):
    if digest(path) != expected:
        raise ValueError(f"pinned artifact differs: {path}")


def plan(parent, source, output):
    argv = parent["link_argv"]
    if (argv[0], parent["library_SHA256"], parent["all50_objects_SHA256"][BASE_OBJECT]) != (
            "g++", NATIVE, OBJECT_SHA):
        raise ValueError("expected exact CPU750 build")
    output_flag = argv.index("-o")
    if argv[output_flag + 1:] != [parent["library"]]:
        raise ValueError("unexpected original link output")
    original = argv[1:output_flag]
    flags = [x for x in original if x.startswith(("-std=", "-O", "-D", "-I"))
             or x in ("-pthread", "-fPIC")]
    adapter = source / "deep_routed_down_w16_adapter.cpp"
    kernel = source / "deep_routed_down_w16_c1.cu"
    host = output / "adapter.o"
    gpu = output / "down-w16.o"
    lib = output / "libdeep_routed_w16_component.so"
    compile_base = ["g++", "-I" + str(source), *flags]
    return {
        "host_deps": [*compile_base, "-M", "-MT", str(host), str(adapter)],
        "host": [*compile_base, "-MD", "-MF", str(host.with_suffix(".d")),
                 "-c", str(adapter), "-o", str(host)],
        "device": ["/opt/dtk/cuda/cuda/bin/nvcc", "-std=c++17", "-O3", "-arch=gfx936",
                   "-Xcompiler=-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1", "-c", str(kernel), "-o", str(gpu)],
        "link": ["g++", "-shared", "-fPIC", str(host), str(gpu), parent["library"],
                 *[x for x in original if x.startswith(("-L", "-l"))],
                 "-Wl,--no-undefined", "-Wl,-rpath," + str(Path(parent["library"]).parent),
                 "-o", str(lib)],
    }


def deps_from(text):
    return {str(Path(x).resolve()) for x in shlex.split(text.replace("\\\n", " ").split(":", 1)[1])}


def build(source, output, parent_path):
    commit = os.environ.get("DEEP_W16_PUBLICATION_COMMIT", "")
    job = os.environ.get("SLURM_JOB_ID", "")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not re.fullmatch("[0-9a-f]{40}", commit) or not job.isdecimal() or not step.isdecimal():
        raise ValueError("published source and actual CPU step required")
    if any(os.environ.get(name) not in ("", "-1") for name in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("CPU build requires all device visibility explicitly disabled")
    expect(parent_path, PARENT_RECEIPT)
    parent = json.loads(parent_path.read_bytes())
    expect(source / "SOURCE_MANIFEST.json", SOURCE)
    source_manifest = json.loads((source / "SOURCE_MANIFEST.json").read_bytes())
    for name, expected in source_manifest["files_SHA256"].items():
        expect(source / name, expected)
    expect(BASE_OBJECT, OBJECT_SHA)
    expect(parent["library"], NATIVE)
    runtime_deps = {p: v["SHA256"] for p, v in parent["actual_DTK_bindings"].items()}
    for path, expected in {**runtime_deps, **TOOLS}.items():
        expect(path, expected)

    cmds = plan(parent, source, output)
    output.mkdir()
    start = monotonic()

    def invoke(label, command):
        budget = min(110, 260 - (monotonic() - start))
        if budget <= 0:
            raise TimeoutError("original build budget exhausted")
        result = subprocess.run(command, capture_output=True, text=True, timeout=budget)
        (output / (label + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed with {result.returncode}")
        return result.stdout

    try:
        headers = {p: digest(p) for p in deps_from(invoke("host-M", cmds["host_deps"]))}
        invoke("host", cmds["host"])
        if deps_from((output / "adapter.d").read_text()) != set(headers):
            raise ValueError("host M/MD dependency closure differs")
        for path, expected in headers.items():
            expect(path, expected)
        invoke("device-compile", cmds["device"])
        invoke("link", cmds["link"])
        lib = Path(cmds["link"][-1])
        exports = invoke("exports", ["nm", "-D", "--defined-only", str(lib)])
        symbols = {line.split()[-1] for line in exports.splitlines() if line.strip()}
        needed = {"deep_routed_down_w16_c1"} | {
            "deep_routed_w16_component_" + name
            for name in ("create", "forward", "destroy", "error", "report")
        }
        if not needed <= symbols or symbols & {"deep_routed_dot_r20", "deep_routed_validate_r20"}:
            raise ValueError("unexpected private/parent symbol binding")
        out = dict(completed=True, CPU_only=True, source_SHA256=SOURCE,
                   baseline_object_SHA256=OBJECT_SHA, baseline_native_SHA256=NATIVE,
                   baseline_native=parent["library"], library=str(lib), library_SHA256=digest(lib),
                   deps=runtime_deps, host_headers_SHA256=headers,
                   device_object_SHA256=digest(output/"down-w16.o"), actual_step=job+"."+step,
                   publication_commit=commit, Native_dlopen=False, GPU_initialized=False,
                   elapsed_s=monotonic()-start)
    except BaseException as error:
        out = dict(completed=False, actual_step=job+"."+step, error=type(error).__name__,
                   elapsed_s=monotonic()-start)
        raise
    finally:
        with (output / "CPU-build.json").open("x", encoding="utf8") as handle:
            json.dump(out, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "parent_receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), args.parent_receipt.resolve())
