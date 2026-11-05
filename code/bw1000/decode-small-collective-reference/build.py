"""Prepare or execute the single host translation unit build for the decode CCL probe."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

REL = Path("code/bw1000/decode-small-collective-reference")
HISTORY = Path("evidence/bw1000/head-and-raw-ccl/CCL-raw-AR-private-r9176")
CPU = "910373.3491"
OLD_SRC = "/refactor/incoming/CCL-raw-AR-r9176/raw_ar.cpp"
OLD_OUT = "/refactor/candidates/CCL-raw-AR-r9176/"
BASE = PurePosixPath("/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM")
EXPORTS = {"decode_ccl_ar_" + name for name in
           ("error", "id_size", "id", "geometry", "create", "prepare", "run", "close", "chain")}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"unexpected SHA256: {path}")


def prepare(repo):
    source = repo / REL
    parent_build = repo / HISTORY / "CPU-b-observed-a.json"
    parent_plan = repo / HISTORY / "minimal-AR-plan-r9176.json"
    require_hash(parent_build, "4d27a7f69efd5fe78b27d9efa2720b92094d5c7ce1abad2ff831816269ac3cd7")
    require_hash(parent_plan, "ed94f694a4dd1322f34a961622ee2df1c2cb9ee3a72689cd9b1da9fea55bee8e")
    observation = json.loads(parent_build.read_bytes())
    recipe = json.loads(parent_plan.read_bytes())
    scope = json.loads((source / "SOURCE_CANDIDATE.json").read_bytes())
    receipt = observation["receipt"]
    if (observation["actual"][0][:4] != [CPU, "CCL-raw-AR-CPU-r9176b", "COMPLETED", "0:0"]
            or receipt["producer"] != CPU or receipt["completed"] is not True
            or receipt["source_SHA256"] != scope["parent_sha256"]):
        raise ValueError("historical producer binding mismatch")
    parent = repo / scope["parent_path"]
    patch = source / "PARENT_DELTA.patch"
    require_hash(parent, scope["parent_sha256"])
    require_hash(patch, scope["patch_sha256"])
    require_hash(source / "decode_geometry.hpp", scope["header_sha256"])
    with tempfile.TemporaryDirectory(prefix="decode-ccl-patch-") as directory:
        work = Path(directory)
        copied = work / scope["parent_path"]
        copied.parent.mkdir(parents=True)
        copied.write_bytes(parent.read_bytes())
        subprocess.run(["git", "apply", str(patch)], cwd=work, check=True, timeout=10)
        # The Windows Git checkout of the patch uses CRLF. Pin the Linux TU
        # after canonicalizing only line endings.
        cpp = copied.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(cpp).hexdigest() != scope["canonical_candidate_cpp_sha256"]:
            raise ValueError("materialized C++ source SHA256 mismatch")

    chain_inputs = {
        "deep-chain/deep_chain.hpp": "036a18a45a8e071f4755dde4351b994ce79751c60cb69c2020818317b39908da",
        "deep-chain/deep_chain_runtime.inc": "bda74b8b57c46877fbc80f30054deac785b44a08e9af69152ec5119f6ed6ed68",
    }
    for name, digest in chain_inputs.items():
        require_hash(source / name, digest)
    cpp += b'#include "deep-chain/deep_chain_runtime.inc"\n'

    historical_paths = {
        OLD_SRC: "{source}/raw_ar.cpp",
        OLD_OUT + "raw_ar.d": "{output}/raw_ar.d",
        OLD_OUT + "libCCL_raw_AR_r9176.so": "{output}/libDecodeSmallCollective.so",
    }
    parent_command = receipt["command"]
    if any(parent_command.count(path) != 1 for path in historical_paths):
        raise ValueError("historical command has unexpected input/output paths")
    command = [historical_paths.get(arg, arg) for arg in parent_command]
    sdk_lib = PurePosixPath(recipe["currentSDK"]) / "lib"
    libraries = {}
    for name, digest in recipe["pinned_libs"].items():
        directory = BASE / "python/infinilm/_lib" if name == "libinfinilm_runtime.so" else sdk_lib
        libraries[str(directory / name)] = digest
    return ({
        "state": "SOURCE_ONLY",
        "historical_cpu": CPU,
        "historical_receipt_sha256": sha(parent_build),
        "historical_plan_sha256": sha(parent_plan),
        "compiler_argv": command,
        "inputs": {"raw_ar.cpp": hashlib.sha256(cpp).hexdigest(),
                   "decode_geometry.hpp": scope["header_sha256"], **chain_inputs},
        "header_sha256": {row["path"]: row["SHA256"] for row in recipe["header_provenance"]},
        "link_input_sha256": libraries,
        "resolved_runtime_paths": {
            "libinfiniccl.so": str(sdk_lib / "libinfiniccl.so"),
            "libinfinirt.so": str(sdk_lib / "libinfinirt.so"),
            "libcudart.so.12": "/opt/dtk/cuda/cuda/lib64/libcudart.so.12",
            "librccl.so.1": "/opt/dtk/lib/librccl.so.1",
        },
        "required_exports": sorted(EXPORTS),
        "sdk_compile_completed": False,
        "gpu_go": False,
    }, cpp)


def build(source, out, expected_plan_sha):
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+", step):
        raise ValueError("expected a numeric Slurm job.step")
    if any(os.environ.get(key) != "" for key in
           ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("CPU build requires explicit empty device masks")
    plan_file = source / "BUILD_CONTRACT.json"
    require_hash(plan_file, expected_plan_sha)
    plan = json.loads(plan_file.read_bytes())
    for name, digest in plan["inputs"].items():
        require_hash(source / name, digest)
    for path, digest in (plan["header_sha256"] | plan["link_input_sha256"]).items():
        require_hash(Path(path), digest)
    out.mkdir()
    native = out / "libDecodeSmallCollective.so"
    command = [arg.format(source=source, output=out) for arg in plan["compiler_argv"]]
    with (out / "compile.log").open("xb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(native)],
                                      text=True, timeout=10)
    actual = {line.split()[-1] for line in symbols.splitlines() if " decode_ccl_ar_" in line}
    if actual != EXPORTS or actual != set(plan["required_exports"]):
        raise ValueError("Native export census mismatch")
    ldd = subprocess.check_output(["ldd", str(native)], text=True, timeout=30)
    if "not found" in ldd:
        raise ValueError("Native runtime dependencies unresolved")
    resolved = dict(re.findall(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(", ldd, re.M))
    needed = plan["resolved_runtime_paths"]
    for name, path in needed.items():
        if resolved.get(name) != path:
            raise ValueError("Native loader selected unexpected path for " + name)
        if path in plan["link_input_sha256"]:
            require_hash(Path(path), plan["link_input_sha256"][path])
    result = {
        "actual_step": step, "build_contract_sha256": expected_plan_sha,
        "native_sha256": sha(native), "dependency_file_sha256": sha(out / "raw_ar.d"),
        "required_runtime_libraries": {name: {"path": path, "sha256": sha(Path(path))}
                                       for name, path in needed.items()},
        "native_exports": sorted(actual), "gpu_executed": False, "python_dlopen": False,
    }
    (out / "CPU_BUILD_RESULT.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("plan")
    plan.add_argument("repo", type=Path)
    plan.add_argument("--cpp-out", type=Path)
    execute = modes.add_parser("compile")
    execute.add_argument("source", type=Path)
    execute.add_argument("out", type=Path)
    execute.add_argument("plan_sha256")
    args = parser.parse_args()
    if args.mode == "plan":
        result, cpp = prepare(args.repo.resolve())
        if args.cpp_out is not None:
            with args.cpp_out.open("xb") as stream:
                stream.write(cpp)
        print(json.dumps(result, indent=2))
    else:
        build(args.source.resolve(), args.out.resolve(), args.plan_sha256)


if __name__ == "__main__":
    main()
