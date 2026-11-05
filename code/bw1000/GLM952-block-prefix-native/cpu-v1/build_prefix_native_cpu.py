"""Execute a published, isolated GLM952 prefix Native CPU rebuild."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from time import monotonic

from bind_prefix_native_rebuild_argv import DEST
from plan_prefix_native_rebuild import NATIVE_SHA, RECEIPT_SHA, checked
from prefix_native_dependency_mirror import SUPPORT, build_mirror, verify_dependencies

PLAN_SHA = "95a34f3b9ceec6cb423c746996f4a2fe81d485794c92edca4a87208aaf9e09ff"
DEPS_SHA = "4c3977c9e26ee5b0c788afaaac8d50421cd1b31482178d20657a369336f8ddb3"
TOOLS_SHA = "bc3dd818bdfe3f70e1e81714e9827083f4b9ab6159022a8d4c262de78c484b8f"
CURRENT = Path("/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json")
ADDED_CAPI = {"modern_glm_block_stage_v1", "modern_glm_block_prepare_v1", "modern_glm_block_publish_v1"}
TOOLS = {"cc1plus": "/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus",
         "collect2": "/usr/lib/gcc/x86_64-linux-gnu/11/collect2",
         "as": "/usr/bin/as", "ld": "/usr/bin/ld"}

SOURCES = (
    "build_prefix_native_cpu.py", "bind_prefix_native_rebuild_argv.py",
    "plan_prefix_native_rebuild.py", "prefix_native_dependency_mirror.py",
    "derive_accepted_prefix_headers.py", "derive_accepted_prefix_bridge.py",
    "accepted_prefix_native.hpp", "accepted_prefix_ticket.hpp", "accepted_prefix_lengths.hpp",
    "CPU_inside_prefix_build.sh",
)
EVIDENCE = ("BUILD_ARGV.json", "DEPENDENCY_PINS.json", "TOOLCHAIN.json")


def file_sha(path):
    with Path(path).open("rb") as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def admission(package, claim_path, claim_sha, environ):
    claim = checked(claim_path.read_bytes(), claim_sha)
    step = ".".join(environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    if (claim.get("CPU_GO") is not True or claim.get("GPU_GO") is not False
            or claim.get("plan_sha256") != PLAN_SHA or claim.get("dependency_sha256") != DEPS_SHA
            or claim.get("toolchain_sha256") != TOOLS_SHA or claim.get("destination") != str(DEST)
            or claim.get("recompile_objects") != 8 or claim.get("retained_objects") != 21
            or claim.get("parent_job") != environ.get("SLURM_JOB_ID")
            or not re.fullmatch(r"\d+\.\d+", step)
            or not re.fullmatch(r"[a-f0-9]{40}", claim.get("publication_commit", ""))
            or environ.get("GLM952_PREFIX_CPU_CLAIM_SHA") != claim_sha
            or type(claim.get("workers")) is not int or not 1 <= claim["workers"] <= 4
            or type(claim.get("inside_seconds")) is not int or not 120 <= claim["inside_seconds"] <= 600):
        raise ValueError("missing published exact CPU build claim")
    if (any(environ.get(key) != "-1" for key in
            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            or any(environ.get(key) for key in ("LD_PRELOAD", "LD_AUDIT", "GCC_EXEC_PREFIX",
                "COMPILER_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"))):
        raise ValueError("unexpected build environment")
    manifest = checked((package / "SOURCE_MANIFEST.json").read_bytes(), claim["source_manifest_sha256"])
    members = manifest["files_SHA256"]
    if (set(members) != set(SOURCES + EVIDENCE)
            or any(manifest.get(key) is not False for key in ("CPU_GO", "GPU_GO", "Model_GO"))):
        raise ValueError("CPU source package inventory differs")
    for name, pin in members.items():
        path = (package / name).resolve(strict=True)
        if not path.is_relative_to(package) or file_sha(path) != pin:
            raise ValueError("source package bytes differ")
    return claim, step, members


def exports(raw):
    result = {}
    for row in raw.splitlines():
        _, kind, name = row.split()
        if kind in {"W", "V", "u", "w", "v"} and name.startswith("_Z"):
            continue
        if name in result:
            raise ValueError("duplicate ELF symbol")
        result[name] = kind
    return result


def check_exports(before, after):
    old, new = exports(before), exports(after)
    if (any(new.get(name) != kind for name, kind in old.items())
            or set(new) - old.keys() != ADDED_CAPI or any(new[name] != "T" for name in ADDED_CAPI)):
        raise ValueError("GLM Native exported ABI differs outside the three private additions")
    return len(old)


def dynamics(raw):
    return [(kind, value) for kind, value in
            re.findall(r"\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[([^\]]+)\]", raw)]


def compile_after_dependency_check(plan, expected, command, read_file, workers):
    def verify(row, md):
        output = row["compile_argv"][-1]
        return verify_dependencies(read_file(md), output, expected[output])

    for index, row in enumerate(plan["compiles"]):
        command(f"preprocess-{index}", row["preprocess_argv"])
        verify(row, row["preprocess_argv"][row["preprocess_argv"].index("-MF") + 1])

    def compile_one(item):
        index, row = item
        argv = row["compile_argv"]
        command(f"compile-{index}", argv)
        return verify(row, argv[argv.index("-MF") + 1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compile_one, enumerate(plan["compiles"])))


def run(package, claim_path, claim_sha):
    claim, step, members = admission(package, claim_path, claim_sha, os.environ)
    plan = checked((package / "BUILD_ARGV.json").read_bytes(), PLAN_SHA)
    deps = checked((package / "DEPENDENCY_PINS.json").read_bytes(), DEPS_SHA)
    tools = checked((package / "TOOLCHAIN.json").read_bytes(), TOOLS_SHA)["tools"]
    current = checked(CURRENT.read_bytes(), RECEIPT_SHA)
    if current["library_SHA256"] != NATIVE_SHA:
        raise ValueError("wrong original GLM Native")
    deadline = monotonic() + claim["inside_seconds"]
    root = Path(str(DEST))
    phase = "pins"

    def remaining():
        budget = deadline - monotonic()
        if budget <= 5:
            raise TimeoutError("original CPU budget exhausted")
        return budget

    def command(label, args):
        timeout = f"{remaining() - 4:.3f}s"
        with (root / f"{label}.stdout").open("x", encoding="utf-8") as out, (
                root / f"{label}.stderr").open("x", encoding="utf-8") as err:
            result = subprocess.run(["/usr/bin/timeout", "--kill-after=3s", timeout, *args],
                                    stdout=out, stderr=err, check=False)
        if result.returncode:
            raise RuntimeError(f"{label} returned {result.returncode}")
        return (root / f"{label}.stdout").read_text(encoding="utf-8")

    report = dict(completed=False, actual_CPU=step, publication_commit=claim["publication_commit"],
                  source_manifest_sha256=claim["source_manifest_sha256"], claim_sha256=claim_sha,
                  plan_sha256=PLAN_SHA, dependency_sha256=DEPS_SHA, GPU=False, Model=False, dlopen=False)
    for path, pin in {current["library"]: NATIVE_SHA, **current["final_objects_SHA256"],
                      **current["actual_dynamic_dependency_SHA256"]}.items():
        remaining()
        if file_sha(path) != pin:
            raise ValueError("original object or dependency bytes differ")
    for path, pin in tools.items():
        if str(Path(path).resolve(strict=True)) != pin["canonical"] or file_sha(path) != pin["sha256"]:
            raise ValueError("effective compiler tool differs")

    prepared = build_mirror(plan, deps, {name: members[name] for name in SUPPORT}, package)
    try:
        phase = "compiler-selection"
        for tool, pin in TOOLS.items():
            selected = command(f"selected-{tool}", ["/usr/bin/g++", f"-print-prog-name={tool}"]).strip()
            path = shutil.which(selected)
            if path is None or str(Path(path).resolve(strict=True)) != tools[pin]["canonical"]:
                raise ValueError("g++ selected a different compiler subprocess")
        phase = "preprocess-and-compile"
        counts = compile_after_dependency_check(plan, prepared["expected_dependencies_by_object"], command,
                                                lambda path: Path(path).read_bytes(), claim["workers"])
        phase = "link"
        command("link", plan["link_argv"])
        library = plan["link_argv"][plan["link_argv"].index("-o") + 1]
        phase = "elf-verification"
        nm = ["/usr/bin/nm", "-D", "--defined-only"]
        count = check_exports(command("parent-exports", [*nm, current["library"]]),
                              command("candidate-exports", [*nm, library]))
        elf = ["/usr/bin/readelf", "-d"]
        parent = dynamics(command("parent-dynamic", [*elf, current["library"]]))
        if not parent or parent != dynamics(command("candidate-dynamic", [*elf, library])):
            raise ValueError("candidate dynamic dependency metadata differs")
        report.update(completed=True, library=library, library_SHA256=file_sha(library),
                      compile_count=len(counts), retained_objects=21, dependency_counts=counts,
                      original_strong_or_CABI_exports=count, added_CAPI=sorted(ADDED_CAPI),
                      dynamic_metadata_equal=True, effective_loader_resolution_tested=False)
    except BaseException as error:
        report.update(failed_phase=phase, failure_type=type(error).__name__)
        raise
    finally:
        with (root / "CPU-build.json").open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("claim", type=Path)
    cli.add_argument("claim_sha")
    args = cli.parse_args()
    run(Path(__file__).resolve().parent, args.claim, args.claim_sha)
