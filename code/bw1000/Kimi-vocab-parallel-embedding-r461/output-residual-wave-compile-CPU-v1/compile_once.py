"""Compile the frozen full-tree and wave32 score objects on an admitted CPU slot."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time


HERE = Path(__file__).resolve().parent
LABEL = "Kimi461-output-residual-wave-score-gfx936-CPU-v1"
ARMS = ("full-tree", "wave32")
SYMBOL = "kimi_output_residual_scores_v1_launch"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def dependencies(path, obj):
    left, separator, right = path.read_text().replace("\\\n", " ").partition(":")
    deps = set(shlex.split(right))
    require(left.strip() == str(obj) and separator and deps and all(p.startswith("/") for p in deps),
            "EXACT_DEVICE_DEPENDENCY_TARGET")
    return deps


def admitted_sources(where, manifest):
    index = where / "SOURCE_FILES.json"
    require(re.fullmatch("[0-9a-f]{64}", manifest) and digest(index) == manifest, "SOURCE_MANIFEST_DRIFT")
    pins = json.loads(index.read_bytes())["files_SHA256"]
    plan = json.loads((where / "BUILD_PLAN.json").read_bytes())
    expected = {f"{a}/{n}" for a, items in plan["rendered_files_SHA256"].items() for n in items}
    require(set(plan["rendered_files_SHA256"]) == set(ARMS) and plan["label"] == LABEL
            and set(pins) == expected | {"compile_once.py", "BUILD_PLAN.json"}, "SOURCE_PACK_MEMBERS")
    for path, pin in pins.items():
        require((where / path).resolve(strict=True).relative_to(where).as_posix() == path
                and digest(where / path) == pin, "SOURCE_PACK_BYTES")
    require(all(pins[f"{a}/{n}"] == pin for a, items in plan["rendered_files_SHA256"].items()
                for n, pin in items.items()), "SOURCE_PLAN_DRIFT")
    return plan


def producer(closure):
    for tool, qualified in closure["tools"].items():
        require(str(Path(tool).resolve(strict=True)) == qualified["canonical"]
                and digest(tool) == qualified["sha256"], "ORIGINAL_DTK_TOOLCHAIN")
    receipt = Path(closure["producer_receipt"])
    require(receipt.stat().st_size == closure["producer_bytes"] and digest(receipt) == closure["producer_SHA256"],
            "ORIGINAL_DTK_PRODUCER_PIN")
    data = json.loads(receipt.read_bytes())
    maps = data["fresh_dependencies_SHA256"]
    require(data["actual_CPU"] == "910407.1113" and data["completed"] is True and data["passed"] is True
            and len(maps) == 19 and data["all19_fresh_M_before_compile"] is True
            and data["all19_MD_equal_fresh_M"] is True, "ORIGINAL_DTK_PRODUCER_QUALIFICATION")
    known = {}
    for paths in maps.values():
        for path, sha in paths.items():
            require(known.setdefault(path, sha) == sha, "ORIGINAL_DTK_DEPENDENCY_CONFLICT")
    require(known.get(closure["cuda_runtime_header"]) == closure["cuda_runtime_header_SHA256"],
            "ORIGINAL_DTK_HEADER_PIN")
    return known


def compile_arms(source_dir, output, plan, known, command, report, save):
    flags = plan["compiler"]["compiler_prefix"]
    require(flags == ["/opt/dtk/cuda/cuda/bin/nvcc", "-std=c++17", "-O3", "-arch=gfx936",
                      "-Xcompiler=-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1"], "FROZEN_COMPILER_RECIPE")
    header = plan["compiler"]["cuda_runtime_header"]
    sources = plan["rendered_files_SHA256"]
    common = None
    for arm in ARMS:
        work = output / arm
        work.mkdir(mode=0o700)
        src = source_dir / arm / "fused_scores.cu"
        obj, fresh = work / "fused_scores.o", work / "fused_scores.before.d"
        command(arm + "-M", flags + ["-M", "-MF", str(fresh), "-MT", str(obj), str(src)], 55)
        names = dependencies(fresh, obj)
        owned = {str(src.parent / filename): sha for filename, sha in sources[arm].items()
                 if filename in ("fused_scores.cu", "score_math.hpp", "reduction.hpp")}
        require(set(owned) <= names and header in names and all(
                (owned.get(name) if name in owned else known.get(name)) == digest(name) for name in names),
                "FRESH_M_DEPENDENCY_DRIFT")
        shared_names = names - set(owned)
        require(common is None or common == shared_names, "UNEQUAL_BASELINE_CANDIDATE_DTK_CLOSURE")
        common = shared_names
        report["arms"][arm].update(fresh_M_SHA256=digest(fresh), dependencies_SHA256={
            name: (owned.get(name) if name in owned else known[name]) for name in sorted(names)})
        save()
    report["all_fresh_M_before_compile"] = True
    save()

    for arm in ARMS:
        work = output / arm
        src = source_dir / arm / "fused_scores.cu"
        obj, md = work / "fused_scores.o", work / "fused_scores.d"
        command(arm + "-compile", flags + ["-MD", "-MF", str(md), "-MT", str(obj), "-c", str(src),
                                           "-o", str(obj)], 65)
        report["actual_device_TUs"] += 1
        row = report["arms"][arm]
        expected = row["dependencies_SHA256"]
        require(dependencies(md, obj) == set(expected) and all(digest(p) == h for p, h in expected.items()),
                "ACTUAL_MD_DEPENDENCY_DRIFT")
        symbols = command(arm + "-symbols", ["/usr/bin/nm", "-g", "--defined-only", str(obj)], 12)
        exports = {words[-1] for line in symbols.read_text().splitlines() if len(words := line.split()) >= 3
                   and words[-2] == "T" and words[-1].startswith("kimi_output_residual_")}
        require(exports == {SYMBOL} and obj.stat().st_size > 0, "EXACT_STANDALONE_SCORE_SYMBOL")
        row.update(object=str(obj), object_SHA256=digest(obj), actual_MD_SHA256=digest(md), exports=sorted(exports))
        save()


def main():
    job, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    node = socket.gethostname().split(".")[0]
    require(job.isdecimal() and job == os.environ["TUNING_CPU_PARENT"] and step.isdecimal()
            and node == os.environ["TUNING_CPU_NODE"], "ADMITTED_SLURM_PLACEMENT")
    require(all(os.environ.get(key) == "-1" for key in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and not any(os.environ.get(key) for key in ("SLURM_STEP_GPUS", "LD_PRELOAD", "LD_AUDIT"))
            and os.environ.get("ROCM_PATH") == "/opt/dtk", "CPU_ONLY_DTK")
    epoch, deadline = (float(os.environ[key]) for key in ("TUNING_CPU_EPOCH", "TUNING_CPU_DEADLINE"))
    wall, monotonic = time.time(), time.monotonic()
    require(math.isfinite(epoch) and math.isfinite(deadline) and deadline == epoch + 300
            and epoch <= wall < deadline - 12, "SINGLE_ORIGINAL_EPOCH")
    manifest = os.environ["TUNING_CPU_MANIFEST_SHA"]
    plan = admitted_sources(HERE, manifest)
    output = Path("/refactor/candidates") / LABEL
    output.mkdir(mode=0o700)
    report = dict(original_step=job + "." + step, node=node, manifest_SHA256=manifest, epoch=epoch, deadline=deadline,
                  planned_device_TUs=2, actual_device_TUs=0, actual_GPU_steps=0, runtime_calls=0,
                  arms={arm: {} for arm in ARMS}, commands=[], completed=False, passed=False)

    def save():
        temp = output / "build.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temp, output / "CPU-result.json")

    def command(label, argv, limit):
        remaining = min(deadline - time.time(), (deadline - wall) - (time.monotonic() - monotonic)) - 12
        require(remaining > 1, "ORIGINAL_COMPILER_BUDGET")
        row = dict(label=label, argv=argv, rc=None)
        report["commands"].append(row)
        save()
        out, err = (output / (label + suffix) for suffix in (".stdout", ".stderr"))
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     timeout=min(limit, remaining), check=False)
        row.update(rc=process.returncode, stdout_SHA256=digest(out), stderr_SHA256=digest(err))
        save()
        require(process.returncode == 0, label + "_FAILED")
        return out

    try:
        compile_arms(HERE, output, plan, producer(plan["compiler"]), command, report, save)
        require(report["arms"][ARMS[0]]["object_SHA256"] != report["arms"][ARMS[1]]["object_SHA256"],
                "CANDIDATE_OBJECTS_MUST_DIFFER")
        report.update(passed=True, device_qualified=False)
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        report.update(error_type=type(error).__name__, error_summary=str(error)[:300])
    finally:
        report["completed"] = True
        save()
    print(json.dumps({key: report.get(key) for key in ("original_step", "passed", "actual_device_TUs", "error_summary")}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
