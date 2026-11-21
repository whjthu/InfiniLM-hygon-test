"""Compile one admitted gfx936 object on a CPU-only Slurm step."""
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


PACKAGE = Path(__file__).resolve().parent
LABEL = "Deep-inverse-output-rope-prefix-pack32-gfx936-CPU-v1"
EXPORTS = {"deep_inverse_rope_h1_pack32_v1"}


def require(value, label):
    if not value:
        raise ValueError(label)


def pin(path):
    content = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content.update(chunk)
    return content.hexdigest()


def dependencies(path, target_object):
    target, colon, fields = path.read_text().replace("\\\n", " ").partition(":")
    names = set(shlex.split(fields))
    require(colon and target.strip() == str(target_object) and names
            and all(name.startswith("/") for name in names), "EXACT_DEVICE_DEPENDENCY_FORMAT")
    return names


def checked_source():
    manifest = PACKAGE / "SOURCE_FILES.json"
    expected = os.environ["TUNING_CPU_MANIFEST_SHA"]
    require(re.fullmatch("[0-9a-f]{64}", expected) and pin(manifest) == expected, "ADMITTED_SOURCE_MANIFEST")
    files = json.loads(manifest.read_bytes())["files_SHA256"]
    require(all(Path(name).name == name and pin(PACKAGE / name) == digest
                for name, digest in files.items()), "ADMITTED_SOURCE_BYTES")
    plan = json.loads((PACKAGE / "PLAN.json").read_bytes())
    require(plan["label"] == LABEL and plan["source_SHA256"] == files["inverse_output_rope.cu"],
            "ADMITTED_COMPILER_PLAN")
    return expected, plan


def compiler_dependencies(plan, source):
    for tool, entry in plan["tools"].items():
        require(str(Path(tool).resolve(strict=True)) == entry["canonical"] and pin(tool) == entry["sha256"],
                "PREVIOUS_DTK_TOOLCHAIN_DRIFT")
    receipt = Path(plan["producer_receipt"])
    require(receipt.stat().st_size == plan["producer_bytes"] and pin(receipt) == plan["producer_SHA256"],
            "PREVIOUS_DEPENDENCY_RECEIPT_DRIFT")
    earlier = json.loads(receipt.read_bytes())
    maps = earlier["fresh_dependencies_SHA256"]
    require(earlier["actual_CPU"] == "910407.1113" and earlier["completed"] and earlier["passed"]
            and earlier["all19_fresh_M_before_compile"] and earlier["all19_MD_equal_fresh_M"] and len(maps) == 19,
            "PREVIOUS_DTK_PRODUCER_IDENTITY")
    allowed = {str(source): plan["source_SHA256"]}
    for group in maps.values():
        for path, digest in group.items():
            require(allowed.setdefault(path, digest) == digest, "PREVIOUS_DEPENDENCY_IDENTITY_CONFLICT")
    require(allowed.get(plan["cuda_runtime_header"]) == plan["cuda_runtime_header_SHA256"],
            "PREVIOUS_DTK_HEADER_IDENTITY")
    return allowed


def main():
    job, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    node = socket.gethostname().split(".")[0]
    require(job.isdecimal() and job == os.environ["TUNING_CPU_PARENT"] and step.isdecimal()
            and node == os.environ["TUNING_CPU_NODE"], "ADMITTED_SLURM_STEP_IDENTITY")
    require(all(os.environ.get(key) == "-1" for key in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and not any(os.environ.get(key) for key in ("SLURM_STEP_GPUS", "LD_PRELOAD", "LD_AUDIT"))
            and os.environ.get("ROCM_PATH") == "/opt/dtk", "OFFLINE_DTK_ENVIRONMENT")
    epoch, deadline = (float(os.environ[key]) for key in ("TUNING_CPU_EPOCH", "TUNING_CPU_DEADLINE"))
    wall_start, mono_start = time.time(), time.monotonic()
    require(math.isfinite(epoch) and math.isfinite(deadline) and deadline == epoch + 300
            and epoch <= wall_start < deadline - 12, "ADMITTED_SINGLE_EPOCH")
    mf, plan = checked_source()
    output = Path("/refactor/candidates") / LABEL
    output.mkdir(mode=0o700)
    source, obj = PACKAGE / "inverse_output_rope.cu", output / "inverse_output_rope.o"
    before, actual = output / "inverse_output_rope.before.d", output / "inverse_output_rope.d"
    receipt = output / "CPU-result.json"
    report = dict(label=LABEL, manifest_SHA256=mf, original_step=job + "." + step, node=node, epoch=epoch,
                  deadline=deadline, GRES="none", planned_device_TUs=1, actual_device_TUs=0,
                  actual_GPU_steps=0, explicit_Runtime_calls=0, commands=[], completed=False, passed=False)

    def save():
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, receipt)

    def command(name, argv, cap):
        left = min(deadline - time.time(), deadline - wall_start - (time.monotonic() - mono_start)) - 12
        require(left > 1, "ORIGINAL_EPOCH_TOOL_BUDGET")
        row = dict(label=name, argv=argv, returncode=None)
        report["commands"].append(row)
        save()
        stdout, stderr = output / (name + ".stdout"), output / (name + ".stderr")
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.run(argv, stdout=out, stderr=err, timeout=min(cap, left), check=False)
        row.update(returncode=child.returncode, stdout_SHA256=pin(stdout), stderr_SHA256=pin(stderr))
        save()
        require(child.returncode == 0, name + "_FAILED")
        return stdout

    try:
        allowed = compiler_dependencies(plan, source)
        flags = plan["compiler_prefix"]
        command("fresh-M", flags + ["-M", "-MF", str(before), "-MT", str(obj), str(source)], 70)
        fresh = dependencies(before, obj)
        require({str(source), plan["cuda_runtime_header"]} <= fresh
                and all(allowed.get(name) == pin(name) for name in fresh), "FRESH_M_DEPENDENCY_DRIFT")
        report.update(fresh_M_SHA256=pin(before), dependencies_SHA256={name: allowed[name] for name in sorted(fresh)})
        command("compile", flags + ["-MD", "-MF", str(actual), "-MT", str(obj), "-c", str(source), "-o", str(obj)], 100)
        report["actual_device_TUs"] = 1
        require(dependencies(actual, obj) == fresh and all(pin(name) == allowed[name] for name in fresh),
                "ACTUAL_MD_DEPENDENCY_DRIFT")
        symbols = command("symbols", ["/usr/bin/nm", "-g", "--defined-only", str(obj)], 20)
        exports = {fields[-1] for line in symbols.read_text().splitlines() if len(fields := line.split()) >= 3
                   and fields[-2] == "T" and fields[-1].startswith("deep_inverse_rope_")}
        require(exports == EXPORTS, "EXACT_PREFIX_PACK32_CABI_EXPORT")
        require(not re.search(r"lib(?:infinilm_runtime|infiniops|infinirt|torch|amdhip64|galaxyhip|cudart)",
                              Path("/proc/self/maps").read_text()), "NO_CONTROLLER_RUNTIME_MAPPING")
        report.update(passed=True, object=str(obj), object_SHA256=pin(obj), actual_MD_SHA256=pin(actual),
                      exports=sorted(exports), device_qualified=False)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report.update(error_type=type(error).__name__, error_summary=str(error)[:320])
    finally:
        report.update(completed=True, elapsed=float(time.monotonic() - mono_start))
        save()
    print(json.dumps({key: report.get(key) for key in ("original_step", "passed", "object_SHA256", "error_summary")}),
          flush=True)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
