"""Build exactly one isolated Kimi residual HOST translation unit."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess


ROOT = Path(__file__).resolve().parent
LABEL = "Kimi461-output-residual-stage-diagnostic-CPU-v1"
OUTPUT = Path("/refactor/candidates") / LABEL


def support(root):
    path = root / "build_support.py"
    spec = importlib.util.spec_from_file_location("pinned_build_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pins_for(root, manifest_sha):
    import hashlib

    def check(path, sha):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("pinned component source differs")

    manifest = root / "SOURCE_FILES.json"
    check(manifest, manifest_sha)
    files = json.loads(manifest.read_bytes())["files"]
    if set(files) != {
        "CLAIM.json", "CPU_build.py", "build_support.py", "CPU_inside.sh", "CPU_outer.sh",
        "launch_on_host.py", "component.cpp", "residual_method.inc", "output_residual_tiles.hpp", "stage.hpp",
    }:
        raise ValueError("invalid component package file set")
    for name, sha in files.items():
        check(root / name, sha)
    claim = json.loads((root / "CLAIM.json").read_bytes())
    if claim["label"] != LABEL or claim["GPU_GO"] is not False or claim["actual_trial_limit"] != 1:
        raise ValueError("invalid component CPU claim")
    return claim


def main():
    manifest = os.environ["KIMI_TILE_MANIFEST_SHA256"]
    claim = pins_for(ROOT, manifest)
    lib = support(ROOT)
    epoch = float(os.environ["KIMI_TILE_EPOCH"])
    deadline = float(os.environ["KIMI_TILE_INNER_DEADLINE"])
    lib.budget(epoch, deadline, 260)
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    lib.need(socket.gethostname().split(".")[0] == claim["node"]
             and os.environ["SLURM_JOB_ID"] == claim["parent"]
             and os.environ["SLURM_STEP_ID"].isdigit() and len(os.sched_getaffinity(0)) >= 4
             and all(os.environ.get(k) == "" for k in
                     ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
             and not any(os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT")),
             "wrong bounded component CPU resource")
    kfd = Path("/sys/class/kfd/kfd/proc")
    lib.need(kfd.is_dir() and not any(kfd.iterdir()), "compute node has device owners")
    OUTPUT.mkdir(mode=0o700)
    result = dict(actual_CPU=step, epoch=epoch, deadline=deadline, manifest_SHA256=manifest,
                  completed=False, passed=False, device_calls=0, weights_read=0, Native_dlopen=False,
                  commands=[], logs={})

    def run(name, args, cap):
        result["phase"] = name
        result["commands"].append(args)
        log = OUTPUT / (name + ".log")
        try:
            with log.open("xb") as stream:
                subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=lib.budget(epoch, deadline, cap), check=True)
        finally:
            if log.is_file():
                result["logs"][name] = dict(bytes=log.stat().st_size, SHA256=lib.checksum(log))
        return log.read_text(errors="replace")

    try:
        parent = lib.read_pin(claim["CPU330_receipt"])
        native = parent["candidate"]
        lib.need(parent["passed"] is True and parent["completed"] is True
                 and parent["actual_CPU"] == "1103606.330"
                 and lib.checksum(native) == parent["candidate_SHA256"] == claim["CPU330_native_SHA256"],
                 "CPU330 candidate identity differs")
        compile, = [c for c in parent["commands"] if "-c" in c]
        oldlink, = [c for c in parent["commands"] if "-shared" in c]
        opts = compile[:compile.index("-MD")]
        source = str(ROOT / "component.cpp")
        obj, before, after, output = map(str, (OUTPUT/"component.o", OUTPUT/"before.d", OUTPUT/"actual.d",
                                               OUTPUT/"libKimi_output_residual_stage_diagnostic_r461.so"))
        run("fresh-M", [*opts, "-M", "-MF", before, "-MT", obj, source], 35)
        paths = lib.dep_paths(Path(before), obj)
        owned = {str(ROOT / f) for f in ("component.cpp", "residual_method.inc", "output_residual_tiles.hpp", "stage.hpp")}
        lib.need(owned <= paths, "component did not use its frozen generated method and constants")
        deps = {p: lib.checksum(p) for p in sorted(paths)}
        common = paths & parent["exact_new_MD"].keys()
        lib.need(common and all(deps[p] == parent["exact_new_MD"][p] for p in common),
                 "shared headers differ from CPU330")
        run("component-HOST-compile", [*opts, "-MD", "-MF", after, "-c", source, "-o", obj], 100)
        lib.need(lib.dep_paths(Path(after), obj) == paths
                 and all(lib.checksum(p) == digest for p, digest in deps.items()), "fresh M/MD closure differs")
        flags = [x for x in oldlink if x.startswith(("-L", "-l"))]
        run("component-private-link", [opts[0], "-shared", "-pthread", "-Wl,-Bsymbolic",
            "-Wl,--no-undefined", obj, *flags, "-o", output], 45)
        exports = lib.exports(run("exports", ["/usr/bin/nm", "-D", "--defined-only", output], 12))
        required = {"kimi_output_residual_component_r461", "kimi_output_residual_error_r461"}
        lib.need(all(exports.get(name) == "T" for name in required), "component C ABI exports differ")
        loader = run("passive-relocations", ["/usr/bin/ldd", "-r", output], 25)
        providers = lib.providers(loader)
        required_owners = {"_infinilm.cpython-310-x86_64-linux-gnu.so", "libinfinilm_runtime.so"}
        lib.need(required_owners <= {Path(p).name for p in providers}, "expected runtime or layer provider missing")
        reused = providers & parent["actual_passive_providers"].keys()
        lib.need(reused and all(lib.checksum(p) == parent["actual_passive_providers"][p] for p in reused),
                 "shared ELF provider bytes differ from CPU330")
        result.update(passed=True, DSO=output, DSO_SHA256=lib.checksum(output), object_SHA256=lib.checksum(obj),
                      dependency_SHA256=deps, CPU330_shared_dependency_count=len(common),
                      passive_providers=sorted(providers), CPU330_shared_provider_count=len(reused),
                      C_ABI=sorted(required), host_TUs=1)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, reason=str(exc)[:300])
        raise
    finally:
        result["completed"] = True
        with (OUTPUT / "CPU-result.json").open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")


if __name__ == "__main__":
    main()
