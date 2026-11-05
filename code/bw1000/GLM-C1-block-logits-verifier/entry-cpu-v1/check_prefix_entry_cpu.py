"""Check the new GLM continuation entry and ELF resolution on one masked CPU step."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from time import monotonic

OLD_NATIVE = "/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/libGLM_V_on_demand_Model_r403.so"
OLD_NATIVE_SHA = "9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b"


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def check(package, claim_path, claim_sha):
    raw = claim_path.read_bytes()
    if sha256(raw).hexdigest() != claim_sha:
        raise ValueError("CPU claim bytes differ")
    claim = json.loads(raw)
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    if (os.environ["SLURM_JOB_ID"], socket.gethostname().split(".")[0]) != (
            claim["parent"], claim["node"]):
        raise ValueError("CPU step is outside its assigned node")
    if any(os.environ.get(key) != "-1" for key in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("CPU validation requires the original negative device masks")
    if digest(package / "SOURCE_MANIFEST.json") != claim["package_manifest_SHA256"]:
        raise ValueError("private package manifest differs")
    if digest(package / "config.template.json") != claim["config_SHA256"]:
        raise ValueError("private config differs")
    if digest(Path(__file__)) != claim["checker_SHA256"]:
        raise ValueError("CPU checker source differs")

    started = monotonic()
    deadline = started + claim["inside_seconds"]
    result = dict(completed=False, passed=False, actual_CPU=step, GPU=False, compile=False,
                  candidate_Native_dlopen=False, Model=False, HTTP=False,
                  package_manifest_SHA256=claim["package_manifest_SHA256"], claim_SHA256=claim_sha)
    output = Path(claim["output"])
    if output.exists():
        raise FileExistsError("original CPU output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    def remaining():
        seconds = deadline - monotonic()
        if seconds < 5:
            raise TimeoutError("original CPU deadline reached")
        return seconds - 3

    def closure(label, native):
        run = subprocess.run(["/usr/bin/ldd", "-r", str(native)], capture_output=True,
                             text=True, timeout=remaining(), stdin=subprocess.DEVNULL)
        text = run.stdout + run.stderr
        result[label] = dict(rc=run.returncode, output_SHA256=sha256(text.encode()).hexdigest(),
                             bytes=len(text.encode()),
                             errors=[line.strip() for line in text.splitlines()
                                     if "not found" in line or "undefined symbol" in line][:12])
        if run.returncode or result[label]["errors"]:
            raise ValueError("CPU ELF resolution failed: " + label)
        paths = {}
        for line in run.stdout.splitlines():
            fields = line.strip().split()
            if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith("/"):
                path = Path(fields[2]).resolve(strict=True)
                paths[fields[0]] = dict(path=str(path), SHA256=digest(path))
        if not paths:
            raise ValueError("ldd produced no dependency paths")
        return paths

    try:
        remaining()
        sys.path.insert(0, str(package))
        from public_glm_native import prefix_native_guard as guard
        from public_glm_native import stage_compare
        from public_glm_native.freeze_continuation_inputs import freeze, summary

        config = json.loads((package / "config.template.json").read_bytes())
        result["native_identity"] = guard.verify(config)
        captured = io.StringIO()
        with redirect_stdout(captured):
            stage_compare.main([
                "--config", str(package / "config.template.json"),
                "--config-sha", claim["config_SHA256"],
                "--source-sha", claim["package_manifest_SHA256"],
            ])
        entry = json.loads(captured.getvalue())
        if entry != dict(source_verified=True, GPU=False, Slurm=False):
            raise ValueError("source-only entry returned an unexpected receipt")
        result["actual_entry"] = entry
        plan = Path(config["block_compare_private_plan"])
        result["plan"] = summary(freeze(plan.read_bytes()))
        if result["plan"]["CAPI_forwards_per_owner"] != 55:
            raise ValueError("unexpected continuation work count")
        if any(name in sys.modules for name in ("torch", "infinilm", "_infinilm")):
            raise ValueError("entry validation imported an execution dependency")
        if digest(Path(OLD_NATIVE)) != OLD_NATIVE_SHA:
            raise ValueError("original GLM Native differs")
        old = closure("original_loader", OLD_NATIVE)
        candidate = closure("candidate_loader", Path(config["Native_library"]))
        if old != candidate:
            raise ValueError("candidate effective loader dependencies differ")
        result.update(passed=True, completed=True, effective_dynamic_dependencies=candidate,
                      explicit_Native_calls=0, explicit_SDK_calls=0,
                      active_preload_initialization="unmeasured")
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = monotonic() - started
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("package", type=Path)
    cli.add_argument("claim", type=Path)
    cli.add_argument("claim_sha")
    args = cli.parse_args()
    value = check(args.package, args.claim, args.claim_sha)
    print(json.dumps({key: value[key] for key in ("passed", "actual_CPU", "elapsed_seconds")}))
