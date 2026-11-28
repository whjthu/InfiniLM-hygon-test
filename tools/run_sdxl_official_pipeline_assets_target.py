"""Assemble and verify the exact official SDXL pipeline in one zero-GRES step."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE = "1103606", "f13r1n14"
MANIFEST_SHA = "b0a2515ad9a6422d83991788f6af7749bc50812ad0b4e028e52b86350c519175"
LABEL = "sdxl-official-pipeline18-verify"
ASSETS = ROOT / "incoming"
CONFIGS = ASSETS / "sdxl-official-index-scheduler-cbec4ec42d8343b7af9c00a6643f9b4c"
PATHS = {
    "scheduler": CONFIGS / "scheduler",
    "text_encoder": ASSETS / "sdxl-text-encoder-official-4621659-v1",
    "text_encoder_2": ASSETS / "sdxl-text-encoder-2-official-4621659-v1",
    "tokenizer": ASSETS / "sdxl-dual-tokenizer-toolchest-4fd51d9d4754419faefebc46653c9b9f/tokenizer",
    "tokenizer_2": ASSETS / "sdxl-dual-tokenizer-toolchest-4fd51d9d4754419faefebc46653c9b9f/tokenizer_2",
    "unet": ROOT / "results/training-t06-sdxl-unet-login-stage-v1/model/unet",
    "vae": ASSETS / "sdxl-vae-official-4621659-abe87f3e68b84aed83a7b887e138bd32/vae",
}
COLS = "JobIDRaw,JobName%128,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160,Start,End"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def manifest(staged):
    raw = (staged / "pipeline.json").read_bytes()
    ensure(sha256(raw).hexdigest() == MANIFEST_SHA, "Original official assets manifest differs")
    result = json.loads(raw)
    ensure(len(result) == 18 and sum(record["bytes"] for record in result.values()) == 13878864542,
           "Expected exact official pipeline extent")
    return result


def link_assets(work, known):
    root = work / "pipeline"
    root.mkdir(mode=0o700)
    for name, record in sorted(known.items()):
        relative = Path(name)
        if name == "model_index.json":
            source = CONFIGS / name
        else:
            component, member = relative.parts
            source = PATHS[component] / member
        current = source.lstat()
        ensure(source.resolve(strict=True) == source and stat.S_ISREG(current.st_mode)
               and current.st_size == record["bytes"], f"Official component extent or pathname differs: {name}")
        target = root / relative
        target.parent.mkdir(mode=0o700, exist_ok=True)
        os.link(source, target, follow_symlinks=False)
        alias = target.lstat()
        ensure((current.st_dev, current.st_ino, current.st_size) == (alias.st_dev, alias.st_ino, alias.st_size),
               f"Existing official file was not bound by inode: {name}")


def compute(work, request):
    ensure(os.environ.get("SLURM_JOB_ID") == PARENT and os.environ.get("SLURMD_NODENAME") == NODE
           and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Expected owned no-GRES step")
    source = work / "source"
    environment = dict(os.environ, APPTAINERENV_CUDA_VISIBLE_DEVICES="-1", APPTAINERENV_HIP_VISIBLE_DEVICES="-1",
                       APPTAINERENV_ROCR_VISIBLE_DEVICES="-1", APPTAINERENV_PYTHONNOUSERSITE="1")
    command = [APPTAINER, "exec", "--contain", "--no-mount", "home,cwd", "--pwd", "/",
               "--bind", f"{source}:/scope:ro", "--bind", f"{work / 'pipeline'}:/official-sdxl:ro",
               str(ROOT / "containers/dtk26-sandbox-r1"), "/usr/bin/python3", "-B", "/scope/run.py", "verify-assets",
               "--pipeline", "/official-sdxl", "--pipeline-manifest", "/scope/pipeline.json",
               "--pipeline-manifest-sha", MANIFEST_SHA]
    intent = json.loads((work / "intent.json").read_bytes())
    remaining = min(760, intent["deadline"] - time.time() - 30)
    ensure(intent["request"] == request and remaining > 0, "Original verification budget exhausted")
    started = time.monotonic()
    result = subprocess.run(["timeout", "--kill-after=5", str(remaining), *command], capture_output=True, env=environment,
                            check=False)
    (work / "verify.stdout").write_bytes(result.stdout)
    (work / "verify.stderr").write_bytes(result.stderr)
    if result.returncode:
        return result.returncode
    proof = json.loads(result.stdout.splitlines()[-1])
    ensure(proof == {"exact_readonly_pipeline_files": 18, "actual_optimizer_updates": 0},
           "Exact original asset verifier did not return")
    save(work / "compute.json", dict(request=request, step=PARENT + "." + os.environ["SLURM_STEP_ID"],
                                    full_verified_bytes=13878864542, full_manifest_SHA256=MANIFEST_SHA,
                                    wall_seconds=time.monotonic() - started, **proof))
    return 0


def accounting(epoch, name):
    earliest = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch - 120))
    result = subprocess.run(["sacct", "-j", PARENT, "--starttime", earliest, "-n", "-P", "-o", COLS], capture_output=True,
                            timeout=10, text=True, check=True, env=dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="standard"))
    return [row for line in result.stdout.splitlines() if len(row := line.split("|")) == 9 and row[1] == name
            and re.fullmatch(PARENT + r"\.\d+", row[0])]


def launch(work, request):
    epoch, name = time.time(), f"{LABEL}-{request[:12]}"
    ensure(not accounting(epoch, name), "This exact verification step already exists")
    save(work / "intent.json", {"request": request, "epoch": epoch, "label": name, "deadline": epoch + 840})
    try:
        link_assets(work, manifest(work / "source"))
    except (OSError, ValueError) as exc:
        save(work / "terminal.json", dict(request=request, accounting=[], result=None, passed=False,
                                           prelaunch_error_type=type(exc).__name__, prelaunch_errno=getattr(exc, "errno", None),
                                           prelaunch_error_SHA256=sha256(str(exc).encode()).hexdigest()))
        return 1
    command = ["srun", "--exact", "--immediate=5", f"--jobid={PARENT}", "-N1", "-n1", "-c2", "--mem=8G",
               "--gres=none", f"--nodelist={NODE}", "--time=00:13:00", f"--job-name={name}",
               HOST_PYTHON, "-B", str(Path(__file__).resolve()), "compute", str(work), request]
    with (work / "step.log").open("xb") as log:
        rc = subprocess.run(["timeout", "--kill-after=5", str(epoch + 820 - time.time()), *command], stdout=log,
                            stderr=subprocess.STDOUT, check=False).returncode
    rows = []
    for _ in range(5):
        if time.time() + 12 >= epoch + 840:
            break
        rows = accounting(epoch, name)
        if len(rows) == 1 and rows[0][2] not in ("RUNNING", "PENDING", "COMPLETING"):
            break
        time.sleep(1)
    proof = work / "compute.json"
    result = json.loads(proof.read_bytes()) if proof.is_file() else None
    passed = (rc == 0 and len(rows) == 1 and rows[0][2:4] == ["COMPLETED", "0:0"] and "gres/" not in rows[0][6]
              and result is not None and result["step"] == rows[0][0])
    save(work / "terminal.json", dict(request=request, accounting=rows, result=result, passed=passed, returncode=rc))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "compute"))
    parser.add_argument("work", type=Path)
    parser.add_argument("request")
    options = parser.parse_args()
    ensure(re.fullmatch(r"[a-f0-9]{32}", options.request) and options.work == ROOT / "results" / f"{LABEL}-{options.request}"
           and options.work.resolve(strict=True) == options.work and options.work.stat().st_uid == os.getuid(),
           "Exact original owned work scope required")
    raise SystemExit((launch if options.mode == "launch" else compute)(options.work, options.request))
