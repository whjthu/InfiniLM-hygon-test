"""Read-only DTK dependency and source checks under one zero-GRES step."""

import argparse
import ast
from hashlib import sha256
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE, STANDBY_STEP = "1103606", "f12r2n10", "1103606.2480"
STANDBY_REQUEST = "9454531ca7ef48e9a18f73d5ed411683"
STANDBY_LABEL = "training80-dcu-standby-" + STANDBY_REQUEST[:12] + "-" + NODE
WORKER_JSON = ROOT / "results" / ("training80-dcu-standby-" + STANDBY_REQUEST) / NODE / "worker.json"
REQUEST = "30f9d5dd39bc49c1b8d52312ba2951d6"
LABEL = "O01-pack16-deps-" + REQUEST[:12]
STAGE = ROOT / "incoming" / (LABEL + "-reviewed")
OUT = ROOT / "results" / LABEL
GPU_REQUEST = "a31421758fc74d319923dbce7045cc12"
GPU_PUBLICATION = "c4da359b16d2eea35300547f42e9967dd3199b86"
RUNTIME_SHA = "999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
RUNTIME_NAMES = {"worker.py", "candidate.py", "baseline.py", "contract.py", "hipprof_trace.py"}
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
DTK = ROOT / "containers/dtk26-sandbox-r1"
SITE = ROOT / "results/training-t06-sdxl-runtime-v1"
BOOT = Path("/proc/sys/kernel/random/boot_id")
ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
TOOLS = (
    (Path("/opt/dtk/hip/bin/hipprof"), 15_528_344, "5fe2db93fb58b7c4e6e5cbf792fa90f81bb660ff54c3556dead235294f3b6a94"),
    (Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0"), 12_349_016,
     "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"),
)


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def read(path, cap, expected=None):
    require(not path.is_symlink(), "PRECHECK_SYMLINK")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    require(0 < len(raw) <= cap and (expected is None or sha256(raw).hexdigest() == expected), "PRECHECK_SOURCE_PIN")
    return raw


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def command(argv, timeout=4, cap=1048576):
    raw = subprocess.check_output(argv, timeout=timeout)
    require(0 < len(raw) <= cap, "PRECHECK_COMMAND_OUTPUT")
    return raw


def number():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(PARENT + r"\.[0-9]+", step) and step != STANDBY_STEP
            and socket.gethostname().split(".")[0] == NODE
            and len(os.sched_getaffinity(0)) == 1
            and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "PRECHECK_ZERO_GRES_SCOPE")
    return step


def rows():
    raw = command(["scontrol", "show", "step", PARENT, "--oneliner"])
    return [dict(field.split("=", 1) for field in line.split() if "=" in field)
            for line in raw.decode().splitlines() if line.strip()]


def live_pair(step):
    visible = rows()
    specs = ((STANDBY_STEP, STANDBY_LABEL,
              {"cpu=8,gres/dcu=8,node=1,mem=16G", "cpu=8,gres/dcu=8,node=1,mem=16384M"}),
             (step, LABEL, {"cpu=1,node=1,mem=256M"}))
    for handle, label, resources in specs:
        matched = [r for r in visible if r.get("StepId") == handle]
        require(len(matched) == 1, "PRECHECK_STEP_AMBIGUOUS")
        [row] = matched
        tres = row["TRES"].split(",")
        require(row.get("Name") == label and row.get("State") == "RUNNING" and row.get("NodeList") == NODE
                and len(tres) == len(set(tres)) and any(set(tres) == set(x.split(",")) for x in resources),
                "PRECHECK_ACTUAL_STEP_DIFFERS")


def worker_identity():
    raw = read(WORKER_JSON, 16384)
    worker = json.loads(raw)
    pid = worker["worker_pid"]
    pci = worker["pci"]
    require(worker["step"] == STANDBY_STEP and worker["node"] == NODE and worker["synthetic"] is True
            and len(pci) == len(set(pci)) == 8 and type(pid) is int and pid > 1, "PRECHECK_STANDBY_OWNER")
    parsed = command(["scontrol", "listpids", STANDBY_STEP]).decode().splitlines()
    require(any(row.split()[:3] == [str(pid), PARENT, "2480"] for row in parsed), "PRECHECK_STANDBY_LISTPIDS")
    process = Path("/proc") / str(pid)
    first = read(process / "stat", 4096).decode().rsplit(") ", 1)[1].split()
    require(first[0] not in ("Z", "X") and int(first[19]) > 0 and process.stat().st_uid == os.getuid(),
            "PRECHECK_STANDBY_PID")
    cgroup = read(process / "cgroup", 16384)
    require(any(("job_1103606", "step_2480") in zip(parts, parts[1:])
                for parts in (Path(line.split(":", 2)[-1]).parts for line in cgroup.decode().splitlines())),
            "PRECHECK_STANDBY_CGROUP")
    return (pid, int(first[19]), pin(raw), sha256(read(BOOT, 128)).hexdigest(), sha256(cgroup).hexdigest())


def runtime(path):
    manifest = json.loads(read(path / "SCOPED_FILES.json", 8192, RUNTIME_SHA))
    files = manifest["files"]
    require(manifest["request"] == GPU_REQUEST and set(files) == RUNTIME_NAMES, "PRECHECK_RUNTIME_MEMBERS")
    for name, attrs in files.items():
        data = read(path / name, 65536, attrs["SHA256"])
        require(len(data) == attrs["bytes"], "PRECHECK_RUNTIME_MEMBER_SIZE")
        ast.parse(data, filename=name, feature_version=(3, 10))
    return {"members": len(files), "manifest_SHA256": RUNTIME_SHA}


def torch_metadata():
    found = PathFinder.find_spec("torch", sys.path)
    require(found is not None and found.origin and found.submodule_search_locations, "PRECHECK_TORCH_LOCATION")
    path = Path(found.origin).parent
    raw = read(path / "version.py", 16384)
    values = {}
    for node in ast.parse(raw).body:
        assignments = [node.target] if isinstance(node, ast.AnnAssign) else node.targets if isinstance(node, ast.Assign) else []
        for target in assignments:
            if isinstance(target, ast.Name) and target.id in ("__version__", "hip"):
                require(target.id not in values, "PRECHECK_TORCH_VERSION_DUPLICATE")
                values[target.id] = ast.literal_eval(node.value)
    require(values == {"__version__": "2.10.0", "hip": "6.3.26113"}, "PRECHECK_TORCH_VERSION")
    extension = list(path.glob("_C*.so"))
    require(len(extension) == 1 and extension[0].is_file(), "PRECHECK_TORCH_EXTENSION_PRESENT")
    return {"declared_torch": values["__version__"], "declared_hip": values["hip"],
            "package_version_pin": pin(raw), "extension_present": True, "package_imported": False}


def inspect_elf(path, extent, expected):
    require(path.resolve(strict=True) == path, "PRECHECK_ELF_CANONICAL")
    raw = read(path, extent, expected)
    require(len(raw) == extent and raw[:5] == b"\x7fELF\x02", "PRECHECK_ELF_HEADER")
    program = command(["readelf", "-l", str(path)], cap=32768)
    dynamic = command(["readelf", "-d", str(path)], cap=32768)
    interp = re.findall(rb"Requesting program interpreter: (/[^ \]]+)\]", program)
    if path.name == "hipprof":
        require(interp == [b"/lib64/ld-linux-x86-64.so.2"] and Path(os.fsdecode(interp[0])).is_file(),
                "PRECHECK_HIPPROF_INTERPRETER")
    names = re.findall(rb"Shared library: \[([A-Za-z0-9_.+-]+)\]", dynamic)
    require(len(names) == len(set(names)) and names, "PRECHECK_ELF_NEEDED")
    return {"artifact": pin(raw), "program_sha256": pin(program)["SHA256"], "needed": [os.fsdecode(n) for n in names],
            "resolution_executed": False, "binary_executed": False}


def inside():
    step = number()
    require(all(os.environ.get(key) == "" for key in ENV) and "torch" not in sys.modules, "PRECHECK_NO_GPU_MASK")
    return {"request": REQUEST, "step": step, "runtime": runtime(Path("/pair-package")),
            "providers": {p.name: inspect_elf(p, size, sha) for p, size, sha in TOOLS},
            "torch": torch_metadata(), "GPU_context_initializations": 0, "hipprof_invocations": 0}


def container_argv(source):
    shell = ('set -euo pipefail; source /opt/dtk/env.sh; '
             'export PYTHONPATH=/training-runtime/site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
             'CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; '
             'exec python3 -B /precheck.py inside')
    return [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
            "--bind", f"{source.as_posix()}:/precheck.py:ro", "--bind", f"{(STAGE / 'runtime').as_posix()}:/pair-package:ro",
            "--bind", f"{SITE.as_posix()}:/training-runtime:ro", DTK.as_posix(),
            "/bin/bash", "--noprofile", "--norc", "-c", shell]


def srun_argv(source, sha):
    require(source == STAGE / "precheck.py" and re.fullmatch("[0-9a-f]{64}", sha), "PRECHECK_STAGE")
    return ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--overlap", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=1",
            "--threads-per-core=1", "--cpu-bind=threads", "--gres=none", "--mem=256M",
            "--time=00:01:00", "--kill-on-bad-exit=1", "--output=" + (OUT / "step.stdout").as_posix(),
            "--error=" + (OUT / "step.stderr").as_posix(), "/usr/local/python3.12/bin/python3",
            "-I", "-S", "-B", source.as_posix(), "host", "--source-sha256", sha]


def host(source, sha):
    require(source == STAGE / "precheck.py" and isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha), "PRECHECK_STAGE")
    read(source, 32768, sha)
    step = number()
    live_pair(step)
    worker = worker_identity()
    checked = runtime(STAGE / "runtime")
    allowed = {"PATH", "LANG", "LC_ALL", "HOME", "USER", "LOGNAME", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in allowed or k.startswith("SLURM_")}
    got = subprocess.run(container_argv(source), env=env, cwd="/", stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=32, check=False)
    require(got.returncode == 0 and 0 < len(got.stdout) <= 8192, "PRECHECK_CONTAINER_FAILED")
    result = json.loads(got.stdout)
    require(result["request"] == REQUEST and result["step"] == step and result["runtime"] == checked
            and result["GPU_context_initializations"] == result["hipprof_invocations"] == 0, "PRECHECK_INSIDE_RESULT")
    live_pair(step)
    require(worker_identity() == worker, "PRECHECK_STANDBY_CHANGED")
    return {**result, "precheck_passed": True, "node": NODE, "target_source_SHA256": sha,
            "original_standby": STANDBY_STEP, "standby_worker_receipt": worker[2], "same_standby_unchanged": True,
            "Slurm_GPU_resources": 0, "target_GPU_GO": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("host", "inside"))
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    try:
        source = Path(__file__).resolve(strict=True)
        result = host(source, args.source_sha256) if args.action == "host" else inside()
    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as error:
        text = str(error)
        print(json.dumps({"request": REQUEST, "precheck_passed": False, "failure_type": type(error).__name__,
                          "gate": text if re.fullmatch("[A-Z][A-Z0-9_]+", text) else None, "target_GPU_GO": False}))
        return 1
    sys.stdout.buffer.write(encode(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
