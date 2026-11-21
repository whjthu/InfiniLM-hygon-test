"""Compile the frozen pack32 and pack64 variants in one admitted zero-GRES step."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import socket
import subprocess
import sys
import time


@dataclass(frozen=True)
class Variant:
    width: int
    manifest: str

    @property
    def label(self):
        return f"Deep-inverse-output-rope-prefix-pack{self.width}-gfx936-CPU-v1"

    @property
    def export(self):
        return f"deep_inverse_rope_h1_pack{self.width}_v1"


VARIANTS = (
    Variant(32, "dfb1012575cb5c9615a9dbbc471734f527b7a4adb6a98235e99220060b90538d"),
    Variant(64, "bbea9ca9524d1ac20ea666eabd302847c608980d5c45ea7d09d2e50ac5bd23f7"),
)
PAIR = "Deep-inverse-output-rope-prefix-width-pair-gfx936-CPU-v1"
PACKAGES = Path("/refactor/incoming")
OUTPUTS = Path("/refactor/candidates")
PROJECT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
SOURCE = PurePosixPath("code/bw1000/Deep-C4-causal-consumer-r483")
SOURCE_ENTRY = PurePosixPath("tools/run_deep_rope_prefix_width_compile_pair.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def host_step_command(repo, parent, node, epoch):
    """Build argv after the coordinator has bound its fresh, reserved placement."""
    repo = PurePosixPath(repo)
    require(repo.is_relative_to(PROJECT) and ".." not in repo.parts
            and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in repo.parts[1:])
            and re.fullmatch(r"[0-9]+", parent) and re.fullmatch(r"[a-zA-Z0-9-]+", node)
            and math.isfinite(epoch) and epoch > 0, "pair command placement or epoch differs")
    entry = PurePosixPath("/refactor/incoming") / SOURCE_ENTRY.name
    producer = "Deep-C4-constant-slots-r482-Native-CPU-v1"
    binds = [(PROJECT, "/refactor", "ro"), (PROJECT / "candidates", "/refactor/candidates", "rw"),
             (PROJECT / "candidates" / producer, "/refactor/candidates/" + producer, "ro"),
             (repo / SOURCE_ENTRY, str(entry), "ro"), ("/opt/hyhal", "/opt/hyhal", "ro")]
    binds.extend((repo / SOURCE / f"inverse-output-rope-prefix-pack{arm.width}-target-v1",
                  f"/refactor/incoming/{arm.label}", "ro") for arm in VARIANTS)
    masked = ["CUDA_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "ROCR_VISIBLE_DEVICES=-1", "LC_ALL=C"]
    inner = ["env", *masked, f"TUNING_CPU_PARENT={parent}", f"TUNING_CPU_NODE={node}",
             f"TUNING_CPU_EPOCH={epoch}", f"TUNING_CPU_DEADLINE={epoch+300}",
             "/usr/bin/python3", "-I", "-S", "-B", "-u", str(entry)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH; "
             "exec " + shlex.join(inner))
    container = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    for source, path, mode in binds:
        container.extend(["--bind", f"{source}:{path}:{mode}"])
    return ["srun", f"--jobid={parent}", "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=4", "--mem=8192M", "--gres=none", f"--nodelist={node}", "--time=00:05:00",
            f"--job-name={PAIR}", *container, str(PROJECT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", shell]


def read(path, cap, expected=None):
    require(path.resolve(strict=True) == path and path.is_file(), "expected original canonical regular file")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    require(0 < len(raw) <= cap and (expected is None or sha256(raw).hexdigest() == expected),
            "original byte limit or SHA differs: " + path.name)
    return raw


def package(packages, arm):
    directory = packages / arm.label
    manifest = json.loads(read(directory / "SOURCE_FILES.json", 8192, arm.manifest))["files_SHA256"]
    require(manifest.keys() == {"PLAN.json", "compile_once.py", "inverse_output_rope.cu"},
            "variant package inventory differs")
    for path, wanted in manifest.items():
        read(directory / path, 128 * 1024, wanted)
    return directory / "compile_once.py"


def save_once(path, report):
    with path.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_result(output, arm, step, node, epoch):
    raw = read(output / "CPU-result.json", 1024 * 1024)
    receipt = json.loads(raw)
    obj = output / "inverse_output_rope.o"
    expected = dict(label=arm.label, original_step=step, node=node, epoch=epoch, deadline=epoch+300,
                    manifest_SHA256=arm.manifest, GRES="none", planned_device_TUs=1, actual_device_TUs=1,
                    actual_GPU_steps=0, explicit_Runtime_calls=0, completed=True, passed=True,
                    device_qualified=False, exports=[arm.export], object=str(obj))
    require(all(type(receipt.get(key)) is type(value) and receipt[key] == value for key, value in expected.items()),
            "variant receipt identity or state differs")
    read(obj, 16 * 1024**2, receipt["object_SHA256"])
    read(output / "inverse_output_rope.before.d", 2 * 1024**2, receipt["fresh_M_SHA256"])
    read(output / "inverse_output_rope.d", 2 * 1024**2, receipt["actual_MD_SHA256"])
    return dict(object_SHA256=receipt["object_SHA256"], worker_receipt_SHA256=sha256(raw).hexdigest(),
                fresh_M_SHA256=receipt["fresh_M_SHA256"], actual_MD_SHA256=receipt["actual_MD_SHA256"])


def invoke_worker(worker, env, log, seconds):
    with log.open("xb") as handle:
        child = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-u", str(worker)], env=env,
                                 stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            return child.wait(seconds)
        except subprocess.TimeoutExpired:
            # The leader has not been reaped and owns this new process group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(5)
            raise


def run_pair(packages, outputs, step, node, epoch, left, *, invoke=invoke_worker):
    entries = {arm: package(packages, arm) for arm in VARIANTS}
    scope = outputs / PAIR
    require(not any(path.exists() or path.is_symlink()
                    for path in (scope, *(outputs / arm.label for arm in VARIANTS))),
            "a variant or pair scope already exists; reconcile the original files")
    scope.mkdir(mode=0o700)
    save_once(scope / "intent.json", dict(label=PAIR, original_step=step, node=node, epoch=epoch,
              deadline=epoch+300, manifests={arm.label: arm.manifest for arm in VARIANTS}, planned_device_TUs=2))
    rows = []
    env = dict(os.environ)
    for arm, worker in entries.items():
        remaining = left() - 12
        row = dict(width=arm.width, manifest_SHA256=arm.manifest, invoked=False, verified=False)
        rows.append(row)
        try:
            require(remaining > 10, "shared compile budget exhausted before this arm")
            row["invoked"] = True
            row["returncode"] = invoke(worker, env | {"TUNING_CPU_MANIFEST_SHA": arm.manifest},
                                       scope / f"pack{arm.width}.log", remaining)
            if row["returncode"] == 0:
                row.update(verify_result(outputs / arm.label, arm, step, node, epoch), verified=True)
        except (OSError, TypeError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            row["error"] = str(error)[:180]
        finally:
            save_once(scope / f"pack{arm.width}-result.json", row)
    qualified = sum(row["verified"] for row in rows)
    result = dict(label=PAIR, original_step=step, node=node, epoch=epoch, deadline=epoch+300, planned_device_TUs=2,
                  verified_device_TUs=qualified, actual_GPU_steps=0, variant_results=rows,
                  completed=True, passed=qualified == len(VARIANTS), device_qualified=False)
    save_once(scope / "CPU-pair-result.json", result)
    return result


def main():
    env = os.environ
    parent, step, node = env["SLURM_JOB_ID"], env["SLURM_STEP_ID"], socket.gethostname().split(".")[0]
    require(parent == env["TUNING_CPU_PARENT"] and parent.isdecimal() and step.isdecimal()
            and node == env["TUNING_CPU_NODE"] and re.fullmatch(r"[A-Za-z0-9-]+", node), "Slurm step identity differs")
    require(env["SLURM_CPUS_PER_TASK"] == "4" and not env.get("SLURM_STEP_GPUS")
            and all(env.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and env.get("ROCM_PATH") == "/opt/dtk", "expected admitted four-CPU hidden-device step")
    epoch = float(env["TUNING_CPU_EPOCH"])
    wall, mono = time.time(), time.monotonic()
    require(math.isfinite(epoch) and epoch <= wall < epoch + 278
            and float(env["TUNING_CPU_DEADLINE"]) == epoch + 300, "shared original compile epoch differs")

    def left():
        return min(epoch+300-time.time(), epoch+300-wall-(time.monotonic()-mono))

    output = run_pair(PACKAGES, OUTPUTS, parent + "." + step, node, epoch, left)
    print(json.dumps(output, sort_keys=True, allow_nan=False), flush=True)
    return int(not output["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
