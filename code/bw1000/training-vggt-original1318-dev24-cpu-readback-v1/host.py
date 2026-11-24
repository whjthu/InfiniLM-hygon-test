"""Install the source-only closure and run one separately admitted zero-GRES readback."""

import argparse
from base64 import b64decode
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import worker


ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PROJECT = Path(str(ROOT))
CPU_STAGE = "training-vggt-original1318-dev24-cpu-readback-"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
SANDBOX = ROOT / "containers/dtk26-sandbox-r1"
T06 = ROOT / "results/training-t06-sdxl-runtime-v1"
SITE_PRECEDENT = "06c10ce305624a7bf8fada9dac4cf5ecae237ed95ce47e077c3483a2db9ebfb1"
SOURCE_MANIFEST = "219283e9f67337f60d9f99bdbdc82caa7515f1aff573c590c9356f63061ee812"
SOURCE_PACKET = "375840bc8ffff40481140b44b32ac089da761ca7c163941da0838f0eed6e7b81"
SHA = re.compile(r"[0-9a-f]{64}")
NODE = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
REQUEST = re.compile(r"[0-9a-f]{32}")
PARENTS = ("1103606", "1194900")
BOOT_NAMES = ("worker.py", "host.py", "verify_terminal.py")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
MAX = 300000


need = worker.need
encode = worker.encode
pin = worker.pin


def scope_name(request):
    need(REQUEST.fullmatch(request), "FRESH_REQUEST_ID")
    return CPU_STAGE + request


def read(path, maximum, expected=None):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    need(path.resolve(strict=True) == path.absolute() and 0 < len(raw) <= maximum
         and (expected is None or pin(raw)["SHA256"] == expected), "EXACT_PRIVATE_ARTIFACT")
    return raw


def save_file(path, raw):
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def decode(raw, digest, boot):
    need(SHA.fullmatch(digest) and 0 < len(raw) <= MAX and pin(raw)["SHA256"] == digest, "EXACT_SOURCE_ONLY_OFFER")
    offer = json.loads(raw)
    need(set(offer) == {"old_source_packet", "cpu_sources"}, "EXACT_CLOSED_SOURCE_ONLY_OFFER")
    old = b64decode(offer["old_source_packet"], validate=True)
    need(pin(old)["SHA256"] == SOURCE_PACKET, "EXACT_REUSE_OLD_SOURCE_PACKET")
    content = json.loads(old)
    manifest_raw = b64decode(content["manifest"], validate=True)
    need(pin(manifest_raw)["SHA256"] == SOURCE_MANIFEST, "EXACT_REUSE_OLD_SOURCE_MANIFEST")
    manifest = json.loads(manifest_raw)
    need(len(manifest) == 18 and set(content["members"]) == set(manifest)
         and set(offer["cpu_sources"]) == set(boot) == set(BOOT_NAMES), "EXACT_18_PLUS_3_CLOSURE")
    files = {}
    for name, item in manifest.items():
        path = PurePosixPath(name)
        need(not path.is_absolute() and ".." not in path.parts and str(path) == name
             and path.suffix in (".py", ".json"), "SOURCE_MEMBER_SCOPE")
        data = b64decode(content["members"][name], validate=True)
        need(pin(data) == item, "FROZEN_SOURCE_MEMBER")
        files["checkout/" + name] = data
    for name in BOOT_NAMES:
        data = b64decode(offer["cpu_sources"][name], validate=True)
        need(0 < len(data) <= 65536 and pin(data) == boot[name], "CPU_IMPLEMENTATION_PIN")
        compile(data, name, "exec")
        files["checkout/" + name] = data
    return files


def boot_identities():
    path = Path(__file__).resolve().parent
    return {name: pin(read(path / name, 65536)) for name in BOOT_NAMES}


def install(parent, request, raw, offer_sha, boot):
    members = decode(raw, offer_sha, boot)
    name = scope_name(request)
    final, lock = parent / name, parent / ("." + name + ".lock")
    need(parent.resolve(strict=True) == parent and not final.exists() and not final.is_symlink(), "SCOPED_INSTALL_ABSENT")
    lock.mkdir(mode=0o700)
    stage = None
    try:
        need(not final.exists() and not final.is_symlink(), "SCOPED_INSTALL_ABSENT")
        stage = Path(tempfile.mkdtemp(prefix="." + name + ".pending-", dir=parent)).resolve()
        need(stage.parent == parent and stage.stat().st_dev == parent.stat().st_dev, "SAME_FILESYSTEM_NEW_STAGE")
        pins = {name: pin(data) for name, data in members.items()}
        for name, data in members.items():
            file = stage / name
            file.parent.mkdir(exist_ok=True, parents=True)
            save_file(file, data)
            need(read(file, len(data), pins[name]["SHA256"]) == data, "INDEPENDENT_SOURCE_READBACK")
        record = {"schema": "vggt.1318.cpu-readonly.install.v1", "request": request, "offer_SHA256": offer_sha,
                  "original_source_manifest_SHA256": SOURCE_MANIFEST, "members": pins}
        save_file(stage / "INSTALL.json", encode(record))
        os.rename(stage, final)
        stage = None
    finally:
        if stage is not None:
            need(stage.parent == parent and stage.name.startswith("." + name + ".pending-"), "SCOPED_PENDING")
            shutil.rmtree(stage)
        lock.rmdir()
    return pin(encode(record))


def installed(stage, request, install_sha, boot):
    need(stage.name == scope_name(request), "EXACT_INSTALLED_REQUEST")
    record = json.loads(read(stage / "INSTALL.json", 16384, install_sha))
    wanted = record["members"]
    need(record["request"] == request and record["original_source_manifest_SHA256"] == SOURCE_MANIFEST
         and len(wanted) == 21 and {name: wanted["checkout/" + name] for name in BOOT_NAMES} == boot,
         "PINNED_INSTALLED_SOURCE_CLOSURE")
    for name, item in wanted.items():
        path = PurePosixPath(name)
        need(path.parts[0] == "checkout" and ".." not in path.parts and str(path) == name, "SOURCE_MEMBER_SCOPE")
        bytes_ = read(stage / name, item["bytes"], item["SHA256"])
        need(pin(bytes_) == item, "EXACT_INSTALLED_READBACK")
    return record


def render(parent, node, request, epoch, stage):
    need(parent in PARENTS and NODE.fullmatch(node) and stage == ROOT / "incoming" / scope_name(request),
         "EXACT_SCOPED_CPU_GEOMETRY")
    app = [APPTAINER, "exec", "--contain", "--no-home", "--pwd", "/"]
    for source, target, mode in ((stage / "checkout", "/checkout", "ro"),
                                 (stage / "evidence", "/evidence", "rw"),
                                 (ROOT / "results", str(ROOT / "results"), "ro"),
                                 (T06, "/training-runtime", "ro")):
        app.extend(["--bind", f"{source}:{target}:{mode}"])
    worker_argv = ["python3", "-B", "-u", "/checkout/worker.py", request, parent, node, repr(epoch)]
    deadline = epoch + worker.LIMIT
    compute_left = shlex.quote(f"import time;print(int({deadline!r}-time.time()))")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 "
             "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; "
             f'LEFT=$(python3 -B -c {compute_left}); test "$LEFT" -gt 10; '
             'exec /usr/bin/timeout --kill-after=5s "$LEFT" ' + shlex.join(worker_argv))
    app.extend([str(SANDBOX), "/bin/bash", "--noprofile", "--norc", "-c", shell])
    return ["srun", "--jobid=" + parent, "--job-name=" + CPU_STAGE + request, "--exact", "--nodes=1", "--ntasks=1",
            "--nodelist=" + node, "--cpus-per-task=2", "--cpu-bind=threads", "--mem=2048M", "--gres=none",
            "--immediate=5", "--time=2", "--kill-on-bad-exit=1", *app]


def query(args):
    env = dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S")
    raw = subprocess.check_output(args, timeout=6, env=env)
    need(len(raw) <= 196608, "CONTROL_QUERY_EXTENT")
    return raw.decode("ascii")


def live(parent, node, request):
    descriptor = dict(item.split("=", 1) for item in query(["scontrol", "show", "job", parent, "--oneliner"]).split()
                      if "=" in item)
    need(descriptor["JobId"] == parent and descriptor["JobState"] == "RUNNING"
         and descriptor["UserId"].endswith(f"({os.getuid()})"), "CURRENT_PARENT_OWNERSHIP")
    hosts = query(["scontrol", "show", "hostnames", descriptor["NodeList"]]).splitlines()
    need(node in hosts, "EXACT_CURRENT_NODE_MEMBERSHIP")
    current = query(["squeue", "--steps", "-h", "-j", parent, "-o", "%i|%N"]).splitlines()
    for line in current:
        step, nodelist = line.split("|", 1)
        if re.fullmatch(parent + r"\.[0-9]+", step) and nodelist not in ("", "(null)"):
            need(node not in query(["scontrol", "show", "hostnames", nodelist]).splitlines(), "NO_LIVE_NUMBERED_STEP_ON_CANDIDATE")
    prior = query(["sacct", "-nP", "-j", parent, "--format=JobName%100"]).splitlines()
    need(CPU_STAGE + request not in prior, "ORIGINAL_NUMERIC_CPU_REQUEST_ALREADY_PRESENT")


def submit(argv, output, deadline):
    with (output / "srun.private.log").open("xb") as stream:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if proc.poll() is not None:
                    break
                try:
                    os.killpg(proc.pid, sig)
                    proc.wait(timeout=2)
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise


def once(stage, request, placement, admission_sha, install_sha, *, dispatch, verify_live, clock=time.time):
    parent, node, epoch = placement["parent"], placement["node"], placement["epoch"]
    need(placement["request"] == request and placement["deadline"] == epoch + worker.LIMIT
         and placement["CPU_only_admitted"] is True and placement["resources"] == {"nodes": 1, "CPU": 2, "memory_MiB": 2048, "GPU": 0}
         and 0 <= clock() - placement["node_observed_epoch"] <= 35 and epoch <= clock() < epoch + 25,
         "FRESH_EXACT_TWO_CPU_ADMISSION")
    argv = render(parent, node, request, epoch, ROOT / "incoming" / scope_name(request))
    output = stage / "evidence"
    need(not output.exists() and not output.is_symlink(), "SINGLE_CPU_SUBMISSION_ONLY")
    verify_live(parent, node, request)
    need(clock() < epoch + 25, "CPU_PLACEMENT_WINDOW_EXPIRED")
    output.mkdir(mode=0o700)
    intent = {"request": request, "parent": parent, "node": node, "label": CPU_STAGE + request, "epoch": epoch,
              "deadline": epoch + worker.LIMIT, "submission_epoch": clock(), "install_SHA256": install_sha,
              "admission_SHA256": admission_sha, "argv_SHA256": pin(encode(argv))["SHA256"], "replay_allowed": False}
    intent_pin = worker.save(output / "INTENT.json", intent)
    result = {"state": "INDEPENDENT_SLURM_TERMINAL_REQUIRED", "intent": intent_pin, "request": request, **worker.DENOM}
    try:
        result["srun_rc"] = dispatch(argv, output, intent["deadline"] + 5)
    except (OSError, subprocess.SubprocessError) as error:
        result.update(srun_rc=None, transport=type(error).__name__)
    worker.save(output / "OWNER_OUTCOME.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("install", "readback-install", "run"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--offer-sha")
    parser.add_argument("--install-sha")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha")
    args = parser.parse_args()
    need(sys.platform == "linux" and sys.flags.dont_write_bytecode, "REAL_SHARED_HOST_ONLY")
    os.umask(0o077)
    stage = PROJECT / "incoming" / scope_name(args.request)
    boot = boot_identities()
    if args.mode == "install":
        value = install(stage.parent, args.request, sys.stdin.buffer.read(MAX + 1), args.offer_sha, boot)
    else:
        need(SHA.fullmatch(args.install_sha or ""), "EXACT_INSTALL_RECEIPT_SHA")
        record = installed(stage, args.request, args.install_sha, boot)
        if args.mode == "readback-install":
            value = {"request": args.request, "source_files": len(record["members"]), "install_SHA256": args.install_sha}
        else:
            need(args.admission is not None and args.admission.is_absolute() and SHA.fullmatch(args.admission_sha or "")
                 and args.admission.parent == stage, "PRIVATE_REQUEST_ADMISSION_RECEIPT")
            admission = json.loads(read(args.admission, 16384, args.admission_sha))
            value = once(stage, args.request, admission, args.admission_sha, args.install_sha, dispatch=submit, verify_live=live)
    print(json.dumps(value, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
