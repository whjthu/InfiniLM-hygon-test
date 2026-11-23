"""Submit one CPU-only validation of the Slurm task attestation entry."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time


PARENT, NODE, STANDBY = "1103606", "f12r2n11", "1103606.2075"
OLD_REQUEST, OLD_LABEL = "af580d6d58004e5ebfe582d5d53d55ed", "O01-zimage-hybrid-pid-none-af580d6d5800"
OLD_RETURN_SHA256 = "4b90536aa065713226417749c4e9a99a6f9b2938af05972c557fc6a132927a85"
REQUEST, LABEL = "0eac6c1ec15f47a08298fa0272d0e540", "O01-zimage-hybrid-pid-none-0eac6c1ec15f"
BASE = Path(__file__).resolve().parent.with_name("zimage-lt-eight-card-attested-entry-v1") / "node_entry.py"
BASE_SHA256 = "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6"
R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PYTHON = "/usr/local/python3.12/bin/python3"


def require(value, message):
    if not value:
        raise ValueError(message)


def render():
    raw = BASE.read_bytes()
    require(sha256(raw).hexdigest() == BASE_SHA256, "adopted attestation source differs")
    for before, after in ((OLD_REQUEST, REQUEST), (OLD_LABEL, LABEL)):
        require(raw.count(before.encode()) == 1, "probe identity anchor differs")
        raw = raw.replace(before.encode(), after.encode())
    compile(raw, "node_entry.py", "exec")
    return raw


def query(*cmd):
    return subprocess.check_output(cmd, text=True, timeout=8)


def one_row(data, key, identity):
    result = []
    for line in data.splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        if fields.get(key) == identity:
            result.append(fields)
    require(len(result) == 1, "exact Slurm identity missing")
    return result[0]


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def srun_args(path):
    return ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=12",
            "--threads-per-core=1", "--cpu-bind=threads", "--gres=none", "--mem=1024M", "--time=00:01:00",
            "--output=" + str(path / "slurm.stdout"), "--error=" + str(path / "slurm.stderr"),
            PYTHON, "-I", "-S", "-B", str(path / "node_entry.py"), "probe", "--run", str(path)]


def run(source, source_sha256):
    raw = source.read_bytes()
    require(len(raw) < 64 << 10 and sha256(raw).hexdigest() == source_sha256, "node source hash differs")
    old = R / OLD_LABEL
    old_raw = (old / "return.json").read_bytes()
    old_result = json.loads(old_raw)
    require(sha256(old_raw).hexdigest() == OLD_RETURN_SHA256
            and old_result["request"] == OLD_REQUEST and old_result["srun_rc"] == 255
            and not (old / "zero-GRES-owner.json").exists(), "prior request has an unresolved target step")
    records = query("sacct", "-nP", "-j", PARENT, "--format", "JobIDRaw,JobName%100")
    require(not any(line.rsplit("|", 1)[-1] in (OLD_LABEL, LABEL) for line in records.splitlines()),
            "exact prior or current accounting label already exists")
    parent = one_row(query("scontrol", "show", "job", PARENT, "--oneliner"), "JobId", PARENT)
    require(parent["JobState"] == "RUNNING" and parent["UserId"].endswith(f"({os.getuid()})"),
            "original parent ownership differs")
    current = one_row(query("scontrol", "show", "step", PARENT, "--oneliner"), "StepId", STANDBY)
    require(current["NodeList"] == NODE and current["State"] == "RUNNING"
            and "gres/dcu=8" in current["TRES"].split(","), "original standby moved")
    path = R / LABEL
    path.mkdir(mode=0o700)
    with (path / "node_entry.py").open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    save(path / "intent.json", {"request": REQUEST, "node_source_SHA256": source_sha256, "epoch": time.time()})
    child = subprocess.run(srun_args(path), capture_output=True, timeout=75)
    result = {"request": REQUEST, "srun_rc": child.returncode, "returned_epoch": time.time(),
              "launcher_stderr_SHA256": sha256(child.stderr).hexdigest()}
    save(path / "return.json", result)
    receipt = path / "zero-GRES-owner.json"
    if receipt.is_file():
        raw_receipt = receipt.read_bytes()
        result.update(receipt_SHA256=sha256(raw_receipt).hexdigest(), receipt=json.loads(raw_receipt))
    else:
        stderr = path / "slurm.stderr"
        body = stderr.read_bytes() if stderr.is_file() and stderr.stat().st_size < 8192 else child.stderr
        result["error_types"] = [category for category in ("ValueError", "CalledProcessError", "TimeoutExpired",
                                 "unrecognized option", "unavailable") if category in body.decode(errors="replace")]
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("render").add_argument("output", type=Path)
    deploy = commands.add_parser("run")
    deploy.add_argument("source", type=Path)
    deploy.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "render":
        with args.output.open("xb") as handle:
            handle.write(render())
    else:
        run(args.source, args.source_sha256)


if __name__ == "__main__":
    main()
