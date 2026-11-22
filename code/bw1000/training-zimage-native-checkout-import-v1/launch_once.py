"""Stage the pinned native source and run its one zero-GRES import step."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time


BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-zimage-native-checkout-import-v1"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=10).stdout


def fields(line):
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def extract_source(archive, target, identity):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        members = {}
        for member in source:
            path = PurePosixPath(member.name)
            need(member.isfile() and not path.is_absolute() and path.as_posix() == member.name
                 and not set(path.parts).intersection({"..", "."}) and "\\" not in member.name
                 and member.name not in members and 0 <= member.size <= 16 << 20, "Unsafe archive member")
            members[member.name] = member
            need(len(members) <= identity["files"], "Too many archive files")
        need(len(members) == identity["files"] and sum(m.size for m in members.values()) == identity["source_bytes"],
             "Archive file or byte count differs")
        target.mkdir(mode=0o700)
        digest = hashlib.sha256()
        for name, member in sorted(members.items()):
            raw = source.extractfile(member).read()
            need(len(raw) == member.size, "Archive member size differs")
            for part in (name.encode(), raw):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
            path = target / name
            path.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
            with path.open("xb") as file:
                file.write(raw)
        need(digest.hexdigest() == identity["source_tree_SHA256"], "Archive tree differs")


def main(data):
    config = {name: base64.b64decode(value, validate=True) for name, value in data["files"].items()}
    manifest = json.loads(config["SOURCE_FILES.json"])["files"]
    need(set(config) == set(manifest) | {"SOURCE_FILES.json"}, "Execution source inventory differs")
    for name, pin in manifest.items():
        need(name in {"CLAIM.json", "launch_once.py", "worker.py"} and sha(config[name]) == pin["SHA256"]
             and len(config[name]) == pin["bytes"], "Execution source differs")
    claim = json.loads(config["CLAIM.json"])
    node, parent = claim["node"], claim["parent"]
    need(claim["label"] == LABEL and claim["cpus"] == 4 and claim["mem_MiB"] == 8192 and claim["dcu"] == 0
         and claim["work_seconds"] == 170 and re.fullmatch(r"[a-f0-9]{32}", claim["request"])
         and re.fullmatch(r"[a-f0-9]{40}", data["publication"]), "CPU execution scope differs")
    parent_row = fields(command("scontrol", "show", "job", "-o", parent))
    need(parent_row["JobState"] == "RUNNING" and parent_row["UserId"].endswith(f"({os.getuid()})")
         and node in command("scontrol", "show", "hostnames", parent_row["NodeList"]).splitlines(),
         "Parent allocation owner or target node differs")
    accounting = command("sacct", "-n", "-P", "-j", parent, "-o", "JobIDRaw,JobName%100")
    need(not any(line.split("|")[-1] == LABEL for line in accounting.splitlines()), "Step label already exists")
    for line in command("squeue", "-h", "-s", "-j", parent, "-o", "%i|%N").splitlines():
        step, nodes = line.strip().split("|", 1)
        if step.rsplit(".", 1)[-1] in {"batch", "extern"}:
            continue
        need(re.fullmatch(parent + r"\.\d+", step) and nodes not in {"", "(null)"}
             and node not in command("scontrol", "show", "hostnames", nodes).splitlines(),
             "Target has an active application step")

    archive = base64.b64decode(data["archive"], validate=True)
    identity = claim["original_checkout"]
    need(len(archive) == identity["local_transfer_archive_bytes"]
         and sha(archive) == identity["local_transfer_archive_SHA256"], "Transfer archive identity differs")
    target = BASE / "results" / LABEL
    target.mkdir(mode=0o700)
    for name, raw in config.items():
        with (target / name).open("xb") as file:
            file.write(raw)
    extract_source(archive, target / "source", identity)
    intent = {"request": claim["request"], "publication": data["publication"], "start_epoch": time.time(),
              "manifest_SHA256": sha(config["SOURCE_FILES.json"]), "source_tree_SHA256": identity["source_tree_SHA256"]}
    with (target / "intent.json").open("x") as file:
        json.dump(intent, file)
        file.flush()
        os.fsync(file.fileno())
    invocation = ["srun", "--jobid=" + parent, "--job-name=" + LABEL, "--exact", "--nodes=1",
                  "--ntasks=1", "--nodelist=" + node, "--immediate=5", "--cpus-per-task=4",
                  "--mem=8192M", "--gres=none", "--time=00:03:00",
                  "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    mounts = [(target, "/import-run"), (target / "source", "/native-source"),
              (BASE / "results/training-zimage-target-native-meta-CPU-v2/dependencies/site", "/zimage-runtime"),
              (BASE / "results/training-t06-sdxl-runtime-v1", "/training-runtime")]
    for source, destination in mounts:
        need(source.is_dir(), "Frozen mount unavailable")
        invocation.extend(("--bind", f"{source}:{destination}:ro"))
    shell = ("set -euo pipefail; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
             "CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
             "PYTHONPATH=/native-source:/zimage-runtime:/training-runtime/site "
             "ZIMAGE_IMPORT_REQUEST=" + claim["request"] + "; exec python3 -B -u /import-run/worker.py")
    invocation.extend((str(BASE / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell))
    with (target / "stdout.private").open("xb") as out, (target / "stderr.private").open("xb") as err:
        result = subprocess.run(invocation, stdout=out, stderr=err, timeout=claim["work_seconds"])
    print(json.dumps(dict(request=claim["request"], srun_rc=result.returncode), sort_keys=True), flush=True)


if __name__ == "__main__":
    main(DATA)
