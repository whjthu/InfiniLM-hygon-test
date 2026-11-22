"""Run the frozen Z-Image import with only the original two dependency versions overlaid."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import zipfile

from launch_once import BASE, command, fields, need


LABEL = "training-zimage-native-checkout-import-v2"
V1 = BASE / "results" / "training-zimage-native-checkout-import-v1"
V2 = BASE / "results" / LABEL


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def install_wheel(raw, destination, pin):
    need(len(raw) == pin["bytes"] and digest(raw) == pin["SHA256"], "Public wheel differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        seen = set()
        extracted = 0
        for member in wheel.infolist():
            path = PurePosixPath(member.filename)
            need(not member.is_dir() and path.parts
                 and path.parts[0] in pin["roots"] and path.as_posix() == member.filename
                 and not path.is_absolute() and not set(path.parts).intersection({"..", "."})
                 and "\\" not in member.filename and member.filename not in seen,
                 "Wheel member outside the frozen distribution")
            seen.add(member.filename)
            extracted += member.file_size
            need(len(seen) <= 20000 and extracted <= 160 << 20, "Wheel exceeds extraction budget")
            output = destination / member.filename
            output.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
            with output.open("xb") as file:
                file.write(wheel.read(member))
        return {"files":len(seen),"bytes":extracted}


def main():
    import os

    root = Path(__file__).resolve().parent
    raw_claim = (root / "CLAIM.json").read_bytes()
    claim = json.loads(raw_claim)
    parent, node = claim["parent"], claim["node"]
    need(root == V2 and claim["label"] == LABEL and claim["dcu"] == 0
         and re.fullmatch(r"[a-f0-9]{32}", claim["request"]), "Unexpected execution scope")
    need(digest((V1 / "worker.py").read_bytes()) == claim["original_worker_SHA256"]
         and digest((V1 / "launch_once.py").read_bytes()) == claim["original_launcher_SHA256"],
         "Original verified runtime changed")
    current = fields(command("scontrol", "show", "job", "-o", parent))
    need(current["JobState"] == "RUNNING" and current["UserId"].endswith(f"({os.getuid()})")
         and node in command("scontrol", "show", "hostnames", current["NodeList"]).splitlines(), "Allocation differs")
    accounting = command("sacct", "-n", "-P", "-j", parent, "-o", "JobIDRaw,JobName%100,State,ExitCode")
    exact_v1 = [line.split("|") for line in accounting.splitlines() if line.split("|")[0] == "1087656.1125"]
    need(exact_v1 == [["1087656.1125", V1.name, "FAILED", "1:0"]]
         and all(line.split("|")[1] != LABEL for line in accounting.splitlines()), "Missing original or duplicate v2")
    for line in command("squeue", "-h", "-s", "-j", parent, "-o", "%i|%N").splitlines():
        step, hosts = line.split("|",1)
        if step.rsplit(".",1)[-1] not in {"batch", "extern"}:
            need(hosts and node not in command("scontrol", "show", "hostnames", hosts).splitlines(),
                 "Target node has an active application step")
    site = root / "site"
    site.mkdir(mode=0o700)
    metrics = {}
    for name, pin in claim["wheel_files"].items():
        metrics[name] = install_wheel((root / name).read_bytes(), site, pin)
    dependency = site / "transformers/models/siglip/modeling_siglip.py"
    need(digest(dependency.read_bytes()) == claim["siglip_provider_SHA256"], "Frozen SigLIP provider differs")
    with (root / "intent.json").open("x") as file:
        json.dump({"request":claim["request"],"claim_SHA256":digest(raw_claim),"epoch":time.time(),
                   "overlay":metrics,"publication":sys.argv[1]},file)
        file.flush()
        os.fsync(file.fileno())
    invocation = ["srun","--jobid="+parent,"--job-name="+LABEL,"--exact","--nodes=1","--ntasks=1",
                  "--nodelist="+node,"--immediate=5","--cpus-per-task=4","--mem=8192M","--gres=none","--time=00:03:00",
                  "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/"]
    mounts = [(root,"/import-v2"),(V1,"/import-run"),(V1/"source","/native-source"),(site,"/zimage-compat"),
              (BASE/"results/training-zimage-target-native-meta-CPU-v2/dependencies/site","/zimage-runtime"),
              (BASE/"results/training-t06-sdxl-runtime-v1","/training-runtime")]
    for source, destination in mounts:
        need(source.is_dir(), "Frozen dependency mount unavailable")
        invocation.extend(("--bind",f"{source}:{destination}:ro"))
    script = ("set -euo pipefail; source /opt/dtk/env.sh; "
              "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
              "CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= "
              "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
              "PYTHONPATH=/native-source:/zimage-compat:/zimage-runtime:/training-runtime/site "
              "ZIMAGE_IMPORT_REQUEST="+claim["request"]+"; exec python3 -B -u /import-v2/verify_overlay.py")
    invocation.extend((str(BASE/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",script))
    with (root/"stdout.private").open("xb") as out, (root/"stderr.private").open("xb") as err:
        result = subprocess.run(invocation,stdout=out,stderr=err,timeout=170)
    print(json.dumps({"request":claim["request"],"srun_rc":result.returncode}),flush=True)


if __name__ == "__main__":
    main()
