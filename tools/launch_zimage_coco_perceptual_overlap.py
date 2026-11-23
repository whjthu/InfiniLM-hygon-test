"""Submit one CPU-only screening step under the existing Slurm allocation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time


REF = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PARENT = "1103606"
NODE = "f13r2n08"
LABEL = "training-zimage-coco-perceptual-958b18734f32"
REQUEST = "958b18734f324e55b404355d1a884282"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def pin(path, digest):
    require(sha256(path.read_bytes()).hexdigest() == digest, f"target pin differs: {path.name}")


def lines(*command):
    return subprocess.check_output(command, timeout=7, text=True).splitlines()


def start(claim_bytes, source, publication):
    claim = json.loads(claim_bytes)
    require(re.fullmatch(r"[a-f0-9]{40}", publication) and claim["request"] == REQUEST
            and claim["label"] == LABEL and claim["maximum_target_invocations"] == 1, "original request differs")
    require(claim["resources"] == {"parent": PARENT, "node": NODE, "CPU": 2, "memory_MiB": 4096, "DCU": 0,
                                   "slurm_seconds": 420, "worker_seconds": 390}, "resource claim differs")
    require(sha256(source).hexdigest() == claim["source"]["SHA256"], "worker source differs")
    inputs = claim["inputs"]
    archive = REF / inputs["archive_directory"]
    manifest = REF / inputs["training_manifest"]
    root = REF / claim["result_directory"]
    require(root == REF / ("results/training-zimage-t07-coco-perceptual-screen-" + REQUEST),
            "result namespace differs")

    [allocation] = lines("scontrol", "show", "job", "-o", PARENT)
    fields = dict(item.split("=", 1) for item in allocation.split() if "=" in item)
    require(fields["JobId"] == PARENT and fields["JobState"] == "RUNNING"
            and fields["UserId"].endswith(f"({os.getuid()})")
            and NODE in lines("scontrol", "show", "hostnames", fields["NodeList"]), "parent allocation differs")
    require(not root.exists() and not any(row.split("|")[1:2] == [LABEL] for row in lines(
        "sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS))
        and not any("Name=" + LABEL in row.split() for row in lines(
            "scontrol", "show", "step", PARENT, "--oneliner")), "original label or result already exists")
    pin(archive / "receipt.json", inputs["terminal_receipt_SHA256"])
    pin(archive / "private-media.json", inputs["media_manifest_SHA256"])
    pin(manifest, inputs["training_manifest_SHA256"])

    image = REF / "containers/dtk26-sandbox-r1"
    deps = REF / "results/training-zimage-target-native-meta-CPU-v2/dependencies/site"
    apptainer = [APPTAINER, "exec", "--cleanenv", "--no-home", "--pwd", "/"]
    runtime = ["env", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1",
               "ROCR_VISIBLE_DEVICES=", "HIP_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES="]
    if (deps / "PIL" / "__init__.py").is_file():
        apptainer += ["--bind", f"{deps}:/audit-deps:ro"]
        runtime += ["PYTHONPATH=/audit-deps"]
    sanity = 'from PIL import Image, features; assert hasattr(Image,"Resampling") and features.check_codec("jpg")'
    subprocess.run(apptainer + [str(image), *runtime, "python3", "-B", "-c", sanity],
                   check=True, timeout=12, capture_output=True)

    os.umask(0o077)
    root.mkdir(mode=0o700)
    save(root / "worker.py", source)
    save(root / "claim.json", claim_bytes)
    epoch = time.time()
    intent = dict(request=REQUEST, label=LABEL, parent=PARENT, node=NODE, epoch=epoch,
                  deadline_epoch=epoch + 420, publication=publication, DCU=0, replay=False)
    save(root / "intent.json", (json.dumps(intent, sort_keys=True) + "\n").encode())
    bounded_worker = ('deadline="$1"; shift; seconds=$((deadline - $(date +%s) - 10)); '
                      '[ "$seconds" -gt 0 ] || exit 124; '
                      '[ "$seconds" -le 390 ] || seconds=390; '
                      'exec timeout --kill-after=5s "${seconds}s" "$@"')
    execution = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=6",
                 "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=4096M", "--gres=none",
                 "--nodelist=" + NODE, "--time=00:07:00", "--kill-on-bad-exit=1",
                 *apptainer, "--bind", f"{root}:/screen:rw", "--bind", f"{archive}:/coco:ro",
                 "--bind", f"{manifest.parent}:/training:ro", str(image), *runtime,
                 "/bin/sh", "-c", bounded_worker, "zimage-perceptual", str(int(intent["deadline_epoch"])),
                 "python3", "-B", "-u", "/screen/worker.py",
                 "--archive-directory", "/coco", "--original-training-manifest", "/training/" + manifest.name,
                 "--result-directory", "/screen"]
    with (root / "worker.stdout.private").open("xb") as out, (root / "worker.stderr.private").open("xb") as err:
        child = subprocess.Popen(execution, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    submission = dict(intent, launcher_pid=child.pid)
    save(root / "submit.json", (json.dumps(submission, sort_keys=True) + "\n").encode())
    print(json.dumps(submission, sort_keys=True), flush=True)
