"""Run two bounded process-local HIP order checks on one Slurm card."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
ORIGINAL = ROOT / "incoming/training-internvl-native64-first2-hip-v3-reviewed"
LABEL = "training-internvl-hip-tmp-bind-v1"
REQUEST = "514cf54ca81e48ef8b86933ba169a91e"
PARENT, NODE = "1087656", "a14r3n04"
ARMS = ("container_tmp", "bound_tmp")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path):
    return sha256(path.read_bytes()).hexdigest()


def records(raw):
    for line in raw.splitlines():
        if line.startswith(b'{"') and len(line) < 8192:
            value = json.loads(line)
            if isinstance(value, dict) and "phase" in value:
                yield value


def main(result, epoch):
    need(os.environ.get("SLURM_JOB_ID") == PARENT
         and socket.gethostname().split(".")[0] == NODE, "unexpected diagnostic placement")
    step = os.environ["SLURM_STEP_ID"]
    need(step.isdecimal() and result == ROOT / "results" / LABEL, "unexpected diagnostic namespace")
    need(0 <= time.time() - epoch <= 40, "diagnostic start is outside its original budget")
    need(pin(ORIGINAL / "SOURCE_FILES.json") ==
         "9b9354a2b36c1cc12a840602d8c504fb3a1425902e1ceb3edf0bf26649c205ab", "v3 source manifest differs")
    sys.path.insert(0, str(ORIGINAL))
    import node
    from scope import pin as pin_original

    manifest = json.loads((ORIGINAL / "SOURCE_FILES.json").read_bytes())
    for name in ("native.py", "node.py", "scope.py"):
        item = manifest["files"][name]
        pin_original(ORIGINAL / name, expected=item["SHA256"], extent=item["bytes"])
    pins = json.loads((result / "admitted-PCIs.json").read_bytes())
    need(len(pins) == len(set(pins)) == 8, "full-node PCI membership required")
    arguments = [node.APPTAINER, "exec", "--rocm", "--contain", "--no-home", "--pwd", "/"]
    for mount in (f"{HERE}:{HERE}:ro", f"{ORIGINAL}:{ORIGINAL}:ro",
                  f"{node.RUNTIME}:/swift-runtime:ro", "/usr/local/hyhal:/opt/hyhal:ro"):
        arguments += ["--bind", mount]
    arguments.extend((str(node.IMAGE), "/bin/bash", "--noprofile", "--norc", "-c",
                      node.SHELL, LABEL, str(HERE / "probe.py")))
    scratch = result / "scratch"
    scratch.mkdir(mode=0o700)
    volume = os.statvfs(scratch)
    need(volume.f_bavail * volume.f_frsize >= 2**30, "private scratch has under 1 GiB available")
    outcomes = {}
    for mode in ARMS:
        argv = arguments
        if mode == "bound_tmp":
            argv = [*arguments[:7], "--bind", f"{scratch}:/tmp:rw", *arguments[7:]]
        log = result / (mode + ".log")
        left = min(35, epoch + 100 - time.time())
        need(left > 1, "diagnostic budget exhausted before next process")
        with log.open("xb") as stream:
            try:
                run = subprocess.run([*argv, "runtime_first", str(ORIGINAL)], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=left)
                rc = run.returncode
            except subprocess.TimeoutExpired:
                rc = None
        with log.open("rb") as stream:
            raw = stream.read(131073)
        need(len(raw) <= 131072, "diagnostic log exceeds bound")
        seen = list(records(raw))
        pci = [item["PCI"] for item in seen if item["phase"] == "hip_pci_return"]
        need(not pci or len(pci) == 1 and pci[0] in pins, "enumeration left the admitted PCI set")
        outcomes[mode] = dict(returncode=rc, phases=seen, PCI=pci[0] if pci else None,
                              LLVM_ENOSPC=b"LLVM ERROR: IO failure on output stream: No space left on device" in raw,
                              driver_warning=b"Could not find any rocm files on this host!" in raw,
                              log_sha256=sha256(raw).hexdigest())
    enumerated = [entry["PCI"] for entry in outcomes.values() if entry["PCI"]]
    need(len(set(enumerated)) <= 1, "the two processes did not see the same allocated card")
    with (result / "outcome.json").open("x", encoding="utf8") as stream:
        json.dump(dict(request=REQUEST, step=PARENT + "." + step, node=NODE, planned_arms=2,
                       observed_arms=len(outcomes), arms=outcomes), stream)


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]))
