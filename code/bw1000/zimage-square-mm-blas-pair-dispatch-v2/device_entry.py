"""Bind the scheduled gfx936 card to the original independent PCI inventory."""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys


PACKAGE = Path("/pair-package")
RUN = Path("/pair-run")
HIP = Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
WORKER_SHA = "fd6a3c8a20e9f598d515550e7d2ffc89464f75bb62292cf24f4459b97f2e8a7b"
CONTRACT_SHA = "956c0892690a554fb31549051797f9dff82d851f1c4c79e7a3e7a2e07059f148"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def file_digest(path):
    with path.open("rb") as stream:
        digest = sha256()
        for raw in iter(lambda: stream.read(1 << 20), b""):
            digest.update(raw)
    return digest.hexdigest()


def main():
    intent = json.loads((RUN / "intent.json").read_bytes())
    pre = json.loads((RUN / "pre.json").read_bytes())
    step = os.environ["SLURM_STEP_ID"]
    require(intent["node"] == pre["node"] == socket.gethostname().split(".")[0]
            and intent["parent"] == os.environ["SLURM_JOB_ID"] == "1103606"
            and step.isdecimal() and pre["step"] != "1103606." + step,
            "Device step is outside its independent preflight")
    for name, digest in (("worker.py", WORKER_SHA), ("contract.py", CONTRACT_SHA)):
        require(file_digest(PACKAGE / name) == digest, "Bound operator source changed")
    boot = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    require(sha256(boot).hexdigest() == pre["boot_id_SHA256"], "Boot differs from preflight")
    require(HIP.resolve(strict=True) == HIP and HIP.stat().st_size == 12_349_016
            and file_digest(HIP) == HIP_SHA, "HIP provider differs")
    hip = ctypes.CDLL(str(HIP), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    count = ctypes.c_int()
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1,
            "Expected exactly one visible physical accelerator")
    buffer = ctypes.create_string_buffer(24)
    require(hip.hipDeviceGetPCIBusId(buffer, len(buffer), 0) == 0, "Assigned PCI is unavailable")
    pci = buffer.value.decode("ascii").lower()
    if len(pci) == 7:
        pci = "0000:" + pci
    require(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci)
            and len(set(pre["PCIs"])) == 8 and pci in pre["PCIs"], "Assigned PCI differs")
    scope = {key: intent[key] for key in ("request", "label", "node", "parent", "epoch", "deadline_epoch")}
    scope.update(root_admitted=True, pre_step=pre["step"], PCI=pci, boot_id=boot.decode("ascii").strip(),
                 torch_version="2.10.0", torch_hip="6.3.26113")
    raw = (json.dumps(scope, sort_keys=True, allow_nan=False) + "\n").encode()
    path = RUN / "worker-scope.json"
    with path.open("xb") as stream:
        stream.write(raw)
    worker = PACKAGE / "worker.py"
    sys.argv = [str(worker), "--scope", str(path), "--scope-sha256", sha256(raw).hexdigest(),
                "--out", str(RUN / "worker-output")]
    runpy.run_path(str(worker), run_name="__main__")


if __name__ == "__main__":
    main()
