"""Run the original GLM timing audit with its Linux provider path spelling."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import sys


PACKAGE = (Path(__file__).resolve().parents[1] / "code/bw1000/GLM-C1-layer-observer-r474"
           / "fresh-kv-pack-event-paired-v1")
FINISH_SHA256 = "956cb25c4e0346aca2ba716494244e22500fcaad9576cb72a3a893d7af5d8322"
HIP_PROVIDER = PurePosixPath("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")


def linux_provider_path(path):
    provider = PurePosixPath(path.as_posix())
    if provider != HIP_PROVIDER:
        raise ValueError("the original absolute HIP provider differs")
    return provider


def main():
    if sha256((PACKAGE / "finish.py").read_bytes()).hexdigest() != FINISH_SHA256:
        raise ValueError("the original GLM reader differs")
    sys.path.insert(0, str(PACKAGE))
    import finish

    verify_stage = finish.H.verify_stage

    def verify_linux_stage(checkout):
        probe, waiter, tres = verify_stage(checkout)
        # The trace contains a Linux path. Compare that exact path even when
        # the unchanged original probe is loaded by a Windows reader.
        probe.HIP_CANONICAL = linux_provider_path(probe.HIP_CANONICAL)
        return probe, waiter, tres

    finish.H.verify_stage = verify_linux_stage
    finish.main()


if __name__ == "__main__":
    main()
