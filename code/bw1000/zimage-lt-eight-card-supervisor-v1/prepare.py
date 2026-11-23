"""Render one exact runtime from the preserved Z-Image witness and standby worker."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from supervisor import LABEL, REQUEST, RUNTIME_MEMBERS

ORIGINAL = HERE.with_name("zimage-transposed-lt-unseen-lease-2075-v3")
ORIGINAL_RENDERER_SHA = "5d36ed6c85130d086e4edc9ce6db36a3aeac85b0f5a4205e7a28d42ac36889af"
ORIGINAL_WORKER_SHA = "8ec35f54147239393b505989af52aa628bd481225726648351add784c351d1a1"
ORIGINAL_RUNTIME_SHA = "b32631db350fffe3db34088bf4650518702e1c59caa7d6ee24cfc68c1a71631a"
ORIGINAL_REQUEST = "c1935bca14ff46b89d61abe7808d63ac"
ORIGINAL_LABEL = "O01-zimage-transposed-lt-unseen-solution-v3"
STANDBY_SHA = "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059"
MASK_SHA = "1f915bd4f493178217ac3fe9b18cf9da0d08d788cffb3c1b286b6722822e9be1"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pin(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def exact(path, expected):
    raw = path.read_bytes()
    if pin(raw)["sha256"] != expected:
        raise ValueError("Source SHA differs: " + path.name)
    return raw


def unique(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen witness boundary differs: " + old[:60])
    return source.replace(old, new, 1)


def adopted_witness():
    path = ORIGINAL / "prepare.py"
    previous = ModuleType("frozen_operator_v3")
    previous.__file__ = str(path)
    exec(compile(exact(path, ORIGINAL_RENDERER_SHA), str(path), "exec"), vars(previous))
    files = previous.frozen_renderer().source_files()
    if (pin(files["worker.py"])["sha256"] != ORIGINAL_WORKER_SHA
            or pin(files["SOURCE_FILES.json"])["sha256"] != ORIGINAL_RUNTIME_SHA):
        raise ValueError("Frozen v3 witness runtime SHA differs")
    worker = files["worker.py"].decode()
    worker = unique(worker, ORIGINAL_REQUEST, REQUEST)
    worker = unique(worker, ORIGINAL_LABEL, LABEL)
    worker = unique(worker, 'os.environ["SLURM_CPUS_PER_TASK"] == "4" and len(os.sched_getaffinity(0)) >= 4',
                    'os.environ["SLURM_CPUS_PER_TASK"] == "12" and len(os.sched_getaffinity(0)) == 4')
    worker = unique(worker, '    address = C.create_string_buffer(32)\n',
                    '    count = C.c_int()\n'
                    '    spec.need(hip.hipGetDeviceCount(C.byref(count)) == 0 and count.value == 1, '
                    '"WITNESS_MASKED_HIP_COUNT")\n'
                    '    address = C.create_string_buffer(32)\n')
    worker = unique(worker, '    if args.child:\n        return witness(scope, args.out, step)',
                    '    pci(scope)\n    if args.child:\n        return witness(scope, args.out, step)')
    worker = unique(worker, 'start_new_session=True, stdin=subprocess.DEVNULL,', 'stdin=subprocess.DEVNULL,')
    worker = unique(worker, 'os.killpg(child.pid, signal.SIGTERM)', 'child.terminate()')
    worker = unique(worker, 'os.killpg(child.pid, signal.SIGKILL)', 'child.kill()')
    return {"witness.py": worker.encode(), "contract.py": files["contract.py"]}


def render():
    files = {
        **{leaf: (HERE / leaf).read_bytes() for leaf in ("supervisor.py", "linux_owner.py", "CLAIM.json")},
        "standby_worker.py": exact(HERE.with_name("fleet-dcu-standby-v2") / "worker.py", STANDBY_SHA),
        "mask_preflight.py": exact(HERE.with_name("zimage-lt-eight-card-mask-preflight-v1") / "mask_preflight.py", MASK_SHA),
        **adopted_witness(),
    }
    if set(files) != RUNTIME_MEMBERS:
        raise ValueError("Runtime members differ")
    for path, raw in files.items():
        if path.endswith(".py"):
            ast.parse(raw, filename=path)
    manifest = encoded({"request": REQUEST, "label": LABEL, "files": {leaf: pin(raw) for leaf, raw in files.items()},
                        "inherited_witness_sha256": ORIGINAL_WORKER_SHA})
    return {**files, "RUNTIME_FILES.json": manifest}


def stage(output):
    files = render()
    output.mkdir(mode=0o700)
    for name, raw in files.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return pin(files["RUNTIME_FILES.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.output), sort_keys=True))
