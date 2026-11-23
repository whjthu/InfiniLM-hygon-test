"""Render the immutable Linux HOST stage for the existing O01 eight-card request."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[1]
HOST = CODE / "zimage-lt-eight-card-host-dispatch-v1"
GUARD = CODE / "zimage-lt-eight-card-pending-guard-v1"
RUNTIME = CODE / "zimage-lt-eight-card-supervisor-v1"
RESULTS = REPO / "results/bw1000/2026-11-24"
PUBLIC = {
    "host2105.json": ("O01-zimage-hybrid-host-original2105-zero-GRES-terminal.json",
                      "10a0070a7f5759dee3bc8f463a336ef6140908e600e6e0c66b644d3d7313fda0"),
    "owner2108.json": ("O01-zimage-hybrid-original2108-zero-GRES-pid-cgroup-terminal.json",
                       "4d26986955950b073ec03d4227f3895e3a693802eb885aca1b4dc729cbb84b37"),
}
SOURCES = {
    "host_scope.py": (HOST / "host_scope.py", "a50bfad3b2e2a7616c464b44ddd5adacbabb8979e79bf94e2c64ae15963d811f"),
    "device_host.py": (HOST / "device_host.py", "402fce88ebf96b57ed474059e7bc0c080dc377abfebb52dce6447c85336a4259"),
    "pending.py": (GUARD / "pending.py", "36a2bee98795384bb7ab4f7810e3dcdbfbe2e958aadecdb78bcd8576e8f1e8b9"),
    "node_entry.py": (CODE / "zimage-lt-eight-card-attested-entry-v1/node_entry.py",
                      "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6"),
    "node_inventory.py": (CODE / "zimage-transposed-lt-unseen-dispatch-v1/node_inventory.py",
                          "dfb047f40a1f21e9ff606afeb95e60026f9d350b5bb35e3b447d8e5b86fa7efc"),
}


def pin(raw):
    return {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def exact(path, expected):
    raw = path.read_bytes()
    if pin(raw)["sha256"] != expected:
        raise ValueError(f"frozen source SHA differs: {path.name}")
    return raw


def build(output):
    files = {key: exact(path, expected) for key, (path, expected) in SOURCES.items()}
    files.update({key: exact(RESULTS / leaf, expected) for key, (leaf, expected) in PUBLIC.items()})
    files.update({key: (HERE / key).read_bytes() for key in ("host_run.py", "scheduler.py")})
    output.mkdir(mode=0o700)
    for key, raw in files.items():
        with (output / key).open("xb") as handle:
            handle.write(raw)
    runtime = output / "runtime"
    bridge = output / "guarded-standby-bridge.py"
    subprocess.run([sys.executable, "-I", "-S", "-B", str(GUARD / "render_bridge.py"), str(bridge)],
                   check=True, capture_output=True, timeout=15)
    subprocess.run([sys.executable, "-I", "-S", "-B", str(RUNTIME / "prepare.py"), str(runtime)],
                   check=True, capture_output=True, timeout=15)
    exact(bridge, "785ed7e948c0484243c7c325eb470d4555b6304e24759667f7b1289f9d67676f")
    runtime_pin = "7ba1d9355838c09d41176f67708d892528832971f83f036be63d6ddacbbdb35d"
    native = json.loads(exact(runtime / "RUNTIME_FILES.json", runtime_pin))
    for name, source in native["files"].items():
        if len(exact(runtime / name, source["sha256"])) != source["bytes"]:
            raise ValueError("runtime source byte count differs")
    manifest = {"original_request": "e23d24ea0c9c4d00af06154d82dc22e8",
                "files": {p.relative_to(output).as_posix(): pin(p.read_bytes())
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    with (output / "HOST_STAGE_FILES.json").open("xb") as handle:
        handle.write((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    result = build(cli.parse_args().output)
    print(json.dumps({"request": result["original_request"], "stage_members": len(result["files"])}))
