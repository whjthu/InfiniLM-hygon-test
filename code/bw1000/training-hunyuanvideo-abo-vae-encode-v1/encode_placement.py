"""Hardware and provider custody around one published ABO VAE encode."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PREVIOUS = PROJECT / "results/training-hunyuanvideo-t09-sft-repeat3-n14-untraced-v1/src"
INSPECT = PROJECT / "results/training-hunyuanvideo-abo-vae-inspect-v1/a6b6685a35a52543d3f6eb19ff141d31"
LIB = Path("/opt/dtk/lib/libMIOpen.so.1.0")
LABEL = "training-hunyuanvideo-abo-vae-native-encode-v1"
NODE, PARENT = "f13r3n14", "1103606"
OLD = {
    "PLACEMENT.json": "4d3c4a3d9b983fb2496a491f413826149af393a5cf23f117960f6825ba72ffc4",
    "placement.py": "63eba182554badd1d9e8fb0dd1971099f402b2edebbb8447d8ad4be25ad27c89",
    "n14_gate.py": "2f94793dcf6664fb7ae27b2c73718b7c695565de0e23b6dedc6954bc6a4c3011",
    "snapshot.py": "20ce55bcd12fa7d579fa2cf6c618f4aa93eedcb2c10b04fb580f280b0dae9413",
    "miopen_wall.py": "f9283976125178412db8ffd5989eeaa9c04b82848fe53b5ee55e402d8e0cd968",
    "maps_probe.py": "a832c2f29071a9e4f756bd16df1e5a5cd3df2f84db02b561ddc6506437a24121",
    "runtime_pci.py": "b42fa0870e69aa7ccf572f132939be3f024d58a7304b64d34ab618be502af401",
}


def require(ok, name):
    if not ok:
        raise ValueError(name)


def read_pin(path, expected, maximum=100_000):
    with path.open("rb") as stream:
        b = stream.read(maximum + 1)
    require(0 < len(b) <= maximum and sha256(b).hexdigest() == expected, "PINNED_RUNTIME_SOURCE")
    return b


def use_previous():
    for name, digest in OLD.items():
        read_pin(PREVIOUS / name, digest)
    sys.path.insert(0, str(PREVIOUS))


def write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return sha256(data).hexdigest()


def host(mode, root, image_command):
    use_previous()
    import miopen_wall
    import placement
    import snapshot

    phase = "gpu" if mode == "gate" else mode
    launch = json.loads((root / "intent.json").read_bytes())
    require(launch["request"] == root.name and launch["node"] == NODE and launch["parent"] == PARENT,
            "ORIGINAL_TARGET_SCOPE")
    identity = placement.authentic_step(phase, launch["step_prefix"] + "-" + phase)
    row = snapshot.snapshot(False)
    require(row["step"] == identity and row["node"] == NODE and row["kfd_holder_count"] == 0,
            "CLEAN_EIGHT_PCI_AND_NUMERIC_STEP")
    if phase != "pre":
        before = json.loads((root / "pre.json").read_bytes())
        require(row["boot_id"] == before["boot_id"] and row["pci"] == before["pci"], "SAME_EIGHT_PCI_AND_BOOT")
        if phase == "gpu":
            require(0 <= row["monotonic_ns"] - before["monotonic_ns"] <= 60_000_000_000,
                    "FRESH_PHYSICAL_CUSTODY")
    if phase == "gpu":
        require(not any(os.environ.get(k) for k in ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND",
            "SINGULARITY_BINDPATH", "LD_PRELOAD", "LD_AUDIT", "LD_PROFILE")), "NO_INHERITED_BIND_OR_LOADER_OVERRIDE")
        require(image_command and image_command[0] == "/public/software/apps/apptainer/1.3.4/bin/apptainer",
                "EXACT_APPTAINER_ENTRY")
        row["provider"] = miopen_wall.stage(root / "miopen-private", min(launch["deadline"], time.time() + 25))
        os.chdir("/")
    write(root / ("gpu-gate.json" if phase == "gpu" else phase + ".json"), row)
    if phase == "gpu":
        os.execv(image_command[0], image_command)


def device(root, args):
    use_previous()
    import miopen_wall
    from runtime_pci import runtime_pci

    sys.path.insert(0, str(INSPECT / "src"))
    import worker

    original = worker.native_encode

    def with_provider(frames, asset, source, report):
        gate = json.loads((root / "gpu-gate.json").read_bytes())
        expected = gate["provider"]
        launch = json.loads((root / "intent.json").read_bytes())
        require("torch" not in sys.modules and os.statvfs(LIB).f_flag & os.ST_RDONLY, "COLD_READONLY_LIBRARY")
        require(all(not os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT", "LD_PROFILE")), "NATURAL_LOADER")
        fd = miopen_wall.open_exact(LIB)
        try:
            s = miopen_wall.digest(fd, launch["deadline"])
            require(miopen_wall.key(s) == expected["provider_dev_inode"] and expected["provider"] == miopen_wall.PIN,
                    "EXACT_HOST_AND_CONTAINER_MIOPEN")
            before = miopen_wall.maps_pin()
            import torch
            after = miopen_wall.maps_pin()
            identity = [os.major(s.st_dev), os.minor(s.st_dev), s.st_ino]
            miopen_wall.verify_mapping(before, after, identity)
            require(torch.cuda.device_count() == 1, "EXACT_ONE_VISIBLE_DEVICE")
            pci = runtime_pci()
            require(pci in gate["pci"], "ASSIGNED_DEVICE_IN_ORIGINAL_EIGHT_PCI")
            payload, runtime = original(frames, asset, source, report)
            os.lseek(fd, 0, os.SEEK_SET)
            final = miopen_wall.digest(fd, launch["deadline"])
            require(miopen_wall.stamp(final) == miopen_wall.stamp(s), "STABLE_NATIVE_MIOPEN_FD")
            miopen_wall.verify_mapping(before, miopen_wall.maps_pin(), identity)
            runtime.update(actual_pci=pci, mapped_MIOpen=miopen_wall.PIN, actual_qualified_mapping=True,
                           torch_allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
                           torch_reserved_peak_bytes=torch.cuda.max_memory_reserved(0))
            return payload, runtime
        finally:
            os.close(fd)

    worker.native_encode = with_provider
    sys.argv = [str(INSPECT / "src/worker.py"), *args]
    return worker.main()


if __name__ == "__main__":
    mode, root, *args = sys.argv[1:]
    if mode == "image":
        raise SystemExit(device(Path(root), args))
    require(mode in {"pre", "gate", "post"}, "EXACT_HARDWARE_PHASE")
    host(mode, Path(root), args)
