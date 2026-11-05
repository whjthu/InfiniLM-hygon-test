from pathlib import Path
import json, os, re, socket, sys

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
NODE = "a02r4n05"
FIXTURE = "de85e0806699b422cae9cfa1beed43c84598c684a2c0ea9236dbcea29f11a6f8"


def observe():
    assert socket.gethostname().split(".")[0] == NODE
    pci = sorted({p.resolve(strict=True).name for p in Path("/sys/class/drm").glob("renderD*/device")})
    assert len(pci) == 8 and all(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", p) for p in pci)
    kfd = sorted(int(p.name) for p in Path("/sys/class/kfd/kfd/proc").iterdir() if p.name.isdigit())
    matching = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.endswith((b"/device_probe.py", b"/GPU_inside.sh")) for arg in args):
            matching.append(int(proc.name))
    return dict(node=NODE, PCI=pci, kfd_pids=kfd, matching=matching, CPU_only=True)


def main(phase, directory):
    from hashlib import sha256
    from freeze_inputs import freeze

    out = Path(directory)
    census = observe()
    assert not census["kfd_pids"] and not census["matching"]
    if phase == "pre":
        freeze(out / "fixtures",
               archive=ROOT / "results/Deep438-1378-analysis-terminal-v1.tar.gz",
               receipt_path=ROOT / "candidates/Deep-T1-MoE-activation-quant-r438-model-diagnostic-cpu-v2/CPU-build.json",
               reader_path=Path(__file__).with_name("analyze_archive.py"))
        assert sha256((out / "fixtures/FIXTURE_MANIFEST.json").read_bytes()).hexdigest() == FIXTURE
        assert observe() == census
    elif phase == "post":
        assert json.loads((out / "pre.json").read_bytes())["PCI"] == census["PCI"]
    else:
        raise ValueError(phase)
    with (out / (phase + ".json")).open("x", encoding="utf8") as stream:
        json.dump(census, stream, indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
