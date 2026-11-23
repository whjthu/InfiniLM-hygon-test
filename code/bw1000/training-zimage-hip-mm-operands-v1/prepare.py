"""Render a two-file diagnostic overlay against the published .1768 recorder."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
PATH = Path("generation/src/gencuda/zimage_native_profile.py")
RECORD = PATH.with_name("zimage_native_record.py")
PROFILE_SHA = "9d3ffc0bdfac9bb0eb4dad3bef5576360917d4bf6b10ededd366600722ca5b77"
RECORD_SHA = "8290e9ecd1f0accf42ad21a8e4bc8050a46d535ad3dccba71fb83ad3a18221f1"
MODULE = "zimage_mm_operands.py"


def checked(root, relative, expected):
    source = root / relative
    raw = source.read_bytes()
    if source.resolve(strict=True) != source or len(raw) > 64 * 1024 or sha256(raw).hexdigest() != expected:
        raise ValueError("published .1768 framework member differs: " + str(relative))
    return raw


def replace_once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError("published profiler observation boundary changed")
    return raw.replace(old, new)


def render(profile):
    if sha256(profile).hexdigest() != PROFILE_SHA:
        raise ValueError("published native profiler bytes changed")
    raw = replace_once(profile, b"import time\n",
                       b"import time\n\nfrom .zimage_mm_operands import verify_smoke, write_capture\n")
    raw = replace_once(raw, b"    result=summarize_events(prof,output)\n",
                       b"    result=summarize_events(prof,output)\n"
                       b"    result['backward_mm_operands']=write_capture(prof,output,check=check)\n")
    raw = replace_once(raw, b"        y=(x@x).sum();y.backward();torch.cuda.synchronize()\n",
                       b"        y=(x@x.transpose(0,1)).sum();y.backward();torch.cuda.synchronize()\n"
                       b"    verify_smoke(p)\n")
    return raw


def stage(framework, output):
    framework = framework.resolve(strict=True)
    checked(framework, RECORD, RECORD_SHA)
    profile = checked(framework, PATH, PROFILE_SHA)
    files = {PATH: render(profile), PATH.with_name(MODULE): (HERE / MODULE).read_bytes()}
    for path, raw in files.items():
        compile(raw, str(path), "exec")
    manifest = (json.dumps({"files": {path.as_posix(): {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}
                                    for path, raw in files.items()},
                            "base": "8db9440c5de49712c88dbeadf1e0ecff9f31dd72", "source_step": "1103606.1768",
                            "actual_device_steps_by_renderer": 0}, sort_keys=True, indent=2) + "\n").encode()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="zimage-mm-operands-", dir=output.parent) as tmp:
        tree = Path(tmp) / "overlay"
        tree.mkdir(mode=0o700)
        for path, raw in files.items():
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (tree / "SOURCE_FILES.json").write_bytes(manifest)
        tree.rename(output)
    return {"file_count": len(files), "manifest_SHA256": sha256(manifest).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.framework, args.output), sort_keys=True))
