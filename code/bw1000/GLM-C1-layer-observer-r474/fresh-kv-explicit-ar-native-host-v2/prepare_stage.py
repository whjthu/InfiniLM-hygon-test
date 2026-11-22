"""Freeze all bytes for explicit-AR Native v2 without target actions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory

import host_once as host
import kernel_855 as kernel
import worker


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = HERE.parent.relative_to(REPO).as_posix()
DEPTH = BASE + "/depth-stratified-observer-CPU-v1/"
FROZEN = {
    **worker.SOURCE_PINS,
    "tools/prepare_glm474_fresh_kv_native_build.py": "12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd",
    "tools/run_glm474_fresh_kv_native_once.py": "2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b",
    "tools/read_vggt_tum_bidirectional30_terminal.py": "597e61786d3b038421d64ae5e11697914b0bc8c9d366947bbaf251dcff25e3c3",
    host.MEMBERSHIP: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    host.ACCOUNTING: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    host.LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    host.EXPLICIT_HOST.relative_to(REPO).as_posix(): host.EXPLICIT_HOST_SHA,
    BASE + "/fresh-kv-pack-native-host-v1/host_once.py": "b7845bd133a642877b43b803780df6ae8e19f3d9ca5598c028f791a34911bc7b",
    DEPTH + "ENV.sh": "147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9",
    DEPTH + "build_once.py": "7e786da991f1c5f83d44645d12a8aaa7e8606b2b0e5bdabd6ae1b0b7ca944e46",
    DEPTH + "PLAN.json": "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6",
    kernel.PUBLIC: kernel.PUBLIC_SHA,
    BASE + "/fresh-kv-explicit-ar-source-v1/SOURCE_FILES.json": worker.SOURCE_MANIFEST_SHA256,
}


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def digest(raw):
    return sha256(raw).hexdigest()


def generate():
    data = {path: kernel.read(REPO / path, pin) for path, pin in FROZEN.items()}
    kernel.archive(data[kernel.PUBLIC])
    old = runpy.run_path(str(REPO / worker.EXPLICIT_PREPARE))
    generated, pins = old["caller_sources"]()
    leaf = HERE.relative_to(REPO)
    for name, raw in generated.items():
        data[(leaf / "generated" / name).as_posix()] = raw
    for name in ("host_once.py", "worker.py", "kernel_855.py"):
        data[(leaf / name).as_posix()] = (HERE / name).read_bytes()
    worker_sha = digest(data[(leaf / "worker.py").as_posix()])
    claim = dict(label=host.LABEL, request=host.REQUEST, placement=host.PLACEMENT, planned=host.PLANNED,
                 original_CPU308="1103606.308", kernel_prerequisite_request=kernel.REQUEST,
                 kernel_object_SHA256=kernel.OBJECT_SHA, kernel_public_evidence_SHA256=kernel.PUBLIC_SHA,
                 kernel_compute_terminal="1103606.855", caller_manifest_SHA256=worker.SOURCE_MANIFEST_SHA256,
                 whole_seconds=1800, work_seconds=1680, recipe_tool_SHA256=worker_sha, runner_SHA256=worker_sha,
                 CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False)
    claim_raw = packed(claim)
    data[(leaf / "SCOPE.json").as_posix()] = claim_raw
    manifest_raw = packed({"files_SHA256": {name: digest(raw) for name, raw in data.items()}})
    data[(leaf / "STAGE_FILES.json").as_posix()] = manifest_raw
    return data, dict(label=host.LABEL, request=host.REQUEST, claim_SHA256=digest(claim_raw),
                      stage_manifest_SHA256=digest(manifest_raw), source_files=len(data),
                      total_bytes=sum(map(len, data.values())), actual_target_steps=0)


def materialize(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError("v2 materialization requires a new directory")
    files, receipt = generate()
    with TemporaryDirectory(dir=output.parent, prefix="glm-explicit-ar-v2-") as temp:
        staged = Path(temp) / "reviewed"
        staged.mkdir()
        for name, raw in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        if output.exists() or output.is_symlink():
            raise FileExistsError("v2 staging scope appeared")
        staged.rename(output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True, allow_nan=False))
