"""Create a fresh exact HOST checkout. This path performs no target action."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_once as host
import worker


HERE, REPO = worker.HERE, worker.REPO
DEPTH = worker.AREA / "depth-stratified-observer-CPU-v1"
CALLER = worker.CALLER.relative_to(REPO)
NEW = HERE.relative_to(REPO)
PRIOR = worker.PRIOR_DIR.relative_to(REPO)
FROZEN = {
    Path("tools/prepare_glm474_fresh_kv_native_build.py"): "12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd",
    Path("tools/run_glm474_fresh_kv_native_once.py"): "2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b",
    Path("tools/read_vggt_tum_bidirectional30_terminal.py"): "597e61786d3b038421d64ae5e11697914b0bc8c9d366947bbaf251dcff25e3c3",
    Path(host.MEMBERSHIP): "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    Path(host.ACCOUNTING): "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    Path(host.LEDGER): "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    host.EXPLICIT_HOST.relative_to(REPO): host.EXPLICIT_HOST_SHA,
    worker.AREA / "fresh-kv-pack-native-host-v1/host_once.py": "b7845bd133a642877b43b803780df6ae8e19f3d9ca5598c028f791a34911bc7b",
    DEPTH / "ENV.sh": "147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9",
    DEPTH / "build_once.py": "7e786da991f1c5f83d44645d12a8aaa7e8606b2b0e5bdabd6ae1b0b7ca944e46",
    DEPTH / "PLAN.json": "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6",
    CALLER / "SOURCE_FILES.json": worker.CALLER_MANIFEST,
    PRIOR / "worker.py": worker.LEGACY_WORKER_SHA,
    PRIOR / "kernel_855.py": worker.KERNEL_SHA,
    worker.ORIGINAL_ARCHIVE: worker.ORIGINAL_ARCHIVE_SHA,
}


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def digest(raw):
    return sha256(raw).hexdigest()


def generate():
    worker.require_request(worker.REQUEST)
    kernel, prior = worker.baseline()
    pins = dict(FROZEN)
    pins.update({Path(name): pin for name, pin in prior["SOURCE_PINS"].items()})
    pins.update({Path(kernel.PUBLIC): kernel.PUBLIC_SHA,
                 prior["CALLER"].relative_to(REPO) / "SOURCE_FILES.json": prior["SOURCE_MANIFEST_SHA256"]})
    data = {name.as_posix(): worker.checked(REPO / name, pin) for name, pin in pins.items()}
    worker.verify_original_archive()
    kernel.archive(data[kernel.PUBLIC])
    declaration = json.loads(data[(CALLER / "SOURCE_FILES.json").as_posix()])
    for entry in declaration["files"]:
        worker.checked(REPO / entry["path"], entry["SHA256"], int(entry["bytes"]))
    script = REPO / CALLER / "render.py"
    namespace = worker.execute(script, script.read_bytes())
    generated = namespace["sources"]()
    if {name: digest(raw) for name, raw in generated.items()} != declaration["generated_sources_SHA256"]:
        raise ValueError("the actual published 15-source caller rendering differs")
    for name, raw in generated.items():
        data[(NEW / "generated" / name).as_posix()] = raw
    for name in ("host_once.py", "worker.py"):
        data[(NEW / name).as_posix()] = worker.checked(HERE / name)
    worker_sha = digest(data[(NEW / "worker.py").as_posix()])
    claim = dict(label=worker.LABEL, request=worker.REQUEST, placement=worker.PLACEMENT, planned=worker.PLANNED,
                 original_CPU308="1103606.308", original_918_library_SHA256=worker.ORIGINAL_NATIVE_SHA,
                 original_918_archive_SHA256=worker.ORIGINAL_ARCHIVE_SHA, kernel_prerequisite_request=kernel.REQUEST,
                 kernel_object_SHA256=kernel.OBJECT_SHA, caller_publication=worker.CALLER_PUBLICATION,
                 caller_manifest_SHA256=worker.CALLER_MANIFEST, new_export=worker.NEW_EXPORT,
                 expected_exports=285, expected_providers=71, whole_seconds=1800, work_seconds=1680,
                 stage_members=len(data)+1, recipe_tool_SHA256=worker_sha, runner_SHA256=worker_sha,
                 CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False)
    claim_raw = packed(claim)
    if worker.checked(HERE / "CLAIM.json", limit=16384) != claim_raw:
        raise ValueError("reviewed native claim does not match the actual generated scope")
    data[(NEW / "SCOPE.json").as_posix()] = claim_raw
    manifest = packed(dict(files_SHA256={name: digest(raw) for name, raw in data.items()}))
    data[(NEW / "STAGE_FILES.json").as_posix()] = manifest
    return data, dict(label=worker.LABEL, request=worker.REQUEST, claim_SHA256=digest(claim_raw),
                      stage_manifest_SHA256=digest(manifest), source_files=len(data),
                      total_bytes=sum(map(len, data.values())), actual_target_steps=0)


def materialize(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError("original T512 checkout already exists")
    data, receipt = generate()
    with TemporaryDirectory(dir=output.parent, prefix="glm-t512-reviewed-") as temp:
        stage = Path(temp) / "checked"
        stage.mkdir()
        for name, raw in data.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        stage.rename(output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True, allow_nan=False))
