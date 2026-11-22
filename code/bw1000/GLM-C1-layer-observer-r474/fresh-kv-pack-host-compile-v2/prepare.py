"""Stage the complete reviewed checkout and the isolated v2 compile namespace."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FAMILY = HERE.parent
V1 = "GLM474-fresh-kv-pack-gfx936-CPU-v1"
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v2"
SCOPE = dict(parent="1103606", node="f12r2n13", zone="O01", cpus=4, memory_MiB=8192, DCU=0)
SOURCE_INDEX_SHA = "a5e43581817e7fad431440bfa79ff3eba3708dec095dc05b76675b6d7388d259"
OLD_CLAIM_SHA = "005a011e94cbe1fa5f6f11a735d1deb1afd9ee88656632656b48e091c576a5f9"
OLD_MANIFEST_SHA = "333551bc79cbd2e56e18274737b00c57c0a918cc00700b167a61b9a316ac7850"
OLD_COMPILER_SHA = "d598ee49072a880a0c2a07f1cc7043e8e4d28f406bfe9a5e1352bb14e8723280"
KERNEL_SHA = "2ed3e3e021546bd7fc56225c18c59cee51b621af252e36f3de06bb2a0da342c1"
UPSTREAM_TASK = "coordination/tasks/glm-r474-fresh-kv-pack-gfx936-cpu-v1.json"
UPSTREAM_TASK_SHA = "6c20deb38e5a726501f044588afc24047a07fd76e3701fe9f2104694f9da2723"
SOURCE_FILES = {"LOCAL_RESULT.json", "PLAN.json", "README.zh-CN.md", "check_local.cpp", "check_local.py",
                "fresh_kv_pack.cu", "prepare_target.py"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def pinned(path, pin):
    need(path.resolve(strict=True) == path, "noncanonical frozen input")
    raw = path.read_bytes()
    need(0 < len(raw) <= 256 * 1024 and digest(raw) == pin, f"frozen source bytes differ: {path.name}")
    return raw


def generate():
    files = {}

    def take(path, pin):
        raw = pinned(path, pin)
        files[path.relative_to(REPO).as_posix()] = raw
        return raw

    source = FAMILY / "fresh-kv-pack-CPU-v1"
    source_index = json.loads(take(source / "SOURCE_FILES.json", SOURCE_INDEX_SHA))["files_SHA256"]
    need(source_index.keys() == SOURCE_FILES, "exact archived source index differs")
    original = {name: take(source / name, pin) for name, pin in source_index.items()}
    old = FAMILY / "fresh-kv-pack-host-compile-v1"
    legacy = json.loads(take(old / "CLAIM.json", OLD_CLAIM_SHA))
    take(old / "host_once.py", legacy["host_entry_SHA256"])
    take(REPO / UPSTREAM_TASK, UPSTREAM_TASK_SHA)
    for path, pin in legacy["shared_files"].items():
        take(REPO / path, pin)

    old_plan_raw = original["PLAN.json"]
    plan = json.loads(old_plan_raw)
    template = take(REPO / plan["compiler_template_path"], plan["compiler_template_SHA256"])
    import_ns = {"__file__": str(source / "prepare_target.py"), "__name__": "frozen_glm_pack_renderer"}
    exec(compile(original["prepare_target.py"], import_ns["__file__"], "exec"), import_ns)
    old_compiler = import_ns["render"](template)
    kernel = original["fresh_kv_pack.cu"]
    old_stage = {"PLAN.json": old_plan_raw, "compile_once.py": old_compiler, "fresh_kv_pack.cu": kernel}
    need(digest(old_compiler) == OLD_COMPILER_SHA and digest(kernel) == KERNEL_SHA
         and digest(json_bytes({"files_SHA256": {name: digest(raw) for name, raw in old_stage.items()}})) ==
         OLD_MANIFEST_SHA and old_compiler.count(V1.encode()) == 1 and plan["label"] == V1,
         "original generated worker differs")
    compiler = old_compiler.replace(V1.encode(), LABEL.encode(), 1)
    need(compiler.replace(LABEL.encode(), V1.encode(), 1) == old_compiler, "v2 changed compiler semantics")
    plan.update(label=LABEL, compile_entry_SHA256=digest(compiler))
    stage = {"PLAN.json": json_bytes(plan), "compile_once.py": compiler, "fresh_kv_pack.cu": kernel}
    manifest = json_bytes({"files_SHA256": {name: digest(raw) for name, raw in stage.items()}})
    root = HERE.relative_to(REPO)
    for name, raw in dict(stage, **{"SOURCE_FILES.json": manifest}).items():
        files[(root / "compile-stage" / name).as_posix()] = raw
    for path in (HERE / "host_once.py", HERE / "prepare.py"):
        files[path.relative_to(REPO).as_posix()] = path.read_bytes()
    target_sha = digest(manifest)
    request = digest(f"{LABEL}:{SCOPE['parent']}:{SCOPE['node']}:{target_sha}".encode())[:32]
    claim = dict(label=LABEL, request=request, placement=SCOPE, predecessor=legacy["request"], replay_allowed=False,
                 stage_manifest_SHA256=target_sha, kernel_SHA256=KERNEL_SHA, compiler_SHA256=digest(compiler),
                 planned_device_translation_units=1, local_CPU_preamble_only=True, GPU_GO=False,
                 files_SHA256={name: digest(raw) for name, raw in files.items()})
    files[(root / "HOST_CLAIM.json").as_posix()] = json_bytes(claim)
    return files, claim


def materialize(output):
    need(not output.exists() and not output.is_symlink(), "v2 checkout must be new")
    files, claim = generate()
    with TemporaryDirectory(dir=output.parent, prefix="glm-fresh-kv-v2-") as temporary:
        stage = Path(temporary) / "reviewed"
        stage.mkdir(mode=0o700)
        for name, raw in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        stage.rename(output)
    return dict(request=claim["request"], label=LABEL, staged_files=len(files),
                claim_SHA256=digest(json_bytes(claim)), stage_manifest_SHA256=claim["stage_manifest_SHA256"],
                planned_device_translation_units=1, target_step_invocations=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True))
