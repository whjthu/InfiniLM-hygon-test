"""Stage the pinned target worker and exact already-present train-only inputs."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from render_worker import CORE_SHA, FRAMEWORK_SHA, check_scope, command_plan, render_worker
from render_owner import OWNER_DEPS, render_owner
from scope_contract import INPUT_ROOT, INPUT_SHA, exact_4479, need, sha, view_bytes

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGET = Path(INPUT_ROOT).parents[2]
ORIGINAL_LABEL = "training-minicpm-off-native6-cpu-1fde0c1744ce"
CORE = REPO / "code/bw1000/training-minicpm-usgs-exposure25-resume20-hip-v1"
PREPARE_SHA = "0193860f93482ce22ef23a9193740ef9fb603389e2e1b4f5cd598a3c735109bc"
CONTROL_SHA = "11e6629bd3c4d1e8e3158ef2a188b80f296178a6bf63abe9169191cc9a875ba3"
OFF_INPUT_CODE_SHA = "4f827910e7875cb646eac872e6aa90d3c09bfcf46b7b9cca397859606334a88d"
ORIGINAL_READER_SHA = "4740a8b26f47ba419c77b59e45476673ed2973e30869b7d8f806713eb6ada69c"
ORIGINAL_BUNDLE_SHA = "20bba70ad1655e9579c93b5a9f6128d06b3653b23482989234c855602fbcff9b"
MODEL_SHA = "ddcb8756dc66474f348bd81bb1730874ba78af8b46567e6a99596cedf5594ae6"
OVERLAY = ("scope_contract.py", "off_callback.py", "render_worker.py", "reload_cp2.py")
EXACT_DEPS = {
    "tools/read_training_slurm_membership.py": "9b3b0ca57c7ade5c28874faf46aa6925178ec0536f666842a085a76825d06de7",
    "code/bw1000/training-qwen-native25-hip-v3/slurm_accounting.py": "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
WORKER_DEPS = {
    "snapshot.py": "be40cbf5d125be73362cab83c718169a974f8f3d9f02c977f65fc97ea0088134",
    "owned_processes.py": "6c99392ee4210e23fa4e9e5a6be4290ca2c12fa9d0dd3e4c7b40d415977950c7",
}


def module(name, path, digest):
    raw = path.read_bytes()
    need(sha(raw) == digest, "OFF_HISTORICAL_SOURCE_DIGEST")
    ns = {"__name__": name, "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), ns)
    return ns


def stage(scope_file, publication, source_manifest_sha):
    scope_raw = scope_file.read_bytes()
    scope = check_scope(json.loads(scope_raw))
    need(scope["publication"] == publication and scope["source_manifest_SHA256"] == source_manifest_sha
         and scope["stage_GO"] is True and scope["GPU_GO"] is False and scope["training_GO"] is False, "OFF_ONLY_SOURCE_STAGE_AUTHORIZED")
    source_manifest = (HERE / "SOURCE_FILES.json").read_bytes()
    need(sha(source_manifest) == source_manifest_sha, "OFF_FIXED_SOURCE_MANIFEST")
    source_files = json.loads(source_manifest)["files"]
    for name, pin in source_files.items():
        need(Path(name).name == name and (HERE/name).is_file() and not (HERE/name).is_symlink(), "OFF_STAGE_MEMBER_PATH")
        raw = (HERE/name).read_bytes()
        need(len(raw) == pin["bytes"] and sha(raw) == pin["SHA256"], "OFF_STAGE_MEMBER_DIGEST")
    original = module("off_original_prepare", REPO / "tools/prepare_minicpm_usgs_pair_target_v3.py", PREPARE_SHA)
    need(original["FRAMEWORK_SHA"] == FRAMEWORK_SHA and original["RESULTS"].parent == TARGET, "OFF_ORIGINAL_STAGE_ROOT")
    stable, small, write = (original[x] for x in ("stable_file", "small", "write"))
    owner_sources = {
        Path(name).name: small(REPO / name, digest)
        for name, digest in {**EXACT_DEPS, "tools/prepare_minicpm_usgs_pair_target_v3.py": PREPARE_SHA}.items()
    }
    need({name: sha(raw) for name, raw in owner_sources.items()} == OWNER_DEPS, "OFF_OWNER_DEPENDENCY_CLOSURE")
    run = TARGET / "results" / scope["label"]
    need(not run.exists() and not run.is_symlink(), "OFF_STAGE_ALREADY_PRESENT")
    original_run = TARGET / "results" / ORIGINAL_LABEL
    original_reader = module("off_original_4479_reader", original_run / "src/independent.py", ORIGINAL_READER_SHA)

    def original_accounting(parent):
        need(parent == "1103606", "OFF4479_PARENT")
        return subprocess.check_output(
            ["sacct", "-j", "1103606.4479", "-nP", "-o", original_reader["COLUMNS"]],
            timeout=12, text=True,
        )

    independent = original_reader["read"](original_run, ORIGINAL_BUNDLE_SHA, original_accounting)
    exact_4479(independent, small(original_run / "worker-result.json"))
    validator = REPO / "code/bw1000/training-minicpm-off-native6-CPU-v1/off_inputs.py"
    off = module("off_frozen_cpu_inputs", validator, OFF_INPUT_CODE_SHA)
    body, ledger = view_bytes(off["read_inputs"](Path(INPUT_ROOT), INPUT_SHA))
    core = small(CORE / "SOURCE_FILES.json", CORE_SHA)
    sources = {}
    for name, pin in json.loads(core)["files"].items():
        need(Path(name).name == name, "OFF_ORIGINAL_CORE_MEMBER")
        sources[name] = stable(CORE / name, size=pin["bytes"], digest=pin["SHA256"], ceiling=2*1024**2, collect=True)
    assets = json.loads(sources["ASSETS.json"])
    initial = assets["initial"]
    init_file = stable(Path(initial["path"]), size=initial["bytes"], digest=initial["SHA256"])
    model = TARGET / "results/training-minicpm-original-model-v1/MiniCPM-V-4.6"
    weights = json.loads(small(REPO / "code/bw1000/training-minicpm-final-tail-r2/MODEL_FILES.json", MODEL_SHA))["files"]
    need(len(weights) == 10 and "model.safetensors" in weights and all(Path(name).name == name for name in weights), "OFF_ORIGINAL_MODEL_MANIFEST")
    model_pins = {name: stable(model/name, size=pin["bytes"], digest=pin["SHA256"]) for name, pin in weights.items()}
    need(shutil.disk_usage(run.parent).free >= 2 * 1024**3, "OFF_CP2_STORAGE")
    sys.path.insert(0, str(CORE))
    import minicpm_domain_recipe as recipe
    run.mkdir(mode=0o700)
    write(run / "off-scope.json", scope_raw)
    (run / "src").mkdir(mode=0o700)
    for name, raw in {**sources, "SOURCE_FILES.json": core}.items():
        write(run / "src" / name, raw)
    (run / "src/owner-deps").mkdir(mode=0o700)
    for name, raw in owner_sources.items():
        write(run / "src/owner-deps" / name, raw)
    overlay = {name: small(HERE / name) for name in OVERLAY}
    overlay["off_inputs.py"] = small(validator, OFF_INPUT_CODE_SHA)
    overlay_pins = {name: {k: v for k, v in write(run/"src"/name, raw).items() if k != "path"} for name, raw in overlay.items()}
    (run / "view").mkdir(mode=0o700)
    write(run / "view/training800.jsonl", body)
    write(run / "view/exposure-ledger-private.json", ledger)
    framework = original["stage_framework"](TARGET/"results/training-qwen-a68952c-swift-initial-CPU-v1", run)
    plan = command_plan(recipe, run/"framework/tools/hygon_vlm_recipe.py", sources["original-launch.json"], run, sha(body), scope["port"])
    plan_pin = write(run/"continuous-plan.json", plan)
    stage = dict(state="OFF_SIX_TO_800_STAGED", request=scope["request"], parent=scope["parent"], node=scope["node"], port=scope["port"],
                 publication=publication, scope_SHA256=sha(scope_raw), source_manifest_SHA256=source_manifest_sha,
                 source_rows=6, train_media=4, input_manifest_SHA256=INPUT_SHA, framework=framework,
                 plan=plan_pin, model_files=model_pins, initial=init_file, overlay=overlay_pins, owner_deps=OWNER_DEPS,
                 actual_GPU_steps=0, actual_optimizer_updates=0)
    stage_pin = write(run/"asset-stage-record.json", stage)
    original_target = REPO / "code/bw1000/training-minicpm-usgs-exposure25-resume20-target-v2"
    worker_raw = render_worker(small(original_target/"worker.py"), scope, stage_pin["SHA256"])
    (run / "target-worker").mkdir(mode=0o700)
    target_members = {"worker.py": worker_raw, **{name: small(original_target/name, digest) for name, digest in WORKER_DEPS.items()}}
    target_pins = {name: write(run/"target-worker"/name, raw) for name, raw in target_members.items()}
    owner_raw = render_owner(small(REPO/"tools/control_minicpm_usgs_pair_target_v3.py", CONTROL_SHA), scope, stage_pin["SHA256"])
    owner = write(run/"owner.py", owner_raw)
    ready = write(run/"asset-stage-complete.json", dict(request=scope["request"], stage=stage_pin, target=target_pins,
                                                       owner=owner, scope_SHA256=sha(scope_raw)))
    return dict(request=scope["request"], label=scope["label"], planned_rows=6, physical_rows=800, train_media=4,
                planned_native_forwards=64, planned_optimizer_updates=2, actual_GPU_steps=0,
                asset_stage_complete_SHA256=ready["SHA256"], scope_SHA256=sha(scope_raw), view_SHA256=sha(body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--source-manifest", required=True)
    args = parser.parse_args()
    def expired(*unused):
        raise TimeoutError("OFF_SOURCE_STAGE_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(240)
    try:
        print(json.dumps(stage(args.scope, args.publication, args.source_manifest), sort_keys=True))
    finally:
        signal.alarm(0)
