"""Render the frozen retained28 score candidate with an exact one-object receipt."""

import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "output-residual-retain-score-input-v1"
SOURCE_PIN = "27a0c112c5eec86a964bdcf8e9b4d1567335e531f8f5dcc050fb1becafbfa2b9"
OLD_LABEL = "Kimi461-output-residual-retained28-score-gfx936-CPU-v1"
LABEL = "Kimi461-output-residual-retained28-score-gfx936-CPU-v2"
OLD = dict(target_manifest_SHA256="abd00263d999296d5d3634cfa482b80535052efad2ba3e127db3aaf836bfa50c",
           candidate_id="b4184862968118fddedc612ebb6d04433733d562e9817cdec831145ad886eb39",
           source_SHA256="bc530d46b82723ddb1279f1000af1fef10dba32003902a862ab474d4648db410",
           worker_SHA256="277667e96c468baf89569605742f88f8c026ee22540b697218baca70a244a818",
           target_files=7)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("frozen member differs: " + path.name)
    return raw


def once(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("expected one frozen transform anchor")
    return raw.replace(before, after)


def render(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads(checked(SOURCE / "SOURCE_FILES.json", SOURCE_PIN))
    for name, expected in manifest["files_SHA256"].items():
        if Path(name).name != name:
            raise ValueError("unexpected score source path")
        checked(SOURCE / name, expected)
    with TemporaryDirectory(prefix="kimi-retained28-target-", dir=output.parent) as directory:
        stage = Path(directory) / "generated"
        previous = runpy.run_path(str(SOURCE / "render.py"))["materialize"](stage)
        if previous != OLD:
            raise ValueError("original retained28 render differs")
        prior = json.loads(checked(stage / "SOURCE_FILES.json", OLD["target_manifest_SHA256"]))["files_SHA256"]
        for name, pin in prior.items():
            checked(stage / name, pin)
        worker_name = "compile_once.py"
        worker = checked(stage / worker_name, OLD["worker_SHA256"]).decode()
        worker = once(worker, f'LABEL = "{OLD_LABEL}"', f'LABEL = "{LABEL}"')
        worker = once(worker, "planned_device_TUs=2,", "planned_device_TUs=1,")
        compile(worker, worker_name, "exec")
        (stage / worker_name).write_bytes(worker.encode())
        plan_name = "BUILD_PLAN.json"
        plan = json.loads(checked(stage / plan_name, prior[plan_name]))
        if plan["label"] != OLD_LABEL or set(plan["rendered_files_SHA256"]) != {"retained28"}:
            raise ValueError("old retained28 arm or label differs")
        plan.update(label=LABEL, planned_device_TUs=1, prior_target_manifest_SHA256=OLD["target_manifest_SHA256"])
        (stage / plan_name).write_bytes((json.dumps(plan, sort_keys=True, indent=2) + "\n").encode())
        pins = {name: sha((stage / name).read_bytes()) for name in prior}
        if {name for name, value in pins.items() if value != prior[name]} != {plan_name, worker_name}:
            raise ValueError("expected identical kernel, math and launch shim")
        manifest_raw = (json.dumps(dict(files_SHA256=pins), sort_keys=True, indent=2) + "\n").encode()
        (stage / "SOURCE_FILES.json").write_bytes(manifest_raw)
        stage.rename(output)
    return dict(manifest_SHA256=sha(manifest_raw), files=len(pins), target_compiles=0)
