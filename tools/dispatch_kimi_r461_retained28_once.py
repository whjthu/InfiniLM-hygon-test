"""Bind the single-score retained28 build to the existing atomic O01 dispatch path."""

import json
from pathlib import Path

from report_training80_backlog import REPO, need, pin_file
from verify_training80_manifest import verify_package_members


PACKAGE = "code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-retained28-host-compile-v1"
HOST = dict(path=PACKAGE + "/host_once.py", SHA256="ad9aafce3d9e86d8892d24fb97ed9b77eb4a043bcaa1b498ac2d4c90e5bcbac4")
MANIFEST = {"path":"code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-retained28-host-compile-v1/SOURCE_FILES.json","bytes":567,"SHA256":"49c91149a1c8f73c0eb824491577e7a1829813023c3bcfe7723ff9d69cea5ddb"}
CLAIM = {"path":"code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-retained28-host-compile-v1/CLAIM.json","bytes":777,"SHA256":"95853c3128de4a4fbeb46fce5552cffb6e11020025b64b373972f4fc44154ed3"}
TASK = {"path":"coordination/tasks/kimi-r461-retained28-score-gfx936-host-compile-v1.json","bytes":1478,"SHA256":"66484828892537e94917e4edb4018a026faf66f2c3f6a47a7c4966a7f9256b35"}
TEMPLATE = {"path":"tools/dispatch_kimi_r461_output_residual_wave_once.py","bytes":10355,"SHA256":"47fe73f0845f7244868cef520e464555971bd1c4cf2c5076ad3bee9d084d33aa"}
BASE = Path(PACKAGE).parent.as_posix()
SOURCE = dict(path=BASE + "/output-residual-retain-score-input-v1/SOURCE_FILES.json",
              SHA256="27a0c112c5eec86a964bdcf8e9b4d1567335e531f8f5dcc050fb1becafbfa2b9")
WORKER = dict(path=BASE + "/output-residual-wave-compile-CPU-v1/SOURCE_FILES.json",
              SHA256="e55041e1f3b5b48c776ce0a3983fce550559dbdc0daa4a50e4dacd4319b13efe")
HOST_TEMPLATE = dict(path=BASE + "/output-residual-wave-host-compile-v1/host_once.py",
                     SHA256="ad70c04a2279a2eebf7fd51e07b714b890bb430d259b60aa114ca55c2d557a08")
DIAGNOSTIC = dict(path="results/bw1000/2026-11-20/Kimi461-original337-residual-stage-diagnostic-terminal.json",
                  SHA256="1d656f3da9182fb90a7c4e5a2c4aec98db846a769f86e7d0ac2b06a649692075")
LABEL = "Kimi461-output-residual-retained28-score-gfx936-CPU-v2"
REQUEST = "7f4239bb5f8fd6dc21da6a248b5c7469"
PARENT, NODE, ZONE = "1103606", "f12r2n10", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
RENDERED_SHA = "9d3efa3a83f6733a479b23f95bbf4dcadb94a88c7b1a01216f2b889b2d6caf04"
IDENT = "kimi-r461-retained28-score-gfx936-host-compile-v1"
CASE = "I01.kimi-r461-retained28-score-compile"
INTENT = f"local-only/{LABEL}-original-dispatch/intent.json"
DISPATCHER = "tools/dispatch_kimi_r461_retained28_once.py"


def frozen_inputs(repo):
    claim = json.loads(pin_file(repo, CLAIM, ("code/bw1000/",)))
    task = json.loads(pin_file(repo, TASK, ("coordination/tasks/",)))
    need(claim["request"] == task["request"] == REQUEST and claim["label"] == LABEL
         and claim["placement"] == task["placement_candidate"] == PLACEMENT and task["case"] == CASE
         and task["backlog_id"] == IDENT and task["source_manifest"] == MANIFEST and task["claim"] == CLAIM
         and claim["planned_device_TUs"] == 1 and claim["wall_seconds"] == 300
         and claim["rendered_manifest_SHA256"] == RENDERED_SHA
         and claim["frozen_package_manifest_SHA256"] == SOURCE["SHA256"]
         and claim["CPU_GO"] is True and claim["GPU_GO"] is claim["Model_GO"] is claim["replay_allowed"] is False,
         "retained28 compile claim differs")
    names = {DISPATCHER, CLAIM["path"], TASK["path"]}
    for pin in (TEMPLATE, HOST_TEMPLATE, DIAGNOSTIC):
        pin_file(repo, pin, ("code/bw1000/", "tools/", "results/bw1000/"))
        names.add(pin["path"])
    for manifest, required in ((MANIFEST, (HOST, CLAIM)), (SOURCE, ()), (WORKER, ())):
        verify_package_members(repo, manifest, required_files=required)
        path = repo / manifest["path"]
        names.add(manifest["path"])
        names.update((path.parent / name).relative_to(repo).as_posix()
                     for name in json.loads(path.read_bytes())["files_SHA256"])
    original = json.loads((repo / WORKER["path"]).with_name("BUILD_PLAN.json").read_bytes())
    for path, pin in (original["repo_source_SHA256"] | _WAVE["SHARED"]).items():
        pin_file(repo, dict(path=path, SHA256=pin),
                 ("code/bw1000/", "tools/", "results/bw1000/", "coordination/training/"))
        names.add(path)
    return names


_WAVE = dict(__name__="frozen_atomic_score_dispatch", __file__=str(REPO / TEMPLATE["path"]))
exec(compile(pin_file(REPO, TEMPLATE, ("tools/",)), _WAVE["__file__"], "exec"), _WAVE)
_WAVE.update(PACKAGE=PACKAGE, HOST=HOST, MANIFEST=MANIFEST, CLAIM=CLAIM, TASK=TASK,
             LABEL=LABEL, REQUEST=REQUEST, PARENT=PARENT, NODE=NODE, ZONE=ZONE, PLACEMENT=PLACEMENT,
             RENDERED_SHA=RENDERED_SHA, IDENT=IDENT, CASE=CASE, INTENT=INTENT, DISPATCHER=DISPATCHER,
             TARGET=_WAVE["PROJECT"] / "results" / (LABEL + "-controller"), frozen_inputs=frozen_inputs)
backlog_identity = _WAVE["backlog_identity"]
original_argv = _WAVE["original_argv"]
dispatch = _WAVE["dispatch"]


if __name__ == "__main__":
    print(json.dumps(backlog_identity(), ensure_ascii=False, indent=2, allow_nan=False))
