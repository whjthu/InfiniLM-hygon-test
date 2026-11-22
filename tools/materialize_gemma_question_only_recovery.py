"""Generate a checkpoint-20 recovery scope for the measured question-only arm."""

import ast
from hashlib import sha256
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "code/bw1000"
OLD = BASE / "training-gemma-usgs72-resume20to25-supervisor-v1"
DOMAIN = BASE / "training-gemma-usgs72-fresh-initial-domain25-v1"
QUESTION = BASE / "training-gemma-usgs72-question-only-control25-v1"
LABEL = "training-gemma-usgs72-question-only-resume20-v2"
DISPATCH = sha256((LABEL + "/dispatch").encode()).hexdigest()[:32]
PORT = 29965
VIEW = "a2d2411778774856889797c3f1851b6f5df7fe67fd6a81ff113ab03d3da3d144"
OUTCOME = "f2d166439f7b38e3820da24cb8c91af7aeea8ca59fcfe11dc69007237be8e5e9"
NATIVE = "4db77eea4cce17bf30868bb2d6f7ddc0aff483f7549ef8f17276f727b1630e9c"
QUESTION_MF = "d94f952d1303961036f14f2c054ad5100569faf8297e87b0f1d766e16a6ae473"


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def exact(raw, mapping):
    text = raw.decode()
    for old, new in mapping:
        if text.count(old) != 1:
            raise ValueError(f"Original anchor differs: {old}")
        text = text.replace(old, new, 1)
    return text.encode()


def freeze(root, digest, selected):
    index = (root / "SOURCE_FILES.json").read_bytes()
    if pin(index)["SHA256"] != digest:
        raise ValueError(f"Original source manifest differs: {root.name}")
    pins = json.loads(index)["files"]
    files = {name: (root / name).read_bytes() for name in selected}
    if any(pin(raw) != pins[name] for name, raw in files.items()):
        raise ValueError(f"Original source member differs: {root.name}")
    return files


def resumed_plan(raw, root, plugin):
    if pin(raw)["SHA256"] != "939b112dcc16a3e8d3203f61203f4b0b3ba5572325c2e6814de0645693049537":
        raise ValueError("Original qualified resume command differs")
    plan = json.loads(raw)
    old_root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results") / DOMAIN.name
    previous_plugin = old_root.as_posix() + "/framework/vlm/plugins/gemma_domain_resume.py"

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(item) for k, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if value == previous_plugin:
            return plugin
        if isinstance(value, str) and value.startswith(old_root.as_posix() + "/"):
            return root + value[len(old_root.as_posix()):]
        return value

    plan = relocate(plan)
    argv = plan["command"]
    value = lambda key: argv[argv.index(key) + 1]
    expected = {"--dataset": root + "/view/train800-swift.jsonl", "--resume_from_checkpoint": root + "/continuous/train/checkpoint-20",
                "--output_dir": root + "/resumed/train", "--external_plugins": plugin, "--max_steps": "25",
                "--gradient_accumulation_steps": "4", "--per_device_train_batch_size": "1"}
    if any(argv.count(key) != 1 or value(key) != target for key, target in expected.items()):
        raise ValueError("Original recovery recipe or checkpoint destination differs")
    plan["environment_overrides"]["MASTER_PORT"] = str(PORT)
    return plan


def materialize(output):
    files = freeze(OLD, "a0ae36f5c810ee4294fda03be4c0cc340a52b413ceb2534d455c4441f81d37db",
                   ("CLAIM.json", "compare_recovery.py", "contract.py", "controller.py", "devices.py", "held_child.py", "launch.py",
                    "owned_processes.py", "read_terminal.py", "rss_monitor.py", "slurm_accounting.py", "snapshot.py", "supervisor.py"))
    resume = freeze(DOMAIN, "d3e8a6432e89db845e527a0d32b9ff39139416d2fb08885babcb5eba67156eb1",
                    ("resumed-plan.json", "overlay/vlm/plugins/gemma_domain_resume.py"))
    question = freeze(QUESTION, QUESTION_MF, ("worker.py", "control_contract.py"))
    root = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/" + QUESTION.name
    plugin = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/" + LABEL + "/src/gemma_question_resume.py"
    files["resumed-plan.json"] = encode(resumed_plan(resume["resumed-plan.json"], root, plugin))
    files["gemma_question_resume.py"] = exact(resume["overlay/vlm/plugins/gemma_domain_resume.py"], [
        ("74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc", VIEW),
        ("class GemmaResumeWitness(ResumeWitness):\n", """class GemmaResumeWitness(ResumeWitness):
    def before(self, module, argv, kwargs):
        fields = ("pixel_values", "pixel_values_videos", "image_sizes", "image_grid_thw", "video_grid_thw")
        if any(kwargs.get(field) is not None for field in fields):
            raise ValueError("question-only restoration received a media input")
        return super().before(module, argv, kwargs)

""")])
    old_worker = question["worker.py"].decode()
    start = old_worker.index("    arm=ROOT/'continuous'")
    stop = old_worker.index("if __name__=='__main__':")
    worker_tail = '''    gate=prior_continuous()
    arm=ROOT/'resumed';arm.mkdir(mode=0o700)
    config=arm/'witness-config.json'
    checkpoint=ROOT/'continuous/train/checkpoint-20'
    cfg=dict(case='T02.sft',dataset_SHA256=stage['view']['SHA256'],binding=str(ROOT/'structural-binding.json'),
             checkpoint_step=20,checkpoint=str(checkpoint),checkpoint_files=gate['checkpoint20_files'],
             trainer_SHA256=runtime['trainer']['SHA256'])
    config.write_bytes(canonical(cfg))
    env=dict(os.environ,**plan['environment_overrides'])
    env.update(MASTER_ADDR='127.0.0.1',NATIVE_RESUME_CONFIG=str(config),
        NATIVE_RESUME_CONFIG_SHA256=sha(config.read_bytes()),NATIVE_RESUME_EPOCH=str(run['epoch']),
        NATIVE_RESUME_DEADLINE=str(run['native_deadline']),NATIVE_RESUME_MAX_SECONDS='2400')
    entry=dict(request_id=TRAINING_REQUEST,arm='resumed',epoch=run['epoch'],deadline=run['deadline'],
        step=step,native_deadline=run['native_deadline'],source_manifest=source,
        actual_updates=None,output_directory_initially_absent=True,admission_SHA256=admission_sha,actual_visible_PCI=pcis)
    (arm/'entry.json').write_bytes(canonical(entry))
    need(plan['command'][0]==str(ROOT/'framework/vlm/runtime/swift_entry.py'),'original common entry must be used')
    need(time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'admission expired before common exec')
    os.execve(sys.executable,[sys.executable,'-B',*plan['command']],env)


'''
    files["worker.py"] = exact((old_worker[:start] + worker_tail + old_worker[stop:]).encode(), [
        ("from control_contract import *", "from control_contract import *\nfrom contract import verify as verify_recovery, prior_continuous, STAGE_PIN"),
        ("source=verify('GPU')", "source=verify_recovery('GPU')"),
        ("run['source_manifest']==source", "run['operational_source_manifest']==source"),
        ("run['stage_complete']", "STAGE_PIN"),
        ("stage['source_manifest']==source", "stage['source_manifest']==run['source_manifest']"),
        ("plan=json.loads(read(ROOT/'continuous-plan.json',stage['plan']))", "plan=json.loads((D/'resumed-plan.json').read_bytes())"),
        ("choices=('continuous',)", "choices=('resumed',)")])
    files["control_contract.py"] = exact(question["control_contract.py"], [("29961", str(PORT))])
    files["contract.py"] = exact(files["contract.py"], [
        (OLD.name, LABEL), ("'results/" + DOMAIN.name + "'", "'results/" + QUESTION.name + "'"),
        ("CORE = R / 'incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source'", "CORE = D"),
        ("16081fc061703371d8e19012ded9e073a87f98c3", "2c470a93a42d8d9723f63c95e2a0644141c4ed7f"),
        ("d3e8a6432e89db845e527a0d32b9ff39139416d2fb08885babcb5eba67156eb1", QUESTION_MF),
        ("dict(bytes=151938, SHA256='f10d27753331d6ea085084af202a571a09880d69e86164e4c914cbcfbfe59fa0')",
         "dict(bytes=150566, SHA256='02c31f7cc5b5f5c6dcdc24e173fc2ac3ba18c2acf44c9c1a597827f7d3afff65')"),
        ("40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329",
         "9b7b5be546b9f6b4c259a949e1ef2b4948397032375c2b1556f8ddbc972a7642"),
        ("247fa3682a114542ac3640fbc726aaae", "2ea01ac64fa07c326d825f0e073b254b"),
        ("66e39b4b1aeb4c129646b5c0db3c33cb", DISPATCH),
        ("'results/training-gemma-usgs72-continuous25-supervisor-v2'", "'results/" + QUESTION.name + "-native'"),
        ("dict(bytes=7528,SHA256='ba9e7917eb04b298de2104aa5fce80f38b19aa39486d18a2f1f681f446c07a09')",
         "dict(bytes=8493,SHA256='f9956d84c779559d8b8b68de3ef24942fab35605832f99b209ac56802641a129')"),
        ("e033cf6ca72c8d7cd4ad73e39b11b463a4d98ab384cb59d4ce407a29b581bfbe", OUTCOME),
        ("34b621bffcb505620ef45b68acf3be778fc555face98d558038ef9a64dbdfeb1", NATIVE),
        ("ASSET/'continuous-independent-terminal.json'", "CONTINUOUS_ROOT/'independent-terminal.json'"),
        ("original851 native/outcome identity changed", "original1206 native/outcome identity changed"),
        ("29932", str(PORT))])
    files["controller.py"] = exact(files["controller.py"], [
        ("(D,D,'ro'),(CORE,CORE,'ro'),", "(D,D,'ro'),"),
        ("        (ASSET/'continuous-independent-terminal.json',ASSET/'continuous-independent-terminal.json','ro'),\n", ""),
        ("stage['training_request_id']==REQUEST", "stage['training_request']==REQUEST"),
        ("actual845 stage prerequisite missing", "question-only1203 stage prerequisite missing"),
        ("(runtime/'training-gemma-usgs-render72-native-CPU-v1/data',runtime/'training-gemma-usgs-render72-native-CPU-v1/data','ro'),",
         "(runtime/'training-gemma-usgs72-question-only-view800-v1',runtime/'training-gemma-usgs72-question-only-view800-v1','ro'),"),
        ("(runtime/'training-gemma-usgs-render72-exposure25-resume20-v1',runtime/'training-gemma-usgs-render72-exposure25-resume20-v1','ro'),",
         "(runtime/'training-gemma-usgs72-fresh-initial-domain25-v1',runtime/'training-gemma-usgs72-fresh-initial-domain25-v1','ro'),")])
    files["supervisor.py"] = exact(files["supervisor.py"], [
        ("    verify('GPU')", "    source = verify('GPU')"),
        ("stage['training_request_id']==REQUEST", "stage['training_request']==REQUEST"),
        ("original845 stage missing", "question-only1203 stage missing"),
        ("pin(CORE/'SOURCE_FILES.json')['SHA256']==CORE_MF_SHA", "pin(CORE/'SOURCE_FILES.json')==source")])
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=LABEL, request_id=DISPATCH, training_request_id="2ea01ac64fa07c326d825f0e073b254b",
                 state="SOURCE_READY_FOR_QUESTION_ONLY_CHECKPOINT20_RECOVERY", schema="GEMMA_USGS72_QUESTION_ONLY_CHECKPOINT20_RESUME_V1")
    claim["target"]["port"] = PORT
    claim["prior_actual_continuous"] = dict(GPU_step="1103606.1206", actual_updates=25, independent_read="1103606.1208",
                                             outcome_SHA256=OUTCOME, native_outcome_SHA256=NATIVE)
    claim["core"] = dict(current_recovery_source=True, original_training_source_manifest_SHA256=QUESTION_MF)
    files["CLAIM.json"] = encode(claim)
    for name, raw in files.items():
        if name.endswith(".py"):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    files["SOURCE_FILES.json"] = encode({"files": {name: pin(raw) for name, raw in files.items()}})
    output.mkdir()
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return dict(label=LABEL, request=DISPATCH, port=PORT, files=sorted(files), manifest=pin(files["SOURCE_FILES.json"]))


if __name__ == "__main__":
    print(json.dumps(materialize(BASE / LABEL), sort_keys=True))
