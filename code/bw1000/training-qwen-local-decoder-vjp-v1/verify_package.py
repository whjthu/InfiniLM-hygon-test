"""Verify real target sources and assets before reserving a training step."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
from scope import CONTINUOUS, ROOT

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
BASE = R/"training-qwen-a68952c-swift-initial-CPU-v1"
STAGE = BASE/"source"
ORIGINAL = R/"training-qwen-continuous-branches21-hip-v2"
FRAMEWORK = ORIGINAL/"framework"
MODEL = R/"training-qwen-original-public-model-v1/model"
INITIAL = R/"training-qwen-initial-trainables-v1/initial.pt"
ASSETS = R/"training-vlm-original-assets-v1"
SWIFT = R/"training-vlm-swift-offline-runtime-v1"
VISION = R/"training-qwen-vision014-CPU-v2/vision"
PROCESSOR = R/"training-minicpm-hygon-processor-r2/cpu/runtime-site"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(path, digest, size=None):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and (size is None or before.st_size == size), str(path)
    got = sha(path)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    assert identity(before) == identity(after) and got == digest, str(path)


def verify():
    src = ROOT/"src"
    intent = json.loads((ROOT/"intent.json").read_bytes())
    exact(src/"SOURCE_FILES.json", intent["source_manifest_SHA256"])
    for name, pin in json.loads((src/"SOURCE_FILES.json").read_bytes())["files"].items():
        exact(src/name, pin["SHA256"], pin["bytes"])
    exact(R/"training-qwen-a68952c-swift-initial-CPU-v4/outcome.json",
          "f396253981fd63f9f997f7c05383d2dd37bc9943079a0a172db6e1b747cfacbb")
    inputs=json.loads((src/"RESUME_INPUTS.json").read_bytes())
    bound=inputs["terminal"]
    exact(ORIGINAL/"outcome.json", bound["SHA256"])
    mf=ORIGINAL/"src/FRAMEWORK.json"
    bound=inputs["framework_manifest"]
    exact(mf, bound["SHA256"], bound["bytes"])
    frozen=json.loads(mf.read_bytes())
    assert frozen["commit"]=="096157e7b63f7e9cd60cb53f31b10f2810ac2ddc"
    files=frozen["files"]
    assert len(files)==406 and {p.relative_to(FRAMEWORK).as_posix()
            for p in FRAMEWORK.rglob("*") if p.is_file()}==set(files)
    for name,pin in files.items():
        path=FRAMEWORK/name
        exact(path, pin["SHA256"], pin["bytes"])
        assert stat.S_IMODE(path.stat().st_mode)==pin["mode"], name
    checkpoint=ORIGINAL/"train/checkpoint-20"
    pins={**inputs["checkpoint20"], **inputs["extra_files"]}
    assert len(pins)==17 and {p.name for p in checkpoint.iterdir()}==set(pins)
    for name,pin in pins.items():
        exact(checkpoint/name,pin["SHA256"],pin["bytes"])
    assert json.loads((checkpoint/"trainer_state.json").read_bytes())["global_step"]==20
    pin = json.loads((src/"ASSETS.json").read_bytes())
    assert {p.name for p in MODEL.iterdir()} == set(pin["model"]) and len(pin["model"]) == 14
    for name, file in pin["model"].items():
        exact(MODEL/name, file["SHA256"], file["bytes"])
    for name, file in pin["images"].items():
        exact(ASSETS/name, file["SHA256"], file["bytes"])
    exact(ASSETS/"vlm_report_coco64.hip.jsonl", "1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f", 26817)
    exact(ASSETS/"plugins/swift_callback_nvidia_baseline_v2.py", "d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8", 4692)
    exact(INITIAL,"2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f",174821899)
    for name, digest in pin["staged_source"].items():
        exact(STAGE/name,digest)
    exact(SWIFT/"site/swift/cli/main.py", "d84a44f9911c01c9af83db19baf2068ad39a8ec926f6e2c58f6f69779bfdad68")
    exact(SWIFT/"site/swift/cli/sft.py", "97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0")
    exact(VISION.parent/"outcome.json", "235851389a1a6a39c1cf5762c89af2ce77802ae7ed45b45365f5863524a983f3")
    vision = json.loads((src/"VISION_FILES.json").read_bytes())
    assert len(vision)==7 and {p.relative_to(VISION).as_posix()
        for p in VISION.rglob("*") if p.is_file()} == set(vision)
    for name, file in vision.items():
        exact(VISION/name, file["SHA256"], file["bytes"])
    transformer = PROCESSOR/'transformers'
    providers = {
        'integrations/sdpa_attention.py': ('87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564', 4851),
        'models/qwen3_vl/modeling_qwen3_vl.py': ('f39b3621fea3f757a01d2e00ebd3a2be3516b7df3048d862a365988aba9fa994', 74723),
        'modeling_utils.py': ('98a17fb9c5a04127f770da4d686932a5c5321cfb25833c057df8ab9f9577ce43', 267972),
        'trainer.py': ('c1a56423fcfcf9cfec6847467ffb2e2c8a9a9e8cc1836b82c87ed0c81e504be0', 217203),
        'modeling_layers.py': ('f56c853f48f1d9a86026aa0f88ed1c33e629483e4a84b933cd4dddbcb00e1500', 11566),
    }
    for relative, (digest, length) in providers.items():
        exact(transformer/relative, digest, length)
    prior = CONTINUOUS.parent
    original_pins = {
        prior/'outcome.json': 'be4aed751c7493df441e67b2f4183d307bf84802467ca0b70cd6c8d63614f45a',
        prior/'native-outcome.json': '3a2ea0e5dab51377128de13a8253ecc1778f4cfe7a38980a817496d5a24d9fa9',
        CONTINUOUS/'native-outcome.json': 'a22c51549395c8f912a79e5d59c93f04fc81db5168b9050811e73b89cc61f041',
        prior/'intent.json': '17af52e9fd8285516f4cbe363c3eb687c95ea2a4f0241116cc50b9716051fc68',
    }
    for path, digest in original_pins.items():
        exact(path, digest)
    old_outcome = json.loads((prior/'outcome.json').read_bytes())
    old_native = json.loads((prior/'native-outcome.json').read_bytes())
    continuous = json.loads((CONTINUOUS/'native-outcome.json').read_bytes())
    assert old_outcome['keeper_passed'] is False and old_outcome['after']['kfd_holder_count'] == 0
    assert old_outcome['after']['owned_processes_remaining'] == [] and not (prior/'resumed').exists()
    assert old_native['error'] == {'type':'AttributeError','message':"module 'hashlib' has no attribute 'file_digest'"}
    assert continuous['normal_native_exit'] and set(continuous['rank_record_counts'].values()) == {8}
    for source in ('candidate_plugin.py','candidate_cli.py'):
        original=(prior/'src'/source).read_bytes()
        expected=(original.replace(b'state.global_step != 25',b'state.global_step != 20') if source=='candidate_plugin.py' else original)
        assert (src/source).read_bytes() == expected
    from checkpoint_pins import checkpoints
    followup=R/'training-qwen-math-sdpa-resume20-v1'
    exact(followup/'outcome.json','06bc62709cc88b459f11c73fdb701afafd9e742e16c867cde7de10ed32bcafe5')
    exact(followup/'native-outcome.json','14f174a0aff8c666dd4152f07ead9748f8e45f19772527aafafa9887298199b6')
    prior_resume=json.loads((followup/'native-outcome.json').read_bytes())
    assert prior_resume['normal_native_exit']
    with (ROOT/'checkpoint20-pins.json').open('x', encoding='utf-8') as f:
        pins=checkpoints(CONTINUOUS/'train/checkpoint-20', 20)
        assert pins == prior_resume['result']['checkpoint20']
        json.dump(pins, f, allow_nan=False)
    pins=json.loads((src/'REFERENCE_PINS.json').read_bytes())
    assert set(pins)=={str(i) for i in range(8)}
    input_references={}
    for rank, pin in pins.items():
        p=followup/'resumed/train/resume-witness'/f'rank-{int(rank):02d}'/'result.json'
        exact(p,pin['SHA256'],pin['bytes'])
        native=json.loads(p.read_bytes())
        selected=native['inputs'][:2]
        assert native['rank']==int(rank) and len(selected)==2 and all(r['global_step']==20 and r['full_input']['finite'] for r in selected)
        input_references[rank]=[m['full_input']['SHA256'] for m in selected]
    with (ROOT/'input-references.json').open('x',encoding='utf-8') as f:
        json.dump(input_references,f)
    assert shutil.disk_usage(ROOT).free >= 80*1024**3
    return dict(passed=True, framework_files=406, checkpoint20_files=17, model_files=14,
                model_bytes_rehashed=sum(p["bytes"] for p in pin["model"].values()),
                image_files=len(pin["images"]), GPU_calls=0)
