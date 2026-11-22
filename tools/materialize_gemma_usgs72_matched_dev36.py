"""Build the matched USGS effect scope from the qualified Gemma dev36 supervisor."""

import ast
from hashlib import sha256
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "training-gemma-generic64-dev36-effect-supervisor-v2"
BASE = REPO / "code/bw1000" / PARENT_NAME
BASE_SHA = "ae9a8fa80bcb28dac66235a2edc4f3a1bd212d09030cac6ba235ea7aab3f8636"
EFFECT = "training-gemma-usgs72-matched-dev36-effect-v1"
LABEL = "training-gemma-usgs72-matched-dev36-supervisor-v1"
QUESTION = "training-gemma-usgs72-question-only-control25-v1"
QUESTION_TRAINING = "2ea01ac64fa07c326d825f0e073b254b"
QUESTION_GATE = dict(bytes=8493, SHA256="f9956d84c779559d8b8b68de3ef24942fab35605832f99b209ac56802641a129")
QUESTION_CHECKPOINT = dict(bytes=2977, SHA256="04804488f5ab562243b537a4e6dc3cfad7263bd2bec6794cfe46e8eb2aa0baa4")
QUESTION_STAGE = dict(bytes=150566, SHA256="02c31f7cc5b5f5c6dcdc24e173fc2ac3ba18c2acf44c9c1a597827f7d3afff65")
DISPATCH, GENERATE, READ, SCORE = (sha256((EFFECT + "/" + role).encode()).hexdigest()[:32]
                                  for role in ("dispatch", "generate", "read", "score"))


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def exact(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Original source anchor differs: {old}")
    return text.replace(old, new, 1)


def materialize(output):
    original_manifest = (BASE / "SOURCE_FILES.json").read_bytes()
    if pin(original_manifest)["SHA256"] != BASE_SHA:
        raise ValueError("Qualified supervisor manifest changed")
    pins = json.loads(original_manifest)["files"]
    outer = ("assigned_kfd.py", "contract.py", "controller.py", "devices.py", "gpu_entry.py", "held_child.py",
             "keeper.py", "launch.py", "owned_processes.py", "read_terminal.py", "rss_monitor.py", "runtime_pci.py",
             "slurm_accounting.py", "snapshot.py", "CLAIM.json")
    inner = ("bind_ready.py", "content_vs_format.py", "contracts.py", "dev_inputs.py", "effect_contract.py",
             "frozen_score.py", "infer_dev36.py", "MODEL_BINDING.json", "read_terminal.py", "response_protocol.py",
             "score_after.py", "state_loader.py", "STRUCTURAL_BINDING.json", "target_contract.py", "CLAIM.json")
    files = {name: (BASE / name).read_bytes() for name in (*outer, *("core/" + name for name in inner))}
    if any(pin(raw) != pins[name] for name, raw in files.items()):
        raise ValueError("Qualified source member changed")

    namespaces = {PARENT_NAME: LABEL, "training-gemma-generic64-dev36-effect-v2": EFFECT,
                  "training-gemma-generic64-control25-inputs-v1-stage": QUESTION + "-stage",
                  "training-gemma-generic64-control25-supervisor-v1": QUESTION + "-native",
                  "training-gemma-generic64-control25-v1": QUESTION,
                  "9d3f3582e39948a28b166873e3ccb49d": DISPATCH, "8d6304c0da154864aa036e8129d53ac5": GENERATE,
                  "40c0a366cd6a4eeaa11fb689be92e688": READ, "30205be083fb49ebad1e55596796d4ce": SCORE}
    for name in ("contract.py", "controller.py", "core/effect_contract.py"):
        text = files[name].decode()
        for old, new in namespaces.items():
            text = text.replace(old, new)
        files[name] = text.encode()

    transforms = {
        "contract.py": [("29946", "29963")],
        "core/effect_contract.py": [
            ("dict(bytes=8507,SHA256='261fc4ebbd519bb5126243440c3f65691d5b3a795331c320ba37ce5b75d94184')", repr(QUESTION_GATE)),
            ("dict(bytes=2977,SHA256='0e484f8efe89b58db5c6251954b9bc5e25e5098a3b4429834954d7cc7cf83ae3')", repr(QUESTION_CHECKPOINT)),
        ],
        "core/bind_ready.py": [
            ("GEMMA_GENERIC64_CONTROL25_INDEPENDENT_TERMINAL_V1", "GEMMA_QUESTION_ONLY_CONTROL25_INDEPENDENT_TERMINAL_V1"),
            ("stage_raw=read(GENERIC/'stage-complete.json')", f"stage_raw=read(GENERIC/'stage-complete.json',{QUESTION_STAGE!r})"),
            ("actual1080 original checkpoint25 manifest differs", "actual1208 question-only checkpoint25 manifest differs"),
            ("captions_unmodified_format_confound=True", "same_train_FINAL_format_across_adapters=True"),
        ],
        "core/score_after.py": [("caption_format_differs_from_domain=True", "uses_same_train_FINAL_format=True")],
    }
    for name, replacements in transforms.items():
        text = files[name].decode()
        for old, new in replacements:
            text = exact(text, old, new)
        files[name] = text.encode()
    for name in ("core/effect_contract.py", "core/bind_ready.py", "core/content_vs_format.py", "core/infer_dev36.py"):
        text = files[name].decode().replace("generic25", "question25").replace("domain_vs_generic", "domain_vs_question")
        if name == "core/bind_ready.py":
            if text.count("2e96e8db20ee44c1a6a6a277e66f0d5e") != 2:
                raise ValueError("Original training request anchors differ")
            text = text.replace("2e96e8db20ee44c1a6a6a277e66f0d5e", QUESTION_TRAINING)
        for old, new in (("GENERIC_OP", "QUESTION_OP"), ("GENERIC_GATE", "QUESTION_GATE"), ("GENERIC_CP", "QUESTION_CP"),
                         ("GENERIC", "QUESTION"), ("generic_", "question_"), ("generic", "question")):
            text = text.replace(old, new)
        files[name] = text.encode()
    for name in ("CLAIM.json", "core/CLAIM.json"):
        claim = json.loads(files[name])
        for old in ("prior_v1_failed_preserved", "source_frozen_core_manifest_before_adoption", "v2_only_validation_delta",
                    "generic_actual_terminal", "caption_format_confound"):
            claim.pop(old, None)
        claim.update(request_id=DISPATCH, request=GENERATE, core_request=GENERATE, read_request=READ,
                     score_request=SCORE, schema="GEMMA_USGS72_MATCHED_DEV36_SINGLE_DCU_V1",
                     state="ROOT_ADOPTED_MATCHED_IMAGE_AND_QUESTION_CONTROL", same_train_FINAL_format=True,
                     question_actual_terminal=QUESTION_GATE, question_checkpoint25=QUESTION_CHECKPOINT)
        if name == "CLAIM.json":
            claim["port"] = 29963
        files[name] = encode(claim)
    files["core/SOURCE_FILES.json"] = encode({"files": {name: pin(files["core/" + name]) for name in inner}})
    files["SOURCE_FILES.json"] = encode({"files": {name: pin(raw) for name, raw in files.items()}})
    for name, raw in files.items():
        if name.endswith(".py"):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    output.mkdir()
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    return dict(label=LABEL, effect=EFFECT, files=sorted(files), manifest=pin(files["SOURCE_FILES.json"]),
                core=pin(files["core/SOURCE_FILES.json"]), dispatch=DISPATCH, generate=GENERATE, read=READ, score=SCORE)


if __name__ == "__main__":
    print(json.dumps(materialize(REPO / "code/bw1000" / LABEL), sort_keys=True))
