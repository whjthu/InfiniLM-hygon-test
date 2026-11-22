"""Derive a separate native Gemma control from the completed generic supervisor."""

import ast
from hashlib import sha256
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "code/bw1000/training-gemma-generic64-control25-inputs-v1"
BASE_PIN = "5e3054150b49d7a980a9b749e6b5e5f8caa74a47f7676d9dda35e6475000fbaf"
NAME = "training-gemma-usgs72-question-only-control25-v1"
LABEL = NAME + "-native"
BASE_LABEL = "training-gemma-generic64-control25-supervisor-v1"
STAGE = NAME + "-stage"
VIEW = "a2d2411778774856889797c3f1851b6f5df7fe67fd6a81ff113ab03d3da3d144"
LEDGER = "9e2ab684c5ae9079d6e593388da64ab729d56a7dc4c2d94b1385da4d48a2eaa6"
TRAIN, DISPATCH, STAGE_REQUEST = (sha256((NAME + "/" + role).encode()).hexdigest()[:32]
                                  for role in ("training", "dispatch", "stage"))


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def rewrite(source, replacements):
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Ambiguous source anchor: {old}")
        source = source.replace(old, new, 1)
    return source


VIEW_VALIDATION = '''        staged_view = R/'results/training-gemma-usgs72-question-only-view800-v1'
        view = bound(staged_view/'train800-swift.jsonl', dict(bytes=477994, SHA256=VIEW_SHA))
        ledger_raw = bound(staged_view/'exposure-ledger-private.json')
        need(sha(ledger_raw)==EXPOSURE_LEDGER_SHA, 'original question exposure ledger differs')
        ledger = json.loads(ledger_raw)
        paired = bound(DOMAIN/'view/train800-swift.jsonl', cap=2<<20)
        need(sha(paired)=='74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc',
             'original image-arm exposure view differs')
        view_lines, paired_lines = view.splitlines(), paired.splitlines()
        need(len(view_lines)==len(paired_lines)==800 and ledger['view_SHA256']==VIEW_SHA
             and ledger['source_index_by_physical_row']==[i%72 for i in range(800)],
             'original complete exposure order differs')
        ids=set()
        for left,right in zip(view_lines,paired_lines,strict=True):
            text,image=json.loads(left),json.loads(right)
            need(len(image.pop('images'))==1 and image['messages'][0]['content'].startswith('<image>\\n'),
                 'original one-image arm shape differs')
            image['messages'][0]['content']=image['messages'][0]['content'][len('<image>\\n'):]
            need(text==image and not set(text)&{'images','videos','audios','objects'},
                 'question, label or source order changed beyond removing training images')
            ids.add(text['id'])
        need(len(ids)==72, 'original 72 distinct source IDs required')
'''


def materialize(output):
    raw = (BASE / "SOURCE_FILES.json").read_bytes()
    if pin(raw)["SHA256"] != BASE_PIN:
        raise ValueError("native supervisor original manifest differs")
    source_pins = json.loads(raw)["files"]
    names = ("DOMAIN_CALLBACK_SOURCE.py", "MODEL_BINDING.json", "ORIGINAL_PLAN.json", "ORIGINAL_PROCESSOR_FILES.json",
             "STRUCTURAL_BINDING.json", "contract.py", "control_contract.py", "control_initial.py", "control_plan.py",
             "controller.py", "devices.py", "held_child.py", "launch.py", "owned_processes.py", "read_terminal.py",
             "rss_monitor.py", "slurm_accounting.py", "snapshot.py", "stage.py", "supervisor.py", "worker.py")
    files = {name: (BASE / name).read_bytes() for name in (*names, "CLAIM.json")}
    if any(pin(content) != source_pins[name] for name, content in files.items()):
        raise ValueError("original native sleeve bytes differ")

    def change(name, *replacements):
        files[name] = rewrite(files[name].decode(), replacements).encode()

    change("control_contract.py", ("training-gemma-generic64-control25-v1", NAME),
           ("2e96e8db20ee44c1a6a6a277e66f0d5e", TRAIN), ("5c892c45f21f4f678174025a60ca4b23", STAGE_REQUEST),
           ("29944", "29961"), ("INPUT_BYTES=26817", "INPUT_BYTES=43014"),
           ("1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f",
            "21637fd30cab21ef005c64d6eddb09657e6933fd7817c75cba75622a1aaa79fa"),
           ("NATIVE64_RESULT_SHA='5c1c4ac14f68892a211254de805451941c78fd344b8d73a878c57bb30612014b'",
            f"VIEW_SHA='{VIEW}'\nEXPOSURE_LEDGER_SHA='{LEDGER}'"))
    change("contract.py", (BASE_LABEL, LABEL), ("training-gemma-generic64-control25-v1", NAME),
           ("2e96e8db20ee44c1a6a6a277e66f0d5e", TRAIN), ("c2c9327d509249e4b329f6ece13d2a5e", DISPATCH),
           ("training-gemma-generic64-control25-inputs-v1-stage", STAGE), ("29944", "29961"),
           ("GEMMA_GENERIC64_CONTROL_INPUTS_AND_TARGET_INITIAL_STAGED", "GEMMA_QUESTION_ONLY_CONTROL_INPUTS_STAGED"),
           ("value['source_rows']==64", "value['source_rows']==72"), ("value['unique_media']==64", "value['unique_media']==0"),
           ("value['source_dataset_SHA256']=='1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f'",
            f"value['view']['SHA256']=='{VIEW}'"))
    change("control_plan.py", ("gemma_generic_continuous_witness", "gemma_question_only_continuous_witness"),
           ("    ids,labels=kwargs.get('input_ids'),kwargs.get('labels')", """    media = ('pixel_values', 'pixel_values_videos', 'image_sizes', 'image_grid_thw', 'video_grid_thw')
    if any(kwargs.get(field) is not None for field in media):
        raise ValueError('question-only training unexpectedly reached an image input')
    ids,labels=kwargs.get('input_ids'),kwargs.get('labels')"""),
           ("experiment='generic64-control25',source_unique_rows=64", "experiment='usgs72-question-only-control25',source_unique_rows=72"),
           ("caption_target_format='original unmodified; differs from domain FINAL answers'", "answer_target_format='same verbatim FINAL labels as image arm'"))
    change("controller.py", ("(DOMAIN,DOMAIN,'ro'),(SOURCE_ASSETS,SOURCE_ASSETS,'ro'),",
                            "(DOMAIN,DOMAIN,'ro'),\n        (runtime/'training-gemma-usgs72-question-only-view800-v1',runtime/'training-gemma-usgs72-question-only-view800-v1','ro'),"),
           ("gemma-generic-control25", "gemma-question-only-control25"))
    change("supervisor.py", ("gemma_generic_continuous_witness", "gemma_question_only_continuous_witness"),
           ("original845 stage missing", "matched question-only stage missing"))
    change("worker.py", ("source_rows=64,physical_exposures=800", "source_rows=72,physical_exposures=800"))
    change("read_terminal.py", ("GEMMA_GENERIC64_CONTROL25_INDEPENDENT_TERMINAL_V1", "GEMMA_QUESTION_ONLY_CONTROL25_INDEPENDENT_TERMINAL_V1"),
           ("unique_train_QA=64,train_media=64", "unique_train_QA=72,train_media=0"))
    source = files["stage.py"].decode()
    source = source[:source.index("def make_view(raw):")] + source[source.index("def execute("):]
    start, stop = source.index("        # Existing original approved64"), source.index("        gate=json.loads(")
    files["stage.py"] = (source[:start] + VIEW_VALIDATION + source[stop:]).encode()
    change("stage.py", ("GEMMA_GENERIC64_CONTROL_INPUTS_AND_TARGET_INITIAL_STAGED", "GEMMA_QUESTION_ONLY_CONTROL_INPUTS_STAGED"),
           ("source_rows=64,source_dataset_SHA256=INPUT_SHA,unique_media=64", "source_rows=72,source_dataset_SHA256=INPUT_SHA,unique_media=0"),
           ("native64_qualification='original160 unchanged exact64 rows/256 fields; no new encode'",
            "native72_qualification='original1197 exact72 text-only rows; original image-arm order verified'"),
           ("planned_source_rows=64", "planned_source_rows=72"))
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=LABEL, port=29961, request_id=DISPATCH, training_request=TRAIN,
                 training_request_id=TRAIN, stage_request=STAGE_REQUEST, stage_label=STAGE,
                 source_rows=72, unique_media=0, format_confounded=False,
                 schema="GEMMA_USGS72_QUESTION_ONLY_CONTROL25_SCOPE_V1",
                 state="ROOT_ADOPTED_PENDING_ZERO_GRES_STAGE", target_calls_this_turn=0,
                 gate="Match the original image-arm exposure order, answers and initial state on the compute node. Require new Slurm CPU stage and eight-device admission.")
    files["CLAIM.json"] = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode()
    for name, content in files.items():
        if name.endswith(".py"):
            ast.parse(content, filename=name, feature_version=(3, 10))
    output.mkdir()
    for name, content in files.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    manifest = (json.dumps({"files": {name: pin(content) for name, content in files.items()}},
                           indent=2, sort_keys=True) + "\n").encode()
    with (output / "SOURCE_FILES.json").open("xb") as stream:
        stream.write(manifest)
    return {"manifest": pin(manifest), "files": sorted([*files, "SOURCE_FILES.json"]), "claim": claim}


if __name__ == "__main__":
    print(json.dumps(materialize(REPO / "code/bw1000" / NAME), sort_keys=True))
