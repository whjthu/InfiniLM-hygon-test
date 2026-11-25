"""Use the frozen checker producer with real inputs and isolated fake processor tensors."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import owner
from contract import digest, encode


def child_module(name,**values):
    value=ModuleType(name)
    value.__dict__.update(values)
    return value


def exact_checker_grid(repo, private):
    source=repo/"code/bw1000/training-domain-pilot-v1"
    frozen=source/"check_textvqa_street_scene_native.py"
    assert digest(frozen.read_bytes())=="0ee5897963121c600b98339e48269a194cb9d45f0087c94d8970acf70597c02f"
    with patch.object(sys,"path",[str(source),*sys.path]):
        original=runpy.run_path(str(frozen))
    check=original["check"]
    globals_=check.__globals__
    source_path=private/"frozen-qwen-template.py"
    config_path=private/"fixture-model/config.json"
    cpu=SimpleNamespace(type="cpu")
    versions=original["SOURCE_VERSIONS"]
    n=0

    def template_encode(_):
        nonlocal n
        n+=1
        size=3+(n%2)
        ids=[99,*range(100+n*8,100+n*8+size-1)]
        return {"input_ids":ids,"labels":[-100,*ids[1:]],
                "pixel_values":SimpleNamespace(device=cpu,numel=lambda:24),
                "image_grid_thw":SimpleNamespace(device=cpu,shape=(1,3),tolist=lambda:[[1,2,4]])}

    class Attention:
        def __init__(self,lengths):
            self.shape=(2,max(lengths))
            self.lengths=lengths

        def bool(self):
            return self

        def sum(self,dim):
            assert dim==1
            return SimpleNamespace(tolist=lambda:self.lengths)

        def __invert__(self):
            return object()

    class Labels:
        def __init__(self,shape):
            self.shape=shape

        def __getitem__(self,key):
            return self

        def __eq__(self,integer):
            assert integer==-100
            return SimpleNamespace(all=lambda:True)

    def collate(rows):
        lengths=[len(row["input_ids"]) for row in rows]
        attention=Attention(lengths)
        return {"labels":Labels(attention.shape),"attention_mask":attention,"image_grid_thw":SimpleNamespace(shape=(2,3))}

    def get_model(*_,**arguments):
        if arguments.get("return_dummy_model"):
            return SimpleNamespace(parameters=lambda:[SimpleNamespace(device=SimpleNamespace(type="meta"))]),None
        assert arguments["load_model"] is False
        return None,object()

    template=SimpleNamespace(set_mode=lambda mode:None,encode=template_encode,data_collator=collate)
    fake={
        "torch":child_module("torch",cuda=SimpleNamespace(is_available=lambda:False),isfinite=lambda p:SimpleNamespace(all=lambda:True),
                             device=lambda value:nullcontext()),
        "swift":child_module("swift"),
        "swift.model":child_module("swift.model",get_model_processor=get_model),
        "swift.template":child_module("swift.template",get_template=lambda *a,**k:template),
        "swift.template.templates":child_module("swift.template.templates",qwen=SimpleNamespace(__file__=str(source_path))),
    }
    true_read=original["read_pinned"]
    def input_read(path,sha):
        if path==source_path:
            assert sha==original["TEMPLATE_SHA256"]
            return b"fixture-template"
        if path==config_path:
            assert sha==original["CONFIG_SHA256"]
            return b'{"image_token_id":99}'
        return true_read(path,sha)
    with patch.dict(sys.modules,fake),patch.dict(globals_,{"read_pinned":input_read}), \
         patch.object(original["importlib"].metadata,"version",side_effect=versions.__getitem__):
        report=check(private/"TRAIN_SWIFT.jsonl",private/"MEDIA_PINS.json",
                     "0b2ff0a69ccee6bde9d91445b66754de46b80691a1bf93e5d1fea74d3450947d",config_path.parent)
    assert report["processed"]==n==41 and report["collator_passed"]
    assert all(row["grid"]==[[1,2,4]] for row in report["rows"])
    return report


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_owner_consumes(run,source_sha,repo,temporary):
    private=repo/"local-only/specialized01-textvqa-original-train16-private-v1"
    report=exact_checker_grid(repo,private)
    prior=repo/"local-only/qwen-textvqa-stage-04a2af4f4b5b4e44bf5411cf32606bb9.outcome.json"
    previous=json.loads(prior.read_bytes())["response"]
    stage=temporary/"original1251"
    stage.mkdir()
    raw=encode(previous["receipt"])
    assert digest(raw)==previous["stage_identity_SHA256"]=="0e070b110cbf3a951f4ad6107c16d18a59e71ac4ba01173d29715a5511430f60"
    (stage/"staged.json").write_bytes(raw)
    log=stage/"worker.out"
    line=encode(report)
    log.write_bytes(b"EXACT_STEP=1194900.1251\n"+line)
    src=run/"src/trainer"
    evidence=load_module(src/"evidence.py","_actual_installed_evidence_v2")
    calls=[]
    with patch.dict(sys.modules,{"evidence":evidence}):
        prepare=load_module(src/"prepare.py","_actual_installed_prepare_v2")
        prepare.STAGE,prepare.DATASET,prepare.NATIVE_OUT=stage,private/"TRAIN_SWIFT.jsonl",log
        evidence.NATIVE_REPORT_SHA=digest(line.rstrip(b"\n"))
        def run_gate(argv,**kwargs):
            assert argv[1:4]==["-I","-B","-X"] and argv[4]=="pycache_prefix="+str(run/"never-populated-python-cache")
            assert argv[5]=="-c" and Path(argv[7])==src
            assert kwargs=={"timeout":25,"stdin":subprocess.DEVNULL,"check":True,"capture_output":True}
            calls.append(1)
            with patch.dict(sys.modules,{"prepare":prepare}),patch.object(sys,"argv",["-c",argv[7]]), \
                 patch.object(sys,"path",sys.path.copy()):
                exec(compile(argv[6],"<actual owner assets gate>", "exec"),{})
            return subprocess.CompletedProcess(argv,0,b"",b"")
        binary=stage/"apptainer"
        binary.write_bytes(b"fixture")
        sandbox=stage/"sandbox"
        sandbox.mkdir()
        with patch.object(owner,"APPTAINER",str(binary)),patch.object(owner,"SANDBOX",sandbox), \
             patch.object(owner.subprocess,"run",run_gate),patch.object(owner,"inventory",return_value={
                 "state":"FIXTURE_ONLY_ORIGINAL_MODEL_ASSETS","model_bytes":17545907231}):
            confirmed=owner.prepare_assets(run,source_sha)
        assert len(calls)==1 and confirmed["model_files"]==14
        index=prepare.stage_proof()
        assert len(index)==41 and all(row["grid"]==[1,2,4] for row in index.values())
        raw_rows=evidence.dataset_rows(private/"TRAIN_SWIFT.jsonl",private/"MEDIA_PINS.json")
        bad=[[1,2],[[1,2]],[[1,2,4],[1,2,4]],[[1,2,4,8]],[[1,0,4]],[[1,True,4]],"/absolute/path"]
        rejected=[]
        for grid in bad:
            rows=[dict(report["rows"][0],grid=grid),*report["rows"][1:]]
            try:
                evidence.fingerprints(rows,raw_rows)
            except ValueError as err:
                assert str(err)=="EXACT_1251_IMAGE_GRID_1_BY_3"
                rejected.append(str(err))
            else:
                raise AssertionError("invalid native grid accepted")
        try:
            evidence.fingerprints([dict(report["rows"][0],row=1),*report["rows"][1:]],raw_rows)
        except ValueError as err:
            assert str(err)=="NATIVE_ROW_FINGERPRINT_MISSING_OR_AMBIGUOUS"
            rejected.append(str(err))
        else:
            raise AssertionError("wrong native ordinal accepted")
        log.write_bytes(b"EXACT_STEP=1194900.1251\n"+line[:-30])
        try:
            prepare.stage_proof()
        except ValueError as err:
            assert str(err)=="EXACT_NATIVE_1251_REPORT_LINE"
            rejected.append(str(err))
        else:
            raise AssertionError("truncated .1251 report accepted")
    return {"installed_trainer_true_frozen_checker_fixture_rows":41,"owner_assets_real_stage_gate_calls":1,
            "fixture_uses_real_original_data_and_mock_processor":True,"wire_grid_shape":[1,3],
            "training_grid_shape":[3],"rejections":rejected}
