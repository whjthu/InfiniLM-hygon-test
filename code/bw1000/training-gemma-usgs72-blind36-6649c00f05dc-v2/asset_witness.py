"""Lock the production owner to the independently verified, unchanged .3538 assets."""
import os
from pathlib import Path
import re
import subprocess

from contract import LABEL, NODE, PARENT, R, need, resources_match

ORIGINAL_SOURCE=R/'incoming/training-gemma-usgs72-blind36-58925c79ef98-v1-source'
ORIGINAL_ASSET=R/'results/training-gemma-usgs72-blind36-58925c79ef98-v1/inference'
ORIGINAL_INPUTS=ORIGINAL_ASSET/'inputs-only'
ORIGINAL_AUDIT=R/'incoming/training-gemma-usgs72-blind36-58925c79ef98-v1-zero-gres-assets-audit-v2'
ORIGINAL_SOURCE_SHA='34380f1660e572ecbf232ee7e1ba1be672fd06c8789e2d603ca95e5c9b3a90bd'
PUBLIC_TERMINAL_SHA='08e8466cb6ef94fddd9c468e43ce61745a1284a153a48d97a5e23e9566a075ac'
INDEPENDENT_SHA='ff486bd07191cfb6c5621bef6e4fc9f10f3e782ec96d19361ab6961768cc2d0f'
OWNER_SHA='9079a6ff434acc1a2dbd1e517f646a0178c2967934d7e285008f0bb0bd95d123'
INTENT_SHA='858544327ecbb9694f56f04eeec867a17402f8a3557e6b2588ff29a1c7b8f599'
OWNER_REQUEST='2f1d0c62a2a4496f811d94f2f3f581e5'
OWNER_SOURCE='cf6256c446abb3201c322831936b857fdfea7f6f'
COMPANION_SHA='57b19373b70d87af43ddd1967df57df62f2c5f0176aa81eb87a51ac3f2247b77'
INPUT_PIN={'bytes':1203,'SHA256':'831a01dc1c047d8446847e1e93d1d1a3fabba068064066e8b6d291a2b72e2434'}
INIT_PIN={'bytes':119382791,'SHA256':'da110892d3f9cac70993d14e8ea1dfb39bae64631b08051007a276f94e067de7'}
INIT_IDENTITY=[49,10896934479,119382791,1795308436000000000,1795308436000000000]


def check_original():
    """Verify the original source and both installed .3538 owner objects, without loading models."""
    import json
    from core.effect_contract import pin, read
    raw=read(ORIGINAL_SOURCE/'SOURCE_FILES.json')
    need(pin(raw)['SHA256']==ORIGINAL_SOURCE_SHA,'original installed source manifest differs')
    manifest=json.loads(raw)['files']
    need(len(manifest)==33,'original source denominator differs')
    for relative, expected in manifest.items():
        name=Path(relative)
        path=ORIGINAL_SOURCE/name
        need(not name.is_absolute() and '\\' not in relative and '..' not in name.parts
             and path.resolve(strict=True)==path,'unexpected original source member')
        read(path,expected)
    checked={}
    decoded={}
    for name,digest in [('intent.json',INTENT_SHA),('outcome.json',OWNER_SHA)]:
        path=ORIGINAL_AUDIT/name
        need(path.resolve(strict=True)==path,'original audit path differs')
        record=read(path,cap=65536)
        actual=pin(record)
        need(actual['SHA256']==digest,'original .3538 installed owner object differs')
        checked[name]=actual
        decoded[name]=json.loads(record)
    intent=decoded['intent.json']
    outcome=decoded['outcome.json']
    provider=outcome['input_provider']
    need(intent['request']==outcome['request']==OWNER_REQUEST and outcome['intent_SHA256']==INTENT_SHA
         and outcome['publication']==OWNER_SOURCE and outcome['companion_manifest_SHA256']==COMPANION_SHA
         and outcome['owner_invocation_rc']==outcome['actual_GPU']==outcome['actual_model_calls']==outcome['gold_files_opened']==0
         and provider['actual_step']=='1103606.3538' and provider['input_manifest']==INPUT_PIN
         and provider['initial']==dict(INIT_PIN,observed_file_identity=INIT_IDENTITY),
         'original independently verified .3538 owner/initial/input evidence differs')
    return dict(original_source_manifest=pin(raw), owner_objects=checked,
                independent_SHA256=INDEPENDENT_SHA, terminal_SHA256=PUBLIC_TERMINAL_SHA)


def current_gpu_step(step, work_clock):
    need(re.fullmatch(PARENT+r'\.[0-9]+',step) is not None,'actual numeric inference step required')
    raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],text=True,
                                timeout=min(5,work_clock.left()))
    need(0<len(raw)<=2<<20 and len(raw.splitlines())<=8192,'unexpected controller response extent')
    selected=[]
    for line in raw.splitlines():
        tokens=[word.split('=',1) for word in line.split() if '=' in word]
        row=dict(tokens)
        if row.get('StepId')==step:
            exact={'StepId','Name','JobName','State','NodeList','TRES','AllocTRES'}
            keys=[key for key,_ in tokens if key in exact]
            need(len(keys)==len(set(keys)),'ambiguous current step fields')
            selected.append(row)
    need(len(selected)==1,'unique controller step record required')
    row=selected[0]
    labels=[row[k] for k in ('Name','JobName') if k in row]
    allocated=[row[k] for k in ('TRES','AllocTRES') if k in row]
    need(labels and all(name==LABEL+'-gpu' for name in labels)
         and row.get('State')=='RUNNING' and row.get('NodeList')==NODE
         and allocated and all(resources_match('gpu',tres) for tres in allocated),
         'actual same step one DCU, eight CPU and 64000 MiB required')
    return dict(step=step,node=NODE,one_DCU=True,cpu=8,mem_MiB=64000)


def verify_source_on_host(source, expected):
    import json
    from core.effect_contract import pin, read
    path=source/'SOURCE_FILES.json'
    raw=read(path)
    needed=pin(raw)
    need(needed==expected,'installed production manifest differs')
    actual=json.loads(raw)['files']
    need({file.relative_to(source).as_posix() for file in source.rglob('*') if file.is_file()}
         ==set(actual)|{'SOURCE_FILES.json'},'installed source member set differs')
    for relative,pinned in actual.items():
        target=source/relative
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts
             and target.resolve(strict=True)==target,'installed source member path differs')
        read(target,pinned)
    return needed
