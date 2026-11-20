"""Actual frozen function invocation, intercepted before subprocess/training.

No image re-decoding, Torch, weights, tokenizers, VAE or official trainer execution.
The explicitly empty sentinel paths satisfy only path-existence admission.
"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace

from adapter import G, build_config

REQUEST='31a8d1b0bcee46548902a4cf7fe0fc3a'
ARCHIVE='code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
ARCHIVE_SHA='85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'
MEMBER='framework/generation/src/gencuda/backends/sdxl_training.py'
BACKEND_SHA='1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7'


class BoundaryObserved(Exception):
    pass


def main():
    origin=time.time()
    out=G/'local-only/training-sdxl-USGS-style18-canonical-adapter-v1-original'
    out.mkdir(exist_ok=False)
    raw=(G/ARCHIVE).read_bytes()
    assert len(raw)==738802 and hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        member=tar.getmember(MEMBER)
        assert member.isfile() and member.size==11576
        source=tar.extractfile(member).read()
    assert hashlib.sha256(source).hexdigest()==BACKEND_SHA
    common=json.loads((G/'code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json').read_bytes())
    registered=common['files'][MEMBER.removeprefix('framework/')]
    assert registered['SHA256']==BACKEND_SHA and registered['bytes']==len(source)
    names={'run_sdxl_training','_positive_config_int','_external_checkpoint'}
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==3
    # No modified implementation: compile exact AST nodes from the pinned archive.
    code=compile(ast.Module(body=functions,type_ignores=[]),MEMBER,'exec')
    captures=[]
    def intercept(command,**kwargs):
        captures.append(list(command))
        raise BoundaryObserved('native subprocess boundary reached; not executed')
    scope=dict(Path=Path,Any=object,re=__import__('re'),os=os,sys=sys,
               subprocess=SimpleNamespace(run=intercept,STDOUT=-2))
    exec(code,scope)
    weights=out/'EMPTY_WEIGHTS_SENTINEL';weights.mkdir()
    trainer=out/'EMPTY_SOURCE_SENTINEL/examples/text_to_image/train_text_to_image_lora_sdxl.py'
    trainer.parent.mkdir(parents=True);trainer.write_bytes(b'')
    config=build_config()
    try:
        scope['run_sdxl_training'](weights,config,out/'intercepted-training-output',out/'EMPTY_SOURCE_SENTINEL')
    except BoundaryObserved:
        pass
    else:
        raise AssertionError('canonical native command boundary not observed')
    assert len(captures)==1
    command=captures[0]
    def value(key):return command[command.index(key)+1]
    assert value('--train_data_dir')==config['train_data_dir'] and Path(value('--train_data_dir')).name=='train'
    assert value('--resolution')=='256' and value('--train_batch_size')=='1'
    assert value('--max_train_steps')=='25' and value('--gradient_accumulation_steps')=='1'
    assert value('--rank')=='4' and value('--mixed_precision')=='bf16'
    assert value('--lr_scheduler')=='constant' and value('--lr_warmup_steps')=='0'
    assert value('--gencuda_checkpoint_steps')=='20,25' and value('--seed')=='20261121'
    assert '--resume_from_checkpoint' not in command
    assert 'dev' not in Path(value('--train_data_dir')).parts
    # Confirm obsolete keys are still rejected by the real entry, not silently adapted.
    try:
        scope['run_sdxl_training'](weights,{**config,'dataset_root':'obsolete'},out/'reject',out/'EMPTY_SOURCE_SENTINEL')
    except ValueError as error:
        assert 'obsolete keys' in str(error)
    else:
        raise AssertionError('obsolete input contract unexpectedly accepted')
    assert len(captures)==1 and time.time()-origin<30
    result=dict(schema='SDXL_STYLE18_CANONICAL_ENTRY_CPU_V1',passed=True,request_id=REQUEST,
        backend_member=MEMBER,backend_bytes=len(source),backend_SHA256=BACKEND_SHA,
        archive_SHA256=ARCHIVE_SHA,common_commit=common['framework_commit'],
        accepted_canonical_entry_calls=1,rejected_obsolete_contract_calls=1,
        train_images=12,dev_images=6,dev_passed_to_training=False,blind_body_read=False,QA_or_gold_read=False,
        exact_canonical_AST_functions=sorted(names),native_subprocess_executed=False,empty_sentinel_paths_used=True,
        effective_native_scheduler='constant with 0 LR warmup; warmup_steps=5 means metrics exclusion only',
        config=config,wall_seconds=time.time()-origin,HTTP=0,Torch=0,GPU=0,weights_body_read=0,
        native_imagefolder_decoder_qualified=False,tokenizer_or_VAE_qualified=False,
        training_READY_increment=0,original_T06_FAIL_retained=True)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['passed','request_id','accepted_canonical_entry_calls','rejected_obsolete_contract_calls','native_subprocess_executed','training_READY_increment']}))


if __name__=='__main__':main()
