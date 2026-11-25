"""One Gemma base load, actual851 initial/domain25/new question25, 108 returns before gold.

Only ROOT's bounded device supervisor may invoke. No SSH, Slurm, training, gold CLI or GPU default.
"""
import argparse
import copy
import functools
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

from effect_contract import *
from blind_inputs import TARGET,check_bundle
from slots import failed_slot,seal_arm
from asset_witness import INDEPENDENT_SHA, INIT_IDENTITY, INIT_PIN, INPUT_PIN, OWNER_SHA
from target_contract import validate_recipe,OBSERVED_ADAPTER_CONFIG_SHA256

ENGINE_PINS={
 'infer_engine/transformers_engine.py':'b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb',
 'infer_engine/infer_engine.py':'857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969',
 'infer_engine/utils.py':'5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644',
 'infer_engine/protocol.py':'f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e'}

def checked_adapter(path,expected,check):
    import hashlib
    before=path.lstat();need(identity(before)==expected['file_identity'],'actual checkpoint stat differs')
    h=hashlib.sha256();total=0
    with path.open('rb') as f:
        need(identity(os.fstat(f.fileno()))==identity(before),'adapter opened different lifetime')
        while True:
            check();raw=f.read(min(1<<20,before.st_size-total+1))
            if not raw:break
            total+=len(raw);need(total<=before.st_size,'adapter grew');h.update(raw)
        need(identity(os.fstat(f.fileno()))==identity(before),'adapter changed during SHA')
    need(identity(path.lstat())==identity(before) and total==expected['bytes'] and
         h.hexdigest()==expected['SHA256'],'actual checkpoint bytes/SHA differs')

def generate(a,binding,result,rows,predictions):
    def check():clock(a.epoch,a.deadline,time.time())
    check()
    binding_scope(binding,result['source_manifest'])
    need(binding['state']=='ACTUAL_THREE_ARM_GEMMA25_BINDING' and binding['request']==REQUEST and
         binding['source_manifest']==result['source_manifest'] and binding['initial_named_SHA256']==NAMED and
         binding['planned_total']==108 and binding['data_root']==TARGET and binding['blind_gold_read'] is False,
         'actual full ready metadata binding required before model import')
    need(binding['same_compute_host']==socket.gethostname() and binding['same_compute_boot_id']==
         Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'same compute mount/boot required, no device-ID relaxation')
    for path,value in binding['model'].items():
        need(identity(Path(path).lstat())==value['file_identity'],'actual model16 lifetime differs from stage SHA')
    framework=Path(binding['framework_root'])
    fw=json.loads(read(DOMAIN/'fresh-framework-files.json',binding['framework']))
    need(len(fw['files'])==396,'qualified396 required')
    for name,value in fw['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe framework member')
        check();read(framework/name,value)
    runtime=json.loads(read(DOMAIN/'runtime-binding.json',binding['runtime']))
    need({k:md.version(k) for k in runtime['versions']}==runtime['versions'],'actual native runtime versions differ')
    for name,wanted in ENGINE_PINS.items():
        need(pin(read(Path('/swift-runtime/site/swift')/name))['SHA256']==wanted,'new inference provider qualification differs')
    sys.path.insert(0,str(framework))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.infer_engine import AdapterRequest,InferRequest,RequestConfig,TransformersEngine
    from vlm.runtime.first_update_audit import fingerprint
    from state_loader import bounded_load
    need(torch.version.hip and not torch.version.cuda and torch.cuda.device_count()==1,'one actually allocated visible HIP device required')
    torch.cuda.set_device(0)
    manifest=json.loads(read(D/'STRUCTURAL_BINDING.json'))['trainable_manifest']
    expected={r['name'].replace('.default.','.'):r for r in manifest}
    need(len(expected)==476 and sum(r['numel'] for r in manifest)==29802496,'Gemma476 native target population required')
    initial=binding['initial']
    witness=binding['independent_asset_witness']
    need(binding['input_manifest']==INPUT_PIN and witness['independent_SHA256']==INDEPENDENT_SHA
         and witness['owner_objects']['outcome.json']['SHA256']==OWNER_SHA
         and {key:initial[key] for key in INIT_PIN}==INIT_PIN and initial['file_identity']==INIT_IDENTITY,
         'actual .3538 initial and independent asset witness required')
    payload=bounded_load(torch,initial['path'],initial,check)
    need(set(payload)=={'schema','seed','named_SHA256','parameters'} and payload['schema']=='GEMMA_TARGET_FRESH_INITIAL_V1' and
         payload['seed']==42 and payload['named_SHA256']==NAMED and
         fingerprint(payload['parameters'])['SHA256']==NAMED,'actual851 target-created initial named values differ')
    states={'base':{k.replace('.default.','.'):v for k,v in payload['parameters'].items()}}
    adapters={};configs={}
    for arm in ('domain25','question25'):
        cp=binding['checkpoints'][arm];root=Path(cp['root'])
        config=read(root/'adapter_config.json',cp['files']['adapter_config.json'])
        checked_adapter(root/'adapter_model.safetensors',cp['files']['adapter_model.safetensors'],check)
        states[arm]=load_file(str(root/'adapter_model.safetensors'),device='cpu')
        need(identity((root/'adapter_model.safetensors').lstat())==cp['files']['adapter_model.safetensors']['file_identity'],
             'adapter changed during native safetensors load')
        adapters[arm]=root;configs[arm]=config
    need(configs['domain25']==configs['question25'] and pin(configs['domain25'])['SHA256']==OBSERVED_ADAPTER_CONFIG_SHA256,
         'actual original byte-identical two25 adapter config differs')
    configs['base']=configs['domain25']
    initial_dir=a.output/'derived-initial476';initial_dir.mkdir(mode=0o700)
    (initial_dir/'adapter_config.json').write_bytes(configs['base'])
    save_file({k:v.contiguous() for k,v in states['base'].items()},str(initial_dir/'adapter_model.safetensors'))
    adapters['base']=initial_dir
    for arm in ARMS:
        need(set(states[arm])==set(expected),'actual arm476 keys differ')
        cfg=json.loads(configs[arm]);struct=json.loads(read(D/'STRUCTURAL_BINDING.json'))
        validate_recipe(cfg,struct)
        for name,value in states[arm].items():
            ref=expected[name]
            need(value.device.type=='cpu' and list(value.shape)==ref['shape'] and str(value.dtype)==ref['dtype'] and
                 bool(torch.isfinite(value).all()),'arm name/shape/dtype/finite differs')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    started=time.monotonic()
    model,processor=get_model_processor(binding['model_root'],model_type='gemma3_vision',load_model=True,
        download_model=False,device_map={'':0},torch_dtype=torch.bfloat16,attn_impl='sdpa')
    result['model_loads']=1;result['model_load_seconds']=time.monotonic()-started
    need(not hasattr(model,'peft_config') and {str(v.device) for v in model.parameters()}=={'cuda:0'},'one original Gemma base object required')
    template=get_template(processor,template_type='gemma3_vision',max_length=512,max_pixels=262144,
        padding_side='right',padding_free=False,truncation_strategy='raise')
    need(template.template_meta.template_type=='gemma3_vision','actual Gemma native template differs')
    engine=TransformersEngine(model,template=template,max_batch_size=1)
    config=RequestConfig(max_tokens=CAP,temperature=0,num_beams=1,n=1,stream=False,return_details=True,stop=[])
    effective=engine._prepare_generation_config(config)
    need(not effective.do_sample and effective.max_new_tokens==CAP and effective.num_beams==1,'fixed greedy256 policy differs')
    model.eval()
    for v in model.parameters():v.requires_grad_(False)
    current={};input_refs={}
    collate=template.data_collator;get_ids=template.get_generate_ids;generate_fn=template.generate
    @functools.wraps(collate)
    def collator(*args,**kwargs):
        value=collate(*args,**kwargs)
        need(value.get('labels') is None and 0<value['input_ids'].shape[-1]<=512,'full untruncated user-only512 native input required')
        current['collator_calls']+=1
        identity={k:{key:v for key,v in fingerprint(x).items() if key!='floating_l2'} for k,x in value.items()}
        if current['arm']=='base':
            input_refs[current['id']]=identity
        else:
            need(input_refs.get(current['id'])==identity,'same full user/image native fields required for all three arms')
        return value
    @functools.wraps(get_ids)
    def generated_ids(*args,**kwargs):
        value=get_ids(*args,**kwargs)
        need(current['raw_ids'] is None and value.ndim==2 and value.shape[0]==1,'one raw native generation required')
        current['raw_ids']=value[0].detach().cpu().tolist();return value
    @functools.wraps(generate_fn)
    def observed_generate(*args,**kwargs):
        cfg=kwargs['generation_config'];arm=current['arm'];name=current['adapter_name']
        need(kwargs.get('past_key_values') is None and kwargs.get('adapter_names')==[name] and
             not cfg.do_sample and cfg.max_new_tokens==CAP and cfg.num_beams==1,'actual native route/greedy/KV policy differs')
        # Dynamic adapter loading precedes template.generate in pinned Swift4.4.2.
        # Verify all476 live values BEFORE the arm's first generated forward.
        if not result['arms'][arm].get('actual476_pre_forward_equal'):
            actual=dict(engine.model.named_parameters())
            for key,value in states[arm].items():
                target=key.replace('.lora_A.','.lora_A.'+name+'.').replace('.lora_B.','.lora_B.'+name+'.')
                need(target in actual and torch.equal(actual[target].detach().cpu(),value),'actual476 active state differs before generation')
            need(engine.model.get_base_model() is model,'three arms must share one original base object')
            result['arms'][arm]['actual476_pre_forward_equal']=True
        need(not engine.model.training,'all arms actual eval mode required')
        value=cfg.eos_token_id;current['EOS']=value if isinstance(value,list) else [value]
        current['generate_calls']+=1;return generate_fn(*args,**kwargs)
    template.data_collator=collator;template.get_generate_ids=generated_ids;template.generate=observed_generate
    try:
        for arm in ARMS:
            check();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            name='gemma_blind36_'+arm;adapter=AdapterRequest(name=name,path=str(adapters[arm]))
            for ordinal,row in enumerate(rows):
                check()
                current.clear()
                current.update(arm=arm,ordinal=ordinal,raw_ids=None,EOS=None,adapter_name=name,
                               collator_calls=0,generate_calls=0,id=row['id'])
                started=time.monotonic()
                try:
                    answer,=engine.infer(
                        [InferRequest(messages=copy.deepcopy(row['messages']),
                                      images=[str(Path(TARGET)/row['images'][0])])],
                        config,use_tqdm=False,adapter_request=adapter)
                    choice,=answer.choices
                    need(current['collator_calls']==current['generate_calls']==1,'one exact native request required')
                    need(choice.token_ids==[v for v in current['raw_ids'] if v!=engine.tokenizer.pad_token_id],
                         'raw pre-padding EOS provenance differs')
                    record=dict(id=row['id'],arm=arm,ordinal=ordinal,text=choice.message.content,status='COMPLETED',
                        finish_reason=choice.finish_reason,raw_generated_token_ids=current['raw_ids'],raw_EOS_ids=current['EOS'],
                        native_seconds=time.monotonic()-started,**eos(current['raw_ids'],current['EOS']))
                    need(isinstance(record['text'],str) and len(record['text'])<=16384,'bounded output required')
                    dump(a.output/(arm+'-return-%02d.json'%ordinal),record)
                    predictions[arm][ordinal]=record
                except Exception as error:
                    predictions[arm][ordinal]=failed_slot(row,arm,ordinal,'ERROR',type(error).__name__)
                    if isinstance(error,TimeoutError):
                        raise
    finally:
        template.data_collator=collate;template.get_generate_ids=get_ids;template.generate=generate_fn

def main(a):
    source=verify('GPU');clock(a.epoch,a.deadline,time.time())
    binding_raw=read(a.binding,dict(bytes=a.binding_bytes,SHA256=a.binding_sha));binding=json.loads(binding_raw)
    admission=json.loads(read(a.admission,dict(bytes=a.admission_bytes,SHA256=a.admission_sha)))
    need(a.output==ROOT/'generated' and admission['request']==REQUEST and admission['passed'] is True and
         admission['epoch']==a.epoch and admission['deadline']==a.deadline and admission['source_manifest']==source and
         admission['fresh_exact_GPU_PCI_KFD_port_scope_satisfied'] is True and admission['actual_visible_devices']==1,
         'actual bounded ROOT device admission required; worker is not a launcher')
    need(admission['host']==socket.gethostname() and admission['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
         and 0<=time.monotonic_ns()-admission['monotonic_ns']<=45_000_000_000,
         'fresh same-host monotonic device admission required')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original900s inference')))
    signal.setitimer(signal.ITIMER_REAL,clock(a.epoch,a.deadline,time.time()))
    a.output.mkdir(mode=0o700)
    result=dict(request=REQUEST,epoch=a.epoch,deadline=a.deadline,source_manifest=source,binding=pin(binding_raw),
        passed=False,planned_each=36,planned_total=108,arms={arm:dict(planned=36,attempted=0,completed=0,missing=36) for arm in ARMS},
        model_loads=0,all108sealed_before_gold=False,blind_gold_read=False,dev_inputs_read=False,actual_updates=0,
        base_is_actual851_initial_LoRA=True,generation_cap=CAP,context_max512_no_truncation=True,
        Model_GO=False,formal17=False,format_confounded_caption_control=True)
    rows,input_manifest=check_bundle(Path(TARGET))
    need(input_manifest==binding['input_manifest'],'original input binding differs')
    result['input_manifest']=input_manifest
    predictions={arm:{} for arm in ARMS}
    dump(a.output/'generation-plan.json',result)
    try:generate(a,binding,result,rows,predictions)
    except BaseException as error:result['failure']=dict(type=type(error).__name__,private_reason=str(error)[:512])
    finally:
        for arm in ARMS:
            result['arms'][arm].update(seal_arm(a.output,arm,rows,predictions[arm]))
        result['all_prediction_files_closed']=True
        result['all108sealed_before_gold']=True
        result['passed']=bool('failure' not in result and result['model_loads']==1 and
            all(r.get('actual476_pre_forward_equal') is True for r in result['arms'].values()))
        dump(a.output/'generation-finalized.json',result)
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('binding','admission','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('binding-bytes','admission-bytes'):p.add_argument('--'+key,type=int,required=True)
    for key in ('binding-sha','admission-sha'):p.add_argument('--'+key,required=True)
    for key in ('epoch','deadline'):p.add_argument('--'+key,type=float,required=True)
    raise SystemExit(main(p.parse_args()))
