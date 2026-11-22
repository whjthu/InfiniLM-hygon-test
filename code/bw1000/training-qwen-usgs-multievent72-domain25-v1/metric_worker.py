"""One base load, dev36 under base/new25. No gold path or training operation."""
import argparse
import copy
import functools
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import time

from prepare_dev import TARGET,PINS,need,pin,check_rows

CAP=256
ENGINE_PINS={
 'infer_engine/transformers_engine.py':'b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb',
 'infer_engine/infer_engine.py':'857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969',
 'infer_engine/utils.py':'5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644',
 'infer_engine/protocol.py':'f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e',
}


def eos(ids,eos_ids):
    need(isinstance(ids,list) and 0<len(ids)<=CAP and all(type(x) is int and x>=0 for x in ids),
         'real unpadded generated IDs required')
    need(isinstance(eos_ids,list) and eos_ids and all(type(x) is int and x>=0 for x in eos_ids),
         'actual effective EOS IDs required')
    cap=len(ids)>=CAP
    return dict(natural_EOS=not cap and ids[-1] in eos_ids and not any(x in eos_ids for x in ids[:-1]),
                cap_hit=cap,generated_tokens=len(ids),generated_ids_SHA256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())


def dump(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())


def read(path,expected):
    before=path.stat();raw=path.read_bytes();after=path.stat()
    need(pin(raw)==expected and (before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_ino,after.st_size,after.st_mtime_ns),'exact metric input changed')
    return raw


def main(a):
    # The enclosing original eight-card lease remains held. This new process uses one visible device;
    # cost accounting must still report eight allocated devices, never a one-card performance result.
    need(time.time()<a.deadline and Path(TARGET)==a.data,'exact metric data/deadline required')
    ctx_raw=a.intent.read_bytes()
    need(hashlib.sha256(ctx_raw).hexdigest()==a.intent_sha,'original root intent SHA differs')
    ctx=json.loads(ctx_raw);root=a.intent.parent
    need(root.name=='training-qwen-usgs-multievent72-domain25-v1' and
         ctx['request_id']=='c933ba7359f84e8b9b80ac40d2d30964' and
         ctx['node']=='f13r1n10' and ctx['parent']=='1103606' and
         0<ctx['epoch']<=time.time()<a.deadline<=ctx['deadline'] and ctx['deadline']-ctx['epoch']==2400 and
         a.output==root/'effect/generated' and a.adapter==root/'continuous/train/checkpoint-25' and
         a.inventory==root/'continuous-checkpoint-25-files.json','exact original metric context differs')
    own=Path(__file__).parent
    need(hashlib.sha256((own/'SOURCE_FILES.json').read_bytes()).hexdigest()==ctx['source_manifest_SHA256'] and
         hashlib.sha256((own/'CLAIM.json').read_bytes()).hexdigest()==ctx['claim_SHA256'],'original source seal differs')
    claim=json.loads((own/'CLAIM.json').read_bytes())
    need(claim['root_GO'] is claim['metric_inference_GO'] is True,'metric inference scope not adopted')
    need(md.version('ms-swift')=='4.4.2' and md.version('transformers')=='5.12.1','qualified inference providers differ')
    for name,wanted in ENGINE_PINS.items():
        raw=(Path('/swift-runtime/site/swift')/name).read_bytes()
        need(hashlib.sha256(raw).hexdigest()==wanted,'exact public Swift infer provider differs')
    for name,expected in PINS.items():read(a.data/name,expected)
    rows,shape=check_rows(read(a.data/'dev/inputs.jsonl',PINS['dev/inputs.jsonl']))
    inventory=json.loads(a.inventory.read_bytes())
    need(inventory['global_step']==25 and inventory['member_count']==14,'actual new25 checkpoint inventory required')
    for name in ('adapter_config.json','adapter_model.safetensors'):
        # Stable full-SHA; no tensor or raw state is exported from this process.
        p=a.adapter/name;before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        after=p.stat()
        need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and
             dict(bytes=before.st_size,SHA256=h.hexdigest())==inventory['files'][name],'new25 saved adapter changed')
    a.output.mkdir(mode=0o700)
    from native72 import VERSIONS,TEMPLATE_SHA
    need({k:md.version(k) for k in VERSIONS}==VERSIONS,'actual Qwen inference environment differs')
    import torch
    from torch_provider import write_and_validate
    torch_identity=write_and_validate(torch,a.output/'torch-provider.json')
    need(torch_identity['SHA256']=='9edf6b378be77655e0641130e18edfc32d4415d9355cf027bef3eba54aeeecc9',
         'actual inference Torch source differs from qualified CPU')
    from safetensors.torch import load_file
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.templates import qwen
    from swift.infer_engine import AdapterRequest,InferRequest,RequestConfig,TransformersEngine
    from vlm.runtime.first_update_audit import fingerprint
    need(torch.cuda.device_count()==1 and torch.version.hip and
         hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest()==TEMPLATE_SHA,'actual one-visible-device Qwen template differs')
    states=load_file(str(a.adapter/'adapter_model.safetensors'),device='cpu')
    need(len(states)==504 and sum(x.numel() for x in states.values())==43646976 and
         all(bool(torch.isfinite(x).all()) for x in states.values()),'actual new25 adapter504 finite state required')
    loaded=time.monotonic()
    model,processor=get_model_processor(str(a.model),model_type='qwen3_vl',load_model=True,
        download_model=False,device_map={'':0},torch_dtype=torch.bfloat16,attn_impl='sdpa')
    need(not hasattr(model,'peft_config') and model.device==torch.device('cuda:0') and
         model.dtype==torch.bfloat16 and model.config._attn_implementation=='sdpa','original base infer load differs')
    template=get_template(processor,template_type='qwen3_vl',max_length=4096,max_pixels=262144,
        padding_side='right',padding_free=False,truncation_strategy='raise')
    engine=TransformersEngine(model,template=template,max_batch_size=1)
    config=RequestConfig(max_tokens=CAP,temperature=0,num_beams=1,n=1,stream=False,return_details=True,stop=[])
    effective=engine._prepare_generation_config(config)
    need(not effective.do_sample and effective.max_new_tokens==CAP and effective.num_beams==1,'frozen greedy cap256 changed')
    current={};collate=template.data_collator;get_ids=template.get_generate_ids;generate=template.generate
    @functools.wraps(collate)
    def observed_collator(*args,**kwargs):
        value=collate(*args,**kwargs)
        need(value.get('labels') is None and 0<value['input_ids'].shape[-1]<=4096,'answer-free full inference collator required')
        current['collator_calls']+=1
        current['input_identity']={k:fingerprint(v) for k,v in value.items()}
        return value
    @functools.wraps(get_ids)
    def observed_ids(*args,**kwargs):
        value=get_ids(*args,**kwargs)
        need(current['raw_ids'] is None and value.ndim==2 and value.shape[0]==1,'one actual generation result required')
        current['raw_ids']=value[0].detach().cpu().tolist();return value
    @functools.wraps(generate)
    def observed_generate(*args,**kwargs):
        cfg=kwargs['generation_config'];name=current['adapter_name']
        need(not cfg.do_sample and cfg.max_new_tokens==CAP and cfg.num_beams==1,'native greedy policy differs')
        need(kwargs.get('adapter_names')==[name] if name else kwargs.get('adapter_names') in (None,[]),
             'actual native adapter route differs')
        value=cfg.eos_token_id;current['EOS']=value if isinstance(value,list) else [value]
        current['generate_calls']+=1;return generate(*args,**kwargs)
    template.data_collator=observed_collator;template.get_generate_ids=observed_ids;template.generate=observed_generate
    result=dict(passed=False,planned_per_arm=36,planned_total=72,model_loads=1,
        initial_is_public_base_not_frozen_A100_LoRA=True,load_seconds=time.monotonic()-loaded,
        dev_gold_read=False,blind_read=False,allocated_DCU=8,visible_inference_DCU=1,
        generation_cap=CAP,arms={name:dict(completed=0,missing=36) for name in ('base','domain25')})
    input_refs={}
    try:
        for arm in ('base','domain25'):
            name=None if arm=='base' else 'usgs_multievent72'
            adapter=None if name is None else AdapterRequest(name=name,path=str(a.adapter))
            path=a.output/(arm+'-predictions-private.jsonl')
            with path.open('x',encoding='utf8') as f:
                for ordinal,row in enumerate(rows):
                    need(time.time()<a.deadline,'original whole native deadline exhausted')
                    current.clear();current.update(raw_ids=None,EOS=None,adapter_name=name,collator_calls=0,generate_calls=0)
                    request=InferRequest(messages=copy.deepcopy(row['messages']),images=[str(a.data/row['images'][0])])
                    answer,=engine.infer([request],config,use_tqdm=False,adapter_request=adapter)
                    choice,=answer.choices
                    record=dict(id=row['id'],ordinal=ordinal,arm=arm,text=choice.message.content,
                        status='COMPLETED',finish_reason=choice.finish_reason,raw_generated_token_ids=current['raw_ids'],raw_EOS_ids=current['EOS'])
                    # Preserve every real return before applying post-infer guards.
                    dump(a.output/(arm+'-return-%02d.json'%ordinal),record)
                    need(current['collator_calls']==current['generate_calls']==1,'one full native user-only generation required')
                    observation=eos(current['raw_ids'],current['EOS'])
                    need(choice.token_ids==[x for x in current['raw_ids'] if x!=engine.tokenizer.pad_token_id],
                         'raw returned token provenance differs')
                    if arm=='base':input_refs[row['id']]=current['input_identity']
                    else:
                        need(current['input_identity']==input_refs[row['id']],'base/domain full native inputs differ')
                        if ordinal==0:
                            actual=dict(engine.model.named_parameters())
                            for key,tensor in states.items():
                                target=key.replace('.lora_A.','.lora_A.'+name+'.').replace('.lora_B.','.lora_B.'+name+'.')
                                need(target in actual and torch.equal(actual[target].detach().cpu(),tensor),
                                     'actual active adapter differs from saved new25 tensors')
                            need(engine.model.get_base_model() is model,'updated arm changed base object')
                    record.update(observation)
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    result['arms'][arm].update(completed=ordinal+1,missing=35-ordinal)
                os.fsync(f.fileno())
            result['arms'][arm].update(file=path.name,**pin(path.read_bytes()))
        result.update(passed=True,all72sealed_before_gold=True)
    finally:
        template.data_collator=collate;template.get_generate_ids=get_ids;template.generate=generate
        result['all_prediction_files_closed']=True
        for arm in result['arms']:
            path=a.output/(arm+'-predictions-private.jsonl')
            if path.exists():result['arms'][arm].update(file=path.name,**pin(path.read_bytes()))
        dump(a.output/'generation-finalized.json',result)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','model','adapter','inventory','output','intent'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--intent-sha',required=True)
    p.add_argument('--deadline',type=float,required=True)
    raise SystemExit(main(p.parse_args()))
