"""New Qwen CPU qualification for the exact existing 72 train rows. No gold/weights/GPU."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import time

DATA_SHA='a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd'
TEMPLATE_SHA='d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3'
VERSIONS={'ms-swift':'4.4.2','transformers':'5.12.1','qwen-vl-utils':'0.0.14'}


def need(ok,reason):
    if not ok: raise ValueError(reason)


def check_suffix(ids,labels,suffix_ids,eos_id):
    need(bool(suffix_ids) and eos_id in suffix_ids and len(ids)==len(labels) and
         ids[-len(suffix_ids):]==suffix_ids and
         all(label in (-100,token) for token,label in zip(suffix_ids,labels[-len(suffix_ids):])) and
         any(token==eos_id and label==token for token,label in zip(suffix_ids,labels[-len(suffix_ids):])),
         'actual complete native suffix including supervised EOS differs')


def check(dataset,metadata_only,output,deadline,context=None):
    need(os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)') and
         all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),
         'zero-GRES and explicit three GPU masks required')
    need({k:md.version(k) for k in VERSIONS}==VERSIONS,'original processor providers differ')
    need(not any(metadata_only.glob('*.safetensors')) and not any(metadata_only.glob('*.pt')),
         'CPU model view must contain metadata only')
    need(isinstance(context,dict) and context['request_id']=='c933ba7359f84e8b9b80ac40d2d30964' and
         context['deadline']-context['epoch']==2400 and
         0<context['epoch']<=time.time()<deadline<=context['deadline'],'exact original CPU request/epoch required')
    raw=dataset.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==DATA_SHA,'actual train72 source identity differs')
    rows=[json.loads(line) for line in raw.splitlines()]
    need(len(rows)==72,'full72 source denominator required')
    evidence=[]
    def progress():
        value=dict(planned=72,completed=len(evidence),missing=72-len(evidence),rows=evidence,
                   deadline=deadline,explicit_GPU_queries=0,actual_updates=0)
        temporary=output/'native72-progress.tmp'
        temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf8')
        temporary.replace(output/'native72-progress.json')
    progress()
    import torch
    from torch_provider import write_and_validate
    identity=write_and_validate(torch,output/'native72-torch-provider.json')
    need(identity['SHA256']=='9edf6b378be77655e0641130e18edfc32d4415d9355cf027bef3eba54aeeecc9',
         'actual qualified DTK Torch source differs')
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.templates import qwen
    need(hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest()==TEMPLATE_SHA,'Qwen template source differs')
    model,processor=get_model_processor(str(metadata_only),model_type='qwen3_vl',load_model=False,
        download_model=False,device_map='cpu')
    need(model is None,'CPU qualification returned a materialized model')
    template=get_template(processor,template_type='qwen3_vl',max_length=4096,max_pixels=262144,
        padding_side='right',padding_free=False,truncation_strategy='raise')
    template.set_mode('train')
    tokenizer=processor.tokenizer
    suffix=template.template_meta.suffix
    need(isinstance(suffix,list) and suffix and all(isinstance(x,str) for x in suffix),
         'actual Qwen suffix must be explicit text contexts')
    suffix_ids=tokenizer.encode(''.join(suffix),add_special_tokens=False)
    need(suffix_ids and tokenizer.eos_token_id in suffix_ids,'actual suffix lacks tokenizer EOS')
    config=json.loads((metadata_only/'config.json').read_bytes())
    image_id=config['image_token_id']
    encoded=[]
    started=time.monotonic()
    for index,row in enumerate(rows):
        need(time.time()<deadline,'original native CPU epoch exhausted')
        need(len(row.get('images',[]))==1 and
             [x.get('role') for x in row.get('messages',[])]==['user','assistant'],'train-only supervised row required')
        value=template.encode(row)
        ids,labels=value['input_ids'],value['labels']
        supervised=[i for i,x in enumerate(labels) if x!=-100]
        media=[i for i,x in enumerate(ids) if x==image_id]
        need(0<len(ids)==len(labels)<=4096 and media and supervised and supervised[0]>media[-1] and
             all(labels[i]==-100 for i in media) and all(ids[i]==labels[i] for i in supervised),
             'full4096 image/assistant supervision mask differs')
        check_suffix(ids,labels,suffix_ids,tokenizer.eos_token_id)
        pixels,grid=value['pixel_values'],value['image_grid_thw']
        need(pixels.device.type==grid.device.type=='cpu' and pixels.numel()>0 and
             tuple(grid.shape)==(1,3) and bool(torch.isfinite(pixels).all()),'native CPU vision fields invalid')
        record=dict(ordinal=index,source_line_SHA256=hashlib.sha256(raw.splitlines()[index]).hexdigest(),
                    full_length=len(ids),supervised_tokens=len(supervised),
                    shifted_supervised_tokens=sum(x!=-100 for x in labels[1:]),
                    complete_suffix_with_supervised_EOS=True,image_token_count=len(media),
                    pixel_shape=list(pixels.shape),pixel_dtype=str(pixels.dtype),grid=grid.tolist())
        encoded.append(value);evidence.append(record);progress()
    # The exact previously qualified RoPE factory is called under meta, never a weight loader.
    with torch.device('meta'):
        template.dummy_model,_=get_model_processor(str(metadata_only),model_type='qwen3_vl',
            return_dummy_model=True,download_model=False)
    parameters=list(template.dummy_model.parameters())
    need(parameters and all(x.device.type=='meta' for x in parameters),'native RoPE dummy model materialized weights')
    # Every row, not only two convenient lengths, traverses the native collator.
    for index,value in enumerate(encoded):
        need(time.time()<deadline,'CPU collator original deadline exhausted')
        batch=template.data_collator([value])
        labels,mask=batch['labels'],batch['attention_mask'].bool()
        need(tuple(labels.shape)==tuple(mask.shape)==(1,evidence[index]['full_length']) and
             int(mask.sum().item())==evidence[index]['full_length'] and
             bool((labels[~mask]==-100).all()),'native collator dropped or mislabeled full input')
    result=dict(state='NATIVE_QWEN_MULTIEVENT72_PASS',rows=72,labels_and_EOS_valid=72,
        full_sequence_retained=72,collator_rows=72,dataset_SHA256=DATA_SHA,
        template_SHA256=TEMPLATE_SHA,versions=VERSIONS,max_length=4096,max_pixels=262144,
        step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        actual_GPU_allocated=0,explicit_GPU_queries=0,actual_updates=0,
        request=context['request_id'],epoch=context['epoch'],deadline=context['deadline'],
        source_manifest_SHA256=context['source_manifest_SHA256'],claim_SHA256=context['claim_SHA256'],
        implicit_Torch_GPU_initialization='UNKNOWN_NOT_MEASURED',metadata_only_model_view=True,torch_provider=identity,
        native_seconds=time.monotonic()-started,planned_denominator=72,
        token_range=[min(r['full_length'] for r in evidence),max(r['full_length'] for r in evidence)],
        shifted_supervised_tokens=sum(r['shifted_supervised_tokens'] for r in evidence),
        per_row_counts_SHA256=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest())
    with (output/'native72-result.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,sort_keys=True,allow_nan=False)
    return result
