"""New16 inference-only native T04 processor/collator. Never constructs model or opens weights."""
import argparse
import copy
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

PROMPT='<image>\nDescribe this image in one short sentence.'
VERSIONS={'torch':'2.8.0a0+5228986c39.nv25.6','transformers':'5.12.1','ms-swift':'4.4.2','peft':'0.19.1','accelerate':'1.14.0'}


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_slot(row,pin,i):
    need(pin['slot']==i and len(row['messages'])==1 and row['messages']==[dict(role='user',content=PROMPT)]
        and set(row)=={'id','messages','images'} and row['images']==[pin['name']]
        and row['id']=='coco:'+str(int(Path(pin['name']).stem)),'Answer-bearing/modified image slot rejected')


def device_fds():
    rows=[]
    for p in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(p)
        except OSError:continue
        if target=='/dev/kfd' or target.startswith('/dev/nvidia') or target.startswith('/dev/dri/renderD'):
            rows.append(dict(fd=int(p.name),device=target))
    return rows


def main(a):
    started=time.monotonic();initial=a.deadline-time.time()
    need(all(math.isfinite(v) for v in (a.epoch,a.deadline)) and a.epoch>0 and 0<initial<=a.deadline-a.epoch<=260,'Invalid original CPU epoch')
    need(len(os.sched_getaffinity(0))==2 and not {0,1}&set(os.sched_getaffinity(0)),'Actual2CPU affinity must exclude0/1')
    need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),'All GPU devices must be hidden')
    before_fds=device_fds();need(not before_fds,'GPU device FD already open before CPU preprocessing')
    need(not a.output.exists(),'Existing CPU result namespace; never rerun');a.output.mkdir()
    def dump(name,value):
        with (a.output/name).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    completed=[];phase='identity';blocked=[];expected_probe=False;probe_denied=[];device_denied=[];network_denied=[]
    def check():
        if min(a.deadline-time.time(),initial-(time.monotonic()-started))<=0:raise TimeoutError('Original CPU epoch exhausted')
    def alarm(*args):raise TimeoutError('Original CPU epoch exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,initial)
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo'):
            network_denied.append(event);raise RuntimeError('CPU preprocessing forbids network')
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if path=='/dev/kfd' or path.startswith(('/dev/nvidia','/dev/dri/renderD')):
                device_denied.append('python-device-open');raise PermissionError('GPU device open forbidden')
            if path.endswith(('.safetensors','.pt','.pth','.bin','.ckpt')):
                value=hashlib.sha256(os.fsencode(path)).hexdigest()
                (probe_denied if expected_probe else blocked).append(value)
                raise PermissionError('CPU preprocessing weight-file access forbidden')
    sys.addaudithook(audit)
    def progress(failed=False):
        tmp=a.output/'progress.tmp';tmp.write_text(json.dumps(dict(planned_rows=16,completed_rows=len(completed),failed=failed,phase=phase)),encoding='utf8');tmp.replace(a.output/'progress.json')
    progress()
    result=dict(passed=False,planned_rows=16,completed_rows=0,Dataset_GO=False,training_GO=False,GPU_GO=False)
    try:
        sf=json.loads((a.source/'SOURCE_PINS.json').read_bytes())
        need({p.name for p in a.source.iterdir() if p.is_file()}==set(sf)|{'SOURCE_PINS.json'},'Exact small CPU runtime source inventory differs')
        for name,pin in sf.items():
            check();p=a.source/name;need(p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],'Own source pin differs')
        fm=json.loads((a.source/'FRAMEWORK_FILES.json').read_bytes())
        need(fm['framework_commit']=='b641edb0ad0f5104f18ba4ab45a247fbe1f7c495' and len(fm['files'])==350,'Original350 framework identity differs')
        for name,pin in fm['files'].items():
            check();p=a.framework/name;need(not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],'Original frozen350 provider changed')
        need({k:md.version(k) for k in VERSIONS}==VERSIONS,'Original A100 native environment changed')
        providers=json.loads((a.source/'ORIGINAL_SOURCES.json').read_bytes())['files']
        need(len(providers)==7,'Original7 Swift provider denominator differs')
        for name,pin in providers.items():
            check();p=Path(md.distribution('ms-swift').locate_file(name));need(p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],'Actual original Swift provider differs')
        metadata=json.loads((a.source/'MODEL_METADATA.json').read_bytes());model=a.model_view
        need(len(metadata['files'])==14,'Original14 processor metadata denominator differs')
        need(model.resolve().is_relative_to(a.data.parent.resolve()) and not model.is_symlink()
            and {p.name for p in model.iterdir()}==set(metadata['files'])
            and all(p.is_file() and not p.is_symlink() for p in model.iterdir()),'New metadata-only model view must contain exactly14 approved files')
        for name,pin in metadata['files'].items():
            check();p=model/name;need(p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],'Original model/processor metadata changed')
        # Real Python open attempts stop in audit hook before OS open/read; no shard payload reads.
        expected_probe=True
        for path in [model/f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)]+[a.output/'forbidden-project-weight.pt']:
            try:
                with path.open('rb'):pass
            except PermissionError:pass
            else:raise RuntimeError('Executable Python file-open weight guard failed')
        expected_probe=False;need(len(probe_denied)==5,'Exact4shard plus project.pt guard denominator differs')
        raw_manifest=(a.data/'DATA_MANIFEST.json').read_bytes();need(hashlib.sha256(raw_manifest).hexdigest()==a.data_manifest_sha256,'Private data manifest SHA differs')
        manifest=json.loads(raw_manifest);raw=(a.data/'inputs.jsonl').read_bytes()
        need(sha(a.data/'inputs.jsonl')==manifest['inputs']['SHA256'] and len(raw)==manifest['inputs']['bytes'],'Frozen answer-free inputs changed')
        rows=[json.loads(line) for line in raw.splitlines()]
        need(len({p['name'] for p in manifest['images']})==len({p['SHA256'] for p in manifest['images']})==len({r['id'] for r in rows})==16,'Duplicate image/content/input identity')
        need(len(rows)==len(manifest['images'])==16 and manifest['references_read']==0 and manifest['labels_in_inputs'] is False,'Fixed16 answer-free denominator differs')
        for i,(row,pin) in enumerate(zip(rows,manifest['images'])):
            check();validate_slot(row,pin,i)
            p=a.data/pin['name'];need(p.resolve().is_relative_to(a.data.resolve()) and not p.is_symlink()
                and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],'Original staged image identity differs')
            from PIL import Image
            with Image.open(p) as image:need(image.format=='JPEG' and image.size==(pin['width'],pin['height']),'Original JPEG dimensions differ')
        phase='native-processor-inference';progress()
        sys.path.insert(0,str(a.framework))
        import numpy as np
        import torch
        need(not torch.cuda.is_initialized(),'Device context already initialized');torch.set_num_threads(2)
        def no_device(*args,**kwargs):device_denied.append('lazy_init');raise RuntimeError('GPU initialization forbidden')
        torch.cuda._lazy_init=no_device
        def no_weight_api(*args,**kwargs):blocked.append('weight-load-API');raise PermissionError('Any Torch/safetensors load forbidden in processor-only scope')
        torch.load=no_weight_api;torch.serialization.load=no_weight_api
        import safetensors
        import safetensors.torch
        safetensors.safe_open=no_weight_api;safetensors.torch.safe_open=no_weight_api;safetensors.torch.load_file=no_weight_api
        random.seed(42);np.random.seed(42);torch.random.default_generator.manual_seed(42)
        from vlm.plugins import llava_onevision
        from swift.model import get_model_processor
        from swift.template import get_template
        empty,processor=get_model_processor(str(model),model_type='llava_onevision1_5',load_model=False,download_model=False,device_map='cpu',attn_impl='eager',torch_dtype=torch.bfloat16)
        need(empty is None and processor.model_meta.template=='llava_onevision1_5','Processor constructed model or changed template')
        template=get_template(processor,template_type=None,max_length=4096,max_pixels=262144,padding_side='right',padding_free=False,truncation_strategy='raise')
        template.set_mode('transformers')
        need(getattr(template,'mode',None)=='transformers','Actual native inference mode not observable as transformers')
        need(template.sequence_parallel_size==1 and not template.padding_free,'Unexpected SP/padding mode')
        method_providers={}
        for name,method in [('encode',template.encode),('collator',template.data_collator),('set_mode',template.set_mode)]:
            path=Path(inspect.getsourcefile(method)).resolve();pin=next((v for n,v in providers.items() if Path(md.distribution('ms-swift').locate_file(n)).resolve()==path),None)
            need(pin and sha(path)==pin['SHA256'],'Native encode/collator unbound provider')
            method_providers[name]=dict(module=method.__module__,name=method.__qualname__,file_SHA256=pin['SHA256'],method_source_SHA256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest())
        from vlm.runtime.first_update_audit import fingerprint
        def fields(value):
            result={}
            for name,v in sorted(value.items()):
                pin=fingerprint(v);need(pin['finite'] is True,'Nonfinite native processed input')
                result[name]={k:x for k,x in pin.items() if k!='floating_l2'}
                if isinstance(v,torch.Tensor):need(v.device.type=='cpu','Device tensor in CPU preprocessing')
            return result
        config=json.loads((model/'config.json').read_bytes());image_token=config['image_token_id']
        for i,(row,pin) in enumerate(zip(rows,manifest['images'])):
            check();request=dict(messages=row['messages'],images=[str(a.data/pin['name'])])
            encoded=template.encode(request);need(isinstance(encoded,dict) and encoded.get('labels') is None,'Inference output contains labels')
            ids=encoded['input_ids'];need(0<len(ids)<=4096,'Complete inference sequence exceeds4096; no truncation')
            encoded_fields=fields(encoded);collated=template.data_collator([copy.deepcopy(encoded)])
            need(collated.get('labels') is None and 'logits_to_keep' not in collated,'Inference collator fabricated labels/logit shortening')
            actual=collated['input_ids'];need(actual.device.type=='cpu' and tuple(actual.shape)==(1,len(ids)),'Collator shortened/padded single input')
            need(torch.equal(actual[0],torch.as_tensor(ids,dtype=actual.dtype,device='cpu')),'Collator altered native input IDs')
            count=int((actual==image_token).sum());need(count>0,'Native image tokens missing')
            need('pixel_values' in collated and collated['pixel_values'].numel()>0,'Native pixel input missing')
            completed.append(dict(slot=i,source_image=dict(name=pin['name'],bytes=pin['bytes'],SHA256=pin['SHA256']),input_id=row['id'],sequence_length=len(ids),image_tokens=count,labels_None=True,
                encoded_fields=encoded_fields,collated_fields=fields(collated),full_input_retained=True))
            del encoded,collated;progress()
        after_fds=device_fds()
        check();need(not torch.cuda.is_initialized() and not blocked and not device_denied and not network_denied and not after_fds,'Forbidden operation attempted')
        dump('input-identities-private.json',completed);dump('native-providers.json',method_providers)
        result.update(passed=True,state='NEW16_T04_NATIVE_INFERENCE_CPU_ONLY',completed_rows=16,
            actual_template_type=template.template_meta.template_type,actual_mode=getattr(template,'mode',None),set_mode_argument='transformers',
            token_range=[min(v['sequence_length'] for v in completed),max(v['sequence_length'] for v in completed)],
            identities_SHA256=sha(a.output/'input-identities-private.json'),providers_SHA256=sha(a.output/'native-providers.json'),
            version_pins=VERSIONS,framework_commit=fm['framework_commit'],framework_files=350,
            metadata_files_verified=14,Swift_provider_files_verified=7,
            labels_None_rows=16,full_rows_retained=16,seed=42,max_length=4096,max_pixels=262144,
            original64_train_mode_qualification_inherited=False,new_inference_mode_qualified_here=True,
            raw_processed_payload_saved=False,weight_payload_bytes_read=0,weight_guard_expected_denied=5,
            unexpected_weight_open_or_load_attempts=blocked,network_attempts=network_denied,device_initialization_attempts=device_denied,
            model_loaded=False,model_forward_called=False,CUDA_initialized=False,GPU_cards=0,
            actual_model_path=str(model),original_metadata_source_root=metadata['model_root'],
            new_model_view_exact_metadata_files=14,weight_files_in_model_view=0,
            original_model_loaded=False,metadata_view_is_CPU_input_scope_not_formal_recipe_change=True,
            device_FDs_before=before_fds,device_FDs_after=after_fds,
            photographer_attribution='PENDING',scorer_calls=0,training_updates=0,
            original80_reference_captions_accessed=False)
    except BaseException as error:
        progress(True);dump('failure-private.json',dict(error_type=type(error).__name__,phase=phase,planned_rows=16,completed_rows=len(completed)))
        result.update(state='NEW16_NATIVE_CPU_FAILURE_RETAIN_ALL_SLOTS',error_type=type(error).__name__,phase=phase,completed_rows=len(completed),
            unexpected_weight_open_or_load_attempts=blocked,weight_guard_expected_denied=len(probe_denied),network_attempts=network_denied,device_initialization_attempts=device_denied)
    result.update(epoch=a.epoch,deadline=a.deadline,wall_seconds=time.monotonic()-started,
        owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])),
        actual_CPU_affinity=sorted(os.sched_getaffinity(0)),peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dump('CPU-result.json',result);print(json.dumps({k:result[k] for k in ('passed','state','planned_rows','completed_rows','Dataset_GO','training_GO','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--framework',type=Path,required=True);p.add_argument('--data-manifest-sha256',required=True)
    p.add_argument('--model-view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    raise SystemExit(main(p.parse_args()))
