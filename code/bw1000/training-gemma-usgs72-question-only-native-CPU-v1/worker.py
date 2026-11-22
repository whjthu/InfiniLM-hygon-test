"""Future exact-target train72 question-only CPU qualification. No launcher or transport."""
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import stat
import sys
import time

from transform import R,TRAIN,TRAIN_PIN,convert_rows,canonical,need,sha
from audit_protocol import audit_rows

D=Path(__file__).resolve().parent
REQUEST='eef6ad53dab54c3994d70b3adf25a4fd'
LABEL='training-gemma-usgs72-question-only-native-CPU-v1'
ROOT=Path('/audit-run')
OUTPUT=ROOT/'audit'
PROCESSOR=Path('/gemma-processor')
FRAMEWORK=Path('/input-framework')
HOST_PROCESSOR=R+'/results/training-gemma-processor-assets-v1/models/gemma-3-4b-it'
HOST_FRAMEWORK=R+'/results/training-qwen-a68952c-swift-initial-CPU-v1/framework'
PROCESSOR_SHA='eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b'
FRAMEWORK_SHA='e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41'
VERSIONS={'torch':'2.10.0+das.opt1.dtk2604.2606161421.g91a580','transformers':'5.12.1',
          'ms-swift':'4.4.2','tokenizers':'0.22.2','Pillow':'11.3.0'}

class DeviceInitializationForbidden(BaseException):pass

def read(path,expected=None,cap=2*1024**2):
    p=Path(path);before=p.lstat()
    ident=lambda s:[getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
    need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'bounded regular permitted member required')
    with p.open('rb') as stream:
        need(ident(os.fstat(stream.fileno()))==ident(before),'opened source lifetime differs')
        raw=stream.read(cap+1)
        need(ident(os.fstat(stream.fileno()))==ident(before),'source changed during read')
    need(ident(p.lstat())==ident(before) and len(raw)==before.st_size and
         (expected is None or dict(bytes=len(raw),SHA256=sha(raw))=={k:expected[k] for k in ('bytes','SHA256')}),
         'exact source pin/lifetime differs')
    return raw

def dump(path,value):
    raw=canonical(value)
    need(len(raw)<=2*1024**2,'private metadata output bound')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(bytes=len(raw),SHA256=sha(raw))

def verify(source_sha):
    raw=(D/'SOURCE_FILES.json').read_bytes();need(sha(raw)==source_sha,'published source MF differs')
    for name,expected in json.loads(raw)['files'].items():
        need(Path(name).name==name,'flat exact source package required')
        body=(D/name).read_bytes();need(dict(bytes=len(body),SHA256=sha(body))==expected,'source member differs')
    claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['request']==REQUEST and claim['root_GO'] is claim['CPU_GO'] is True,'new CPU source scope not adopted')
    need(all(claim[k] is False for k in ('GPU_GO','training_GO','Model_GO','formal17_GO','READY')),'qualification drift')
    return dict(bytes=len(raw),SHA256=sha(raw))

def clock(epoch,deadline,now):
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (epoch,deadline,now)) and
         0<epoch<=now<deadline<=epoch+390,'one original CPU390 work epoch required')
    return deadline-now

def run(a):
    source=verify(a.source_manifest_sha);remaining=clock(a.epoch,a.deadline,time.time())
    need(os.name=='posix' and socket.gethostname().split('.')[0]=='f13r1n12' and
         os.environ.get('SLURM_JOB_ID')=='1103606' and os.environ.get('SLURM_STEP_ID','').isdecimal(),
         'exact original new target CPU step required, never run against local data')
    need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')) and
         os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)') and
         os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1',
         'hidden0GPU/offline CPU environment required')
    need(len(a.publication)==40 and not set(a.publication)-set('0123456789abcdef'),'exact source publication required')
    need('torch' not in sys.modules,'fresh CPU worker required')
    allowed=sorted(os.sched_getaffinity(0));need(len(allowed)>=4,'four CPU affinity unavailable')
    os.sched_setaffinity(0,set(allowed[:4]));need(len(os.sched_getaffinity(0))==4,'actual four CPU affinity failed')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original CPU390 deadline')))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    OUTPUT.mkdir(mode=0o700)
    result=dict(state='GEMMA_TRAIN72_QUESTION_ONLY_NATIVE_CPU_V1',request=REQUEST,source_manifest=source,
        publication=a.publication,epoch=a.epoch,deadline=a.deadline,step='1103606.'+os.environ['SLURM_STEP_ID'],
        node='f13r1n12',host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        passed=False,phase='source_train_conversion',planned=72,attempted=0,completed=0,failed=0,missing=72,
        full_encode_attempts=0,capped_encode_attempts=0,collator_attempts=0,source_train=TRAIN_PIN,
        source_images_referenced=12,actual_media_body_reads=0,actual_blank_images_created=0,
        actual_cpu_affinity=sorted(os.sched_getaffinity(0)),memory_limit='External Slurm8GiB; not enforced/measured by this worker',
        model_loaded=None,load_model_false_requested=False,GPU_init_attempts_blocked=0,
        explicit_device_count_or_available_calls=0,implicit_driver_initialization='UNKNOWN_NOT_MEASURED',
        actual_updates=0,dev_or_blind_inputs_labels_media_read=False,original511_qualification_inherited=False,
        GPU_GO=False,training_GO=False,Model_GO=False,formal17_GO=False,READY=False)
    dump(OUTPUT/'intent.json',result)
    records=[];failures=[];native_init=None;native_cuda=None
    def check():clock(a.epoch,a.deadline,time.time())
    try:
        original=read(Path(TRAIN),TRAIN_PIN,54246)
        rows=[json.loads(line) for line in original.splitlines()]
        converted,witness=convert_rows(rows)
        derived=b''.join(canonical(x) for x in converted)
        with (OUTPUT/'train72-question-only-private.jsonl').open('xb') as f:f.write(derived);f.flush();os.fsync(f.fileno())
        provenance=dump(OUTPUT/'source72-provenance-private.json',dict(source_file=TRAIN,source_pin=TRAIN_PIN,
            **witness,derived=dict(bytes=len(derived),SHA256=sha(derived))))
        result.update(source_provenance=provenance,derived_train=dict(bytes=len(derived),SHA256=sha(derived)))
        result['phase']='processor_metadata_verify'
        # Existing processor metadata-only view. No full model path/weights or images are exposed/used.
        raw=(D/'MODEL_PROCESSOR_FILES.json').read_bytes();need(sha(raw)==PROCESSOR_SHA,'original14 manifest differs')
        model_meta=json.loads(raw)['files'];need(len(model_meta)==14,'metadata14 denominator')
        need({p.name for p in PROCESSOR.iterdir()}==set(model_meta),'only the exact original14 processor metadata permitted')
        cache={}
        for name,expected in model_meta.items():
            check();need(Path(name).name==name and not name.endswith(('.safetensors','.pt','.bin')),'no weight member allowed')
            cache[name]=read(PROCESSOR/name,expected,64*1024**2)
        result['phase']='framework_runtime_verify'
        raw=(D/'FRAMEWORK_FILES.json').read_bytes();need(sha(raw)==FRAMEWORK_SHA,'original qualified393 manifest differs')
        framework=json.loads(raw)['files'];need(len(framework)==393,'original393 source denominator')
        for name,expected in framework.items():
            check();need(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe source member required')
            read(FRAMEWORK/name,expected)
        need({k:md.version(k) for k in VERSIONS}==VERSIONS,'actual original CPU runtime metadata differs')
        import numpy as np
        import torch
        native_cuda=torch.cuda
        need(torch.cuda.is_initialized() is False,'Python device context already initialized')
        native_init=torch.cuda._lazy_init
        def forbid(*_,**__):
            result['GPU_init_attempts_blocked']+=1
            raise DeviceInitializationForbidden('device initialization forbidden in metadata-only CPU qualification')
        torch.cuda._lazy_init=forbid
        torch.set_num_threads(2);random.seed(42);np.random.seed(42);torch.random.default_generator.manual_seed(42)
        sys.path.insert(0,str(FRAMEWORK))
        from vlm.runtime import first_update_audit
        need(Path(first_update_audit.__file__).resolve()==(FRAMEWORK/'vlm/runtime/first_update_audit.py').resolve(),
             'actual fingerprint source path differs')
        from swift.model import get_model_processor
        from swift.template import get_template
        result['phase']='native_processor_load';result['load_model_false_requested']=True
        model,processor=get_model_processor(str(PROCESSOR),model_type='gemma3_vision',load_model=False,
            download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
        result['model_loaded']=model is not None
        need(model is None,'metadata-only processor unexpectedly constructed model')
        result['phase']='native_question_only_template'
        options=dict(template_type='gemma3_vision',max_pixels=262144,padding_side='right',padding_free=False,truncation_strategy='raise')
        full=get_template(processor,max_length=None,**options);capped=get_template(processor,max_length=512,**options)
        full.set_mode('train');capped.set_mode('train')
        need(full.template_meta.template_type==capped.template_meta.template_type=='gemma3_vision','actual native Gemma template differs')
        declared=capped.template_meta.suffix
        need(declared and declared==full.template_meta.suffix,'actual native suffix metadata differs')
        suffix=list(capped._encode_context_list(declared,[1.]*len(declared))[0])
        need(suffix==list(full._encode_context_list(declared,[1.]*len(declared))[0]),'actual suffix tokens differ')
        configured=json.loads(cache['generation_config.json'])['eos_token_id']
        eos={configured} if type(configured) is int else set(configured);eos.add(processor.tokenizer.eos_token_id)
        need(eos and all(type(x) is int and x>=0 for x in eos),'actual pinned EOS metadata invalid')
        result['native_provider']=dict(template_class=type(capped).__qualname__,template_module=type(capped).__module__,
            template_source_path=inspect.getsourcefile(type(capped)),encode_source_path=inspect.getsourcefile(capped.encode),
            source_bodies_separately_read=False,suffix_tokens_SHA256=sha(canonical(suffix)),suffix_contains_actual_EOS=bool(set(suffix)&eos))
        result['phase']='native72_no_image_audit'
        with (OUTPUT/'native72-identities-private.jsonl').open('x',encoding='utf8',newline='\n') as stream:
            def event(kind,value):
                if kind=='row_attempt':result['attempted']+=1;result['missing']=72-result['attempted']
                elif kind=='full_encode_attempt':result['full_encode_attempts']+=1
                elif kind=='capped_encode_attempt':result['capped_encode_attempts']+=1
                elif kind=='collator_attempt':result['collator_attempts']+=1
                elif kind=='row_pass':records.append(value);result['completed']+=1
                elif kind=='row_failure':failures.append(value);result['failed']+=1
                stream.write(json.dumps(dict(event=kind,value=value),sort_keys=True,allow_nan=False)+'\n');stream.flush()
            audited=audit_rows(converted,full,capped,first_update_audit.fingerprint,suffix,eos,check,event)
            os.fsync(stream.fileno())
        need(torch.cuda.is_initialized() is False,'Python device context initialized during new CPU ABI')
        result.update(passed=audited['passed'],pixel_fields_absent=audited['passed'],phase='native72_completed',
            original72_ID_order_SHA256=witness['source_ID_order_SHA256'],original12_image_reference_hashes=witness['source_image_reference_SHA256'],
            maximum_input_tokens=max((r['input_tokens'] for r in records),default=None),
            total_input_tokens=sum(r['input_tokens'] for r in records),
            total_shifted_supervised_tokens=sum(r['shifted_supervised_tokens'] for r in records),
            actual_native_identities=dict(bytes=(OUTPUT/'native72-identities-private.jsonl').stat().st_size,
                SHA256=sha((OUTPUT/'native72-identities-private.jsonl').read_bytes())))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        dump(OUTPUT/'failure-private.json',dict(type=type(error).__name__,message=str(error)[:512],phase=result['phase'],
            attempted=result['attempted'],completed=result['completed'],failed=result['failed'],missing=result['missing']))
        result['failure_type']=type(error).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if native_init is not None:native_cuda._lazy_init=native_init
        result['missing']=72-result['attempted']
        result['source_rows_not_new_examples']=72
        dump(OUTPUT/'result.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('epoch','deadline'):p.add_argument('--'+key,type=float,required=True)
    p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha',required=True)
    a=p.parse_args();r=run(a)
    print(json.dumps({k:r[k] for k in ('request','passed','planned','attempted','completed','failed','missing','actual_updates','model_loaded')},sort_keys=True))
    raise SystemExit(0 if r['passed'] else 1)
