"""One zero-GRES compute stage on existing approved assets; stdlib, no Torch/Slurm/network."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import sys
import time

from control_contract import *
from control_plan import derive_callback,make_plan

MAX_LOGICAL=12*1024**3


def make_view(raw):
    need(len(raw)==INPUT_BYTES and sha(raw)==INPUT_SHA,'actual original HIP64 source differs')
    lines=raw.splitlines(keepends=True)
    need(len(lines)==64 and all(x.endswith(b'\n') and x.strip() for x in lines),'full original64 line denominator required')
    rows=[json.loads(line) for line in lines]
    for row in rows:
        need([m.get('role') for m in row.get('messages',[])]==['user','assistant'] and
             len(row.get('images',[]))==1,'original generic supervision/image interface differs')
    need(len({r['images'][0] for r in rows})==64,'original64 distinct image paths required')
    indices=[i%64 for i in range(800)]
    view=b''.join(lines[i] for i in indices)
    ledger=dict(source_rows=64,source_SHA256=INPUT_SHA,physical_rows=800,
        original_index_by_physical_row=indices,original_row_SHA256=[sha(x) for x in lines],
        view_SHA256=sha(view),exposure_counts=[indices.count(i) for i in range(64)],
        original_line_bytes_retained=True,caption_target_format='original unmodified',
        new_distinct_examples=0,dev_or_blind_read=False)
    return view,ledger


def execute(origin,deadline,publication,source_manifest_sha):
    source=verify('CPU')
    need(source['SHA256']==source_manifest_sha,'published source manifest differs')
    clock(origin,deadline,time.time(),210)
    need('torch' not in sys.modules,'zero-GRES stage must not inherit an imported Torch module')
    node=socket.gethostname().split('.')[0]
    step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    need(node==NODE and os.environ.get('SLURM_JOB_ID')==PARENT and
         os.environ.get('SLURM_STEP_ID','').isdecimal() and len(os.sched_getaffinity(0))>=2 and
         os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)'),
         'exact original zero-GRES compute mount required, never infer HOST/compute st_dev equivalence')
    need(len(publication)==40 and set(publication)<=set('0123456789abcdef'),'publication format')
    ROOT.mkdir(mode=0o700)
    def save(name,value):
        raw=canonical(value);target=ROOT/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        return dict(bytes=len(raw),SHA256=sha(raw))
    save('stage-intent.json',dict(stage_request=STAGE_REQUEST,training_request=TRAINING_REQUEST,
        publication=publication,source_manifest=source,origin=origin,deadline=deadline,step=step,node=node))
    total=0;reads={}
    def check():clock(origin,deadline,time.time(),210)
    def bound(path,expected=None,cap=2<<20,body=True):
        nonlocal total
        path=Path(path);check();before=path.lstat()
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap and total+before.st_size<=MAX_LOGICAL,
             'bounded regular source member required')
        import hashlib
        h=hashlib.sha256();count=0;data=[]
        with path.open('rb') as f:
            need(identity(os.fstat(f.fileno()))==identity(before),'source open lifetime differs')
            while True:
                check();raw=f.read(min(1<<20,before.st_size-count+1))
                if not raw:break
                count+=len(raw);need(count<=before.st_size,'source grew beyond frozen length');h.update(raw)
                if body:data.append(raw)
            need(identity(os.fstat(f.fileno()))==identity(before),'source changed during exact read')
        need(count==before.st_size and identity(path.lstat())==identity(before),'source lifetime changed')
        pin=dict(bytes=count,SHA256=h.hexdigest());total+=count
        need(expected is None or pin=={k:expected[k] for k in ('bytes','SHA256')},'source exact bytes/SHA differs')
        reads[str(path)]=dict(**pin,file_identity=identity(before))
        return b''.join(data) if body else pin
    result=None
    try:
        # Existing original approved64 is verified, not downloaded or sent to the client.
        meta=(D/'ORIGINAL_INPUT_FILES.json').read_bytes()
        need(sha(meta)==SOURCE_INPUT_MANIFEST_SHA,'original67 assets identity differs')
        members=json.loads(meta)['files']
        need(len(members)==67 and len({m['name'] for m in members})==67 and
             sum(m['bytes'] for m in members)==10964794,'original approved67 denominator differs')
        original=None
        for item in members:
            rel=relative(item['name']);path=SOURCE_ASSETS.joinpath(*rel.parts)
            raw=bound(path,item,cap=4<<20,body=item['name']=='original.cuda.jsonl')
            if item['name']=='original.cuda.jsonl':original=raw
        hip=bound(SOURCE_ASSETS/'vlm_report_coco64.hip.jsonl',dict(bytes=INPUT_BYTES,SHA256=INPUT_SHA))
        remap={p['source']:(SOURCE_ASSETS/p['name']).as_posix() for p in members}
        def convert(value):
            if isinstance(value,str):return remap.get(value,value)
            if isinstance(value,list):return [convert(x) for x in value]
            if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
            return value
        need([convert(json.loads(x)) for x in original.splitlines()]==[json.loads(x) for x in hip.splitlines()],
             'original HIP64 is not the exact approved path-only remap')
        view,ledger=make_view(hip)
        gate=json.loads(bound(DOMAIN/'continuous-independent-terminal.json',GATE_PIN))
        need(gate['passed'] is gate['independent_passed'] is gate['accounting_C0'] is True and
             gate['request_id']=='247fa3682a114542ac3640fbc726aaae' and gate['actual_updates']==25 and
             gate['fresh_initial_named_SHA256']==FRESH_NAMED_SHA,'actual851 initial origin must be independently qualified')
        initial=dict(named_SHA256=FRESH_NAMED_SHA,rng={})
        for rank,record in enumerate(gate['rank_receipts']):
            need(record['rank']==rank,'original ordered eight-rank metadata required')
            folder=DOMAIN/'continuous/train/continuous-witness'/f'rank-{rank:02d}'
            obj=json.loads(bound(folder/'initialization.json',record['initialization']))
            need(obj['rank']==rank and obj['mode']=='TARGET_FRESH_DOMAIN_ONLY' and obj['target_created'] is True and
                 obj['reference_initial_read'] is False and obj['RNG_unchanged'] is True and
                 obj['named']['SHA256']==FRESH_NAMED_SHA and obj['named']['tensors']==476 and
                 obj['named']['elements']==29802496 and obj['named']['finite'] is True,
                 'original target-created initial/rank values differ')
            expected_names={'target-fresh-rng.pt'}|({'target-fresh-initial.pt'} if rank==0 else set())
            need(set(obj['artifacts'])==expected_names,'unexpected original initial artifacts')
            for name,pin in obj['artifacts'].items():bound(folder/name,pin,cap=128<<20,body=False)
            initial['rng'][str(rank)]=dict(file=obj['artifacts']['target-fresh-rng.pt'],RNG_local=obj['RNG_local'])
            if rank==0:initial['parameters']=obj['artifacts']['target-fresh-initial.pt']
        need(len(initial['rng'])==8,'all eight original rank RNG artifacts required')
        model=json.loads((D/'MODEL_BINDING.json').read_bytes())
        need(len(model['files'])==16 and model['actual_full_target_weight_SHA_verified'] is True,'existing full model16 proof missing')
        processor=json.loads((D/'ORIGINAL_PROCESSOR_FILES.json').read_bytes())['files']
        need(len(processor)==14 and all(name in model['files'] and
             {k:pin[k] for k in ('bytes','SHA256')}=={k:model['files'][name][k] for k in ('bytes','SHA256')}
             for name,pin in processor.items()),'original qualified processor14 is not the identical full-model subset')
        for name,pin in model['files'].items():
            need(relative(name).name==name,'model member is not flat')
            bound(Path(model['target_model_directory'])/name,pin,cap=6*1024**3,body=False)
        framework_raw=bound(DOMAIN/'fresh-framework-files.json')
        need(sha(framework_raw)==FRAMEWORK_SHA,'actual original396 manifest differs')
        files=json.loads(framework_raw)['files'];need(len(files)==396,'framework396 denominator')
        callback=derive_callback((D/'DOMAIN_CALLBACK_SOURCE.py').read_bytes(),sha(view))
        final={}
        for name,pin in files.items():
            relative(name);raw=bound(DOMAIN/'framework'/name,pin)
            if name=='vlm/plugins/gemma_domain_continuous.py':raw=callback
            target=ROOT/'framework'/name;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
            final[name]=dict(bytes=len(raw),SHA256=sha(raw))
        need(final['vlm/plugins/gemma_domain_continuous.py']['SHA256']==sha(callback),'control callback missing')
        runtime_raw=bound(DOMAIN/'runtime-binding.json')
        need(sha(runtime_raw)==RUNTIME_SHA,'existing native runtime binding differs')
        need({k:md.version(k) for k in ('torch','transformers','ms-swift','tokenizers','Pillow')}==
             json.loads(runtime_raw)['versions'],'current native providers differ')
        (ROOT/'view').mkdir()
        (ROOT/'view/train800-swift.jsonl').write_bytes(view)
        ledger_pin=save('view/exposure-ledger-private.json',ledger)
        fw_pin=save('fresh-framework-files.json',dict(files=final))
        (ROOT/'structural-binding.json').write_bytes((D/'STRUCTURAL_BINDING.json').read_bytes())
        (ROOT/'runtime-binding.json').write_bytes(runtime_raw)
        plan_pin=save('continuous-plan.json',make_plan((D/'ORIGINAL_PLAN.json').read_bytes(),ROOT.as_posix()))
        result=dict(state='GEMMA_GENERIC64_CONTROL_INPUTS_AND_TARGET_INITIAL_STAGED',passed=True,
            stage_request=STAGE_REQUEST,training_request=TRAINING_REQUEST,source_manifest=source,
            publication=publication,origin=origin,deadline=deadline,step=step,node=node,
            host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            source_rows=64,source_dataset_SHA256=INPUT_SHA,unique_media=64,physical_rows=800,
            view=dict(bytes=len(view),SHA256=sha(view)),ledger=ledger_pin,
            framework_manifest=fw_pin,runtime_binding=dict(bytes=len(runtime_raw),SHA256=sha(runtime_raw)),
            plan=plan_pin,reference_initial=initial,reads=reads,logical_read_bytes=total,
            native64_qualification='original160 unchanged exact64 rows/256 fields; no new encode',
            original_A100_initial_read=False,dev_or_blind_read=False,model_deserializations=0,
            Torch_imports=0,Torch_module_present_at_return='torch' in sys.modules,
            implicit_GPU_initialization='NOT_QUERIED',GPU_queries=0,actual_updates=0,READY=False)
        need(result['Torch_module_present_at_return'] is False,'stage unexpectedly imported Torch')
        stage_pin=save('stage-complete.json',result)
        return dict(**{k:v for k,v in result.items() if k not in ('reads','reference_initial')},stage_complete=stage_pin)
    except BaseException as exc:
        save('stage-failure.json',dict(stage_request=STAGE_REQUEST,training_request=TRAINING_REQUEST,
            passed=False,error_type=type(exc).__name__,private_reason=str(exc)[:512],
            planned_source_rows=64,planned_exposures=800,logical_read_bytes=total,
            actual_updates=0,GPU_queries=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--publication',required=True);p.add_argument('--source-manifest-sha256',required=True)
    a=p.parse_args();remaining=clock(a.epoch,a.deadline,time.time(),210)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original210s stage')))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    print(json.dumps(execute(a.epoch,a.deadline,a.publication,a.source_manifest_sha256),sort_keys=True),flush=True)
