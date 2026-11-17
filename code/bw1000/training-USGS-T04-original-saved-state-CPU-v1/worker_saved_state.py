"""CPU-only typed original29 audit. Existing qualified safe loader, never unsafe fallback."""
import argparse
from collections import Counter
import gc
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from failure_private import clipped_utf8,failure_record
from common import D,canonical,check_state_contract,need,pinned,sha,verify_source


def main(a):
    verify_source();claim=json.loads((D/'CLAIM.json').read_bytes());need(claim['root_GO'] is True and claim['CPU_GO'] is True,'Root CPU admission required')
    need(a.epoch<=time.time()<a.deadline<=a.epoch+140,'Original180s CPU epoch differs')
    mono=time.monotonic()+a.deadline-time.time()
    def check():
        if min(a.deadline-time.time(),mono-time.monotonic())<=0:raise TimeoutError('Saved-state original epoch exhausted')
    def alarm(*_):raise TimeoutError('Saved-state original epoch alarm')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,a.deadline-time.time()))
    need(len(os.sched_getaffinity(0))==4 and not set(os.sched_getaffinity(0))&{0,1},'Actual4CPUexcluding0/1 required')
    need(not a.output.exists(),'Original CPU namespace exists');a.output.mkdir(mode=0o700)
    def save(name,value):
        raw=canonical(value);need(len(raw)<=128*1024,'Metadata output capped');p=a.output/(name+'.tmp');p.write_bytes(raw);p.replace(a.output/name)
    def device_fds():
        values=[]
        for p in Path('/proc/self/fd').iterdir():
            try:t=os.readlink(p)
            except OSError:continue
            if t=='/dev/kfd' or t.startswith(('/dev/nvidia','/dev/dri/renderD')):values.append(int(p.name))
        return values
    need(not device_fds(),'CPU alreadyowns deviceFD');denied=[]
    def audit(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:denied.append('network');raise PermissionError('Saved-state CPU offline')
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=os.fsdecode(args[0])
            if path=='/dev/kfd' or path.startswith(('/dev/nvidia','/dev/dri/renderD')):denied.append('device');raise PermissionError('CPU no device access')
    sys.addaudithook(audit)
    c=json.loads((D/'PINNED_STATE.json').read_bytes());paths=check_state_contract(c)
    result=dict(passed=False,planned_state_files=29,checked_state_files=0,planned_state_bytes=c['state_total_bytes'],original_failure_preserved=True,
        historical_rank0_receipt_missing=True,original_training_accepted=False,GPU_resume_qualified=False,actual_updates=0,training_GO=False,GPU_GO=False,Model_GO=False,
        unsafe_loader_fallback=False,full_optimizer_vs_original_live='UNAVAILABLE_NO_COMPLETE_LIVE_DIGEST',checkpoints={})
    seen={};start=time.monotonic();phase='admission'
    def progress():save('progress.json',dict(planned_state_files=29,completed_state_files=len(seen),planned_state_bytes=c['state_total_bytes'],completed_state_bytes=sum(v['bytes'] for v in seen.values()),actual_updates=0))
    progress()
    try:
        plan=json.loads((D/'PLAN.json').read_bytes());need(a.framework==Path(plan['framework_root']),'Original400 root exact')
        mf=json.loads((D/'FRAMEWORK_BINDING.json').read_bytes());need(len(mf['files'])==400,'Original400 member denominator')
        for name,pin in mf['files'].items():check();pinned(a.framework/name,pin)
        for row in plan['providers']:check();pinned(Path(row['canonical_path']),row)
        need({k:md.version(k) for k in plan['versions']}==plan['versions'],'Qualified A100 versions differ')
        binding=json.loads(pinned(D/'T04-native-binding.json',plan['native_binding']));manifest=binding['trainable_manifest'];need(len(manifest)==504 and sum(r['numel'] for r in manifest)==43646976,'Canonical504manifest differs')
        phase='import'
        import torch
        need(not torch.cuda.is_initialized(),'CPU audit CUDA initialized');torch.set_num_threads(4)
        def forbid(*_,**__):denied.append('CUDA_lazy_init');raise PermissionError('CPU saved-state neverGPU')
        torch.cuda._lazy_init=forbid
        sys.path.insert(0,str(a.framework))
        from vlm.runtime.cuda_resume_witness import trusted_load,tree_fingerprint,optimizer_steps
        from vlm.runtime.initial_trainables import named_parameter_sha256
        from vlm.runtime.resume_witness import rng_fingerprint
        expected_manifest={r['name']:r for r in manifest};expected_adapter={n.replace('.default.','.'):r for n,r in expected_manifest.items()};need(len(expected_adapter)==504,'Unambiguous canonical adapter names')
        expected_shapes=Counter((tuple(r['shape']),r['numel']) for r in manifest)
        def load(root,name,pin):
            check();result['active_state_file']=dict(path=str(Path(root)/name),bytes=pin['bytes'],SHA256=pin['SHA256']);value=trusted_load(root,name,pin,check);seen[str(Path(root)/name)]=pin;progress();return value
        def tensor(t):need(isinstance(t,torch.Tensor) and t.device.type=='cpu' and (not t.is_floating_point() or bool(torch.isfinite(t).all())),'Finite CPU tensor required')
        phase='initial'
        init=c['initial'];v=load(str(Path(init['path']).parent),'initial.pt',init)
        need(set(v)=={'parameters','RNG','manifest','global_step'} and v['global_step']==0 and v['RNG']['restored'] is False,'Original initialpayload schema')
        parameters=v['parameters'];need(len(parameters)==504 and set(parameters)==set(expected_manifest),'Initial exact504 names')
        for name,t in parameters.items():
            tensor(t);r=expected_manifest[name];need(list(t.shape)==r['shape'] and t.numel()==r['numel'] and str(t.dtype)==r['dtype'],'Initialcanonicalshape/dtype')
        actual_named=named_parameter_sha256(list(parameters.items()));need(actual_named==c['initial_named_SHA256'],'Initial actualnamedSHA differs')
        need(sum(t.numel() for t in parameters.values())==43646976,'Initialelementdenominator')
        result['initial']=dict(file=init,named_SHA256=actual_named,tensors=504,elements=43646976,finite=True,RNG_restored=False);del v,parameters;gc.collect()
        for step in [20,25]:
            phase='checkpoint'+str(step)
            cfg=c['checkpoint'][str(step)];root=cfg['root'];pins=cfg['files'];row=dict(step=step,fullstate_members=14,typed_checked_members=0)
            adapter=load(root,'adapter_model.safetensors',pins['adapter_model.safetensors']);need(len(adapter)==504 and set(adapter)==set(expected_adapter),'Adapter504 normalizedcanonicalnames')
            for name,t in adapter.items():
                tensor(t);r=expected_adapter[name];need(list(t.shape)==r['shape'] and t.numel()==r['numel'] and str(t.dtype)==r['dtype'],'Adapterexactshape/dtype')
            ah=tree_fingerprint(adapter);row.update(adapter_tree_SHA256=ah,adapter_tensors=504,adapter_elements=sum(t.numel() for t in adapter.values()),adapter_finite=True)
            if step==25:need(ah==c['live25']['adapter_tree_SHA256'],'Saved25 adapter not originalall8livehash');row['saved_adapter_matches_all8_original_live']=True
            del adapter;gc.collect()
            opt=load(root,'optimizer.pt',pins['optimizer.pt']);optimizer_steps(opt,step,504);shapes=Counter();total=0
            for state in opt['state'].values():
                tensor(state['step']);x=state['exp_avg'];y=state['exp_avg_sq'];tensor(x);tensor(y);need(x.shape==y.shape and x.dtype==y.dtype,'AdamWtwo moments shape/dtype differ');shapes[(tuple(x.shape),x.numel())]+=1;total+=x.numel()
            need(shapes==expected_shapes and total==43646976,'Optimizer504moment shape multiset/elements differs')
            row.update(optimizer_states=504,optimizer_step=step,optimizer_moment_elements_each=total,optimizer_finite=True,optimizer_tree_SHA256=tree_fingerprint(opt),optimizer_named_ID_map='NOT_PROVEN_COUNT_AND_SHAPE_MULTISET_ONLY',optimizer_vs_historical_live='UNAVAILABLE_NO_COMPLETE_LIVE_DIGEST');del opt;gc.collect()
            scheduler=load(root,'scheduler.pt',pins['scheduler.pt']);need(isinstance(scheduler,dict) and scheduler.get('last_epoch')==step and scheduler.get('_step_count')==step+1,'Actualschedulerstep differs');sh=tree_fingerprint(scheduler);row['scheduler_tree_SHA256']=sh
            if step==25:need(sh==c['live25']['scheduler_tree_SHA256'],'Saved25scheduler not originalall8livehash');row['saved_scheduler_matches_all8_original_live']=True
            del scheduler
            trainer=load(root,'trainer_state.json',pins['trainer_state.json']);need(trainer['global_step']==step and trainer['max_steps']==25 and float(trainer['epoch'])==step/25,'Actualtrainerglobalstep/epochcursor differs');row['trainer_cursor']=dict(global_step=step,max_steps=25,epoch=step/25)
            args=load(root,'args.json',pins['args.json']);need(isinstance(args,dict),'SavedargsJSON schema')
            # Report exact present source values without guessing aliases across Swift releases.
            row['saved_argument_identity']={k:args[k] for k in ['max_steps','gradient_accumulation_steps','per_device_train_batch_size','seed','data_seed','dataset_shuffle','dataloader_drop_last','max_length','learning_rate','lr_scheduler_type','optim','lora_rank','lora_alpha','lora_dropout'] if k in args}
            need(args.get('max_steps')==25 and args.get('gradient_accumulation_steps')==4 and args.get('per_device_train_batch_size')==1 and args.get('seed')==args.get('data_seed')==42,'Savedargs originaltrainingbudget/seed differs')
            ac=load(root,'adapter_config.json',pins['adapter_config.json']);need(ac['r']==16 and ac['lora_alpha']==32 and ac['lora_dropout']==.05 and ac.get('bias')=='none','SavedLoRaconfig differs')
            rngs=[]
            for rank in range(8):
                val=load(root,'rng_state_%d.pth'%rank,pins['rng_state_%d.pth'%rank]);need(isinstance(val,dict) and {'python','numpy','cpu','cuda'}<=set(val),'SavedrankRNG schema')
                tensor(val['cpu']);cuda=val['cuda'];need(isinstance(cuda,(list,tuple,torch.Tensor)),'SavedCUDA RNGcontainer schema')
                for ct in cuda if isinstance(cuda,(list,tuple)) else [cuda]:tensor(ct);need(ct.dtype==torch.uint8,'CUDARNGbytesdtype')
                need(val['cpu'].dtype==torch.uint8,'CPURNGbytesdtype');finger=rng_fingerprint(val);need(finger['finite'] is True,'RNGfingerprintfinite')
                rngs.append(dict(rank=rank,fingerprint_SHA256=finger['SHA256'],restore_executed=False));del val
            row.update(rank_RNG=rngs,typed_checked_members=14,restoration_into_model_executed=False);result['checkpoints'][str(step)]=row
        need(len(seen)==29 and not denied and not device_fds() and not torch.cuda.is_initialized(),'Full29CPU/devicebound differs')
        result.update(passed=True,state='ORIGINAL29_TYPED_SAVED_STATE_AUDIT_PASS_ORIGINAL_TRAINING_FAIL_RETAINED',checked_state_files=29,
            original_rank0_terminal_receipt_reconstructed=False,fullstate_GPU_recovery_qualified=False,formal17=False,effect_qualified=False,performance_qualified=False)
    except BaseException as error:
        result.update(state='KNOWN_SAVED_STATE_CPU_FAILURE_NO_UNSAFE_FALLBACK',error_type=type(error).__name__,phase=phase,
            error='Saved-state CPU audit failed; reason retained only in bounded private diagnostic',checked_state_files=len(seen))
        try:
            diagnostic=failure_record(type(error).__name__,phase,result.get('active_state_file'),clipped_utf8(error),clipped_utf8(traceback.format_exc(limit=12)))
            path=a.output/'failure-private.json'
            with path.open('xb') as stream:stream.write(diagnostic);stream.flush();os.fsync(stream.fileno())
            result['private_failure']=dict(bytes=len(diagnostic),SHA256=sha(diagnostic),error_type=type(error).__name__,phase=phase,scope='private CPU namespace only; do not export body')
        except BaseException as diagnostic_error:
            result['private_failure_write_error_type']=type(diagnostic_error).__name__
    result.update(epoch=a.epoch,deadline=a.deadline,finished=time.time(),CPU_wall_seconds=time.monotonic()-start,last_phase=phase,CPU_affinity=sorted(os.sched_getaffinity(0)),peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,denied_attempts=denied,self_device_FDs=device_fds(),source_manifest_SHA256=sha((D/'SOURCE_FILES.json').read_bytes()))
    save('result.json',result);print(json.dumps({k:result[k] for k in ['passed','planned_state_files','checked_state_files','original_failure_preserved','actual_updates','GPU_GO']},sort_keys=True));return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--framework',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True);raise SystemExit(main(p.parse_args()))
