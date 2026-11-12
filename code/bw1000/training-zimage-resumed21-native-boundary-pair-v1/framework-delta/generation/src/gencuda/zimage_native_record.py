"""Bound original native25 producer and fresh-process resume consumer.

This uses the registered framework's native update, checkpoint and recorder
implementations. Model/data bodies stay on the executing host. Diagnostic I/O
and identity checks are reported separately and never qualify formal throughput.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_binding(binding):
    config=binding['config']
    expected=dict(model='z-image-turbo',seed=20260911,cards='0',precision='bfloat16',lora_rank=4,
                  lora_target_modules='to_q,to_k,to_v,to_out.0,w1,w2,w3',learning_rate=1e-4,weight_decay=.01,
                  max_steps=25,warmup_steps=5,max_grad_norm=1.,gradient_checkpointing=True,height=256,width=256,
                  checkpoint_steps=[10,20,25],validate_recovery=False,validate_turbo_regression=True)
    if any(config.get(n)!=v or type(config.get(n)) is not type(v) for n,v in expected.items()):
        raise ValueError('original Z-Image recipe differs')
    if canonical(config)!=binding['config_canonical_SHA256']:
        raise ValueError('complete original configuration differs')
    if len(binding['data']['images'])!=20 or len(binding['data']['row_order'])!=20:
        raise ValueError('full original COCO20 is required')
    if (binding['initial']['trainable_tensors']!=476 or binding['initial']['trainable_elements']!=9922560
            or binding['model']['model_files']!=77 or binding['model']['preset_files']!=1):
        raise ValueError('original model or trainable extent differs')
    if binding['platform'] not in ('cuda','hip') or binding['logical_device_ids']!=[0]:
        raise ValueError('one original logical accelerator is required')
    if binding['planned_continuous_steps']!=25 or binding['planned_resume_global_steps']!=[21,22,23,24,25]:
        raise ValueError('native25 and fresh resume20 schedule differs')


def run(binding, output, *, mode, epoch, deadline, checkpoint=None, first_resumed_update_only=False, warmup_before_restored_update=False):
    validate_binding(binding)
    if type(first_resumed_update_only) is not bool or (first_resumed_update_only and mode!='resume20'):
        raise ValueError('resumed21 diagnosis requires explicit boolean and original resume20 mode')
    if type(warmup_before_restored_update) is not bool or (warmup_before_restored_update and not first_resumed_update_only):
        raise ValueError('diagnostic warmup requires explicit resumed21 diagnosis')
    if mode not in ('continuous','resume20') or (mode=='resume20')!=(checkpoint is not None):
        raise ValueError('native recording mode or checkpoint differs')
    if not all(math.isfinite(v) for v in (epoch,deadline)) or not 0<deadline-epoch<=3600:
        raise ValueError('invalid original single epoch')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    started=time.monotonic();remaining=deadline-time.time()
    def check():
        if min(deadline-time.time(),remaining-(time.monotonic()-started))<=0:raise TimeoutError('native25 epoch expired')
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while True:
                check();raw=stream.read(4*1024**2)
                if not raw:break
                h.update(raw)
        return h.hexdigest()
    def dump(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2)
    completed=[];warmup_calls_completed=0;warmup_info=None
    def progress(phase,**extra):
        value=dict(mode=mode,planned_continuous_steps=25,planned_resume_steps=[21,22,23,24,25],
                   diagnostic_update_only=first_resumed_update_only,actual_planned_updates=[21] if first_resumed_update_only else None,
                   planned_extra_warmup_calls=int(warmup_before_restored_update),completed_extra_warmup_calls=warmup_calls_completed,
                   completed_global_steps=list(completed),epoch=epoch,deadline=deadline,phase=phase,**extra)
        p=output/'progress.tmp';p.write_text(json.dumps(value,indent=2));p.replace(output/'progress.json')
    owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
    dump('owner.json',owner);phase='frozen-prerequisites';progress(phase)
    try:
        for n,pin in binding['source_files'].items():
            p=Path(binding['source_root'])/n
            if p.is_symlink() or p.stat().st_size!=pin['bytes'] or sha(p)!=pin['SHA256']:raise ValueError('original source differs: '+n)
        def stat_check(path,pin):
            p=Path(path);s=p.stat()
            if p.is_symlink() or any(getattr(s,{'bytes':'st_size','mtime_ns':'st_mtime_ns','inode':'st_ino','device':'st_dev'}[k])!=pin[k] for k in ('bytes','mtime_ns','inode','device')):
                raise ValueError('original file stat differs: '+str(p))
        for name,pin in binding['model']['files'].items():stat_check(Path(binding['model_root'])/name,pin)
        stat_check(binding['preset_path'],binding['model']['preset_stat'])
        for name,pin in binding['data']['images'].items():stat_check(Path(binding['data']['root'])/name,pin)
        for name,pin in binding['data']['small_files'].items():
            if sha(Path(binding['data']['root'])/name)!=pin['SHA256']:raise ValueError('original COCO20 metadata differs')
        for p in binding['initial']['artifacts'].values():
            if Path(p['path']).stat().st_size!=p['bytes'] or sha(p['path'])!=p['SHA256']:raise ValueError('original private initial artifact differs')
        import gc,importlib.metadata as md,random,sys
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from .backends import zimage_training as backend
        from .zimage_first_update import ZImageFirstUpdateRecorder,instrument_update,recorded_native_update
        from . import zimage_training_resume_checkpoint as resume
        from . import hy_resume_checkpoint as common
        check()
        if bool(torch.version.hip)!=(binding['platform']=='hip') or torch.cuda.device_count()!=1:
            raise RuntimeError('expected original platform and one visible accelerator')
        if torch.cuda.get_device_name(0)!=binding['GPU_name']:raise RuntimeError('accelerator identity differs')
        versions={n:md.version(n) for n in binding['versions']}
        if versions!=binding['versions']:raise RuntimeError('original package versions differ')
        torch.set_num_threads(1);torch.manual_seed(20260911);random.seed(20260911);np.random.seed(20260911)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.synchronize();free_before,total=torch.cuda.mem_get_info()
        if total!=binding['GPU_total_bytes'] or free_before<binding['initial_minimum_free_bytes']:raise RuntimeError('fresh accelerator admission differs')
        torch.cuda.reset_peak_memory_stats();phase='original-native-full-load';progress(phase)
        source_root=Path(binding['source_root']);sys.path.insert(0,str(source_root))
        training_class=backend._training_module_class(source_root)
        native_module=sys.modules[training_class.__module__]
        from diffsynth.core import UnifiedDataset
        cfg=binding['config'];model_root=Path(binding['model_root']);loadstart=time.monotonic()
        model=training_class(model_paths=json.dumps(binding['model_paths']),tokenizer_path=str(model_root/'tokenizer'),
             lora_base_model='dit',lora_target_modules=cfg['lora_target_modules'],lora_rank=4,preset_lora_path=binding['preset_path'],
             preset_lora_model='dit',use_gradient_checkpointing=True,device='cuda',task='sft')
        initial=load_file(binding['initial']['artifacts']['trainable']['path'],device='cpu')
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if len(named)!=476 or sum(p.numel() for _,p in named)!=9922560:raise ValueError('native trainable extent differs')
        if {n.removeprefix('pipe.dit.') for n,_ in named}!=set(initial):raise ValueError('actual initial names differ')
        with torch.no_grad():
            for n,p in named:
                value=initial[n.removeprefix('pipe.dit.')]
                if p.shape!=value.shape or p.dtype!=value.dtype or p.device.type!='cuda':raise ValueError('actual initial shape/dtype differs')
                p.copy_(value)
        del initial
        # This is our SHA-bound private producer artifact,including NumPy state.
        rng=torch.load(binding['initial']['artifacts']['rng']['path'],map_location='cpu',weights_only=False)
        torch.set_rng_state(rng['torch_CPU']);torch.cuda.set_rng_state(rng['torch_CUDA']);random.setstate(rng['python']);np.random.set_state(rng['numpy']);del rng
        torch.cuda.synchronize();load_seconds=time.monotonic()-loadstart
        dataset=UnifiedDataset(base_path=cfg['dataset_root'],metadata_path=cfg['dataset_metadata'],data_file_keys=('image',),
            main_data_operator=UnifiedDataset.default_image_operator(base_path=cfg['dataset_root'],height=256,width=256,height_division_factor=16,width_division_factor=16))
        if len(dataset)!=20:raise ValueError('original dataset length differs')
        optimizer=torch.optim.AdamW(list(model.trainable_modules()),lr=1e-4,weight_decay=.01)
        optimizer_progress=common.initial_optimizer_progress(model,optimizer)
        model.pipe.scheduler.set_timesteps(1000,training=True);scheduler=resume.scheduler_spec(model.pipe.scheduler)
        identities=dict(config=binding['config_canonical_SHA256'],source=canonical(binding['execution_identity']),data=canonical(binding['data']),
                        model=canonical(binding['model']),runtime=canonical(dict(versions=versions,platform=binding['platform'],GPU_name=binding['GPU_name'])),
                        scheduler=common.tree_sha256(scheduler))
        contract=resume.build_contract(identity=identities,seed=20260911,max_steps=25,
            sample_order=[binding['data']['small_files']['metadata-zimage-20.csv']['SHA256']+':'+str(i) for i in range(20)],device_ids=[0])
        dump('contract.json',contract)
        begin=0;restore_info=None
        if mode=='resume20':
            payload=resume.load_checkpoint(checkpoint)
            optimizer_progress,restore_info=resume.restore_checkpoint(payload,contract,model,optimizer,model.pipe.scheduler,scheduler)
            resume.assert_restored_boundary(payload,model,optimizer,optimizer_progress,model.pipe.scheduler,[0])
            if restore_info['metadata']['global_step']!=20 or restore_info['next']['sample_cursor']!=0:raise ValueError('not original completed20 boundary')
            begin=20;dump('restore.json',restore_info);del payload
        elif cfg['validate_turbo_regression']:
            progress('original-before-turbo-render')
            backend._render_turbo_preserving_training_boundary(backend.turbo_regression_options(cfg),model,output/'turbo-before.private.png',scheduler)
        model.train()
        def tensor_sha(tensor):
            h=hashlib.sha256();flat=tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            for offset in range(0,flat.numel(),4*1024**2):check();h.update(flat[offset:offset+4*1024**2].cpu().numpy().tobytes())
            return h.hexdigest()
        def frozen_snapshot():
            result={}
            for name,p in model.named_parameters():
                if not p.requires_grad:
                    if p.grad is not None:raise ValueError('frozen parameter received gradient')
                    result[name]=dict(shape=list(p.shape),dtype=str(p.dtype),SHA256=tensor_sha(p))
            return result
        progress('frozen-before');frozen_before=frozen_snapshot();dump('frozen-before.json',frozen_before)
        if warmup_before_restored_update:
            phase='diagnostic-warmup-before-restore';progress(phase)
            check();warmup_data=dataset[0]
            warmup_value=backend._train_step(model,optimizer,warmup_data,20260911+21,1.)
            warmup_calls_completed=1
            warmup_info=dict(actual_optimizer_calls=1,temporary_global_update=21,metrics=warmup_value,restored20_after_warmup=False)
            dump('diagnostic-warmup-returned.json',warmup_info);progress('diagnostic-warmup-returned')
            payload=resume.load_checkpoint(checkpoint)
            optimizer_progress,checked=resume.restore_checkpoint(payload,contract,model,optimizer,model.pipe.scheduler,scheduler)
            optimizer.zero_grad(set_to_none=True)
            resume.assert_restored_boundary(payload,model,optimizer,optimizer_progress,model.pipe.scheduler,[0])
            if checked['metadata']['global_step']!=20 or checked['next']['sample_cursor']!=0:raise ValueError('diagnostic warmup failed to restore original20 boundary')
            warmup_info['restored20_after_warmup']=True;dump('diagnostic-warmup-restored20.json',warmup_info)
            del payload,warmup_data;progress('diagnostic-warmup-restored20')
        recorder=None
        if mode=='continuous':
            recorder=ZImageFirstUpdateRecorder(output/'first-update-private',named,optimizer,epoch=epoch,deadline=deadline,device_ids=[0],
                  binding=dict(case='T07.sft',config_SHA256=binding['config_canonical_SHA256'],full_binding_SHA256=canonical(binding),actual_original_inputs=True),total_limit=2*1024**3)
        elif first_resumed_update_only:
            from .zimage_resumed_update import ZImageResumedUpdateRecorder
            recorder=ZImageResumedUpdateRecorder(output/'resumed-update-private',named,optimizer,epoch=epoch,deadline=deadline,device_ids=[0],
                  binding=dict(case='T07.sft:resumed21-diagnosis',config_SHA256=binding['config_canonical_SHA256'],full_binding_SHA256=canonical(binding),actual_original_inputs=True,
                               checkpoint20_SHA256=sha(checkpoint),actual_global_step=21),total_limit=2*1024**3)
        update=instrument_update(backend._train_step) if recorder else backend._train_step
        metrics=[];checkpoints=[];phase='native-updates';progress(phase)
        for step in range(begin+1,22 if first_resumed_update_only else 26):
            check();data=dataset[(step-1)%20]
            if first_resumed_update_only:
                from .zimage_resumed_update import recorded_resumed_update
                value=recorded_resumed_update(update,native_module,recorder,step,model,optimizer,data,20260911+step,1.)
            else:
                value=recorded_native_update(update,native_module,recorder,step,model,optimizer,data,20260911+step,1.)
            optimizer_progress=common.advance_optimizer_progress(optimizer_progress,model,optimizer)
            row=dict(global_step=step,sample_index=(step-1)%20,**value);metrics.append(row);completed.append(step)
            with (output/'steps.private.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            progress(phase)
            if step in (10,20,25) or first_resumed_update_only:
                check();cp=output/('checkpoint-%d'%step)/'state.private.pt';cpstart=time.monotonic()
                payload=resume.make_payload(contract=contract,global_step=step,model=model,optimizer=optimizer,optimizer_progress=optimizer_progress,
                     scheduler=model.pipe.scheduler,expected_scheduler=scheduler)
                resume.save_atomic(cp,payload)
                checkpoints.append(dict(global_step=step,path=str(cp),bytes=cp.stat().st_size,SHA256=sha(cp),
                      integrity_SHA256=payload['integrity_sha256'],adapter_SHA256=payload['adapter_sha256'],wall_seconds=time.monotonic()-cpstart,
                      next_data_cursor=payload['metadata']['next_data_cursor'],state_fields=list(payload['state']),RNG_devices=list(payload['rng']['device']) if 'device' in payload['rng'] else None))
                del payload;progress('checkpoint-saved',global_step=step)
        phase='frozen-after';progress(phase);frozen_after=frozen_snapshot();dump('frozen-after.json',frozen_after)
        if frozen_before!=frozen_after:raise ValueError('frozen native parameters changed')
        if recorder:recorder.guard(recorder.finish,21 if first_resumed_update_only else 25,True)
        if mode=='continuous' and cfg['validate_turbo_regression']:
            progress('original-after-turbo-render')
            backend._render_turbo_preserving_training_boundary(backend.turbo_regression_options(cfg),model,output/'turbo-after.private.png',scheduler)
        providers={};provider_bytes=0
        for name,module in sorted(sys.modules.items()):
            if name!='diffsynth' and not name.startswith('diffsynth.'):continue
            path=getattr(module,'__file__',None)
            if not path:continue
            p=Path(path);relative=p.resolve().relative_to(source_root.resolve()).as_posix();size=p.stat().st_size
            if p.is_symlink() or size>2*1024**2:raise ValueError('unbounded native provider')
            provider_bytes+=size
            if len(providers)>=512 or provider_bytes>40*1024**2:raise ValueError('native provider bound exceeded')
            providers[name]=dict(relative_path=relative,bytes=size,SHA256=sha(p))
        dump('providers.json',providers)
        for name,pin in binding['model']['files'].items():stat_check(model_root/name,pin)
        for name,pin in binding['data']['images'].items():stat_check(Path(binding['data']['root'])/name,pin)
        result=dict(passed=True,mode=mode,owner=owner,epoch=epoch,deadline=deadline,completed_global_steps=completed,
                    actual_optimizer_updates=len(completed)+warmup_calls_completed,diagnostic_warmup=warmup_info,
                    first_update_capture_complete=bool(recorder and recorder.first_complete),
                    initial_tensors=476,initial_elements=9922560,frozen_tensors=len(frozen_before),frozen_parameters_unchanged=True,
                    load_seconds=load_seconds,wall_seconds=time.monotonic()-started,checkpoints=checkpoints,restore_info=restore_info,
                    first_update_capture_bytes=recorder.total if recorder else 0,recording_seconds=recorder.diagnostic_seconds if recorder else 0,
                    step_metrics=metrics,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    original_full_recipe=not first_resumed_update_only,diagnostic_update_only=first_resumed_update_only,
                    private_values_exported=False,formal_performance=False,cross_platform_numerical_qualified=False,Model_GO=False)
        dump('result.json',result)
        # Cache references can survive Python object deletion. Persist real
        # pre-exit metrics; after-process device cleanup is checked by the owner.
        del named,optimizer,model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        dump('pre-exit-allocator.json',dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0],
             no_zero_allocation_claim=True,requires_after_process_owned_GPU_postcheck=True))
        progress('NATIVE_PHASE_RETURNED')
        return result
    except BaseException as error:
        import traceback
        dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-10000:],completed_global_steps=completed))
        progress('FAILED',error_type=type(error).__name__)
        raise
