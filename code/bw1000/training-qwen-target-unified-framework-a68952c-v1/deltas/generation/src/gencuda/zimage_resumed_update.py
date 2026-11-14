"""Observe one actual global21 update after a full original checkpoint20 restore.

Reuse bounded private serialization and the exact loss/update AST observers.
No global step is relabeled as a fresh first step;existing AdamW state is required.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

from . import hy_resume_checkpoint as common
from .vggt_first_update import FirstUpdateRecorder
from .zimage_first_update import ZImageFirstUpdateRecorder,observe_native_loss,safe_input


class ZImageResumedUpdateRecorder(FirstUpdateRecorder):
    _group=ZImageFirstUpdateRecorder._group

    def __init__(self,root,named,optimizer,*,epoch,deadline,device_ids,binding,total_limit=2*1024**3):
        import torch
        self.root,self.named,self.optimizer=Path(root),list(named),optimizer
        self.epoch,self.deadline=epoch,deadline
        if not all(math.isfinite(x) for x in (epoch,deadline)) or epoch<=0 or not 0<deadline-epoch<=3600:
            raise ValueError('invalid resumed-update recorder epoch')
        self.anchor,self.start_remaining=time.monotonic(),deadline-time.time()
        self.device_ids,self.binding=list(device_ids),binding
        self.artifact_limit=256*1024**2;self.total_limit=total_limit
        if not 0<total_limit<=2*1024**3:raise ValueError('invalid resumed-update byte budget')
        self.files,self.total,self.completed_steps=[],0,20
        self.current_step,self.phase,self.failed,self.first_complete=None,'restored20',False,False
        self.diagnostic_seconds=0.
        if not self.named or len({n for n,_ in self.named})!=len(self.named) or len({id(p) for _,p in self.named})!=len(self.named):
            raise ValueError('duplicate resumed trainable ownership')
        if any(type(p) is not torch.nn.Parameter or not p.requires_grad or p.is_meta or not p.is_leaf or p.grad is not None for _,p in self.named):
            raise ValueError('expected materialized restored trainables with no live gradient')
        owned=[p for group in optimizer.param_groups for p in group['params']]
        if [id(p) for p in owned]!=[id(p) for _,p in self.named]:raise ValueError('restored optimizer ownership differs')
        for _,p in self.named:
            state=optimizer.state.get(p)
            if not state or not {'step','exp_avg','exp_avg_sq'}<=set(state) or float(state['step'])!=20:
                raise ValueError('recorder requires actual completed20 AdamW states')
        self.check();self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        with (self.root/'binding.json').open('x') as f:json.dump(binding,f,indent=2)
        self.progress()

    def progress(self):
        v=dict(case='T07.sft:resumed21-diagnosis',planned_global_steps=[21],actual_start_boundary=20,
            completed_global_steps=[21] if self.completed_steps==21 else [],current_global_step=self.current_step,
            completed_recorded_updates=int(self.first_complete),trainable_tensors=len(self.named),trainable_elements=sum(p.numel() for _,p in self.named),
            epoch=self.epoch,deadline=self.deadline,phase=self.phase,failed=self.failed,files=self.files,artifact_bytes=self.total,diagnostic_seconds=self.diagnostic_seconds)
        p=self.root/'progress.tmp';p.write_text(json.dumps(v,indent=2)+'\n');p.replace(self.root/'progress.json')

    def enter(self,step,batch):
        self.check()
        if type(step) is not int or step!=21 or self.completed_steps!=20 or self.current_step is not None:
            raise ValueError('exactly one actual global21 update required')
        if any(p.grad is not None for _,p in self.named):raise ValueError('restored boundary has live gradients')
        self.current_step=step;self.phase='initial';self._group('initial')
        self._save('initial_optimizer.pt',self.optimizer.state_dict())
        import numpy as np
        import torch
        n=np.random.get_state()
        rng=dict(python=random.getstate(),numpy=dict(algorithm=n[0],keys=n[1].tolist(),position=int(n[2]),has_gauss=int(n[3]),cached_gaussian=float(n[4])),torch_CPU=torch.get_rng_state(),device={})
        for d in self.device_ids:
            torch.cuda.synchronize(d);rng['device'][d]=torch.cuda.get_rng_state(d).cpu()
        self.batch=safe_input(batch);self.input_SHA256=common.tree_sha256(self.batch)
        self._save('input.pt',dict(batch=self.batch,RNG=rng,input_SHA256=self.input_SHA256,actual_global_step=21))
        self.phase='input';self.progress()

    def prepared(self,inputs,noise,timestep_id,timestep,target):
        if self.current_step!=21 or self.phase!='input':raise ValueError('actualglobal21 target order differs')
        actual=safe_input(inputs);self.prepared_identity=common.tree_sha256(actual);self.prepared_inputs=actual
        self._save('prepared.pt',dict(inputs=actual,noise=noise,timestep_id=timestep_id,timestep=timestep));self._save('target.pt',dict(training_target=target))
        self.phase='target';self.progress()

    def forward_value(self,prediction,loss):
        if self.phase!='target' or common.tree_sha256(self.batch)!=self.input_SHA256 or common.tree_sha256(self.prepared_inputs)!=self.prepared_identity:
            raise ValueError('actualglobal21 forward input mutated or boundary differs')
        self._save('forward.pt',dict(noise_prediction=prediction,loss=loss));del self.batch,self.prepared_inputs
        self.phase='forward';self.progress()

    def gradients(self,stage,norm=None):
        if self.current_step!=21 or stage not in ('preclip','postclip') or self.phase!=('forward' if stage=='preclip' else 'preclip'):
            raise ValueError('actualglobal21 gradient order differs')
        self._group(stage,gradients=True)
        if stage=='postclip':self._save('clip_norm.pt',norm)
        self.phase=stage;self.progress()

    def updated(self):
        if self.current_step!=21 or self.phase!='postclip':raise ValueError('actualglobal21 update order differs')
        self._group('updated');states={}
        for name,p in self.named:
            state=self.optimizer.state.get(p)
            if not state or not {'step','exp_avg','exp_avg_sq'}<=set(state) or float(state['step'])!=21:
                raise ValueError('actual AdamW state did not advance to21')
            states[name]=state
        self._save('optimizer.pt',states);self._save('optimizer_groups.pt',self.optimizer.state_dict()['param_groups'])
        self.first_complete=True;self.completed_steps=21;self.phase='updated';self.progress()

    def finish(self,global_step,frozen_parameters_unchanged):
        self.check()
        if global_step!=21 or self.completed_steps!=21 or not self.first_complete or not frozen_parameters_unchanged:
            raise ValueError('actual resumed21 capture or frozen boundary incomplete')
        self.phase='RESUMED21_RETURNED';self.progress()


def recorded_resumed_update(update,native_module,recorder,step,model,optimizer,data,seed,max_grad_norm):
    def run():
        recorder.enter(step,data)
        with observe_native_loss(native_module,recorder):
            return update(model,optimizer,data,seed,max_grad_norm,observer=recorder)
    return recorder.guard(run)
