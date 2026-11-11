"""Observe the first real Trainer forward, then stop before backward or update."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import types
import time
import numpy as np
import torch
from swift.callbacks import TrainerCallback,callbacks_map
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vlm.runtime.dropout_capture import replay_dropout,validate_record
from vlm.runtime.initial_trainables import named_parameter_sha256,restore_initial_trainables
from vlm.runtime.lora_dropout_session import input_identity
from vlm.runtime.minicpm_recipe_contract import validate_native_recipe
from vlm.runtime.provider_counterfactual import selected_callables
from vlm.runtime.forward_provider_values import file_sha,to_cpu,metrics
from vlm.runtime.forward_context_state import context_snapshot,training_mode_changes,input_layouts
KEY='native_first_forward_context'

class NativeFirstForwardContext(TrainerCallback):
    def __init__(self,swift_args,trainer):
        super().__init__(swift_args,trainer)
        self.trainer=trainer;self.handles=[];self.originals=[];self.spy=None;self.spy_records=None;self.entered=0;self.finished=False;self.inner=None

    def save(self,name,value):
        with (self.output/name).open('x') as stream:json.dump(value,stream,indent=2)

    def progress(self,stage,failed=False):
        value=dict(owner=self.owner,stage=stage,planned_forwards=1,entered_forwards=self.entered,completed_forwards=int(self.finished),optimizer_updates=0,failed=failed)
        temp=self.output/'progress.tmp';temp.write_text(json.dumps(value));temp.replace(self.output/'progress.json')

    def check(self):
        if time.time()>=self.deadline:raise TimeoutError('native first-forward deadline exhausted')

    def failure(self,error,stage):
        self.progress(stage,True)
        if not (self.output/'failure.json').exists():self.save('failure.json',dict(owner=self.owner,stage=stage,error_type=type(error).__name__,message=str(error)[:400],planned_forwards=1,entered_forwards=self.entered,completed_forwards=int(self.finished),Model_GO=False))

    def remove(self):
        for h in self.handles:h.remove()
        self.handles=[]
        for module in self.originals:del module.forward
        self.originals=[]
        if self.spy is not None:
            active=self.spy;self.spy=None;active.__exit__(None,None,None)

    def on_train_begin(self,args,state,control,**kwargs):
        path=Path(os.environ['NATIVE_FORWARD_CONFIG']);assert file_sha(path)==os.environ['NATIVE_FORWARD_CONFIG_SHA256'];self.config=json.loads(path.read_bytes())
        epoch=float(os.environ['NATIVE_FORWARD_EPOCH']);self.deadline=float(os.environ['NATIVE_FORWARD_DEADLINE']);assert self.deadline==epoch+1080 and epoch<=time.time()<self.deadline
        def expired(*_):raise TimeoutError('native first-forward hard deadline')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,self.deadline-time.time())
        self.train_begin_ns=time.monotonic_ns()
        self.rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert self.rank==local and int(os.environ['WORLD_SIZE'])==8 and torch.cuda.current_device()==local
        assert torch.version.cuda=='12.9' and not torch.version.hip and self.config['case']=='T05.sft'
        self.output=Path(self.config['output'])/('rank-%02d'%self.rank);self.output.mkdir(parents=True,exist_ok=False)
        self.owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),rank=self.rank,local_rank=local,epoch=epoch,deadline=self.deadline)
        self.save('owner.json',self.owner);self.progress('train_begin')
        try:
            self.model=kwargs.get('model',self.trainer.model);self.named,frozen=validate_native_recipe(self.model,args,10822656)
            assert state.global_step==0 and all(p.grad is None for _,p in self.named)
            row=self.config['ranks'][self.rank];assert row['rank']==self.rank
            def load(kind):
                self.check();item=row[kind];p=Path(item['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and file_sha(p)==item['SHA256']
                return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            initial=row['initial'];receipt=restore_initial_trainables(self.named,self.trainer.optimizer,global_step=0,path=initial['path'],file_bytes=initial['bytes'],file_SHA256=initial['SHA256'],named_SHA256=self.config['initial_named_SHA256']);self.save('initialization.json',receipt)
            self.captured=load('input');self.reference=load('output')['result'];self.packet=load('mask')
            assert self.captured['global_step']==0 and self.captured['micro_ordinal']==1 and self.packet['identity']['global_step']==0 and self.packet['identity']['micro_ordinal']==1
            assert self.packet['contract']['rank']==self.rank and self.packet['contract']['world_size']==8
            assert input_identity(self.captured['args'],self.captured['kwargs'])==self.packet['identity']['input_SHA256']
            self.bindings=[(n,m) for n,m in self.model.named_modules() if type(m).__module__.startswith('transformers.models.qwen3_5.') and hasattr(m,'chunk_gated_delta_rule')];assert len(self.bindings)==18
            source=Path(inspect.getfile(type(self.bindings[0][1])));assert file_sha(source)==self.config['model_source_SHA256']
            defining=sys.modules[type(self.bindings[0][1]).__module__];native=defining.chunk_gated_delta_rule
            assert callable(native) and all(m.chunk_gated_delta_rule is native for _,m in self.bindings)
            self.native_provider=dict(module=native.__module__,name=native.__qualname__);assert self.native_provider['module'].startswith('fla.ops.gated_delta_rule')
            self.begin_context=context_snapshot(self.model)
            self.save('train-begin-context.json',dict(train_begin_monotonic_ns=self.train_begin_ns,template_class=type(self.trainer.template).__module__+'.'+type(self.trainer.template).__qualname__,template_forward_context=self.trainer.template.forward_context.__func__.__qualname__,template_post_encode=self.trainer.template._post_encode.__func__.__qualname__,context=self.begin_context,trainer_model=type(self.trainer.model).__module__+'.'+type(self.trainer.model).__qualname__,wrapped_model=type(self.trainer.model_wrapped).__module__+'.'+type(self.trainer.model_wrapped).__qualname__,native_provider=self.native_provider,frozen=frozen))
            self.progress('waiting_for_real_training_step')
        except BaseException as error:self.failure(error,'train_begin');raise
        return control

    def on_step_begin(self,args,state,control,**kwargs):
        if self.handles or self.entered or state.global_step!=0:raise ValueError('only first native Trainer step may be observed')
        def before(module,argv,kw):
            try:
                self.check();self.entered+=1;assert self.entered==1 and state.global_step==0
                identity=input_identity(argv,kw);assert identity==self.packet['identity']['input_SHA256']
                actual_layout=input_layouts(argv,kw);captured_layout=input_layouts(self.captured['args'],self.captured['kwargs'])
                assert set(actual_layout)==set(captured_layout)
                layout_changes=[]
                for key,value in actual_layout.items():
                    reference=captured_layout[key];assert value['shape']==reference['shape'] and value['dtype']==reference['dtype']
                    changed={name:dict(actual=value[name],captured=reference[name]) for name in ('stride','storage_offset','contiguous') if value[name]!=reference[name]}
                    if changed:layout_changes.append(dict(path=key,changes=changed))
                self.save('input-layouts.json',dict(actual=actual_layout,captured=captured_layout,changes=layout_changes,qualification='Header differences only; no causal or numerical acceptance implied.'))
                self.enter_context=context_snapshot(self.model)
                self.save('real-root-context.json',dict(context=self.enter_context,mode_changes=training_mode_changes(self.begin_context,self.enter_context),input_SHA256=identity,DDP_require_backward_grad_sync=getattr(self.trainer.model_wrapped,'require_backward_grad_sync',None),DDP_require_forward_param_sync=getattr(self.trainer.model_wrapped,'require_forward_param_sync',None),DDP_broadcast_buffers=getattr(self.trainer.model_wrapped,'broadcast_buffers',None),accelerator_sync_gradients=self.trainer.accelerator.sync_gradients,gradient_accumulation=args.gradient_accumulation_steps,bf16=args.bf16))
                rng=self.captured['RNG_before_forward'];random.setstate(rng['python']);r=rng['numpy'];np.random.set_state((r['algorithm'],np.asarray(r['keys'],dtype=np.uint32),r['position'],r['has_gauss'],r['cached_gaussian']));torch.set_rng_state(rng['torch_CPU']);torch.cuda.set_rng_state(rng['torch_device'])
                self.position=0;records=self.packet['records'];dropouts=[(n,m) for n,m in self.model.named_modules() if '.lora_dropout.' in n];assert len(dropouts)==len(records)==186
                for name,m in dropouts:
                    assert type(m) is torch.nn.Dropout and m.training and m.p==.05 and not m.inplace and 'forward' not in m.__dict__;self.originals.append(m)
                    def replay(owner,value,*,_name=name):
                        i=self.position;assert i<len(records);row=records[i];assert row['module']==_name and row['call_ordinal']==i;validate_record(value,row['native_record']);self.position+=1;return replay_dropout(value,row['native_record'])
                    m.forward=types.MethodType(replay,m)
                self.spy=selected_callables(self.bindings);self.spy_records=self.spy.__enter__()
                assert named_parameter_sha256(self.named)==self.config['initial_named_SHA256'] and not self.trainer.optimizer.state
                torch.cuda.synchronize();assert torch.cuda.mem_get_info()[0]>=8*1024**3;self.progress('real_forward_entered');self.forward_start_ns=time.monotonic_ns()
            except BaseException as error:self.failure(error,'root_pre_hook');self.remove();raise
        def inside(module,argv,kw):
            if self.inner is None:self.inner=context_snapshot(self.model)
        def after(module,argv,kw,result):
            if result is None:return
            try:
                self.check();torch.cuda.synchronize();self.forward_end_ns=time.monotonic_ns();self.host_audit_start_ns=time.monotonic_ns();assert self.entered==1 and self.position==186 and self.inner is not None
                assert self.spy_records is not None and len(self.spy_records)==18 and all(r['calls']==1 for r in self.spy_records)
                values=to_cpu(dict(result));assert set(values)==set(self.reference)
                for key,value in values.items():
                    if isinstance(value,torch.Tensor):assert value.shape==self.reference[key].shape and value.dtype==self.reference[key].dtype
                p=self.output/'native-output.pt'
                with p.open('xb') as stream:torch.save(values,stream)
                assert p.stat().st_size<=512*1024**2
                errors={key:metrics(values[key],self.reference[key],self.check) for key in ('loss','logits')}
                assert named_parameter_sha256(self.named)==self.config['initial_named_SHA256'] and not self.trainer.optimizer.state and all(p.grad is None for _,p in self.named) and state.global_step==0
                self.save('inside-linear-context.json',self.inner)
                self.save('calls.json',dict(chunk_calls=self.spy_records,dropout_calls=self.position,first_input_identity=self.packet['identity']['input_SHA256']))
                self.save('result.json',dict(train_begin_monotonic_ns=self.train_begin_ns,forward_start_monotonic_ns=self.forward_start_ns,forward_end_monotonic_ns=self.forward_end_ns,forward_enclosing_seconds=(self.forward_end_ns-self.forward_start_ns)/1e9,post_forward_host_audit_seconds=(time.monotonic_ns()-self.host_audit_start_ns)/1e9,timing_qualification='Instrumented root forward enclosure includes host enqueue and inner observation; not exclusive GPU or formal throughput. Post-forward CPU copy/output write/metric audit recorded separately.',passed=True,owner=self.owner,source_SHA256=file_sha(Path(__file__)),completed_forwards=1,optimizer_updates=0,global_step=0,actual_loss=float(values['loss']),reference_loss=float(self.reference['loss']),comparison=errors,exact_native_reproduction=all(x['exact'] for x in errors.values()),training_mode_changes=training_mode_changes(self.begin_context,self.enter_context),inside_autocast_enabled=self.inner['autocast_enabled'],root_autocast_enabled=self.enter_context['autocast_enabled'],native_provider=self.native_provider,original_numerical_tolerances_available=False,numerical_acceptance=False,raw_tensor_transfer_bytes=0,Model_GO=False))
                self.finished=True;self.progress('complete');self.remove();torch.cuda.synchronize();assert torch.cuda.mem_get_info()[0]>=8*1024**3
                torch.distributed.barrier();torch.distributed.destroy_process_group()
            except BaseException as error:self.failure(error,'root_post_hook');self.remove();raise
            raise SystemExit(0)
        self.handles=[self.model.register_forward_pre_hook(before,with_kwargs=True),self.model.register_forward_hook(after,with_kwargs=True,always_call=True),self.bindings[0][1].register_forward_pre_hook(inside,with_kwargs=True)]
        return control

    def on_train_end(self,*args,**kwargs):
        raise RuntimeError('native first-forward observer must stop before any backward/update')

if KEY in callbacks_map:raise ValueError('duplicate native first-forward callback')
callbacks_map[KEY]=NativeFirstForwardContext
