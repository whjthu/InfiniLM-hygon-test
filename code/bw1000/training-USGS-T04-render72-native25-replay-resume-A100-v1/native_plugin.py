"""Observe native Swift FWD/BWD/AdamW and full checkpoint resume. New opt-in only."""
import functools,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from swift.callbacks import TrainerCallback,callbacks_map
from peft import get_peft_model_state_dict
from common import D,P,need,obj,save,left,sha,canonical,stamp,native_inputs,pinned,own
sys.path.insert(0,P['framework'])
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.provider_sources import model_provider_sources
from vlm.runtime.initial_trainables import named_parameter_sha256
from vlm.runtime.first_update_audit import fingerprint,parameter_delta
from vlm.runtime.cuda_resume_witness import tree_fingerprint,comparison,trusted_load,optimizer_steps,provider_record
from vlm.runtime.resume_witness import rng_values,rng_fingerprint
KEY='native_render72_diagnostic'

def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().contiguous().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v

def fields(v):return {k:{a:b for a,b in fingerprint(cpu(x)).items() if a!='floating_l2'} for k,x in sorted(v.items())}
def signature(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())
CORE=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json','args.json','rng_state.pth']

class NativeRender72(TrainerCallback):
    def __init__(self,swift_args,trainer):
        super().__init__(swift_args,trainer);self.t=trainer;self.c=obj(os.environ['RENDER72_CONFIG']);self.phase=self.c['phase'];self.root=Path(self.c['phase_root']);self.records=[];self.forwards=[];self.losses=[];self.steps=[];self.clips=[];self.saved=[];self.restores=[];self.handles=[];self.originals={};self.capture_pins={};self.capture_bytes=0;self.comparisons=[];self.pending=None;self.entry=dict(initial=[],initialized=[])
        self.start=20 if self.phase=='resume' else 0;self.end=2 if self.phase=='replay2' else 25;self.planned=(self.end-self.start)*4
        for name,key in [('set_initial_training_values','initial'),('_init_training_state','initialized')]:
            original=getattr(trainer,name);self.originals[name]=original
            def observe(*args,_original=original,_key=key,**kw):
                result=_original(*args,**kw);need(isinstance(result,tuple),'NATIVE_ENTRY_TUPLE');self.entry[_key].append(list(result));return result
            setattr(trainer,name,observe)
        self.progress()
    def check(self):left(self.c)
    def progress(self,error=None):save(self.root/'progress.json',dict(phase=self.phase,planned_microbatches=self.planned,planned_updates=self.end-self.start,completed_microbatches=len(self.losses),attempted_microbatches=len(self.records),completed_updates=len(self.steps),saved_steps=self.saved,failed=error is not None,error=error),True)
    def pin(self,path,cap):
        self.check();p=Path(path);a=p.lstat();need(p.is_file() and not p.is_symlink() and 0<a.st_size<=cap,'PRIVATE_FILE_BOUND');h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):self.check();h.update(b)
            need(stamp(a)==stamp(os.fstat(f.fileno()))==stamp(p.lstat()),'PRIVATE_STABLE_FD')
        return dict(bytes=a.st_size,SHA256=h.hexdigest())
    def load(self,path,pin):return trusted_load(Path(path).parent,Path(path).name,pin,self.check)
    def store(self,name,value):
        self.check();p=self.root/name;tmp=p.with_name(p.name+'.tmp');need(not p.exists() and not tmp.exists(),'NEW_PRIVATE_CAPTURE');remaining=P['capture_cap_bytes']-self.capture_bytes;need(remaining>0,'CAPTURE_CAP')
        def upper(v):
            if isinstance(v,torch.Tensor):return v.numel()*v.element_size()+65536
            if isinstance(v,np.ndarray):return v.nbytes+65536
            if isinstance(v,dict):return 4096+sum(upper(k)+upper(x) for k,x in v.items())
            if isinstance(v,(list,tuple)):return 4096+sum(upper(x) for x in v)
            if isinstance(v,str):return len(v.encode())*6+256
            need(v is None or type(v) in (int,float,bool),'TYPED_CAPTURE_ONLY');return 256
        need(upper(value)+1048576<=min(P['capture_file_cap_bytes'],remaining),'BEFORE_WRITE_TYPED_CAPTURE_BOUND')
        with tmp.open('xb') as f:
            torch.save(value,f);f.flush();need(f.tell()<=min(P['capture_file_cap_bytes'],remaining),'CAPTURE_FILE_CAP');os.fsync(f.fileno());a=os.fstat(f.fileno());tmp.rename(p);b=os.fstat(f.fileno());need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and stamp(b)==stamp(p.stat()),'CAPTURE_RENAME_FD')
        fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        pin=self.pin(p,P['capture_file_cap_bytes']);self.capture_bytes+=pin['bytes'];self.capture_pins[name]=pin;save(self.root/'capture-pins.json',dict(files=self.capture_pins,bytes=self.capture_bytes),True);return pin
    def compare(self,name,value):
        ref=Path(self.c['continuous_root']);pins=obj(ref/'capture-pins.json')['files'];expected=self.load(ref/name,pins[name])
        if name.startswith('forward-'):
            need(set(expected)==set(value)=={'logits','loss'},'FULL_FORWARD_KEYS');stats={}
            for k in expected:
                a,b=expected[k],value[k];need(isinstance(a,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,'FULL_FORWARD_SHAPE_DTYPE');exact=True;maximum=0.;err=0.;base=0.
                for i in range(0,a.numel(),1048576):
                    self.check();x=a.reshape(-1)[i:i+1048576];y=b.reshape(-1)[i:i+1048576];need(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'FINITE_FORWARD');exact=exact and torch.equal(x,y);xx=x.double();dd=y.double()-xx;maximum=max(maximum,float(dd.abs().max()));err+=float(dd.square().sum());base+=float(xx.square().sum())
                stats[k]=dict(exact=exact,max_abs=maximum,relative_L2=math.sqrt(err/base) if base else (0. if not err else None),elements=a.numel())
            result=dict(exact=all(x['exact'] for x in stats.values()),fields=stats)
        else:result=comparison(expected,value)
        self.comparisons.append(dict(name=name,**result));save(self.root/'numerical-comparisons.json',self.comparisons,True);need(result['exact'],'SAME_PLATFORM_EXACT_NUMERICAL_DIAGNOSTIC')
        del expected
    def observe_value(self,name,value):
        if self.phase=='continuous':self.store(name,value)
        elif self.phase=='replay2':self.compare(name,value)
    def wrap(self,name,observe):
        original=getattr(self.t,name);self.originals[name]=original
        @functools.wraps(original)
        def run(*a,**kw):
            try:self.check();return observe(original,*a,**kw)
            except BaseException as e:self.progress(type(e).__name__);raise
        setattr(self.t,name,run)
    def rng(self,expected=None):
        allstates=isinstance((expected or {}).get('cuda'),list)
        if allstates:need(len(expected['cuda'])==1,'ONLY_ONE_DEVICE_RNG')
        return rng_values(torch.cuda.get_rng_state_all if allstates else torch.cuda.get_rng_state)
    def grads(self):
        out={}
        for n,p in self.named:
            need(p.grad is not None,'ALL504_GRADIENTS');out[n]=cpu(p.grad);need(bool(torch.isfinite(out[n]).all()),'FINITE_GRADIENT')
        need(len(out)==504,'GRADIENT_DENOMINATOR');return out
    def adapter(self):return cpu(get_peft_model_state_dict(self.model))
    def typed_state(self):return dict(adapter=self.adapter(),optimizer=cpu(self.t.optimizer.state_dict()),scheduler=cpu(self.t.lr_scheduler.state_dict()))
    def on_train_begin(self,args,state,control,**kw):
        self.check();need(torch.version.cuda and not torch.version.hip and torch.cuda.device_count()==1 and torch.cuda.current_device()==0 and int(os.environ.get('WORLD_SIZE','1'))==1,'REAL_SINGLE_CUDA_PROCESS')
        need(args.max_steps==25 and args.gradient_accumulation_steps==4 and args.per_device_train_batch_size==1 and args.dataloader_num_workers==4 and args.seed==args.data_seed==42 and not args.dataloader_drop_last and args.bf16 and not args.fp16,'NEW_SINGLECARD_ORIGINAL_OPTIM_RECIPE')
        need(not self.t.is_deepspeed_enabled and not self.t.is_fsdp_enabled and self.t.template.max_length==4096,'NATIVE4096_NO_OTHER_TRAINER')
        binding=obj(D/'NATIVE_BINDING.json');self.model=kw.get('model',self.t.model);self.named,frozen=validate_native_recipe(self.model,args,binding,checkpointing_method=self.t._prepare_gradient_checkpointing);need(len(self.named)==504 and sum(p.numel() for _,p in self.named)==43646976 and all(p.device.type=='cuda' and p.device.index==0 for _,p in self.named),'504_FROZEN_RECIPE')
        providers=model_provider_sources(self.model,binding['model_source_SHA256'],'T04.sft');save(self.root/'recipe.json',dict(frozen=frozen,model_providers=providers,named_SHA256=named_parameter_sha256(self.named),world_size=1,original_world8=False,GC=False,attention='eager'))
        expected={x['canonical_path']:x for x in P['providers']}
        for name in ['get_train_dataloader','_load_rng_state','_load_optimizer_and_scheduler','_init_training_state','_prepare_for_training','_run_epoch','set_initial_training_values','training_step','_clip_grad_norm','_save_checkpoint']:
            actual=provider_record(self.originals.get(name,getattr(self.t,name)));ref=expected.get(actual['path']);need(ref and all(actual[k]==ref[k] for k in ['bytes','SHA256']),'NATIVE_TRAINER_PROVIDER')
        _,rows,_=native_inputs(self.c);self.lookup={signature(x['collated_fields']):x for x in rows};need(len(self.lookup)==72,'72_UNAMBIGUOUS_NATIVE_FIELDS');self.loader=self.t.callback_handler.train_dataloader;length=len(self.loader);updates=(length+3)//4;epochs=self.start//updates;skip=(self.start%updates)*4
        need(length==72 and len(self.entry['initial'])==len(self.entry['initialized'])==1,'ACTUAL72_LOADER_ENTRY');v=self.entry['initial'][0];need(len(v)==7 and v[1]==18 and v[2]==72 and v[4]==4 and v[5]==72 and v[6]==25 and self.entry['initialized'][0]==[epochs,skip],'ACTUAL_NATIVE_CURSOR_NO800_ASSUMPTION');save(self.root/'cursor.json',dict(actual_entry=self.entry,loader_length=length,updates_per_epoch=updates,trained_epochs=epochs,skipped_microbatches=skip))
        need(state.global_step==self.start,'ACTUAL_GLOBAL_STEP');self.continuous=obj(Path(self.c['continuous_root'])/'microbatches.json') if self.phase!='continuous' else None
        if self.start:
            need(args.ignore_data_skip is False and args.resume_only_model is False,'FULL_NATIVE_RESTORE');cp=obj(Path(self.c['continuous_root'])/'checkpoint-20-files.json');self.expected_rng=self.load(Path(cp['path'])/'rng_state.pth',cp['files']['rng_state.pth'])
            for n,value in [('adapter_model.safetensors',self.adapter()),('optimizer.pt',cpu(self.t.optimizer.state_dict())),('scheduler.pt',cpu(self.t.lr_scheduler.state_dict()))]:need(comparison(self.load(Path(cp['path'])/n,cp['files'][n]),value)['exact'],'EXACT_NATIVE20_RESTORED_STATE')
            def rngload(original,checkpoint):
                need(not self.restores and Path(checkpoint)==Path(self.c['checkpoint20']),'ONE_NATIVE_RNG_RESTORE');r=original(checkpoint);need(rng_fingerprint(self.rng(self.expected_rng))==rng_fingerprint(self.expected_rng),'NATIVE_RNG_CALL_RETURN');self.restores.append(dict(exact=True));return r
            self.wrap('_load_rng_state',rngload)
        else:
            need(not self.t.optimizer.state and all(p.grad is None for _,p in self.named),'EMPTY_FRESH_OPTIMIZER')
            if self.phase=='continuous':self.store('initial.pt',dict(parameters={n:cpu(p) for n,p in self.named},RNG=self.rng(),named_SHA256=named_parameter_sha256(self.named)))
            else:
                pin=obj(Path(self.c['continuous_root'])/'capture-pins.json')['files']['initial.pt'];initial=self.load(Path(self.c['continuous_root'])/'initial.pt',pin);need(set(initial['parameters'])=={n for n,p in self.named},'SAME_INITIAL504')
                with torch.no_grad():
                    for n,p in self.named:p.copy_(initial['parameters'][n].to(device=p.device,dtype=p.dtype))
                r=initial['RNG'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda']);need(named_parameter_sha256(self.named)==initial['named_SHA256'] and rng_fingerprint(self.rng())==rng_fingerprint(r),'EXPLICIT_REPLAY_INITIAL_AND_RNG')
                del initial
        self.wrap('training_step',self.training_step);self.wrap('_clip_grad_norm',self.clip);self.handles.append(self.model.register_forward_pre_hook(self.before_forward,with_kwargs=True));self.handles.append(self.model.register_forward_hook(self.after_forward,with_kwargs=True));return control
    def before_forward(self,module,args,kw):
        need(self.pending is not None,'ONE_FORWARD_PER_NATIVE_MICRO');r=dict(input_SHA256=tree_fingerprint(dict(args=args,kwargs=kw)),RNG_before=rng_fingerprint(self.rng()))
        i=len(self.forwards)
        if self.continuous is not None:
            ref=self.continuous['forwards'][i+(80 if self.start else 0)];need(r==ref,'IDENTICAL_NATIVE_FORWARD_INPUT_RNG')
        self.forwards.append(r)
    def after_forward(self,module,args,kw,output):
        if self.phase!='resume' and len(self.forwards)<=8:
            logits=getattr(output,'logits',None);loss=getattr(output,'loss',None);need(isinstance(logits,torch.Tensor) and logits.ndim==3 and tuple(logits.shape[:2])==(1,self.current_native_ref['full_length']) and isinstance(loss,torch.Tensor) and loss.numel()==1,'FULL_NATIVE_FORWARD_LOGITS_LOSS')
            self.observe_value('forward-%02d.pt'%(len(self.forwards)-1),dict(logits=cpu(logits),loss=cpu(loss)))
    def training_step(self,original,model,inputs,*a,**kw):
        need(len(self.records)<self.planned and self.pending is None and (not self.start or len(self.restores)==1),'FULL_MICRO_AND_RESTORE_DENOMINATOR');ref=self.lookup.get(signature(fields(inputs)));need(ref is not None,'ACTUAL_TRAIN72_NATIVE_INPUT')
        self.current_native_ref=ref;labels=inputs['labels'];mask=inputs['attention_mask'];ids=inputs['input_ids'];tokens=int(((labels[0]!=-100)&(mask[0]!=0))[1:].sum());eos=sum(int(((ids[0]==t)&(labels[0]!=-100)).sum()) for t in ref['native_EOS_ids']);need(tokens==ref['effective_supervised_tokens']>0 and eos==ref['supervised_EOS_count']>0 and ids.shape==(1,ref['full_length']),'FULL_NATIVE_LOSS_MASK_EOS')
        row=dict(global_step_before=self.t.state.global_step,ordinal=ref['ordinal'],source_id_SHA256=ref['source_id_SHA256'],training_inputs_SHA256=tree_fingerprint(inputs),effective_tokens=tokens)
        if self.continuous is not None:need(row==self.continuous['inputs'][len(self.records)+(80 if self.start else 0)],'ACTUAL_REFERENCE_INPUT_ORDER')
        self.records.append(row);self.pending=len(self.records)-1;r=original(model,inputs,*a,**kw);need(isinstance(r,torch.Tensor) and r.numel()==1 and bool(torch.isfinite(r)) and len(self.records)==len(self.forwards),'NATIVE_FWD_BWD_RETURN');value=float(r.detach().cpu());self.losses.append(value);self.pending=None
        if self.continuous is not None:need(value==self.continuous['losses'][len(self.losses)-1+(80 if self.start else 0)],'NATIVE_LOSS_COMPARISON')
        self.progress();return r
    def clip(self,original,model):
        step=self.t.state.global_step+1
        if step<=2:self.observe_value('step-%d-preclip.pt'%step,self.grads())
        result=original(model);need(math.isfinite(float(result)),'FINITE_CLIP')
        if step<=2:self.observe_value('step-%d-postclip.pt'%step,self.grads())
        self.clips.append(step);return result
    def on_pre_optimizer_step(self,args,state,control,**kw):
        need(len(self.records)==4*(len(self.steps)+1) and len(self.clips)==len(self.steps)+1,'FOUR_MICROS_BEFORE_OPTIMIZER')
        if state.global_step<2:self.before={n:cpu(p) for n,p in self.named};self.observe_value('step-%d-before.pt'%(state.global_step+1),self.typed_state())
        return control
    def on_optimizer_step(self,args,state,control,**kw):
        step=state.global_step+1;optimizer_steps(self.t.optimizer.state_dict(),step,504)
        if step<=2:
            delta=parameter_delta(self.before,{n:cpu(p) for n,p in self.named});need((delta['changed_elements']==0 and all(g['lr']==0 for g in self.t.optimizer.param_groups)) if step==1 else delta['changed_elements']>0,'WARMUP0_THEN_NONZERO');self.observe_value('step-%d-after.pt'%step,self.typed_state());save(self.root/('step-%d-delta.json'%step),delta);del self.before
        return control
    def on_step_end(self,args,state,control,**kw):
        need(state.global_step==self.start+len(self.steps)+1,'ACTUAL_STEP_SEQUENCE');self.steps.append(state.global_step);self.progress()
        if self.phase=='replay2' and state.global_step==2:control.should_training_stop=True;control.should_save=False
        if self.phase!='replay2' and state.global_step==25:control.should_save=True
        return control
    def on_save(self,args,state,control,**kw):
        step=state.global_step;need(step in ([2] if self.phase=='replay2' else ([25] if self.start else [10,20,25])) and step not in self.saved,'ACTUAL_NATIVE_SAVED_STEPS');self.saved.append(step);root=Path(args.output_dir)/('checkpoint-%d'%step);pins={n:self.pin(root/n,512*1024**2) for n in CORE};need(sum(x['bytes'] for x in pins.values())<=512*1024**2,'SINGLE_RANK_FULLSTATE_CAP');st=obj(root/'trainer_state.json');need(st['global_step']==step and st['max_steps']==25 and float(st['epoch'])==(step//18+(step%18)*4/72),'ACTUAL_CROSS_EPOCH_CHECKPOINT_CURSOR');save(self.root/('checkpoint-%d-files.json'%step),dict(path=str(root),files=pins));return control
    def on_train_end(self,args,state,control,**kw):
        self.check();need(state.global_step==self.end and self.steps==list(range(self.start+1,self.end+1)) and len(self.records)==len(self.forwards)==len(self.losses)==self.planned and len(self.clips)==self.end-self.start,'COMPLETE_PHASE_DENOMINATOR')
        recovery=None
        if self.phase=='continuous':need(self.saved==[10,20,25] and len(self.capture_pins)==17,'COMPLETE_CONTINUOUS_CAPTURE')
        elif self.phase=='replay2':need(len(self.comparisons)==16 and all(x['exact'] for x in self.comparisons),'ALL16_NUMERICAL_COMPARISONS')
        else:
            need(self.saved==[25] and len(self.restores)==1,'FULL_NATIVE_RESUME');ref=obj(Path(self.c['continuous_root'])/'checkpoint-25-files.json');own_checkpoint=obj(self.root/'checkpoint-25-files.json');recovery={}
            for n in ['adapter_model.safetensors','optimizer.pt','scheduler.pt']:recovery[n]=comparison(self.load(Path(ref['path'])/n,ref['files'][n]),self.load(Path(own_checkpoint['path'])/n,own_checkpoint['files'][n]));need(recovery[n]['exact'],'FULL_NATIVE_FINAL_STATE_EQUAL')
            recovery['saved_RNG_exact']=rng_fingerprint(self.load(Path(ref['path'])/'rng_state.pth',ref['files']['rng_state.pth']))==rng_fingerprint(self.load(Path(own_checkpoint['path'])/'rng_state.pth',own_checkpoint['files']['rng_state.pth']));need(recovery['saved_RNG_exact'],'FULL_SAVED_RNG_EQUAL');save(self.root/'recovery.json',recovery)
        save(self.root/'microbatches.json',dict(inputs=self.records,forwards=self.forwards,losses=self.losses,steps=self.steps));save(self.root/'receipt.json',dict(passed=True,request_id=self.c['request_id'],publication=self.c['publication'],epoch=self.c['epoch'],deadline=self.c['deadline'],phase=self.phase,owner=own(),updates=len(self.steps),microbatches=len(self.losses),final_step=state.global_step,saved_steps=self.saved,capture_files=len(self.capture_pins),capture_bytes=self.capture_bytes,rng_restore_calls=len(self.restores),numerical_comparisons=len(self.comparisons),numerical_exact=all(x['exact'] for x in self.comparisons) if self.phase=='replay2' else None,recovery_exact=bool(recovery and recovery['saved_RNG_exact']) if self.start else None,formal17=False,Model_GO=False,domain_capability_qualified=False))
        for h in self.handles:h.remove()
        for n,v in self.originals.items():setattr(self.t,n,v)
        return control
if KEY in callbacks_map:raise ValueError('DUPLICATE_NEW_OPTIN_CALLBACK')
callbacks_map[KEY]=NativeRender72
