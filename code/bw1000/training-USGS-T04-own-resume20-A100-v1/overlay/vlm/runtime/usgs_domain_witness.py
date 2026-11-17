"""Observe native Swift calls and saved state; no optimizer, loop, RNG reset or input mutation."""
import collections,datetime,functools,hashlib,json,os
from pathlib import Path
import torch
from vlm.runtime.cuda_resume_witness import (CudaResumeWitness,tree_fingerprint,provider_record,
    optimizer_steps,trusted_load,comparison)
from vlm.runtime.resume_witness import rng_values,rng_fingerprint
from vlm.runtime.first_update_audit import fingerprint,parameter_delta
from .usgs_domain_contract import (MODES,MEMBERS,remaining,require,read_pin,cursor_contract,verify_file,validate_supervised_counter)
AUDIT_UPDATES={1,2,20,21,22,23,24,25}

def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def field_ids(value):
    return {k:{n:v for n,v in fingerprint(cpu_tree(x)).items() if n!='floating_l2'} for k,x in sorted(value.items())}

def signature(fields):return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class DomainWitness(CudaResumeWitness):
    def __init__(self,trainer,model,loader,config,binding,rank,entry_observations,entry_methods,*,native_rows,occurrence):
        self.trainer,self.model,self.loader=trainer,model,loader;self.config,self.binding,self.rank=config,binding,rank
        self.root=Path(config['output'])/('rank-%02d'%rank);self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        self.inputs,self.forward_inputs,self.losses,self.steps,self.clips=[],[],[],[],[]
        self.rng_loads,self.saved_steps,self.handles=[],[],[];self.original_methods=dict(entry_methods);self.finished=False
        self.entry_observations=entry_observations;self.loader_epoch_calls=[];self.expected_rng20=None
        self.start,self.end,self.planned=MODES[self.config['mode']]
        self.native_rows=native_rows;self.occurrence=occurrence;self.lookup={}
        for row in native_rows:
            key=signature(row['collated_fields']);require(key not in self.lookup,'AMBIGUOUS_NATIVE45_FIELD_IDENTITY');self.lookup[key]=row
        self.forward_pending=None;self.native_returns=[];self.state_audits=[];self.initial=None
        self.write('domain-plan.json',dict(mode=self.config['mode'],arm=self.config['arm'],pair_id=self.config['pair_id'],planned_updates=self.end-self.start,planned_rank_microbatches=self.planned,planned_global_microbatches=self.planned*8,formal17=False))

    def check(self):remaining(self.config)

    def progress(self,error=None):
        self.write('progress.json',dict(mode=self.config['mode'],arm=self.config['arm'],planned_microbatches=self.planned,completed_microbatches=len(self.losses),completed_updates=self.steps,saved_steps=self.saved_steps,error=error,finished=self.finished))

    def attach(self):
        self.check();a=self.trainer.args;state=self.trainer.state
        require(state.global_step==self.start and a.max_steps==25 and a.gradient_accumulation_steps==4 and a.per_device_train_batch_size==1 and a.dataloader_num_workers==4 and not a.dataloader_drop_last and a.seed==a.data_seed==42,'DOMAIN_NATIVE_RECIPE')
        require(a.bf16 is True and not a.fp16 and self.trainer.template.max_length==4096 and not self.trainer.is_deepspeed_enabled and not self.trainer.is_fsdp_enabled,'DOMAIN_PRECISION')
        expected={r['canonical_path']:r for r in self.config['providers']};active={}
        for name in ['get_train_dataloader','_load_rng_state','_load_optimizer_and_scheduler','_init_training_state','_prepare_for_training','_run_epoch','set_initial_training_values','training_step','_clip_grad_norm','_save_checkpoint']:
            actual=provider_record(self.original_methods.get(name,getattr(self.trainer,name)));pin=expected.get(actual['path'])
            require(pin is not None and all(actual[k]==pin[k] for k in ['bytes','SHA256']),'ACTUAL_TRAINER_PROVIDER');active[name]=actual
        self.write('trainer-method-providers.json',active)
        saved=self.ref(20,'trainer_state.json') if self.start else None
        self.cursor=cursor_contract(self.config['mode'],len(self.loader),self.entry_observations['initial_training_values'],self.entry_observations['init_training_state'],saved['epoch'] if saved else None)
        self.write('loader.json',dict(**self.cursor,loader_provider=provider_record(type(self.loader)),batch_sampler_provider=provider_record(type(self.loader.batch_sampler)),sampler_scalars={k:v for k,v in vars(self.loader.batch_sampler).items() if k in {'seed','epoch','rank','num_replicas','shuffle','drop_last'} and type(v) in (int,bool,str,float)}))
        if self.start:
            require(a.ignore_data_skip is False and a.resume_only_model is False and saved['global_step']==20,'FULL_NATIVE_RESUME_REQUIRED')
            restored=self.state_comparison(20);self.write('restored-state-before-rng.json',restored)
            require(all(restored[k]['exact'] for k in ('adapter','optimizer','scheduler')),'EXACT_RESTORED_STATE')
            self.expected_rng20=self.ref(20,f'rng_state_{self.rank}.pth');self.wrap('_load_rng_state',self.observe_rng)
            reference=json.loads(read_pin(self.config['continuous_reference'][str(self.rank)]))
            require(reference['mode']=='fresh25' and reference['arm']==self.config['arm'] and reference['pair_id']==self.config['pair_id'] and len(reference['training_inputs'])==100,'OWN_CONTINUOUS100_REFERENCE')
            self.tail=reference['training_inputs'][80:];self.forward_tail=reference['forward_inputs'][80:];self.loss_tail=reference['losses'][80:]
            require(len(self.forward_tail)==len(self.loss_tail)==20 and all(type(x) in (int,float) and __import__('math').isfinite(x) for x in self.loss_tail),'CONTINUOUS20_FORWARD_AND_LOSS_REFERENCE')
        else:
            require(not self.trainer.optimizer.state_dict()['state'],'FRESH_OPTIMIZER_MUST_BE_EMPTY')
            self.tail=None
        self.wrap('training_step',self.observe_training_step);self.wrap('_clip_grad_norm',self.observe_clip)
        if callable(getattr(self.loader,'set_epoch',None)):
            original=self.loader.set_epoch;self.original_loader_set_epoch=original
            def observed_epoch(epoch,*args,**kwargs):
                value=original(epoch,*args,**kwargs);self.loader_epoch_calls.append(dict(epoch=epoch));self.write('loader-set-epoch.json',self.loader_epoch_calls);return value
            self.loader.set_epoch=observed_epoch
        self.handles.append(self.model.register_forward_pre_hook(self.before_forward,with_kwargs=True));self.progress()

    def before_forward(self,module,args,kwargs):
        require(self.forward_pending is not None,'UNDECLARED_MODEL_FORWARD')
        before=rng_fingerprint(rng_values(torch.cuda.get_rng_state))
        value=tree_fingerprint(dict(args=args,kwargs=kwargs))
        require(before==rng_fingerprint(rng_values(torch.cuda.get_rng_state)),'OBSERVER_CHANGED_RNG')
        if self.start:
            expected=self.forward_tail[self.forward_pending];require(value==expected['input_SHA256'] and before==expected['RNG_before'],'CONTINUOUS_REAL_FORWARD_INPUT_AND_RNG')
        self.forward_inputs.append(dict(microbatch=self.forward_pending,input_SHA256=value,RNG_before=before))
        require(len(self.forward_inputs)==len(self.inputs),'ONE_NATIVE_FORWARD_PER_MICRO')

    def observe_training_step(self,original,model,inputs,*args,**kwargs):
        require(len(self.inputs)<self.planned and self.forward_pending is None and (not self.start or len(self.rng_loads)==1),'MICRO_OR_RNG_BOUNDARY')
        fields=field_ids(inputs);key=signature(fields);row=self.lookup.get(key)
        if row is None:
            self.write('input-schema-failure.json',dict(actual_field_names=sorted(fields),actual_field_identities=fields,expected_native45_signature_count=45));raise ValueError('ACTUAL_TRAINING_INPUT_NOT_NATIVE45')
        labels=inputs['labels'].detach().cpu();mask=inputs['attention_mask'].detach().cpu();ids=inputs['input_ids'].detach().cpu()
        require(tuple(ids.shape)==(1,row['full_length']) and labels.shape==ids.shape and mask.shape==ids.shape,'FULL_INPUT_LENGTH')
        count=int(((labels[0]!=-100)&(mask[0]!=0))[1:].sum());eos=sum(int(((ids[0]==t)&(labels[0]!=-100)).sum()) for t in row['native_EOS_ids'])
        reference_counter_available=validate_supervised_counter(row,count,eos)
        record=dict(microbatch=len(self.inputs),global_step_before=self.trainer.state.global_step,canonical_ordinal=row['ordinal'],source_id_SHA256=row['source_id_SHA256'],collated_fields_SHA256=key,input_tokens=row['full_length'],effective_supervised_tokens=count,reference_counter_available=reference_counter_available,supervised_EOS_count=eos,training_inputs_SHA256=tree_fingerprint(inputs))
        if self.tail is not None:
            expected=self.tail[len(self.inputs)];require(all(record[k]==expected[k] for k in ['global_step_before','canonical_ordinal','source_id_SHA256','collated_fields_SHA256','training_inputs_SHA256']),'ACTUAL_CONTINUOUS_TAIL_INPUT_MISMATCH')
        self.inputs.append(record);self.forward_pending=len(self.inputs)-1
        result=original(model,inputs,*args,**kwargs)
        require(isinstance(result,torch.Tensor) and result.numel()==1 and bool(torch.isfinite(result)) and len(self.forward_inputs)==len(self.inputs),'NATIVE_TRAINING_STEP_RETURN')
        actual_loss=float(result.detach().cpu())
        if self.start:
            observation=dict(ordinal=self.inputs[-1]['canonical_ordinal'],microbatch=len(self.losses),global_step_before=self.inputs[-1]['global_step_before'],
                expected_loss=self.loss_tail[len(self.losses)],actual_loss=actual_loss,exact=actual_loss==self.loss_tail[len(self.losses)],
                planned_microbatches=self.planned,attempted_microbatches=len(self.inputs),completed_native_returns=len(self.losses)+1,native_training_step_called=True,native_training_step_returned_finite=True)
            self.write('tail-loss-observation.json',observation)
            if not observation['exact']:
                # The real native forward/backward returned before this diagnostic failure.
                self.forward_pending=None;self.losses.append(actual_loss)
                self.progress(dict(type='CONTINUOUS_TAIL_LOSS_MISMATCH',planned_microbatches=self.planned,attempted_microbatches=len(self.inputs),completed_native_returns=len(self.losses)))
                require(False,'ACTUAL_CONTINUOUS_TAIL_LOSS_MISMATCH')
        self.forward_pending=None;self.losses.append(actual_loss);self.progress();return result

    def observe_clip(self,original,model):
        step=self.trainer.state.global_step+1
        before=self.gradient_summary() if step in AUDIT_UPDATES else None
        result=original(model)
        after=self.gradient_summary() if step in AUDIT_UPDATES else None
        norm=float(result.detach().cpu()) if isinstance(result,torch.Tensor) else float(result)
        require(__import__('math').isfinite(norm),'FINITE_NATIVE_CLIP_NORM')
        self.clips.append(dict(global_step=step,pre=before,post=after,native_norm=norm));return result

    def pre_optimizer(self):
        require(len(self.inputs)==(len(self.steps)+1)*4 and len(self.clips)==len(self.steps)+1,'FOUR_NATIVE_MICROS_BEFORE_UPDATE')
        step=self.trainer.state.global_step+1
        if step in (1,2):self.first_before={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad};self.first_lr=[float(g['lr']) for g in self.trainer.optimizer.param_groups]
        if step in AUDIT_UPDATES:self.write(f'update-{step:02d}-before.json',dict(adapter_SHA256=tree_fingerprint(self.live_adapter()),gradient=self.clips[-1]))

    def post_optimizer(self):
        step=self.trainer.state.global_step+1;optimizer_steps(self.trainer.optimizer.state_dict(),step,504)
        if step in (1,2):
            after={n:p.detach().cpu() for n,p in self.model.named_parameters() if p.requires_grad};delta=parameter_delta(self.first_before,after)
            self.write(f'first-update-{step}-actual-delta.json',dict(global_step=step,learning_rates=self.first_lr,parameter_delta=delta,optimizer_state_entries=len(self.trainer.optimizer.state),actual_optimizer_step=step))
            require(delta['parameter_tensors']==504 and ((delta['changed_elements']==0 and all(lr==0 for lr in self.first_lr)) if step==1 else delta['changed_elements']>0),'FIRST2_ACTUAL_UPDATE_CONTRACT')
            del self.first_before,after
        if step in AUDIT_UPDATES:self.write(f'update-{step:02d}-after.json',dict(adapter_SHA256=tree_fingerprint(self.live_adapter()),optimizer_steps=step))

    def step_end(self):
        step=self.trainer.state.global_step;require(step==self.start+1+len(self.steps),'ACTUAL_UPDATE_SEQUENCE');self.steps.append(step)
        self.write(f'update-{step:02d}-scheduler.json',dict(global_step=step,epoch=self.trainer.state.epoch,scheduler_SHA256=tree_fingerprint(self.trainer.lr_scheduler.state_dict())));self.progress()

    def saved(self):
        step=self.trainer.state.global_step;require(step in ([10,20,25] if not self.start else [25]) and step not in self.saved_steps,'CHECKPOINT_STEP_DENOMINATOR');self.saved_steps.append(step);self.progress()

    def finish(self):
        self.check();require(self.forward_pending is None and len(self.inputs)==len(self.forward_inputs)==len(self.losses)==self.planned and self.steps==list(range(self.start+1,26)) and len(self.clips)==25-self.start and self.saved_steps==([10,20,25] if not self.start else [25]),'FULL_NATIVE_DENOMINATORS')
        work=torch.distributed.barrier(device_ids=[torch.cuda.current_device()],async_op=True);require(work.wait(timeout=datetime.timedelta(seconds=max(.001,min(60.,self.check_left()-5)))) is not False,'FINAL_BARRIER')
        if not self.start:
            counts=collections.Counter(r['canonical_ordinal'] for r in self.inputs);actual=torch.tensor([counts[i] for i in range(45)],dtype=torch.int64,device=torch.cuda.current_device());torch.distributed.all_reduce(actual,op=torch.distributed.ReduceOp.SUM)
            expected=collections.Counter(self.occurrence['original_index_by_physical_row']);require(actual.cpu().tolist()==[expected[i] for i in range(45)],'ACTUAL_GLOBAL800_OCCURRENCES');self.write('global800-occurrences.json',dict(counts=actual.cpu().tolist(),total=int(actual.sum().item())))
        self.write('microbatches.json',dict(mode=self.config['mode'],arm=self.config['arm'],pair_id=self.config['pair_id'],training_inputs=self.inputs,forward_inputs=self.forward_inputs,losses=self.losses,canonical_occurrences=dict(collections.Counter(r['canonical_ordinal'] for r in self.inputs)),actual_input_tokens=sum(r['input_tokens'] for r in self.inputs),actual_supervised_tokens=sum(r['effective_supervised_tokens'] for r in self.inputs),global800_requires_all8_aggregation=True))
        compared=None
        if self.start:
            compared=self.state_comparison(25);saved=self.read_own(Path(self.trainer.args.output_dir)/'checkpoint-25',f'rng_state_{self.rank}.pth');compared['saved_rank_RNG_exact']=rng_fingerprint(saved)==rng_fingerprint(self.ref(25,f'rng_state_{self.rank}.pth'))
            compared['loss_tail_exact']=self.losses==self.loss_tail;compared['loss_tail_count']=len(self.losses)
            require(compared['loss_tail_exact'] and compared['loss_tail_count']==20,'COMPLETE20_LOSS_COMPARISON')
            self.write('resume25-comparison.json',compared)
        if self.rank==0:
            for step in self.saved_steps:
                root=Path(self.trainer.args.output_dir)/f'checkpoint-{step}';pins={}
                for name in sorted(MEMBERS):
                    p=root/name;st=p.stat();require(not p.is_symlink() and 0<st.st_size<=512*1024**2,'SAVED_FULLSTATE_MEMBER')
                    h=hashlib.sha256()
                    with p.open('rb') as f:
                        for block in iter(lambda:f.read(1024**2),b''):self.check();h.update(block)
                    after=p.stat();require((st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'SAVED_STATE_STABLE')
                    pins[name]=dict(bytes=st.st_size,SHA256=h.hexdigest())
                require(sum(p['bytes'] for p in pins.values())<=512*1024**2,'FULL_STATE_AGGREGATE')
                state=json.loads((root/'trainer_state.json').read_bytes());require(state['global_step']==step and state['max_steps']==25 and float(state['epoch'])==step/25,'SAVED_ACTUAL_STEP_CURSOR')
                self.write(f'checkpoint-{step}-files.json',dict(path=str(root),files=pins))
            # Actual saved final typed state must equal live state; no producer summary shortcut.
            root=Path(self.trainer.args.output_dir)/'checkpoint-25'
            for name,live in [('adapter_model.safetensors',self.live_adapter()),('optimizer.pt',self.trainer.optimizer.state_dict()),('scheduler.pt',self.trainer.lr_scheduler.state_dict())]:
                require(comparison(self.read_own(root,name),live)['exact'],'SAVED25_EQUALS_LIVE')
        self.finished=True;self.progress();self.write('receipt.json',dict(mode=self.config['mode'],arm=self.config['arm'],pair_id=self.config['pair_id'],rank=self.rank,completed_microbatches=self.planned,updates=len(self.steps),completed_step=25,recovery_exact=(all(compared[k]['exact'] for k in ['adapter','optimizer','scheduler']) and compared['saved_rank_RNG_exact'] and compared['loss_tail_exact']) if compared else None,formal17=False,performance_qualified=False,effect_qualified=False));self.detach()

    def check_left(self):return remaining(self.config)

    def read_own(self,root,name):
        p=root/name;st=p.stat();require(not p.is_symlink() and 0<st.st_size<=512*1024**2,'OWN_STATE_BOUND')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):self.check();h.update(block)
        return trusted_load(root,name,dict(bytes=st.st_size,SHA256=h.hexdigest()),self.check)
