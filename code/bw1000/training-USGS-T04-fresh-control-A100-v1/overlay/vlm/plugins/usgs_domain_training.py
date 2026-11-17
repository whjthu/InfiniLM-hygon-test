"""Explicit T04 USGS callback. Original396 registry entries and math stay unchanged."""
import functools,hashlib,json,os,signal,sys
from pathlib import Path
import torch
from swift.callbacks import TrainerCallback,callbacks_map
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vlm.runtime.usgs_domain_contract import load_config,load_inputs,remaining,require,verify_supervisor,verify_provider_files,read_pin
from vlm.runtime.usgs_domain_witness import DomainWitness,tree_fingerprint,cpu_tree
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.provider_sources import model_provider_sources
from vlm.runtime.initial_trainables import named_parameter_sha256,restore_initial_trainables
from vlm.runtime.resume_witness import rng_values,rng_fingerprint
KEY='native_usgs_domain_witness'

class NativeUSGSDomain(TrainerCallback):
    def __init__(self,swift_args,trainer):
        super().__init__(swift_args,trainer)
        self.trainer=trainer;self.witness=None;self.config=load_config();self.entry_methods={}
        self.entry_observations=dict(initial_training_values=[],init_training_state=[])
        root=Path(self.config['output']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.entry_path=root/('entry-rank-%02d.json'%int(os.environ['RANK']))
        with self.entry_path.open('x') as f:json.dump(dict(phase='CALLBACK_CREATED',mode=self.config['mode'],arm=self.config['arm']),f)
        self.observe_entry('set_initial_training_values','initial_training_values');self.observe_entry('_init_training_state','init_training_state')

    def observe_entry(self,name,key):
        original=getattr(self.trainer,name);self.entry_methods[name]=original
        @functools.wraps(original)
        def call(*args,**kwargs):
            remaining(self.config);result=original(*args,**kwargs)
            require(isinstance(result,tuple) and all(x is None or type(x)is int for x in result),'ACTUAL_TRAINER_ENTRY_SCHEMA')
            row=dict(returned_values=list(result))
            if key=='initial_training_values':
                loader=args[1] if len(args)>1 else kwargs['dataloader'];row.update(actual_loader_length=len(loader),actual_loader_class=type(loader).__module__+'.'+type(loader).__qualname__)
            self.entry_observations[key].append(row)
            temp=self.entry_path.with_suffix('.tmp');temp.write_text(json.dumps(self.entry_observations,allow_nan=False)+'\n');temp.replace(self.entry_path);return result
        setattr(self.trainer,name,call)

    def on_train_begin(self,args,state,control,**kwargs):
        c=self.config;remaining(c);verify_supervisor(c);verify_provider_files(c)
        require(torch.version.cuda and not torch.version.hip and torch.cuda.device_count()==8,'A100_CUDA_ONLY')
        rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);dist=torch.distributed
        require(int(os.environ['WORLD_SIZE'])==8 and rank==local and 0<=rank<8 and torch.cuda.current_device()==local and dist.is_initialized() and dist.get_world_size()==8 and dist.get_rank()==rank,'TRUE8_RANK_DEVICE')
        forbidden={'BaselineCallback','FirstUpdates','NativeFirstUpdates','NativeDropoutAlignment','NativeResume','NativeCudaResume','QwenResume'}
        callbacks=self.trainer.callback_handler.callbacks
        require(sum(cb is self for cb in callbacks)==1 and not any(type(cb).__name__ in forbidden for cb in callbacks),'OLD_CALLBACK_COEXISTENCE_FORBIDDEN')
        binding,native_rows,occurrence=load_inputs(c);model=kwargs.get('model',self.trainer.model)
        require(self.trainer.template.template_meta.template_type==binding['resolved_template_type'],'T04_ACTUAL_TEMPLATE')
        named,frozen=validate_native_recipe(model,args,binding,checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        require(len(named)==504 and sum(p.numel() for _,p in named)==43646976 and all(p.device.type=='cuda' and p.device.index==local for _,p in named),'ACTUAL504_PARAMETERS')
        require(state.global_step==(0 if c['mode']=='fresh25' else 20),'ACTUAL_START_STEP')
        providers=model_provider_sources(model,binding['model_source_SHA256'],'T04.sft')
        self.witness=DomainWitness(self.trainer,model,self.trainer.callback_handler.train_dataloader,c,binding,rank,self.entry_observations,self.entry_methods,native_rows=native_rows,occurrence=occurrence)
        w=self.witness;w.write('owner.json',dict(rank=rank,PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),pair_id=c['pair_id'],arm=c['arm'],mode=c['mode'],request_id=c['request_id'],epoch=c['epoch'],deadline=c['deadline']))
        w.write('native-recipe-providers.json',dict(frozen=frozen,providers=providers))
        before_rng=rng_fingerprint(rng_values(torch.cuda.get_rng_state))
        if c['mode']=='fresh25':
            require(not self.trainer.optimizer.state and all(p.grad is None for _,p in named),'FRESH_EMPTY_OPTIMIZER')
            initial=c.get('initial_state')
            restored=None
            if initial is not None:
                reference=json.loads(read_pin(c['pair_initial_reference']));require(reference['pair_id']==c['pair_id'] and reference['arm']=='domain' and reference['rank']==0 and reference['global_step']==0 and reference['tensors']==504 and reference['elements']==43646976 and reference['initial_state']==initial,'ACTUAL_DOMAIN_INITIAL_REFERENCE')
                restored=restore_initial_trainables(named,self.trainer.optimizer,global_step=0,path=initial['path'],file_bytes=initial['bytes'],file_SHA256=initial['SHA256'],named_SHA256=initial['named_SHA256'])
            else:require(c['arm']=='domain','ONLY_FIRST_DOMAIN_CREATES_INITIAL')
            digest=named_parameter_sha256(named)
            # Existing true process group only; bounded 32-byte digest agreement, no synthetic RankInfo.
            code=torch.tensor(list(bytes.fromhex(digest)),dtype=torch.uint8,device=torch.device("cuda",local));expected=code.clone();dist.broadcast(expected,src=0)
            flag=torch.tensor(int(torch.equal(code,expected)),dtype=torch.int32,device=torch.device("cuda",local));dist.all_reduce(flag,op=dist.ReduceOp.MIN);require(bool(flag.item()),'ALL8_INITIAL_VALUES_EQUAL')
            initial_receipt=dict(pair_id=c['pair_id'],arm=c['arm'],rank=rank,global_step=0,tensors=504,elements=43646976,named_SHA256=digest,restoration=restored,RNG_restored=False)
            if rank==0 and initial is None:
                path=Path(c['output'])/'initial.pt';require(not path.exists(),'INITIAL_NO_REPLAY')
                payload=dict(parameters={n:p.detach().cpu().contiguous() for n,p in named},RNG=dict(restored=False,observed_fingerprint=before_rng),manifest=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),numel=p.numel()) for n,p in named],global_step=0)
                require(all(bool(torch.isfinite(v).all()) for v in payload['parameters'].values()),'FINITE_ACTUAL_INITIAL')
                with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                require(path.stat().st_size<=256*1024**2,'INITIAL_CAP')
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024**2),b''):remaining(c);h.update(chunk)
                initial_receipt['initial_state']=dict(path=str(path),bytes=path.stat().st_size,SHA256=h.hexdigest(),named_SHA256=digest)
                del payload
            w.write('initial-state.json',initial_receipt)
        require(before_rng==rng_fingerprint(rng_values(torch.cuda.get_rng_state)),'INITIAL_OBSERVER_MUST_NOT_RESTORE_OR_CONSUME_RNG')
        w.attach()
        def expired(*_):
            w.progress(dict(type='TimeoutError',reason='DOMAIN_ORIGINAL_EPOCH'));raise TimeoutError('DOMAIN_ORIGINAL_EPOCH')
        previous=signal.getitimer(signal.ITIMER_REAL)[0];signal.signal(signal.SIGALRM,expired);left=remaining(c);signal.setitimer(signal.ITIMER_REAL,min(previous,left) if previous else left)
        return control

    def invoke(self,name):
        require(self.witness is not None,'DOMAIN_OBSERVER_NOT_INITIALIZED')
        try:getattr(self.witness,name)()
        except BaseException as e:self.witness.progress(dict(type=type(e).__name__,reason=str(e)[:180]));raise

    def on_pre_optimizer_step(self,args,state,control,**kw):self.invoke('pre_optimizer');return control
    def on_optimizer_step(self,args,state,control,**kw):self.invoke('post_optimizer');return control
    def on_step_end(self,args,state,control,**kw):
        self.invoke('step_end')
        if state.global_step==25:control.should_save=True
        return control
    def on_save(self,args,state,control,**kw):self.invoke('saved');return control
    def on_train_end(self,args,state,control,**kw):self.invoke('finish');return control

if KEY in callbacks_map:raise ValueError('DUPLICATE_DOMAIN_CALLBACK')
callbacks_map[KEY]=NativeUSGSDomain
