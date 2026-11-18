"""Explicit T03 USGS callback. Original396 registry entries and math stay unchanged."""
import functools,hashlib,json,math,os,random,signal,sys
import numpy as np
from pathlib import Path
import torch
from swift.callbacks import TrainerCallback,callbacks_map
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vlm.runtime.usgs_t03_contract import load_config,load_inputs,remaining,require,verify_supervisor,verify_provider_files,read_pin
from vlm.runtime.usgs_t03_witness import DomainWitness,tree_fingerprint,cpu_tree
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.provider_sources import model_provider_sources
from vlm.runtime.initial_trainables import named_parameter_sha256,restore_initial_trainables
from vlm.runtime.resume_witness import rng_values,rng_fingerprint
KEY='native_usgs_t03_witness'


def restore_rank_original(named, optimizer, rank, pin):
    # Reuse the qualified names/shape/finite/manifest/fullSHA weights-only parameter loader.
    restored = restore_initial_trainables(named, optimizer, global_step=0,
        path=pin['path'], file_bytes=pin['bytes'], file_SHA256=pin['SHA256'], named_SHA256=pin['named_SHA256'])
    # A second bounded same-FD hash+weights-only read obtains only the original rank RNG.
    # This cost is explicit; no arbitrary pickle globals, seed replacement or RNG fabrication.
    path=Path(pin['path']);before=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(path.is_file() and not path.is_symlink() and before.st_size==pin['bytes']<=256*1024**2,'ORIGINAL_INITIAL_RNG_FILE')
    with path.open('rb') as f:
        require(identity(before)==identity(os.fstat(f.fileno())),'ORIGINAL_INITIAL_RNG_OPEN_IDENTITY')
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        require(h.hexdigest()==pin['SHA256'] and identity(before)==identity(os.fstat(f.fileno())),'ORIGINAL_INITIAL_RNG_FULL_SHA')
        f.seek(0);payload=torch.load(f,map_location='cpu',weights_only=True)
        require(identity(before)==identity(os.fstat(f.fileno()))==identity(path.stat()) and not path.is_symlink(),'ORIGINAL_INITIAL_RNG_STABLE_AFTER_LOAD')
    require(isinstance(payload,dict) and set(payload)=={'parameters','RNG','manifest','global_step'} and payload['global_step']==0,'ORIGINAL_INITIAL_SCHEMA')
    r=payload['RNG'];require(isinstance(r,dict) and set(r)=={'python','numpy','torch_CPU','torch_device','device_index'} and r['device_index']==rank,'ORIGINAL_RANK_RNG_SCHEMA')
    n=r['numpy'];require(isinstance(n,dict) and set(n)=={'algorithm','keys','position','has_gauss','cached_gaussian'} and n['algorithm']=='MT19937'
        and len(n['keys'])==624 and all(type(x)is int and 0<=x<2**32 for x in n['keys']) and type(n['position'])is int and 0<=n['position']<=624
        and n['has_gauss'] in (0,1) and math.isfinite(n['cached_gaussian']),'ORIGINAL_NUMPY_RNG')
    numpy_state=(n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian'])
    probe=random.Random(0);probe.setstate(r['python']);np_probe=np.random.RandomState(0);np_probe.set_state(numpy_state)
    for key in ('torch_CPU','torch_device'):
        value=r[key];require(type(value)is torch.Tensor and value.device.type=='cpu' and value.dtype==torch.uint8 and value.ndim==1
            and value.is_contiguous() and 0<value.numel()<=65536,'ORIGINAL_TORCH_RNG')
    expected=rng_fingerprint(dict(python=r['python'],numpy=numpy_state,cpu=r['torch_CPU'],cuda=r['torch_device']))
    random.setstate(r['python']);np.random.set_state(numpy_state);torch.set_rng_state(r['torch_CPU']);torch.cuda.set_rng_state(r['torch_device'],device=rank)
    observed=rng_fingerprint(rng_values(torch.cuda.get_rng_state));require(observed==expected,'ALL_ORIGINAL_RANK_RNG_COMPONENTS_RESTORED')
    restored.update(RNG_restored=True,rank=rank,original_rank_RNG_fingerprint=expected,actual_RNG_after=observed,
        initial_full_SHA_scans=2,full_SHA_logical_bytes=2*pin['bytes'],weights_only_deserializations=2,
        deserialized_parameter_bytes=2*43646976*4,total_logical_file_read_bytes=None,
        physical_disk_IO_bytes=None,seed_substitution=False)
    del payload,r
    return restored

class NativeUSGST03(TrainerCallback):
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
        forbidden={'BaselineCallback','FirstUpdates','NativeFirstUpdates','NativeDropoutAlignment','NativeResume','NativeCudaResume','QwenResume','NativeUSGSDomain'}
        callbacks=self.trainer.callback_handler.callbacks
        require(sum(cb is self for cb in callbacks)==1 and not any(type(cb).__name__ in forbidden for cb in callbacks),'OLD_CALLBACK_COEXISTENCE_FORBIDDEN')
        binding,native_rows,occurrence=load_inputs(c);model=kwargs.get('model',self.trainer.model)
        require(self.trainer.template.template_meta.template_type==binding['resolved_template_type'],'T03_ACTUAL_TEMPLATE')
        named,frozen=validate_native_recipe(model,args,binding,checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        require(len(named)==504 and sum(p.numel() for _,p in named)==43646976 and all(p.device.type=='cuda' and p.device.index==local for _,p in named),'ACTUAL504_PARAMETERS')
        require(state.global_step==(0 if c['mode']=='fresh25' else 20),'ACTUAL_START_STEP')
        checkpointing=validate_checkpointing_lifecycle(model.get_base_model(),args,binding,self.trainer._prepare_gradient_checkpointing)
        providers=model_provider_sources(model,binding['model_source_SHA256'],'T03.sft')
        self.witness=DomainWitness(self.trainer,model,self.trainer.callback_handler.train_dataloader,c,binding,rank,self.entry_observations,self.entry_methods,native_rows=native_rows,occurrence=occurrence)
        w=self.witness;w.write('owner.json',dict(rank=rank,PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),pair_id=c['pair_id'],arm=c['arm'],mode=c['mode'],request_id=c['request_id'],epoch=c['epoch'],deadline=c['deadline']))
        w.write('native-recipe-providers.json',dict(frozen=frozen,providers=providers,checkpointing_lifecycle=checkpointing))
        before_rng=rng_fingerprint(rng_values(torch.cuda.get_rng_state))
        require(c['mode']=='fresh25' and not self.trainer.optimizer.state and all(p.grad is None for _,p in named),'FRESH_ORIGINAL_T03_ARM_ONLY')
        initial=c['initial_states'][str(rank)]
        w.write('initial-attempt.json',dict(case='T03.sft',rank=rank,original_initial=initial,planned_updates=25,completed_updates=0))
        try:
            restored=restore_rank_original(named,self.trainer.optimizer,rank,initial)
        except BaseException as error:
            w.progress(dict(type=type(error).__name__,phase='ORIGINAL_INITIAL_RESTORE',reason_SHA256=hashlib.sha256(str(error).encode()).hexdigest()));raise
        digest=named_parameter_sha256(named);require(digest==initial['named_SHA256'],'ACTUAL_ORIGINAL_INITIAL_VALUES')
        code=torch.tensor(list(bytes.fromhex(digest)),dtype=torch.uint8,device=torch.device('cuda',local));expected=code.clone();dist.broadcast(expected,src=0)
        flag=torch.tensor(int(torch.equal(code,expected)),dtype=torch.int32,device=torch.device('cuda',local));dist.all_reduce(flag,op=dist.ReduceOp.MIN);require(bool(flag.item()),'ALL8_INITIAL_VALUES_EQUAL')
        require(rng_fingerprint(rng_values(torch.cuda.get_rng_state))==restored['actual_RNG_after'],'INITIAL_OBSERVER_AFTER_RESTORE_CHANGED_RNG')
        w.write('initial-state.json',dict(case='T03.sft',pair_id=c['pair_id'],arm=c['arm'],rank=rank,global_step=0,
            tensors=504,elements=43646976,named_SHA256=digest,original_initial=initial,restoration=restored,
            RNG_restored=True,RNG_before_restoration=before_rng))
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
callbacks_map[KEY]=NativeUSGST03
