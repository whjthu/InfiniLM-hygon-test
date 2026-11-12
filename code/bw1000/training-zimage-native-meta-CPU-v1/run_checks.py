"""Full original ZImageDiT/PEFT CPU meta construction; no weight or data access."""
from pathlib import Path
import datetime,hashlib,importlib,importlib.metadata as md,json,os,resource,signal,sys,time

W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE'])
start=time.monotonic();assert 0<deadline-time.time()<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original CPU meta epoch exhausted')))
signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (W/n).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
attempts=[]
def forbid(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0])
        if p.startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):
            attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest())
            raise RuntimeError('original weights and datasets are forbidden in meta qualification')
    if event=='socket.connect':raise RuntimeError('network access forbidden in CPU meta qualification')
sys.addaudithook(forbid);dump('CPU-owner.json',owner());phase='source'
try:
    assert all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
    fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==356
    assert fm['framework_commit']=='8b43976c3f98602391a7f03674092a05ada2df60'
    for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
    binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());root=Path(binding['source_root'])
    for n,pin in binding['original_source_files'].items():
        p=root/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
    original=json.loads((S/'ORIGINAL_HEADERS.json').read_bytes())
    sys.path.insert(0,str(F/'generation/src'))
    from gencuda.zimage_fusion_contract import check_metadata,validate_header
    shards=[];preset=None
    for n,h in original.items():
        validate_header(h['tensors'],file_bytes=h['bytes'],header_bytes=h['header_bytes'])
        if n.startswith('diffusion_pytorch_model-'):shards.append(h['tensors'])
        else:preset=h['tensors']
    expected=check_metadata(shards,preset)
    assert expected['preset_pairs']==240 and expected['new_rank4_elements']==9922560
    phase='original-native-imports'
    import torch
    assert not torch.cuda.is_initialized();torch.set_num_threads(1)
    def forbidden_device(*a,**k):raise RuntimeError('GPU init forbidden in CPU meta qualification')
    torch.cuda._lazy_init=forbidden_device
    sys.path.insert(0,str(root))
    native=importlib.import_module('diffsynth.models.z_image_dit')
    training=importlib.import_module('diffsynth.diffusion.training_module')
    lora=importlib.import_module('diffsynth.utils.lora.general')
    for module,name in [(native,'diffsynth/models/z_image_dit.py'),(training,'diffsynth/diffusion/training_module.py'),(lora,'diffsynth/utils/lora/general.py')]:
        assert Path(module.__file__).resolve()==(root/name).resolve() and sha(Path(module.__file__))==binding['original_source_files'][name]['SHA256']
    phase='full-original-meta-constructor'
    with torch.device('meta'):
        model=native.ZImageDiT()
    state=model.state_dict();base={n:v for shard in shards for n,v in shard.items()}
    assert set(state)==set(base) and len(state)==521
    assert all(t.is_meta and list(t.shape)==base[n]['shape'] and t.dtype==torch.float32 for n,t in state.items())
    assert sum(p.numel() for p in model.parameters())==6154908736
    dump('BASE_METADATA.json',{n:dict(shape=list(t.shape),dtype=str(t.dtype),meta=t.is_meta) for n,t in state.items()})
    phase='actual-preset-target-modules'
    mapper=lora.GeneralLoRALoader(device='cpu',torch_dtype=torch.bfloat16)
    names=mapper.get_name_dict(preset);assert set(names)==set(expected['preset_dimensions']) and len(names)==240
    for name in names:
        actual=model.get_submodule(name);shape=expected['preset_dimensions'][name]['base']
        assert isinstance(actual,torch.nn.Linear) and list(actual.weight.shape)==shape and actual.weight.is_meta
    dump('PRESET_MODULE_METADATA.json',{name:dict(module_class=type(model.get_submodule(name)).__module__+'.'+type(model.get_submodule(name)).__name__,
        shape=list(model.get_submodule(name).weight.shape),dtype=str(model.get_submodule(name).weight.dtype),meta=model.get_submodule(name).weight.is_meta) for name in names})
    phase='native-PEFT-rank4-and-freezing'
    model.to(dtype=torch.bfloat16);model.requires_grad_(False)
    helper=training.DiffusionTrainingModule()
    targets=helper.parse_lora_target_modules(model,binding['target_modules'])
    model=helper.add_lora_to_model(model,target_modules=targets,lora_rank=4,upcast_dtype=torch.bfloat16)
    expected_trainable={}
    for n,shape in expected['new_target_shapes'].items():
        expected_trainable[n+'.lora_A.default.weight']=[4,shape[1]]
        expected_trainable[n+'.lora_B.default.weight']=[shape[0],4]
        module=model.get_submodule(n)
        assert module.r['default']==4 and module.lora_alpha['default']==4 and module.scaling['default']==1
        assert getattr(module.lora_dropout['default'],'p',0)==0
    actual_trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(actual_trainable)==set(expected_trainable) and len(actual_trainable)==476
    assert sum(p.numel() for p in actual_trainable.values())==9922560
    assert all(p.is_meta and p.dtype==torch.bfloat16 and list(p.shape)==expected_trainable[n] for n,p in actual_trainable.items())
    frozen={n:p for n,p in model.named_parameters() if not p.requires_grad}
    assert len(frozen)==521 and sum(p.numel() for p in frozen.values())==6154908736
    assert all(p.is_meta and p.dtype==torch.bfloat16 for p in frozen.values())
    phase='passive-import-provider-closure'
    providers={};total=0
    for name,module in sorted(sys.modules.items()):
        if name!='diffsynth' and not name.startswith('diffsynth.'):continue
        value=getattr(module,'__file__',None)
        if not value:continue
        p=Path(value);assert p.is_file() and not p.is_symlink()
        relative=p.resolve().relative_to(root.resolve()).as_posix();size=p.stat().st_size
        assert size<=2*1024**2;total+=size;assert total<=20*1024**2
        providers[name]=dict(relative_path=relative,bytes=size,SHA256=sha(p))
    dump('IMPORTED_SOURCE_FILES.json',providers)
    assert not torch.cuda.is_initialized() and not attempts
    trainable_shapes={n:dict(shape=list(p.shape),dtype=str(p.dtype),meta=p.is_meta,requires_grad=p.requires_grad) for n,p in actual_trainable.items()}
    dump('TRAINABLE_METADATA.json',trainable_shapes)
    dump('FROZEN_METADATA.json',{n:dict(shape=list(p.shape),dtype=str(p.dtype),meta=p.is_meta,requires_grad=p.requires_grad) for n,p in frozen.items()})
    result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
        framework_commit=fm['framework_commit'],framework_files=356,base_tensors=521,base_elements=6154908736,
        actual_native_preset_linear_targets=240,actual_native_new_LoRA_modules=238,actual_trainable_tensors=476,actual_trainable_elements=9922560,
        frozen_tensors=521,frozen_elements=6154908736,all_parameters_meta=True,all_training_parameters_BF16=True,
        native_rank=4,native_alpha=4,native_scaling=1,native_dropout=0,source_provider_files=len(providers),
        source_provider_manifest_SHA256=sha(W/'IMPORTED_SOURCE_FILES.json'),trainable_metadata_SHA256=sha(W/'TRAINABLE_METADATA.json'),
        base_metadata_SHA256=sha(W/'BASE_METADATA.json'),frozen_metadata_SHA256=sha(W/'FROZEN_METADATA.json'),
        preset_module_metadata_SHA256=sha(W/'PRESET_MODULE_METADATA.json'),
        versions={n:md.version(n) for n in ('torch','transformers','peft','accelerate','safetensors')},
        wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        real_weight_or_data_open_attempts=attempts,original_header_metadata_reused=True,weight_values_read=False,
        actual_fusion_executed=False,full_training_module_pipeline_constructed=False,Torch_imported=True,GPU_initialized=False,GPU_cards=0,
        numerical_acceptance=False,HIP_target_qualified=False,GPU_GO=False,Model_GO=False,
        qualification='Full native6.154B meta DiT and native PEFT target/rank/dtype/freezing qualification only. No pretrained values,preset fusion math,complete pipeline/encoder/VAE,forward/backward/update,recovery,performance or heldout qualification.')
except BaseException as error:
    import traceback
    dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
    result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,
        wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
