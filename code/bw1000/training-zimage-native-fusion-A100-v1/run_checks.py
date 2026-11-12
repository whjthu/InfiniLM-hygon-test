"""One full native original pipeline load,fusion and initialized trainable audit."""
from pathlib import Path
import datetime,gc,hashlib,importlib.util,json,os,random,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE'])
start=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=1100
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native fusion single epoch expired')))
signal.setitimer(signal.ITIMER_REAL,remaining)
def check():
 if min(deadline-time.time(),remaining-(time.monotonic()-start))<=0:raise TimeoutError('native fusion deadline')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def progress(v):
 check();v=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**v)
 with (W/'fusion-progress.private.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 p=W/'progress.tmp';p.write_text(json.dumps(v));p.replace(W/'progress.json')
def stats(p):
 s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('GPU-owner.json',owner());phase='source';model=None
try:
 binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());source=Path(binding['source_root']);observed=json.loads((S/'ORIGINAL_FILE_STATS.json').read_bytes())
 modelroot=Path(observed['model_root']);preset=Path(observed['adapter']['path'])
 for n,pin in binding['original_source_files'].items():assert sha(source/n)==pin['SHA256'] and (source/n).stat().st_size==pin['bytes']
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 for n,pin in observed['files'].items():assert stats(modelroot/n)=={k:pin[k] for k in ('bytes','mtime_ns','inode','device')} and not (modelroot/n).is_symlink()
 assert stats(preset)=={k:observed['adapter'][k] for k in ('bytes','mtime_ns','inode','device')}
 prior=json.loads((S/'PRIOR_IDENTITY.json').read_bytes())
 for n,pin in prior['artifacts'].items():assert sha(Path(prior['namespace'])/n)==pin['SHA256']
 phase='torch-original-environment'
 import torch,numpy as np
 from safetensors.torch import save_file
 import importlib.metadata as md
 assert {n:md.version(n) for n in binding['versions']}==binding['versions']
 assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100-SXM4-40GB'
 torch.set_num_threads(1);torch.manual_seed(20260911);random.seed(20260911);np.random.seed(20260911)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();free_before,total=torch.cuda.mem_get_info()
 assert total==42405855232 and free_before>=38*1024**3
 sys.path.insert(0,str(source))
 from diffsynth.utils.lora.general import GeneralLoRALoader
 path=F/'generation/src/gencuda/zimage_native_fusion_audit.py';spec=importlib.util.spec_from_file_location('published_zimage_native_fusion',path)
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 nativepath=source/'examples/z_image/model_training/train.py';spec=importlib.util.spec_from_file_location('original_zimage_fusion_training',nativepath)
 native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 shapes=json.loads((S/'PRESET_SHAPES.json').read_bytes())
 audit=helper.PresetFusionAudit(GeneralLoRALoader,shapes,deadline=check,sink=progress)
 phase='native-full-pipeline-load-and-fusion';loadstart=time.monotonic()
 paths=[[str(modelroot/'transformer'/('diffusion_pytorch_model-%05d-of-00003.safetensors'%i)) for i in range(1,4)],
        [str(modelroot/'text_encoder'/('model-%05d-of-00003.safetensors'%i)) for i in range(1,4)],str(modelroot/'vae/diffusion_pytorch_model.safetensors')]
 with audit.observe():
  model=native.ZImageTrainingModule(model_paths=json.dumps(paths),tokenizer_path=str(modelroot/'tokenizer'),
   lora_base_model='dit',lora_target_modules='to_q,to_k,to_v,to_out.0,w1,w2,w3',lora_rank=4,
   preset_lora_path=str(preset),preset_lora_model='dit',use_gradient_checkpointing=True,device='cuda',task='sft')
 torch.cuda.synchronize();loadwall=time.monotonic()-loadstart;assert audit.result and audit.result['passed']
 dump('FUSION_IDENTITY.json',audit.result)
 phase='actual-rank4-trainable-and-freeze-boundary'
 assert all(getattr(model.pipe,n) is not None for n in ('dit','text_encoder','vae_encoder','vae_decoder','tokenizer'))
 expected=json.loads((S/'TRAINABLE_METADATA.json').read_bytes())
 trainable={n:p for n,p in model.pipe.dit.named_parameters() if p.requires_grad}
 assert set(trainable)==set(expected) and len(trainable)==476 and sum(p.numel() for p in trainable.values())==9922560
 for n,p in trainable.items():assert p.dtype==torch.bfloat16 and p.device.type=='cuda' and list(p.shape)==expected[n]['shape'] and bool(torch.isfinite(p).all())
 frozen={n:p for n,p in model.pipe.dit.named_parameters() if not p.requires_grad}
 assert len(frozen)==521 and sum(p.numel() for p in frozen.values())==6154908736
 assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in frozen.values())
 assert len([p for p in model.parameters() if p.requires_grad])==476
 for n in shapes:
  m=model.pipe.dit.get_submodule(n)
  if hasattr(m,'r'):assert m.r['default']==4 and m.lora_alpha['default']==4 and m.scaling['default']==1 and getattr(m.lora_dropout['default'],'p',0)==0
 phase='private-initial-state-and-provider-identities'
 initial={n:p.detach().cpu().contiguous() for n,p in trainable.items()}
 save_file(initial,str(W/'trainable-initial.private.safetensors'));del initial
 torch.save(dict(torch_CPU=torch.get_rng_state(),torch_CUDA=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),seed=20260911),W/'post-constructor-RNG.private.pt')
 dump('TRAINABLE_METADATA.json',{n:dict(shape=list(p.shape),dtype=str(p.dtype),elements=p.numel(),requires_grad=p.requires_grad) for n,p in trainable.items()})
 providers={};totalbytes=0
 for n,m in sorted(sys.modules.items()):
  if n!='diffsynth' and not n.startswith('diffsynth.'):continue
  value=getattr(m,'__file__',None)
  if not value:continue
  p=Path(value);relative=p.resolve().relative_to(source.resolve()).as_posix();size=p.stat().st_size;assert not p.is_symlink() and size<=2*1024**2
  totalbytes+=size;assert totalbytes<=40*1024**2
  providers[n]=dict(relative_path=relative,bytes=size,SHA256=sha(p))
 dump('IMPORTED_SOURCE_FILES.json',providers)
 for n,pin in observed['files'].items():assert stats(modelroot/n)=={k:pin[k] for k in ('bytes','mtime_ns','inode','device')}
 assert stats(preset)=={k:observed['adapter'][k] for k in ('bytes','mtime_ns','inode','device')}
 free_loaded,_=torch.cuda.mem_get_info();peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
 # All references to parameters must be released before checking the allocator.
 del trainable,frozen,p,m,model;model=None;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
 free_after,_=torch.cuda.mem_get_info();allocated_after=torch.cuda.memory_allocated();reserved_after=torch.cuda.memory_reserved()
 assert allocated_after==0 and reserved_after==0
 result=dict(passed=True,owner=owner(),epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),
  full_native_pipeline_loaded=True,actual_preset_fusion_executed=True,preset_targets=240,changed_target_tensors=audit.result['changed_target_tensors'],base_tensors=521,
  actual_trainable_tensors=476,actual_trainable_elements=9922560,frozen_DiT_tensors=521,frozen_DiT_elements=6154908736,
  gradient_checkpointing=True,preset_alpha=1,new_LoRA_rank=4,new_LoRA_alpha=4,new_LoRA_scaling=1,new_LoRA_dropout=0,
  FUSION_IDENTITY_SHA256=sha(W/'FUSION_IDENTITY.json'),initial_trainable_SHA256=sha(W/'trainable-initial.private.safetensors'),
  initial_trainable_bytes=(W/'trainable-initial.private.safetensors').stat().st_size,RNG_SHA256=sha(W/'post-constructor-RNG.private.pt'),
  source_provider_files=len(providers),source_provider_manifest_SHA256=sha(W/'IMPORTED_SOURCE_FILES.json'),versions={n:md.version(n) for n in binding['versions']},
  GPU_cards=1,free_before=free_before,free_loaded=free_loaded,free_after=free_after,peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,
  allocated_after=allocated_after,reserved_after=reserved_after,load_and_audit_seconds=loadwall,
  fusion_before_audit_seconds=audit.result['before_audit_seconds'],fusion_after_audit_seconds=audit.result['after_audit_seconds'],
  native_fusion_host_call_seconds=audit.result['native_host_call_seconds'],wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  actual_forward_calls=0,actual_backward_calls=0,actual_optimizer_updates=0,independent_numerical_backend=False,
  weights_exported_from_source_host=False,numerical_training_qualified=False,HIP_target_qualified=False,formal_performance=False,Model_GO=False,
  qualification='One originalA100 complete native pipeline load,actualBF16presetfusion and seeded rank4initialstate. Allbasevalues/hashmapped arithmetic checked on sameCUDAbackend;no forward/update,independentmath,crossplatform/recovery/performance/effects qualification.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(phase=phase,error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=1,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
