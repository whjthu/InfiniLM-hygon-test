"""Qualify full Qwen meta architecture through actual native Swift/PEFT."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];stage=W/'src';framework=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE'])
start=time.monotonic();assert 0<deadline-time.time()<=260
weight_open_attempts=[]
def no_weights(event,arguments):
 if event=='open' and arguments and isinstance(arguments[0],(str,bytes)):
  path=os.fsdecode(arguments[0])
  if path.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
   weight_open_attempts.append(hashlib.sha256(path.encode()).hexdigest())
   raise RuntimeError('weight/checkpoint file opening forbidden in structural CPU scope')
sys.addaudithook(no_weights)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Qwen structural CPU deadline exhausted')))
signal.setitimer(signal.ITIMER_REAL,deadline-time.time());sys.path.insert(0,str(framework))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source';result=None
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():
  p=framework/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 pins=json.loads((stage/'ORIGINAL_SOURCE_METADATA.json').read_bytes())
 for tag,pin in pins['files'].items():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='shared-stdlib-tests';log=io.StringIO()
 suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_native_recipe_contract','vlm.tests.test_qwen_native_recipe_contract'])
 unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue())
 assert unit.wasSuccessful() and unit.testsRun==8
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.native_recipe_contract import native_recipe_spec,parameter_manifest,validate_native_recipe
 root=Path(pins['files']['config']['path']).parent
 launch=build_launch('T01.sft',(stage/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath('/unused/dataset'),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 spec=native_recipe_spec(launch);assert spec['training']['gradient_checkpointing'] is True
 phase='original-model-processor-metadata'
 import torch
 assert not torch.cuda.is_initialized();native_init=torch.cuda._lazy_init
 def forbid(*_args,**_kwargs):raise RuntimeError('device initialization forbidden in structural CPU scope')
 torch.cuda._lazy_init=forbid;torch.set_num_threads(2)
 from swift.model import get_model_processor
 empty,processor=get_model_processor(str(root),model_type='qwen3_vl',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
 assert empty is None
 from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration,Seq2SeqTrainingArguments
 config=Qwen3VLConfig.from_pretrained(root,local_files_only=True)
 config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 phase='full-meta-native-tuner'
 with torch.device('meta'):core=Qwen3VLForConditionalGeneration(config)
 core.to(dtype=torch.bfloat16);core.model_meta=processor.model_meta;core.model_info=processor.model_info;core.requires_grad_(False)
 from swift.arguments.tuner_args import TunerArguments
 from swift.pipelines.train.tuner import get_target_modules,prepare_adapter
 from dataclasses import dataclass
 @dataclass
 class NativeTunerArguments(TunerArguments):
  model_meta:object=None
  tuner_type:str='lora'
 tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta)
 assert tuner.freeze_aligner is True and tuner.freeze_llm is False
 tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
 target=get_target_modules(tuner,core)
 with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
 model.gradient_checkpointing_enable()
 named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
 assert named and all(p.device.type=='meta' for p in model.parameters())
 binding=dict(spec=spec,target_modules=target,trainable_manifest=parameter_manifest(named),
              model_source_SHA256=pins['files']['model_source']['SHA256'],initial_trainables_SHA256=None,
              initial_values_qualified=False)
 args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-training-output'),use_cpu=True,report_to='none',**spec['training'])
 verified,frozen=validate_native_recipe(model,args,binding);assert len(verified)==len(named)
 from vlm.plugins.native_first_updates import NativeFirstUpdates,KEY,model_providers
 from swift.callbacks import callbacks_map
 holder=object();assert NativeFirstUpdates(args,holder).trainer is holder and callbacks_map[KEY] is NativeFirstUpdates
 providers=model_providers(model,binding['model_source_SHA256'],'T01.sft')
 phase='actual-bad-boundaries';rejected=[]
 def reject(label):
  try:validate_native_recipe(model,args,binding)
  except ValueError:rejected.append(label)
  else:raise AssertionError('bad actual Qwen structural boundary accepted: '+label)
 for name,bad in [('max_steps',24),('gradient_accumulation_steps',1),('learning_rate',.01),('warmup_ratio',.2)]:
  old=getattr(args,name);setattr(args,name,bad);reject(name);setattr(args,name,old)
 cfg=model.peft_config[model.active_adapter]
 for name,bad in [('r',4),('lora_alpha',4),('lora_dropout',0)]:
  old=getattr(cfg,name);setattr(cfg,name,bad);reject(name);setattr(cfg,name,old)
 p=next(core.get_submodule('model.visual').parameters());p.requires_grad_(True);reject('visual_unfrozen');p.requires_grad_(False)
 manifest=binding['trainable_manifest'];binding['trainable_manifest']=manifest[:-1];reject('missing_tensor');binding['trainable_manifest']=manifest
 for key,bad in [('shape',[1]),('numel',1),('dtype','torch.bfloat16')]:
  old=manifest[0][key];manifest[0][key]=bad;reject('manifest_'+key);manifest[0][key]=old
 old=binding['target_modules'];binding['target_modules']='wrong-targets';reject('targets');binding['target_modules']=old
 language=core.get_submodule(spec['language_path']);old=language.config._attn_implementation;language.config._attn_implementation='eager';reject('attention');language.config._attn_implementation=old
 model.gradient_checkpointing_disable();reject('checkpointing_off');model.gradient_checkpointing_enable()
 assert len(rejected)==15
 validate_native_recipe(model,args,binding);assert not torch.cuda.is_initialized();torch.cuda._lazy_init=native_init
 dump('QWEN_NATIVE_BINDING.json',binding);dump('providers.json',providers)
 actual_sources={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and (name.startswith(('transformers.models.qwen3_vl.','swift.model.models.qwen','swift.pipelines.train.tuner','peft.tuners.lora.')) or (name.startswith(('vlm.','tools.hygon_vlm_recipe')))):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():actual_sources[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=8,
  actual_model_class=type(core).__name__,meta_parameter_count=sum(p.numel() for p in core.parameters()),
  native_trainable_tensors=len(named),native_trainable_parameters=sum(p.numel() for _,p in named),trainable_dtypes=sorted(set(str(p.dtype) for _,p in named)),
  frozen=frozen,expanded_targets=target,binding_SHA256=sha(W/'QWEN_NATIVE_BINDING.json'),providers_SHA256=sha(W/'providers.json'),actual_sources=actual_sources,
  real_Swift_prepare_adapter=True,real_PEFT=True,gradient_checkpointing=core.is_gradient_checkpointing,effective_warmup_steps=args.get_warmup_steps(25),
  normal_contract=1,rejected_boundaries=rejected,callback_registered=True,versions={k:metadata.version(k) for k in ['torch','transformers','ms-swift','peft','accelerate']},
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
  model_weight_values_read=False,model_weight_values_loaded=False,weight_file_open_attempts=weight_open_attempts,
  initial_values_qualified=False,GPU_initialized=False,GPU_cards=0,
  training_forward_backward=False,capture_size_qualified=False,GPU_GO=False,Model_GO=False,
  qualification='Full original Qwen meta model and actual Swift/PEFT structural contract only; no initialized values,training,numerics,recovery or performance.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,owner=owner(),epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,
             GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
