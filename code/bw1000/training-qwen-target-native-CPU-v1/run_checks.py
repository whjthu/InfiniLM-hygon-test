"""Qualify full Qwen meta architecture through actual native Swift/PEFT."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,importlib.util,io,json,os,resource,signal,sys,time,unittest
from copy import deepcopy
from types import SimpleNamespace
W=Path('/evidence');stage=Path('/input-stage');framework=Path('/input-framework')
epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE'])
start=time.monotonic();assert 0<deadline-time.time()<=220
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
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner()['node']=='a01r4n16' and os.environ['SLURM_CPUS_PER_TASK']=='4'
dump('CPU-owner.json',owner());phase='source';result=None
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():
  p=framework/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 pins=json.loads((stage/'ORIGINAL_SOURCE_METADATA.json').read_bytes())
 for pin in json.loads((stage/'RUNTIME_SOURCE_PINS.json').read_bytes())['files'].values():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 for tag,pin in pins['files'].items():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='shared-stdlib-tests';log=io.StringIO()
 suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_native_recipe_contract','vlm.tests.test_qwen_native_recipe_contract','vlm.tests.test_checkpointing_lifecycle'])
 unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue())
 assert unit.wasSuccessful() and unit.testsRun==13
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
 from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
 from swift.trainers import Seq2SeqTrainingArguments
 from swift.trainers.mixin import SwiftMixin
 from vlm.runtime.checkpointing_lifecycle import provider_identity,validate_checkpointing_lifecycle
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
 named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
 assert named and all(p.device.type=='meta' for p in model.parameters())
 binding=dict(spec=spec,target_modules=target,trainable_manifest=parameter_manifest(named),
              model_source_SHA256=pins['files']['model_source']['SHA256'],initial_trainables_SHA256=None,
              initial_values_qualified=False)
 args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-training-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**spec['training'])
 assert args.gradient_checkpointing is True and args.vit_gradient_checkpointing is False
 phase='actual-Swift-checkpointing-preparation'
 method=SwiftMixin._prepare_gradient_checkpointing
 # The real training loop passes template.get_base_model(unwrapped PEFT).
 # This is exactly the original HF core, with actual Swift model metadata.
 holder=SimpleNamespace(args=args)
 method(holder,core)
 binding['native_checkpointing_lifecycle']=dict(original_requested=True,native_Trainer_flag=False,
  language_path='model.language_model',vision_path='model.visual',language_layer_count=36,provider=provider_identity(method))
 old_binding=json.loads((stage/'ORIGINAL_STRUCTURAL_BINDING.json').read_bytes())
 assert binding['trainable_manifest']==old_binding['trainable_manifest'] and binding['spec']==old_binding['spec']
 assert binding['target_modules'] or old_binding['target_modules']
 from vlm.runtime.native_recipe_contract import target_signature
 assert target_signature(binding['target_modules'])==target_signature(old_binding['target_modules'])
 # Reproduce the exact original observer failure against this actual state.
 oldspec=importlib.util.spec_from_file_location('frozen_failed_native_recipe',stage/'original-failed-validator.py')
 oldmod=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(oldmod)
 try:oldmod.validate_native_recipe(model,args,binding)
 except ValueError as error:
  original_failure=str(error);assert original_failure=='native Trainer argument differs: gradient_checkpointing'
 else:raise AssertionError('original observer failure did not reproduce')
 verified,frozen=validate_native_recipe(model,args,binding,checkpointing_method=method)
 lifecycle=validate_checkpointing_lifecycle(core,args,binding,method)
 assert len(verified)==504 and sum(p.numel() for _,p in verified)==43646976
 from vlm.plugins.native_first_updates import NativeFirstUpdates,KEY,model_providers
 from swift.callbacks import callbacks_map
 callback_holder=object();assert NativeFirstUpdates(args,callback_holder).trainer is callback_holder and callbacks_map[KEY] is NativeFirstUpdates
 providers=model_providers(model,binding['model_source_SHA256'],'T01.sft')
 phase='actual-lifecycle-bad-boundaries';rejected=[]
 def reject(label,method_value=method):
  try:validate_native_recipe(model,args,binding,checkpointing_method=method_value)
  except ValueError:rejected.append(label)
  else:raise AssertionError('bad actual lifecycle boundary accepted: '+label)
 args.gradient_checkpointing=True;reject('Trainer_flag_not_lowered');args.gradient_checkpointing=False
 language=core.get_submodule('model.language_model');layer=language.layers[17]
 layer.gradient_checkpointing=False;reject('one_language_layer_disabled');layer.gradient_checkpointing=True
 language.config.use_cache=True;reject('language_cache_enabled');language.config.use_cache=False
 vision=core.get_submodule('model.visual');flags=[m for _,m in vision.named_modules() if hasattr(m,'gradient_checkpointing')]
 assert flags;flags[0].gradient_checkpointing=True;reject('vision_checkpointing_enabled');flags[0].gradient_checkpointing=False
 for key in ('source_file_SHA256','method_source_SHA256'):
  actual=binding['native_checkpointing_lifecycle']['provider'][key]
  binding['native_checkpointing_lifecycle']['provider'][key]='0'*64;reject(key)
  binding['native_checkpointing_lifecycle']['provider'][key]=actual
 lifecycle_binding=binding.pop('native_checkpointing_lifecycle');reject('missing_lifecycle');binding['native_checkpointing_lifecycle']=lifecycle_binding
 reject('missing_actual_provider',None)
 args.learning_rate=.5;reject('other_recipe_learning_rate');args.learning_rate=spec['training']['learning_rate']
 parameter=next(vision.parameters());parameter.requires_grad_(True);reject('visual_unfrozen');parameter.requires_grad_(False)
 manifest=binding['trainable_manifest'];binding['trainable_manifest']=manifest[:-1];reject('truncated_manifest');binding['trainable_manifest']=manifest
 binding['spec']['training']['gradient_checkpointing']=False;reject('original_recipe_disabled');binding['spec']['training']['gradient_checkpointing']=True
 assert len(rejected)==12
 validate_native_recipe(model,args,binding,checkpointing_method=method)
 assert lifecycle==validate_checkpointing_lifecycle(core,args,binding,method)
 assert not torch.cuda.is_initialized();torch.cuda._lazy_init=native_init
 dump('checkpointing-lifecycle.json',lifecycle)
 dump('QWEN_NATIVE_BINDING.json',binding);dump('providers.json',providers)

 phase='actual-native-64-row-target-preprocessing'
 deps=json.loads((stage/'DEPENDENCIES.json').read_bytes());assets=json.loads((stage/'ORIGINAL_INPUT_FILES.json').read_bytes())
 asset_root=Path(deps['inputs']['host_root']);dataset=Path(deps['inputs']['host_dataset']['path'])
 for pin in assets['files']:
  p=asset_root/pin['name'];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 remap={p['source']:str(asset_root/p['name']) for p in assets['files']}
 def convert(v):
  if isinstance(v,str):return remap.get(v,v)
  if isinstance(v,list):return [convert(x) for x in v]
  if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
  return v
 original_rows=[json.loads(x) for x in (asset_root/'original.cuda.jsonl').read_bytes().splitlines()]
 actual_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()]
 assert len(original_rows)==len(actual_rows)==64 and convert(original_rows)==actual_rows
 from vlm.runtime.template_inputs import capture_native_inputs,compare_input_identities
 processor_binding=json.loads((stage/'PROCESSOR_FILES.json').read_bytes())
 settings=json.loads((stage/'ORIGINAL_TEMPLATE_REFERENCE.json').read_bytes())['settings']
 capture=capture_native_inputs(settings,root,dataset,W/'encoded',processor_binding['files'],deps['inputs']['host_dataset']['SHA256'],epoch=epoch,deadline=deadline-10)
 identities=json.loads((W/'encoded/input-identities.json').read_bytes())
 original_reference=json.loads((stage/'ORIGINAL_TEMPLATE_REFERENCE.json').read_bytes())
 comparison=compare_input_identities(original_reference['identities'],identities)
 dump('comparison.json',comparison)
 assert capture['passed'] and capture['rows']==64 and capture['CUDA_initialized'] is False and capture['model_loaded'] is False
 assert comparison['all_fields_exact'] and not torch.cuda.is_initialized() and torch.version.hip and not torch.version.cuda

 actual_sources={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and (name.startswith(('transformers.models.qwen3_vl.','swift.model.models.qwen','swift.pipelines.train.tuner','peft.tuners.lora.')) or (name.startswith(('vlm.','tools.hygon_vlm_recipe','swift.trainers.mixin','swift.trainers.arguments','swift.trainers.utils')))):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():actual_sources[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=13,
  actual_model_class=type(core).__name__,meta_parameter_count=sum(p.numel() for p in core.parameters()),
  native_trainable_tensors=len(named),native_trainable_parameters=sum(p.numel() for _,p in named),trainable_dtypes=sorted(set(str(p.dtype) for _,p in named)),
  frozen=frozen,expanded_targets=target,binding_SHA256=sha(W/'QWEN_NATIVE_BINDING.json'),providers_SHA256=sha(W/'providers.json'),actual_sources=actual_sources,
  real_Swift_prepare_adapter=True,real_PEFT=True,gradient_checkpointing=core.is_gradient_checkpointing,effective_warmup_steps=args.get_warmup_steps(25),
  normal_contract=1,rejected_boundaries=rejected,callback_registered=True,actual_native_checkpointing_preparation=True,full_Trainer_constructed=False,original_failure_reproduced=original_failure,checkpointing_lifecycle=lifecycle,comparison=comparison,comparison_SHA256=sha(W/'comparison.json'),capture=capture,identities_SHA256=sha(W/'encoded/input-identities.json'),original_asset_files_verified=67,path_rewrite_only_proved=True,HIP=torch.version.hip,versions={k:metadata.version(k) for k in ['torch','transformers','ms-swift','peft','accelerate']},
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
  model_weight_values_read=False,model_weight_values_loaded=False,weight_file_open_attempts=weight_open_attempts,
  initial_values_qualified=False,GPU_initialized=False,GPU_cards=0,
  training_forward_backward=False,capture_size_qualified=False,GPU_GO=False,Model_GO=False,
  qualification='Actual unchanged Swift checkpointing method on the complete original Qwen meta model; original observer failure reproduced and fixed observer qualified. No initialized values,forward/backward,numerics,recovery or performance.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,owner=owner(),epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,
             GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
