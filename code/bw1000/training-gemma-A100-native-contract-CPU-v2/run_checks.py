"""Actual full Gemma meta/Swift/PEFT structural contract with all devices hidden."""
from pathlib import Path
import datetime,hashlib,importlib.metadata as metadata,io,json,os,signal,sys,time,unittest
W=Path(__file__).resolve().parent.parent;stage=Path(__file__).parent;source=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-update-contract-CPU-v1/framework');epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native contract CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,remaining);sys.path.insert(0,str(source))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]));dump('CPU-owner.json',owner);result=None;phase='source'
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==314
 for name,pin in fm['files'].items():assert (source/name).stat().st_size==pin['bytes'] and sha(source/name)==pin['SHA256']
 for name,pin in json.loads((stage/'INSTALLED_SOURCE_FILES.json').read_bytes()).items():
  package='transformers' if name.startswith('transformers/') else 'ms-swift';p=Path(metadata.distribution(package).locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='shared-source-tests';suite=unittest.defaultTestLoader.loadTestsFromName('vlm.tests.test_native_recipe_contract');log=io.StringIO();unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue());assert unit.wasSuccessful() and unit.testsRun==2
 from pathlib import PurePosixPath
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.native_recipe_contract import native_recipe_spec,parameter_manifest,validate_native_recipe
 launch=build_launch('T02.sft',(stage/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath('/mnt/data/nvidia-baseline-20261010/inputs/models'),PurePosixPath('/unused/dataset'),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501);spec=native_recipe_spec(launch)
 phase='original-processor-metadata'
 pins=json.loads((stage/'MODEL_PROCESSOR_FILES.json').read_bytes());root=Path(pins['original_model_root']);assert len(pins['files'])==14
 for name,pin in pins['files'].items():assert (root/name).stat().st_size==pin['bytes'] and sha(root/name)==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized();native_init=torch.cuda._lazy_init
 def forbid(*_a,**_kw):raise RuntimeError('device context forbidden during structural CPU qualification')
 torch.cuda._lazy_init=forbid;torch.set_num_threads(2)
 from swift.model import get_model_processor
 root=Path('/mnt/data/nvidia-baseline-20261010/inputs/models/gemma-3-4b-it');model,processor=get_model_processor(str(root),model_type='gemma3_vision',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16);assert model is None
 from transformers import Gemma3Config,Gemma3ForConditionalGeneration,Seq2SeqTrainingArguments
 config=Gemma3Config.from_pretrained(root,local_files_only=True);config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 phase='full-meta-native-tuner'
 with torch.device('meta'):core=Gemma3ForConditionalGeneration(config)
 core.to(dtype=torch.bfloat16);core.model_meta=processor.model_meta;core.model_info=processor.model_info;core.requires_grad_(False)
 from swift.arguments.tuner_args import TunerArguments
 from swift.pipelines.train.tuner import get_target_modules,prepare_adapter
 from dataclasses import dataclass
 @dataclass
 class NativeTunerArguments(TunerArguments):
  model_meta: object = None
  tuner_type: str = 'lora'
 tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta);assert isinstance(tuner,TunerArguments) and tuner.freeze_aligner is True and tuner.freeze_llm is False
 tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
 target=get_target_modules(tuner,core)
 with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
 named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad);assert named and all(p.device.type=='meta' for p in model.parameters());binding=dict(spec=spec,target_modules=target,trainable_manifest=parameter_manifest(named),model_source_SHA256='c7938a2c0fdf04c55da2dbe8e03d12d22501f06f5db917bb1e7ce0189835418e',initial_trainables_SHA256=None,initial_values_qualified=False)
 args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-training-output'),use_cpu=True,report_to='none',**spec['training']);verified,frozen=validate_native_recipe(model,args,binding);assert len(verified)==len(named)
 from vlm.plugins.native_first_updates import NativeFirstUpdates,KEY
 from swift.callbacks import callbacks_map
 holder=object();assert NativeFirstUpdates(args,holder).trainer is holder and callbacks_map[KEY] is NativeFirstUpdates
 phase='actual-contract-bad-boundaries';rejected=[]
 def reject(label):
  try:validate_native_recipe(model,args,binding)
  except ValueError:rejected.append(label)
  else:raise AssertionError('bad actual structural boundary accepted: '+label)
 for name,bad in [('max_steps',24),('gradient_accumulation_steps',1),('learning_rate',.01),('warmup_ratio',.2)]:
  old=getattr(args,name);setattr(args,name,bad);reject(name);setattr(args,name,old)
 cfg=model.peft_config[model.active_adapter]
 for name,bad in [('r',4),('lora_alpha',4),('lora_dropout',0)]:
  old=getattr(cfg,name);setattr(cfg,name,bad);reject(name);setattr(cfg,name,old)
 for path in spec['frozen_paths']:
  p=next(core.get_submodule(path).parameters());p.requires_grad_(True);reject(path);p.requires_grad_(False)
 manifest=binding['trainable_manifest'];binding['trainable_manifest']=manifest[:-1];reject('missing_manifest_tensor');binding['trainable_manifest']=manifest
 old=manifest[0]['shape'];manifest[0]['shape']=[1];reject('manifest_shape');manifest[0]['shape']=old
 old=manifest[0]['numel'];manifest[0]['numel']=1;reject('manifest_count');manifest[0]['numel']=old
 old=binding['target_modules'];binding['target_modules']='wrong-targets';reject('expanded_targets');binding['target_modules']=old
 old=core.model.language_model.config._attn_implementation;core.model.language_model.config._attn_implementation='eager';reject('attention');core.model.language_model.config._attn_implementation=old
 core.gradient_checkpointing_enable();reject('checkpointing');core.gradient_checkpointing_disable();assert len(rejected)==15
 validate_native_recipe(model,args,binding);assert not torch.cuda.is_initialized();torch.cuda._lazy_init=native_init;dump('GEMMA_NATIVE_BINDING.json',binding)
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=314,shared_tests=2,actual_model_class=type(core).__name__,meta_parameter_count=sum(p.numel() for p in core.parameters()),native_trainable_tensors=len(named),native_trainable_parameters=sum(p.numel() for _,p in named),trainable_dtypes=sorted(set(str(p.dtype) for _,p in named)),frozen=frozen,expanded_targets=target,binding_SHA256=sha(W/'GEMMA_NATIVE_BINDING.json'),real_Swift_TunerArguments=True,real_Swift_prepare_adapter=True,real_PEFT=True,real_TrainerArguments=True,effective_warmup_steps=args.get_warmup_steps(25),normal_contract=1,rejected_boundaries=rejected,callback_registration=True,versions={k:metadata.version(k) for k in ['torch','transformers','ms-swift','peft','accelerate']},model_weight_files_read=0,model_weight_values_loaded=False,trainable_initial_values_qualified=False,GPU_initialized=False,GPU_cards=0,training_forward_backward=False,Model_GO=False,wall_seconds=time.monotonic()-start,qualification='Full Gemma architecture on meta tensors and actual Swift/PEFT training-scope contract only. No weight values,initialization,training forward/update,numerics,recovery or performance qualification.')
except BaseException as error:result=dict(passed=False,owner=owner,epoch=epoch,deadline=deadline,phase=phase,error_type=type(error).__name__,message=str(error)[-500:],GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
