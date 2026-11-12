"""Qualify complete original HF InternVL meta model through real Swift/PEFT."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
from dataclasses import dataclass
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1];stage=W/'src';framework=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();assert 0<deadline-time.time()<=260
attempts=[]
def forbid_weights(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
  attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weight value opening forbidden in original structural CPU scope')
sys.addaudithook(forbid_weights)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original InternVL CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
sys.path.insert(0,str(framework))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source';result=None
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():assert (framework/name).stat().st_size==pin['bytes'] and sha(framework/name)==pin['SHA256']
 sources=json.loads((stage/'ORIGINAL_SOURCES.json').read_bytes())['files']
 for name,pin in sources.items():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 model_metadata=json.loads((stage/'MODEL_METADATA.json').read_bytes());root=Path(model_metadata['model_root'])
 for name,pin in model_metadata['files'].items():assert (root/name).stat().st_size==pin['bytes'] and sha(root/name)==pin['SHA256']
 phase='shared-stdlib-tests';log=io.StringIO()
 suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_native_recipe_contract','vlm.tests.test_qwen_native_recipe_contract','vlm.tests.test_checkpointing_lifecycle','vlm.tests.test_internvl_native_recipe_contract','vlm.tests.test_provider_sources'])
 unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue());assert unit.wasSuccessful() and unit.testsRun==20
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.native_recipe_contract import native_recipe_spec,parameter_manifest,validate_native_recipe
 from vlm.runtime.checkpointing_lifecycle import provider_identity,validate_checkpointing_lifecycle
 launch=build_launch('T03.sft',(stage/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath('/unused/data'),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 spec=native_recipe_spec(launch);assert spec['training']['gradient_checkpointing'] is True and '--template' not in launch['command']
 phase='original-native-HF-processor'
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def no_device(*args,**kwargs):raise RuntimeError('GPU initialization forbidden in CPU meta scope')
 torch.cuda._lazy_init=no_device
 from swift.model import get_model_processor
 empty,processor=get_model_processor(str(root),model_type='internvl',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
 assert empty is None and processor.model_meta.template=='internvl_hf'
 from swift.template import get_template
 template=get_template(processor,template_type=None,max_length=4096,max_pixels=262144,padding_side='right',padding_free=False)
 resolved=template.template_meta.template_type;assert resolved=='internvl_hf';template.set_mode('train')
 phase='complete-original-meta-model-and-native-adapter'
 from transformers import InternVLConfig,InternVLForConditionalGeneration
 config=InternVLConfig.from_pretrained(root,local_files_only=True)
 config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 with torch.device('meta'):core=InternVLForConditionalGeneration(config)
 core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info
 from swift.arguments.tuner_args import TunerArguments
 from swift.pipelines.train.tuner import get_target_modules,prepare_adapter
 @dataclass
 class NativeTunerArguments(TunerArguments):
  model_meta:object=None
  tuner_type:str='lora'
 tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta)
 assert tuner.freeze_aligner is True and tuner.freeze_llm is False
 tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
 target=get_target_modules(tuner,core)
 with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
 from swift.trainers import Seq2SeqTrainingArguments
 from swift.trainers.mixin import SwiftMixin
 args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**spec['training'])
 assert args.gradient_checkpointing is True
 method=SwiftMixin._prepare_gradient_checkpointing;method(SimpleNamespace(args=args),core)
 named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
 assert all(p.device.type=='meta' for p in model.parameters())
 binding=dict(spec=spec,resolved_template_type=resolved,target_modules=target,trainable_manifest=parameter_manifest(named),
  model_source_SHA256={'transformers.models.internvl.':sources['transformers/models/internvl/modeling_internvl.py']['SHA256'],'transformers.models.qwen3.':sources['transformers/models/qwen3/modeling_qwen3.py']['SHA256']},
  initial_trainables_SHA256=None,initial_values_qualified=False,native_checkpointing_lifecycle=dict(original_requested=True,native_Trainer_flag=False,language_path='model.language_model',vision_path='model.vision_tower',language_layer_count=36,provider=provider_identity(method)))
 verified,frozen=validate_native_recipe(model,args,binding,checkpointing_method=method)
 assert len(verified)==504 and sum(p.numel() for _,p in verified)==43646976
 lifecycle=validate_checkpointing_lifecycle(core,args,binding,method)
 from vlm.plugins.native_first_updates import NativeFirstUpdates,KEY,model_providers
 from swift.callbacks import callbacks_map
 holder=object();assert NativeFirstUpdates(args,holder).trainer is holder and callbacks_map[KEY] is NativeFirstUpdates
 providers=model_providers(model,binding['model_source_SHA256'],'T03.sft')
 phase='actual-structural-rejections';rejected=[]
 def reject(label,method_value=method):
  try:validate_native_recipe(model,args,binding,checkpointing_method=method_value)
  except ValueError:rejected.append(label)
  else:raise AssertionError('bad actual InternVL boundary accepted: '+label)
 for name,bad in [('max_steps',24),('gradient_accumulation_steps',1),('learning_rate',.1),('warmup_ratio',.2)]:
  original=getattr(args,name);setattr(args,name,bad);reject(name);setattr(args,name,original)
 cfg=model.peft_config[model.active_adapter]
 for name,bad in [('r',4),('lora_alpha',4),('lora_dropout',0)]:
  original=getattr(cfg,name);setattr(cfg,name,bad);reject(name);setattr(cfg,name,original)
 for path in ('model.vision_tower','model.multi_modal_projector'):
  p=next(core.get_submodule(path).parameters());p.requires_grad_(True);reject('unfrozen_'+path);p.requires_grad_(False)
 manifest=binding['trainable_manifest'];binding['trainable_manifest']=manifest[:-1];reject('missing_tensor');binding['trainable_manifest']=manifest
 binding['resolved_template_type']='internvl3_5';reject('nonHF_template');binding['resolved_template_type']=resolved
 language=core.get_submodule('model.language_model');language.layers[17].gradient_checkpointing=False;reject('one_language_layer_off');language.layers[17].gradient_checkpointing=True
 vision=core.get_submodule('model.vision_tower');flags=[m for _,m in vision.named_modules() if hasattr(m,'gradient_checkpointing')];assert flags
 flags[0].gradient_checkpointing=True;reject('vision_checkpointing_on');flags[0].gradient_checkpointing=False
 language.config.use_cache=True;reject('language_cache_on');language.config.use_cache=False
 reject('missing_native_method',None)
 from vlm.runtime.provider_sources import model_provider_sources
 wrong=dict(binding['model_source_SHA256']);wrong['transformers.models.qwen3.']='0'*64
 try:model_provider_sources(model,wrong,'T03.sft')
 except ValueError:rejected.append('wrong_embedded_Qwen3_source')
 else:raise AssertionError('unverified embedded decoder source accepted')
 assert len(rejected)==16
 validate_native_recipe(model,args,binding,checkpointing_method=method);assert lifecycle==validate_checkpointing_lifecycle(core,args,binding,method)
 assert not torch.cuda.is_initialized() and not attempts
 dump('INTERNVL_NATIVE_BINDING.json',binding);dump('checkpointing-lifecycle.json',lifecycle);dump('providers.json',providers)
 runtime={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and name.startswith(('transformers.models.internvl.','transformers.models.qwen3.','swift.model.models.internlm','swift.template.templates.internvl','swift.trainers.mixin','swift.pipelines.train.tuner','peft.tuners.lora.','vlm.','tools.hygon_vlm_recipe')):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():runtime[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=20,actual_model_class=type(core).__name__,actual_language_class=type(language).__name__,
  resolved_template_type=resolved,actual_template_class=type(template).__module__+'.'+type(template).__qualname__,meta_parameter_count=sum(p.numel() for p in core.parameters()),
  native_trainable_tensors=len(named),native_trainable_parameters=sum(p.numel() for _,p in named),frozen=frozen,real_Swift_prepare_adapter=True,real_PEFT=True,actual_native_checkpointing_preparation=True,
  full_Trainer_constructed=False,checkpointing_lifecycle=lifecycle,rejected_boundaries=rejected,binding_SHA256=sha(W/'INTERNVL_NATIVE_BINDING.json'),providers_SHA256=sha(W/'providers.json'),
  actual_sources=runtime,metadata_files_verified=13,callback_registered=True,versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate')},
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_initialized=False,weight_file_open_attempts=attempts,
  model_weight_values_read=False,initial_values_qualified=False,training_forward_backward=False,GPU_GO=False,Model_GO=False,
  qualification='Original full meta HF InternVL/nativeSwift/PEFT/lifecycle/provider qualification only. No initialized values,actual image encoding,forward/backward,numerics,25steps,recovery,performance or effects.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
