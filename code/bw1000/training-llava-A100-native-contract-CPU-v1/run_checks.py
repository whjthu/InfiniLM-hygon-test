"""Qualify the original full custom OneVision meta model through Swift/PEFT."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
from dataclasses import dataclass
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1];stage=W/'src';framework=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();assert 0<deadline-time.time()<=260
attempts=[]
def forbid_weights(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
  attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weight value opening forbidden in CPU meta scope')
sys.addaudithook(forbid_weights)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original OneVision CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
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
  p=Path(metadata.distribution('ms-swift').locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 model_metadata=json.loads((stage/'MODEL_METADATA.json').read_bytes());root=Path(model_metadata['model_root'])
 for name,pin in model_metadata['files'].items():assert (root/name).stat().st_size==pin['bytes'] and sha(root/name)==pin['SHA256']
 phase='shared-stdlib-tests';log=io.StringIO()
 names=['vlm.tests.test_native_recipe_contract','vlm.tests.test_qwen_native_recipe_contract','vlm.tests.test_checkpointing_lifecycle','vlm.tests.test_internvl_native_recipe_contract','vlm.tests.test_provider_sources','vlm.tests.test_llava_native_recipe_contract']
 unit=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names));(W/'tests.txt').write_text(log.getvalue());assert unit.wasSuccessful() and unit.testsRun==24
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.native_recipe_contract import native_recipe_spec,parameter_manifest,validate_native_recipe
 launch=build_launch('T04.sft',(stage/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath('/unused/data'),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 spec=native_recipe_spec(launch);assert spec['attention']=='eager' and spec['training']['gradient_checkpointing'] is False and '--template' not in launch['command']
 phase='original-native-processor-and-template'
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def no_device(*args,**kwargs):raise RuntimeError('GPU initialization forbidden in CPU meta scope')
 torch.cuda._lazy_init=no_device
 from vlm.plugins import llava_onevision as compatibility
 from swift.model import get_model_processor
 empty,processor=get_model_processor(str(root),model_type='llava_onevision1_5',load_model=False,download_model=False,device_map='cpu',attn_impl='eager',torch_dtype=torch.bfloat16)
 assert empty is None and processor.model_meta.template=='llava_onevision1_5'
 from swift.template import get_template
 template=get_template(processor,template_type=None,max_length=4096,max_pixels=262144,padding_side='right',padding_free=False)
 resolved=template.template_meta.template_type;assert resolved=='llava_onevision1_5';template.set_mode('train')
 phase='complete-original-custom-meta-model'
 from transformers import AutoConfig
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 config=AutoConfig.from_pretrained(root,trust_remote_code=True,local_files_only=True)
 for part in (config,config.text_config,config.vision_config):part._attn_implementation='eager'
 compatibility._prepare_load_config(config)
 target_module=compatibility._preimport_verified_eager_class(str(root),config,{'attn_implementation':'eager'})
 model_class=get_class_from_dynamic_module('modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration',str(root),local_files_only=True)
 assert model_class.__module__.endswith('.modeling_llavaonevision1_5')
 native_source=Path(sys.modules[model_class.__module__].__file__);assert sha(native_source)==model_metadata['files']['modeling_llavaonevision1_5.py']['SHA256']
 with torch.device('meta'):core=model_class(config)
 core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info
 compatibility._verify_loaded_native_eager_classes(core,model_class.__module__)
 # Original post-materialization RoPE repair explicitly forbids meta parameters.
 # Do not bypass it or claim loading/buffer-value qualification from this scope.
 assert not getattr(core,'_vlm_onevision_rope_initialized',False)
 phase='actual-native-Swift-PEFT-adapter'
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
 SwiftMixin._prepare_gradient_checkpointing(SimpleNamespace(args=args),core)
 named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
 assert all(p.device.type=='meta' for p in model.parameters())
 binding=dict(spec=spec,resolved_template_type=resolved,target_modules=target,trainable_manifest=parameter_manifest(named),model_source_SHA256=sha(native_source),initial_trainables_SHA256=None,initial_values_qualified=False)
 verified,frozen=validate_native_recipe(model,args,binding)
 assert len(verified)==504 and sum(p.numel() for _,p in verified)==43646976 and all(p.dtype==torch.float32 for _,p in verified)
 language=core.get_submodule('model.language_model');vision=core.get_submodule('model.visual')
 optimizer=torch.optim.AdamW([p for _,p in verified],lr=spec['training']['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.1)
 assert not optimizer.state and sum(len(g['params']) for g in optimizer.param_groups)==504
 from vlm.plugins.native_first_updates import NativeFirstUpdates,KEY,model_providers
 from swift.callbacks import callbacks_map
 holder=object();assert NativeFirstUpdates(args,holder).trainer is holder and callbacks_map[KEY] is NativeFirstUpdates
 providers=model_providers(model,binding['model_source_SHA256'],'T04.sft')
 phase='actual-structural-rejections';rejected=[]
 def reject(label):
  try:validate_native_recipe(model,args,binding)
  except ValueError:rejected.append(label)
  else:raise AssertionError('bad actual OneVision boundary accepted: '+label)
 for name,bad in [('max_steps',24),('gradient_accumulation_steps',1),('learning_rate',.1),('warmup_ratio',.2),('gradient_checkpointing',True)]:
  old=getattr(args,name);setattr(args,name,bad);reject(name);setattr(args,name,old)
 cfg=model.peft_config[model.active_adapter]
 for name,bad in [('r',4),('lora_alpha',4),('lora_dropout',0)]:
  old=getattr(cfg,name);setattr(cfg,name,bad);reject(name);setattr(cfg,name,old)
 for path in ('model.visual','model.visual.merger'):
  p=next(core.get_submodule(path).parameters());p.requires_grad_(True);reject('unfrozen_'+path);p.requires_grad_(False)
 old=binding['trainable_manifest'];binding['trainable_manifest']=old[:-1];reject('missing_tensor');binding['trainable_manifest']=old
 binding['resolved_template_type']='qwen2_vl';reject('foreign_template');binding['resolved_template_type']=resolved
 for part,label in ((core.config,'root_sdpa'),(language.config,'language_sdpa'),(vision.config,'vision_sdpa')):
  old=part._attn_implementation;part._attn_implementation='sdpa';reject(label);part._attn_implementation=old
 for part,label in ((language,'language_GC'),(vision,'vision_GC')):
  part.gradient_checkpointing=True;reject(label);part.gradient_checkpointing=False
 for component,attr,label in ((language.layers[-1],'self_attn','last_language_attention'),(vision.blocks[-1],'attn','last_vision_attention')):
  old=getattr(component,attr);setattr(component,attr,torch.nn.Identity());reject(label);setattr(component,attr,old)
 old=binding['model_source_SHA256'];binding['model_source_SHA256']='0'*64;reject('wrong_source_binding');binding['model_source_SHA256']=old
 from vlm.runtime.provider_sources import model_provider_sources
 try:model_provider_sources(model,'0'*64,'T04.sft')
 except ValueError:rejected.append('wrong_actual_provider_source')
 else:raise AssertionError('unverified actual dynamic source accepted')
 assert len(rejected)==21
 validate_native_recipe(model,args,binding);assert not torch.cuda.is_initialized() and not attempts
 dump('LLAVA_NATIVE_BINDING.json',binding);dump('providers.json',providers)
 runtime={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and name.startswith(('transformers_modules.','swift.model.models.llava','swift.template.templates.llava','swift.trainers.mixin','swift.pipelines.train.tuner','peft.tuners.lora.','vlm.','tools.hygon_vlm_recipe')):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():runtime[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=24,actual_model_class=type(core).__name__,actual_language_class=type(language).__name__,actual_vision_class=type(vision).__name__,resolved_template_type=resolved,actual_template_class=type(template).__module__+'.'+type(template).__qualname__,meta_parameter_count=sum(p.numel() for p in core.parameters()),native_trainable_tensors=len(named),native_trainable_parameters=sum(p.numel() for _,p in named),frozen=frozen,real_Swift_prepare_adapter=True,real_PEFT=True,actual_native_checkpointing_preparation=True,language_layers=36,vision_layers=24,all_native_eager_attention=True,language_GC=False,vision_GC=False,empty_AdamW_bound=True,full_Trainer_constructed=False,rejected_boundaries=rejected,binding_SHA256=sha(W/'LLAVA_NATIVE_BINDING.json'),providers_SHA256=sha(W/'providers.json'),actual_sources=runtime,metadata_files_verified=len(model_metadata['files']),callback_registered=True,versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate')},peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_initialized=False,weight_file_open_attempts=attempts,model_weight_values_read=False,initial_values_qualified=False,materialized_RoPE_repair_qualified=False,full_weight_loading_qualified=False,training_forward_backward=False,GPU_GO=False,Model_GO=False,qualification='Original complete custom OneVision meta/nativeSwift/PEFT/eager/source contract only. No materialized buffers,weight values,actual encoding,forward/backward,numerics,25steps,recovery,performance or effects.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
