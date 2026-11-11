from pathlib import Path
import hashlib,importlib.metadata as metadata,json,os,resource,sys,time
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1];stage=W/'src';source=W/'framework';sys.path.insert(0,str(source));start=time.monotonic();phase='source';result=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 for name,pin in json.loads((stage/'INSTALLED_SOURCE_FILES.json').read_bytes()).items():
  package='transformers' if name.startswith('transformers/') else 'ms-swift';p=Path(metadata.distribution(package).locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 pins=json.loads((stage/'MODEL_PROCESSOR_FILES.json').read_bytes());root=Path(pins['original_model_root'])
 for name,pin in pins['files'].items():assert (root/name).stat().st_size==pin['bytes'] and sha(root/name)==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
 def forbid(*_a,**_kw):raise RuntimeError('GPU context forbidden during source initialization CPU check')
 torch.cuda._lazy_init=forbid;torch.set_num_threads(2)
 from swift.model import get_model_processor
 _,processor=get_model_processor(str(root),model_type='gemma3_vision',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
 from transformers import Gemma3Config,Gemma3ForConditionalGeneration,Seq2SeqTrainingArguments
 config=Gemma3Config.from_pretrained(root,local_files_only=True);config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 phase='full-meta-Swift-PEFT'
 with torch.device('meta'):core=Gemma3ForConditionalGeneration(config)
 core.to(dtype=torch.bfloat16);core.model_meta=processor.model_meta;core.model_info=processor.model_info;core.requires_grad_(False)
 from swift.arguments.tuner_args import TunerArguments
 from swift.pipelines.train.tuner import prepare_adapter
 from dataclasses import dataclass
 @dataclass
 class NativeTunerArguments(TunerArguments):
  model_meta: object=None
  tuner_type: str='lora'
 tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta);tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
 with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
 binding=json.loads((stage/'GEMMA_BINDING.json').read_bytes());args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-training-output'),use_cpu=True,report_to='none',**binding['spec']['training'])
 from vlm.runtime.native_recipe_contract import validate_native_recipe
 from vlm.runtime.initial_trainables import named_parameter_sha256
 named,frozen=validate_native_recipe(model,args,binding);assert len(named)==476 and sum(p.numel() for _,p in named)==29802496 and all(p.is_meta for p in model.parameters())
 for name,p in named:
  parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,torch.nn.Parameter(torch.zeros(tuple(p.shape),dtype=p.dtype,device='cpu'),requires_grad=True))
 named,_=validate_native_recipe(model,args,binding);assert all(not p.is_meta for _,p in named)
 frozen_before={n:(id(p),tuple(p.shape),str(p.dtype),p.is_meta,p.requires_grad) for n,p in model.named_parameters() if not p.requires_grad};assert frozen_before and all(row[3] for row in frozen_before.values())
 optimizer=torch.optim.AdamW([p for _,p in named],lr=args.learning_rate,betas=(args.adam_beta1,args.adam_beta2),eps=args.adam_epsilon,weight_decay=args.weight_decay)
 from vlm.plugins.native_initial_trainables import NativeInitialTrainables
 from vlm.plugins.native_first_updates import NativeFirstUpdates
 trainer=SimpleNamespace(model=model,optimizer=optimizer,callback_handler=SimpleNamespace(callbacks=[]));initialize=NativeInitialTrainables(args,trainer);recorder=NativeFirstUpdates(args,trainer);trainer.callback_handler.callbacks=[initialize,recorder]
 os.environ.update(FIRST_UPDATE_EPOCH=os.environ['CONTRACT_CPU_EPOCH'],FIRST_UPDATE_DEADLINE=os.environ['CONTRACT_CPU_DEADLINE'],FIRST_UPDATE_MAX_SECONDS='300',FIRST_UPDATE_BINDING=str(stage/'GEMMA_BINDING.json'),FIRST_UPDATE_BINDING_SHA256=sha(stage/'GEMMA_BINDING.json'),FIRST_UPDATE_CAPTURE=str(W/'capture'),RANK='0',WORLD_SIZE='8')
 before=named_parameter_sha256(named);objects=[id(p) for _,p in named];rng=torch.get_rng_state().clone();phase='actual-original476-callback-copy';copy_start=time.monotonic();control=object();assert initialize.on_train_begin(args,SimpleNamespace(global_step=0),control) is control;copy_wall=time.monotonic()-copy_start
 after=named_parameter_sha256(named);assert after==binding['initial_trainables_SHA256'] and after!=before and objects==[id(p) for _,p in named] and not optimizer.state and torch.equal(rng,torch.get_rng_state())
 frozen_after={n:(id(p),tuple(p.shape),str(p.dtype),p.is_meta,p.requires_grad) for n,p in model.named_parameters() if not p.requires_grad};assert frozen_before==frozen_after and not (W/'capture').exists()
 receipt=json.loads((W/'capture-initialization/rank-00.json').read_bytes());assert receipt['passed'] and receipt['parameter_tensors']==476 and receipt['parameter_numel']==29802496 and not receipt['RNG_restored']
 assert not torch.cuda.is_initialized()
 result=dict(passed=True,phase='TERMINAL_ACTUAL_SOURCE476_CPU_RESTORE',parameter_tensors=476,parameter_numel=29802496,actual_initial_file=binding['initial_state'],initial_named_SHA256=after,zero_named_SHA256=before,actual_native_model_contract=True,real_Swift_prepare_adapter=True,real_PEFT=True,real_TrainerCallback=True,real_AdamW=True,parameter_objects_preserved=True,optimizer_state_entries=0,frozen_parameter_tensors=len(frozen_before),frozen_meta_boundary_preserved=True,RNG_unchanged=True,source_base_weight_values_loaded=False,source_initial_trainable_values_read=True,initializer_receipt=receipt,initializer_receipt_SHA256=sha(W/'capture-initialization/rank-00.json'),rank_fixture=0,declared_world_size_fixture=8,actual_CPU_processes=1,actual_distributed_run=False,copy_validation_and_hash_wall_seconds=copy_wall,wall_seconds=time.monotonic()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=False,GPU_cards=0,framework_commit='234288e312efc43d859fde3cfc6176866b097a01',versions={k:metadata.version(k) for k in ('torch','transformers','ms-swift','peft','accelerate')},qualification='Real original476 trainable CPU initialization through actual full-Gemma meta architecture and shared Swift/PEFT callback on the source host. No payload export,base-weight values,GPU forward/backward,Hygon restoration,dropout/RNG alignment,cross-platform numerics,recovery,formal performance or heldout effects.')
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-400:],GPU_cards=0,wall_seconds=time.monotonic()-start)
(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['passed']:raise SystemExit(1)
