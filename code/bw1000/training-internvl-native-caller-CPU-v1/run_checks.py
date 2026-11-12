"""Actual native collator and logits-selection methods on all64 original rows."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import copy,datetime,hashlib,inspect,json,os,random,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();initial=deadline-time.time();assert 0<initial<=260
def check():
 if min(deadline-time.time(),initial-(time.monotonic()-start))<=0:raise TimeoutError('native caller original CPU epoch exhausted')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native caller CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,initial)
sys.path.insert(0,str(F));attempts=[]
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
  attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weight/capture file opens forbidden in CPU caller qualification')
sys.addaudithook(forbid)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source';rows=[]
def progress(failed=False):
 p=W/'caller-progress.tmp';p.write_text(json.dumps(dict(planned_rows=64,completed_rows=len(rows),failed=failed)));p.replace(W/'caller-progress.json')
progress()
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():assert (F/name).stat().st_size==pin['bytes'] and sha(F/name)==pin['SHA256']
 sources=json.loads((S/'ORIGINAL_SOURCES.json').read_bytes())['files']
 for pin in sources.values():
  check();p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 metadata=json.loads((S/'MODEL_METADATA.json').read_bytes());root=Path(metadata['model_root'])
 for name,pin in metadata['files'].items():
  check();p=root/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assets=json.loads((S/'INPUT_FILES.json').read_bytes())
 for pin in assets['files']:
  check();p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 prior=json.loads((S/'INPUT_REFERENCE.json').read_bytes());reference=prior['identities'];assert len(reference)==64
 binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());assert binding['spec']['case']=='T03.sft' and len(binding['trainable_manifest'])==504
 parameters=sum(p['numel'] for p in binding['trainable_manifest']);assert parameters==43646976
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.template_inputs import settings_from_launch
 dataset=Path('/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl')
 launch=build_launch('T03.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 settings=settings_from_launch(launch);assert settings==prior['capture']['settings'] and settings['template_type'] is None
 phase='actual-native-caller-setup'
 import numpy as np
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def no_device(*args,**kwargs):raise RuntimeError('device initialization forbidden in CPU caller scope')
 torch.cuda._lazy_init=no_device
 random.seed(settings['seed']);np.random.seed(settings['seed']);torch.random.default_generator.manual_seed(settings['seed'])
 from swift.model import get_model_processor
 from swift.template import get_template
 empty,processor=get_model_processor(str(root),model_type='internvl',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
 assert empty is None and processor.model_meta.template=='internvl_hf'
 template=get_template(processor,template_type=None,max_length=4096,max_pixels=settings['max_pixels'],padding_side='right',padding_free=False);template.set_mode('train')
 assert template.sequence_parallel_size==1 and not template.padding_free
 from transformers import InternVLConfig,InternVLForConditionalGeneration
 config=InternVLConfig.from_pretrained(root,local_files_only=True)
 config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 with torch.device('meta'):core=InternVLForConditionalGeneration(config)
 core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info
 from swift.trainers import Seq2SeqTrainingArguments
 from swift.trainers.mixin import SwiftMixin
 from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
 args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**binding['spec']['training'])
 SwiftMixin._prepare_gradient_checkpointing(SimpleNamespace(args=args),core)
 holder=SimpleNamespace(args=args,template=template,model=core)
 assert args.use_logits_to_keep is None
 use_logits=SwiftMixin.get_use_logits_to_keep(holder,template.sequence_parallel_size==1)
 assert use_logits is True and args.use_logits_to_keep is True
 from vlm.runtime.native_caller_budget import plan_capture,validate_capture_conditions
 conditions=dict(case='T03.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False)
 guard=validate_capture_conditions(core,conditions)
 providers={}
 for name,method in [('collator',template.data_collator),('base_collator',template._data_collator),('logit_policy',SwiftMixin.get_use_logits_to_keep),('logit_selection',SwiftMixin.prepare_logits_to_keep),('native_prepare_inputs',Seq2SeqTrainer._prepare_inputs),('post_encode',template._post_encode)]:
  p=Path(inspect.getsourcefile(method)).resolve();raw=p.read_bytes();pin=next((v for v in sources.values() if Path(v['path']).resolve()==p),None)
  assert pin and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  providers[name]=dict(module=method.__module__,name=method.__qualname__,path=str(p),bytes=len(raw),SHA256=pin['SHA256'],method_source_SHA256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest())
 from vlm.runtime.first_update_audit import fingerprint
 def identity(v):return {k:x for k,x in fingerprint(v).items() if k!='floating_l2'}
 def fields(v):return {k:identity(x) for k,x in sorted(v.items())}
 class CountSink:
  def __init__(self):self.count=0
  def write(self,data):
   if self.count+len(data)>2*1024**3-1:raise ValueError('complete collated artifact exceeds unchanged per-file limit')
   self.count+=len(data);return len(data)
  def flush(self):pass
  def tell(self):return self.count
 raw_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()];assert len(raw_rows)==64
 phase='actual64-native-collator-and-logit-selection'
 for ordinal,raw_row in enumerate(raw_rows,1):
  check();encoded=template.encode(raw_row);encoded_fields=fields(encoded)
  assert encoded_fields==reference[ordinal-1]['fields'] and len(encoded['input_ids'])==reference[ordinal-1]['sequence_length']
  collated=template.data_collator([copy.deepcopy(encoded)])
  assert collated['input_ids'].ndim==collated['labels'].ndim==2 and collated['input_ids'].shape==collated['labels'].shape and collated['input_ids'].shape[0]==1
  length=collated['input_ids'].shape[1];assert length==len(encoded['input_ids']) and 0<length<=4096
  before=fields(collated);original_labels=collated['labels'].clone();loss_mask=(original_labels!=-100)[0];assert not bool(loss_mask[0])
  image_tokens=int((collated['input_ids']==config.image_token_id).sum());assert image_tokens>0
  supervised=int(loss_mask.sum());assert supervised>0
  SwiftMixin.prepare_logits_to_keep(holder,collated)
  selected=collated['logits_to_keep'];assert selected.dtype==torch.bool and list(selected.shape)==[length]
  assert torch.equal(selected,torch.nn.functional.pad(loss_mask[1:],(0,1),value=True))
  assert torch.equal(collated['labels'],torch.nn.functional.pad(original_labels[:,loss_mask],(1,0),value=-100))
  selected_rows=int(selected.sum());assert selected_rows==supervised+1==collated['labels'].shape[1]
  after=fields(collated)
  unchanged=set(before)-{'labels'};assert all(before[k]==after[k] for k in unchanged) and set(after)==set(before)|{'logits_to_keep'}
  sink=CountSink();torch.save(dict(args=(),kwargs=collated),sink)
  rows.append(dict(ordinal=ordinal,sequence_length=length,selected_logit_rows=selected_rows,supervised_tokens=supervised,image_tokens=image_tokens,
   collated_serialized_bytes=sink.count,original_encoded_fields_exact=True,complete_input_sequence_unchanged=True,non_label_fields_unchanged=True,
   label_and_logit_mask_native_semantics_checked=True,before_fields=before,after_fields=after))
  del encoded,collated,original_labels,loss_mask,selected
  progress()
 plan=plan_capture(rows,parameters=parameters,tensors=504,hidden_size=4096,vocab_size=151936)
 check();assert not torch.cuda.is_initialized() and not attempts and all(p.device.type=='meta' for p in core.parameters())
 dump('native-caller-rows.json',rows);dump('native-caller-providers.json',providers);dump('capture-budget-plan.json',plan);dump('capture-budget-conditions.json',conditions)
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),native_template=template.template_meta.template_type,planned_rows=64,completed_rows=len(rows),
  min_full_sequence=min(r['sequence_length'] for r in rows),max_full_sequence=max(r['sequence_length'] for r in rows),
  min_selected_logit_rows=min(r['selected_logit_rows'] for r in rows),max_selected_logit_rows=max(r['selected_logit_rows'] for r in rows),
  native_logits_policy=use_logits,native_method_providers=providers,capture_budget=plan,capture_conditions=guard,
  original_pre_model_reference_SHA256=sha(S/'INPUT_REFERENCE.json'),rows_SHA256=sha(W/'native-caller-rows.json'),budget_SHA256=sha(W/'capture-budget-plan.json'),
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
  GPU_cards=0,GPU_initialized=False,weight_file_open_attempts=attempts,actual_model_loaded=False,actual_image_embedding_forward=False,full_Trainer_constructed=False,
  actual_GPU_serialized_artifact_bytes_qualified=False,training_forward_backward=False,raw_encoded_values_saved=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False,
  qualification='Actual CPU nativecollator and installed logitpolicy/selection on complete original64 inputs; actual collated serialization counted and conservative unchanged8GiB recorder plan. Bare complete meta HF model for policy signature/cache conditions only. No initialized LoRA,post-encode embedding forward,full Trainer,weights,training,numerics,resume or performance.')
except BaseException as error:
 import traceback
 progress(True);dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,planned_rows=64,completed_rows=len(rows),owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),completed_rows=result.get('completed_rows'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
