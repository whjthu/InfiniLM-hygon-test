"""First original64 OneVision encode/collator and actual native logits-policy audit."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import copy,datetime,hashlib,importlib.metadata as md,inspect,json,os,random,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic();initial=deadline-time.time();assert 0<initial<=260
def check():
 if min(deadline-time.time(),initial-(time.monotonic()-start))<=0:raise TimeoutError('original caller CPU deadline exhausted')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original caller CPU deadline exhausted')));signal.setitimer(signal.ITIMER_REAL,initial)
sys.path.insert(0,str(F));attempts=[]
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
  attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weight/capture-file opening forbidden in CPU input audit')
sys.addaudithook(forbid)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source';rows=[]
def progress(failed=False):
 p=W/'caller-progress.tmp';p.write_text(json.dumps(dict(planned_rows=64,completed_rows=len(rows),failed=failed)));p.replace(W/'caller-progress.json')
progress()
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():assert (F/name).stat().st_size==pin['bytes'] and sha(F/name)==pin['SHA256']
 sources=json.loads((S/'ORIGINAL_SOURCES.json').read_bytes())['files']
 for name,pin in sources.items():
  check();p=Path(md.distribution('ms-swift').locate_file(name));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 metadata=json.loads((S/'MODEL_METADATA.json').read_bytes());root=Path(metadata['model_root'])
 for name,pin in metadata['files'].items():
  check();p=root/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assets=json.loads((S/'INPUT_FILES.json').read_bytes())
 for pin in assets['files']:
  check();p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());assert binding['spec']['case']=='T04.sft' and len(binding['trainable_manifest'])==504
 parameters=sum(p['numel'] for p in binding['trainable_manifest']);assert parameters==43646976
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.template_inputs import settings_from_launch
 dataset=Path('/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl')
 launch=build_launch('T04.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 settings=settings_from_launch(launch);assert settings['template_type'] is None and settings['max_length']==4096 and settings['attention']=='eager'
 phase='native-processor-and-class-signature'
 import numpy as np
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def no_device(*args,**kwargs):raise RuntimeError('device initialization forbidden in CPU input audit')
 torch.cuda._lazy_init=no_device
 random.seed(settings['seed']);np.random.seed(settings['seed']);torch.random.default_generator.manual_seed(settings['seed'])
 from vlm.plugins import llava_onevision as compatibility
 from swift.model import get_model_processor
 from swift.template import get_template
 empty,processor=get_model_processor(str(root),model_type='llava_onevision1_5',load_model=False,download_model=False,device_map='cpu',attn_impl='eager',torch_dtype=torch.bfloat16)
 assert empty is None and processor.model_meta.template=='llava_onevision1_5'
 template=get_template(processor,template_type=None,max_length=4096,max_pixels=settings['max_pixels'],padding_side='right',padding_free=False);template.set_mode('train')
 assert template.sequence_parallel_size==1 and not template.padding_free
 from transformers import AutoConfig
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 config=AutoConfig.from_pretrained(root,trust_remote_code=True,local_files_only=True)
 for part in (config,config.text_config,config.vision_config):part._attn_implementation='eager'
 compatibility._prepare_load_config(config);compatibility._preimport_verified_eager_class(str(root),config,{'attn_implementation':'eager'})
 cls=get_class_from_dynamic_module('modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration',str(root),local_files_only=True)
 native_source=Path(inspect.getfile(cls));assert sha(native_source)==binding['model_source_SHA256']
 # The installed policy only inspects multimodal metadata and the bound native
 # forward signature. Use the actual class without its constructor/parameters.
 shell=object.__new__(cls);torch.nn.Module.__init__(shell);shell.model_meta=processor.model_meta
 assert not list(shell.parameters()) and 'logits_to_keep' not in inspect.signature(shell.forward).parameters
 from swift.trainers.mixin import SwiftMixin
 from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
 args=SimpleNamespace(use_logits_to_keep=None);holder=SimpleNamespace(args=args,template=template,model=shell)
 use_logits=SwiftMixin.get_use_logits_to_keep(holder,template.sequence_parallel_size==1)
 assert use_logits is False and args.use_logits_to_keep is False
 providers={}
 for name,method in [('collator',template.data_collator),('base_collator',template._data_collator),('logit_policy',SwiftMixin.get_use_logits_to_keep),('logit_selection_not_called',SwiftMixin.prepare_logits_to_keep),('native_prepare_inputs_not_called',Seq2SeqTrainer._prepare_inputs),('post_encode_not_called',template._post_encode)]:
  p=Path(inspect.getsourcefile(method)).resolve();raw=p.read_bytes();pin=next((v for n,v in sources.items() if Path(md.distribution('ms-swift').locate_file(n)).resolve()==p),None)
  assert pin and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
  providers[name]=dict(module=method.__module__,name=method.__qualname__,path=str(p),bytes=len(raw),SHA256=pin['SHA256'],method_source_SHA256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest())
 from vlm.runtime.first_update_audit import fingerprint
 def identity(v):return {k:x for k,x in fingerprint(v).items() if k!='floating_l2'}
 def fields(v):return {k:identity(x) for k,x in sorted(v.items())}
 class CountSink:
  def __init__(self):self.count=0
  def write(self,data):
   if self.count+len(data)>2*1024**3-1:raise ValueError('complete collated input exceeds per-file recorder bound')
   self.count+=len(data);return len(data)
  def flush(self):pass
  def tell(self):return self.count
 raw_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()];assert len(raw_rows)==64
 phase='actual64-native-encode-and-collator'
 for ordinal,raw_row in enumerate(raw_rows,1):
  check();encoded=template.encode(raw_row);encoded_fields=fields(encoded)
  assert 0<len(encoded['input_ids'])==len(encoded['labels'])<=4096
  collated=template.data_collator([copy.deepcopy(encoded)])
  assert collated['input_ids'].ndim==collated['labels'].ndim==2 and collated['input_ids'].shape==collated['labels'].shape and collated['input_ids'].shape[0]==1
  length=collated['input_ids'].shape[1];assert length==len(encoded['input_ids']) and 'logits_to_keep' not in collated
  image_tokens=int((collated['input_ids']==config.image_token_id).sum());supervised=int((collated['labels']!=-100).sum());assert image_tokens>0 and supervised>0
  before=fields(collated);sink=CountSink();torch.save(dict(args=(),kwargs=collated),sink);assert fields(collated)==before
  rows.append(dict(ordinal=ordinal,sequence_length=length,selected_logit_rows=length,supervised_tokens=supervised,image_tokens=image_tokens,collated_serialized_bytes=sink.count,encoded_fields=encoded_fields,collated_fields=before,complete_sequence_and_labels_retained=True,native_logits_policy=False))
  del encoded,collated;progress()
 from vlm.runtime.native_caller_budget import describe_capture,plan_capture
 plan=describe_capture(rows,parameters=parameters,tensors=504,hidden_size=4096,vocab_size=151936)
 if plan['within_recorder_limits']:assert plan_capture(rows,parameters=parameters,tensors=504,hidden_size=4096,vocab_size=151936)==plan
 else:
  try:plan_capture(rows,parameters=parameters,tensors=504,hidden_size=4096,vocab_size=151936)
  except ValueError as e:assert str(e)=='complete first-two-update capture exceeds unchanged recorder limits'
  else:raise AssertionError('oversized plan admitted')
 check();assert not torch.cuda.is_initialized() and not attempts and not list(shell.parameters())
 dump('native-caller-rows.json',rows);dump('native-caller-providers.json',providers);dump('capture-budget-description.json',plan)
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),native_template=template.template_meta.template_type,settings=settings,planned_rows=64,completed_rows=len(rows),min_full_sequence=min(r['sequence_length'] for r in rows),max_full_sequence=max(r['sequence_length'] for r in rows),native_logits_policy=False,logit_selection_called=False,actual_native_class_signature_only=True,model_constructor_called=False,forward_signature=str(inspect.signature(shell.forward)),native_method_providers=providers,capture_budget_description=plan,capture_admission_state='CONSERVATIVE_BOUND_FITS' if plan['within_recorder_limits'] else 'CONSERVATIVE_BOUND_EXCEEDS_RECORDER',rows_SHA256=sha(W/'native-caller-rows.json'),budget_SHA256=sha(W/'capture-budget-description.json'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_initialized=False,weight_file_open_attempts=attempts,actual_model_loaded=False,actual_image_embedding_forward=False,full_Trainer_constructed=False,actual_GPU_output_dtype_or_serialization_qualified=False,raw_encoded_values_saved=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False,qualification='First complete original64 native OneVision CPU encode/collator and installed policy on actual bound custom-forward signature. No constructor/weights/postencode forward/fullTrainer/device tensors. Full F32 conservative output budget is analysis,not measured GPU artifact size or training acceptance.')
except BaseException as error:
 import traceback
 progress(True);dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,planned_rows=64,completed_rows=len(rows),owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),completed_rows=result.get('completed_rows'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
