"""Original64 native HF inputs and original header-to-meta key closure, no values."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,json,os,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic()
initial=deadline-time.time();assert 0<initial<=260;anchor=time.monotonic()
def check():
 if min(deadline-time.time(),initial-(time.monotonic()-anchor))<=0:raise TimeoutError('original CPU epoch exhausted')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original CPU epoch exhausted')))
signal.setitimer(signal.ITIMER_REAL,initial);sys.path.insert(0,str(F))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='pins';allowed_header=None;attempts=[]
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=os.fsdecode(args[0])
  if p.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')) and p!=allowed_header:
   attempts.append(hashlib.sha256(os.fsencode(p)).hexdigest());raise RuntimeError('weight values forbidden in input/header scope')
sys.addaudithook(forbid)
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():assert (F/name).stat().st_size==pin['bytes'] and sha(F/name)==pin['SHA256']
 pins=json.loads((S/'MODEL_METADATA.json').read_bytes());root=Path(pins['model_root'])
 for name,pin in pins['files'].items():check();assert (root/name).stat().st_size==pin['bytes'] and sha(root/name)==pin['SHA256']
 sources=json.loads((S/'ORIGINAL_SOURCES.json').read_bytes())['files']
 for name,pin in sources.items():
  check();p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 inputs=json.loads((S/'ORIGINAL_INPUT_FILES.json').read_bytes())
 for pin in inputs['files']:
  check();p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.template_inputs import capture_native_inputs,settings_from_launch
 dataset=Path('/mnt/data/nvidia-baseline-20261010/inputs/vlmdata/vlm_report_coco64.cuda.jsonl')
 dataset_pin=next(p for p in inputs['files'] if p['source']==str(dataset))
 launch=build_launch('T03.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 settings=settings_from_launch(launch);assert settings['model_type']=='internvl' and settings['template_type'] is None and settings['max_length']==4096
 phase='original64-native-HF-inputs'
 capture=capture_native_inputs(settings,root,dataset,W/'encoded',pins['files'],dataset_pin['SHA256'],epoch=epoch,deadline=deadline)
 assert capture['passed'] and capture['rows']==64 and capture['actual_template_type']=='internvl_hf'
 rows=json.loads((W/'encoded/input-identities.json').read_bytes())
 assert len(rows)==64 and all(r['row_ordinal']==i+1 for i,r in enumerate(rows))
 phase='original-four-checkpoint-headers'
 from vlm.runtime.native_checkpoint_schema import read_header,audit_native_schema
 inventory=json.loads((S/'SHARD_INVENTORY.json').read_bytes());headers={}
 for name,pin in inventory.items():
  assert Path(name).name==name and name.endswith('.safetensors')
  assert (root/name).stat().st_mtime_ns==pin['mtime_ns']
  allowed_header=str(root/name)
  try:headers[name]=read_header(root/name,pin['bytes'],check=check)
  finally:allowed_header=None
 assert len(headers)==4
 dump('checkpoint-headers.json',headers)
 phase='native-installed-key-mapping-to-complete-meta-schema'
 import torch
 assert not torch.cuda.is_initialized();torch.set_num_threads(2)
 def no_device(*args,**kwargs):raise RuntimeError('device initialization forbidden in meta schema scope')
 torch.cuda._lazy_init=no_device
 from transformers import InternVLConfig,InternVLForConditionalGeneration
 config=InternVLConfig.from_pretrained(root,local_files_only=True)
 config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
 with torch.device('meta'):model=InternVLForConditionalGeneration(config)
 model.to(dtype=torch.bfloat16)
 native={n:dict(shape=list(p.shape),dtype=str(p.dtype),device=p.device.type) for n,p in model.state_dict().items()}
 dump('native-meta-schema.json',native)
 index=json.loads((root/'model.safetensors.index.json').read_bytes())
 provider_pins={Path(name).name:pin for name,pin in sources.items() if name.startswith('transformers/')}
 audit=audit_native_schema(model,index,headers,provider_pins,check=check)
 assert audit['keys']==audit['meta_state_keys']==841 and audit['payload_bytes_read']==0
 dump('native-checkpoint-key-schema.json',audit)
 check();assert not torch.cuda.is_initialized() and not attempts
 actual_sources={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and name.startswith(('transformers.models.internvl.','transformers.models.qwen3.','transformers.conversion_mapping','transformers.core_model_loading','swift.template.templates.internvl','swift.model.models.internlm','vlm.runtime.')):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():actual_sources[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 lengths=[r['sequence_length'] for r in rows]
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),original_input_files_verified=len(inputs['files']),processor_files_verified=len(pins['files']),
  capture=capture,planned_rows=64,encoded_fields=sum(len(r['fields']) for r in rows),sequence_length_min=min(lengths),sequence_length_max=max(lengths),
  schema_keys=audit['keys'],renamed_keys=audit['changed_names'],checkpoint_shards=len(headers),header_bytes_read=sum(h['bytes_read'] for h in headers.values()),
  header_payload_bytes_read=0,full_checkpoint_SHA256_qualified=False,actual_weight_loading_qualified=False,model_initialized_values=False,
  actual_sources=actual_sources,versions={n:metadata.version(n) for n in ('torch','transformers','ms-swift','peft','accelerate')},
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
  GPU_cards=0,GPU_initialized=False,weight_file_open_attempts=attempts,training_forward_backward=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False,
  qualification='Original64 CPU native template and complete841 HF checkpoint header/key/shape/dtype correspondence. No values,actual loading,post-model collator,forward/backward,updates,resume,performance or effects.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
