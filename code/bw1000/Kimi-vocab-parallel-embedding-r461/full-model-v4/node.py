"""One node's actual TP32 constructor/full loader/typed AR/reset/destroy.

Metadata modes import neither Torch nor Native/SDK. Old real IDs are reused.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,socket,struct,sys,time,traceback
from constructor import check_metadata,resident_constructor
import model_driver
EMBEDDING_KEY='language_model.model.embed_tokens.weight'
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkpoint_header(q):
 root=Path(q['model_directory']);pin=q['checkpoint_embedding'];p=root/pin['filename']
 with p.open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n==pin['header_bytes'];raw=f.read(n)
 assert hashlib.sha256(raw).hexdigest()==pin['header_SHA256']
 actual=json.loads(raw)[EMBEDDING_KEY];assert actual==pin['tensor']
 st=p.stat();assert st.st_size==pin['file_bytes']and st.st_mtime_ns==pin['file_mtime_ns']
 return dict(filename=pin['filename'],header_SHA256=pin['header_SHA256'],tensor=actual,file_bytes=st.st_size,file_mtime_ns=st.st_mtime_ns)
def metadata(node=None,hostname=None,require_gpu_entry_mask=True):
 q=json.loads((D/'config.json').read_bytes());n=int(os.environ['SLURM_PROCID'])if node is None else node
 h=socket.gethostname().split('.')[0]if hostname is None else hostname
 assert n in range(4)and h==q['nodes'][n]
 if require_gpu_entry_mask:assert len(os.sched_getaffinity(0))>=32,'Original entry>=32; no early pin'
 for key,value in q['environment'].items():assert os.environ.get(key)==value,(key,os.environ.get(key))
 root=Path(q['model_directory']);assert sha(root/'config.json')==q['config_SHA256']and sha(root/'model.safetensors.index.json')==q['index_SHA256']
 cfg=json.loads((root/'config.json').read_bytes());wm=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map'];check_metadata(cfg,wm)
 assert wm[EMBEDDING_KEY]==q['checkpoint_embedding']['filename'];checkpoint_header(q)
 return q,n,cfg,wm
def fixtures(q):
 pins=q['input_pins'];b=Path(pins['encoding_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==pins['encoding_SHA256'];enc=json.loads(b)
 assert enc['passed']and enc['actual_CPU']=='910407.937'and len(enc['cases'])==256
 raw=Path(pins['cases_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==pins['cases_SHA256'];cases=json.loads(raw)['cases'];assert len(cases)==256
 output=[]
 for ordinal,(e,case)in enumerate(zip(enc['cases'],cases)):
  assert e['selected_ordinal']==ordinal and e['case_id']==case['id']
  p=Path(e['private_path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==e['private_SHA256'];v=json.loads(raw);ids=v['prompt_token_ids']
  assert v['case_id']==case['id']and model_driver.H(model_driver.compact(ids))==e['prompt_IDs_SHA256']
  assert len(ids)==e['prompt_tokens']and len(ids)+4096<=32768 and v['body']['max_tokens']==4096
  assert model_driver.H(v['prompt'].encode())==e['prompt_SHA256']and model_driver.H(model_driver.compact(v['body']))==e['request_body_SHA256']
  output.append((e,case,v))
 assert q['order']==[0,128]+[i for i in range(256)if i not in(0,128)]
 return output

def first_prefill_affinity_guard(lib,control,record,node,entry_mask):
 # Query bytes are recorded before any predicate. No affinity mutation here.
 raw=json.loads(lib.modern_owner_affinity_report_r2392())
 record('first_prefill_affinity_raw',dict(node=node,report=raw))
 started=time.perf_counter();error=None
 try:
  assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'
  owners=raw['owners'];assert len(owners)==8
  assert {v['device']for v in owners}==set(range(8))
  entry=sorted(entry_mask);assert len(entry)>=32
  masks=[]
  for v in owners:
   assert v['changes']==1,('Runtime owner must bind exactly once after first prefill',v)
   mask=[int(c)for c in v['verified_mask'].split(',')]
   assert mask==entry[4*v['device']:4*v['device']+4],('Exact Runtime owner mask',v,entry)
   masks+=mask
  assert len(masks)==len(set(masks))==32
 except(AssertionError,KeyError,TypeError,ValueError)as failure:error=repr(failure)
 packet=dict(node=node,passed=error is None,error=error,actual_owner_report=raw)
 peers=control.collect('first-real-prefill-affinity-all32',packet)
 record('first_prefill_affinity_all32',dict(peers=peers,wall_s=time.perf_counter()-started))
 assert len(peers)==4 and {v['node']for v in peers}==set(range(4))
 assert all(v['passed']for v in peers),('Actual first-prefill owner binding failure',peers)
 return packet

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--metadata-only',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--metadata-all-recording-hosts',action='store_true');args=parser.parse_args(argv)
 assert sum([args.metadata_only,args.execute,args.metadata_all_recording_hosts])==1
 if args.metadata_all_recording_hosts:
  q=json.loads((D/'config.json').read_bytes());roles=[]
  fixtures(q)
  for n,h in enumerate(q['nodes']):metadata(n,h,False);roles.append(dict(node=n,hostname_leaf='recording:'+h))
  print(json.dumps(dict(passed=True,roles=roles,Native_dlopen=False,GPU=0,real_affinity_count=len(os.sched_getaffinity(0)),GPU_entry_mask_qualified=False,
   planned_questions=256,fixed_order=q['order'],checkpoint_header=checkpoint_header(q))));return
 q,n,cfg,wm=metadata();fixtures(q)
 if args.metadata_only:print(json.dumps(dict(passed=True,node=n,Native_dlopen=False,GPU=0)));return
 epoch=float(os.environ['K461_EPOCH']);deadline=epoch+q['Native_seconds'];assert time.time()<deadline
 assert sys.byteorder=='little'
 out=Path(q['runtime_directory']);out.mkdir(exist_ok=True);report=out/('node%d.json'%n);assert not report.exists()
 result=dict(node=n,hostname=q['nodes'][n],run=q['run'],source_manifest_SHA256=os.environ['K461_SOURCE_SHA'],epoch=epoch,Native_deadline=deadline,
  actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),UID=os.getuid(),PGID=os.getpgrp(),entry_mask=sorted(os.sched_getaffinity(0)),
  completed=False,passed=False,weight_keys=0,fullModel_forward=0,HTTP=0,reset=0,create_calls=0,destroy_calls=0,events=[])
 def save():
  t=report.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(report)
 def store(name,raw):
  path=out/('node%d-'%n+name)
  with path.open('xb')as f:f.write(raw)
  return dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),dtype='BF16',endianness='little',scope='raw CPU reference/output; no activation export to Git')
 def record(event,payload):
  # Actual full key list is validated by the unchanged consumer but stored only
  # once, avoiding repeated 40MB metadata in every memory/query event.
  if event=='constructor_observation'and'keys'in payload:
   count=len(payload['keys']);keys=json.dumps(payload['keys'],separators=(',',':')).encode();keyrecord=store('constructor-keys.json',keys);keyrecord.pop('dtype');keyrecord.pop('endianness')
   keyrecord['scope']='Actual constructor key JSON; no tensor payload'
   payload={k:v for k,v in payload.items()if k!='keys'};payload['keys_manifest']=dict(**keyrecord,count=count)
  result['events'].append(dict(event=event,payload=payload,unix=time.time()))
  if event=='before_create':result['create_calls']+=1
  if event=='destroy_returned':result['destroy_calls']+=1
  if event.startswith('memory_'):result[event]=payload
  save()
 save()
 try:
  import torch
  import numpy as np
  from safetensors import safe_open
  assert torch.cuda.device_count()==8
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];cuda.cudaDeviceGetPCIBusId.restype=C.c_int
  hardware=[]
  for d in range(8):
   prop=torch.cuda.get_device_properties(d);bus=C.create_string_buffer(64)
   assert 'gfx936'in prop.gcnArchName and cuda.cudaDeviceGetPCIBusId(bus,64,d)==0
   hardware.append(dict(device=d,PCI=bus.value.decode().lower(),arch=prop.gcnArchName))
  assert len({v['PCI']for v in hardware})==8 and {v['PCI']for v in hardware}=={p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')}
  result['hardware']=hardware;torch.cuda.set_device(0)
  memory=lambda:[list(torch.cuda.mem_get_info(d))for d in range(8)]
  record('memory_before_create',memory());assert all(t==q['physical_VRAM_bytes']and f>q['floor_bytes']for f,t in result['memory_before_create'])
  assert sha(q['native'])==q['Native_SHA256'];lib=C.CDLL(q['native'],mode=os.RTLD_NOW|os.RTLD_LOCAL)
  class Info(C.Structure):_fields_=[('file',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
  dl=C.CDLL(None);dl.dladdr.argtypes=[C.c_void_p,C.POINTER(Info)];dl.dladdr.restype=C.c_int;providers={}
  for name in ['modern_kimi_engine_create_vocab_parallel_r461','modern_kimi_engine_forward_ar_storage_r456','modern_kimi_engine_load_r2475','modern_kimi_engine_finish_r2475','modern_kimi_engine_reset_r2475','modern_kimi_engine_destroy_r2475','modern_owner_affinity_report_r2392']:
   info=Info();assert dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info))
   path=Path(info.file.decode()).resolve();assert path==Path(q['native']).resolve()and sha(path)==q['Native_SHA256'];providers[name]=dict(path=str(path),SHA256=sha(path),address=int(info.address))
  maps=Path('/proc/self/maps').read_text();result.update(dladdr=providers,Native_maps=maps)
  for basename,pin in q['dependency_libraries'].items():
   found={line.split()[-1]for line in maps.splitlines()if line.split()and line.split()[-1].split('/')[-1]==basename}
   assert len(found)==1 and all(sha(v)==pin['SHA256']for v in found),(basename,found)
  save()
  with resident_constructor(lib,cfg,wm,node=n,address=q['master_address'],port=q['native_port'],record=record)as(ctx,_):
   try:
    record('memory_after_create',memory());assert all(t==q['physical_VRAM_bytes']and f>q['floor_bytes']for f,t in result['memory_after_create']),'Original 4GiB constructor floor'
    from parameter_loader_r461 import load_parameters
    from head_source_r461 import read_head
    from public_kimi_native.runtime.stage_control_r2386 import StageControl
    from public_kimi_native.runtime.long_codec_r179 import KimiLongCodec
    import model_driver
    control=StageControl(n,q['master_address'],q['native_port']+7000,q['run']+os.environ['K461_SOURCE_SHA'],4,1800)
    def check(rc):
     if rc:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    model_driver.declare(lib)
    from public_kimi_native.runtime.memory_guard_r9144 import MemorySampler
    sampler=MemorySampler(torch,interval=.01)
    def memory_guard(label):
     snap=sampler.snapshot();assert not snap['sampler_error'];packets=control.collect('floor:'+label,snap)
     minimum=min(pair[0]for packet in packets for pair in packet['memory']);assert minimum>q['floor_bytes'],'Original sampled all32 floor'
     return minimum
    try:
     keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));assert len(keys)>1000
     key_packets=control.collect('before-complete-loader',dict(node=n,keys_SHA256=model_driver.H(model_driver.compact(sorted(keys))),keys=len(keys)))
     assert len({v['keys_SHA256']for v in key_packets})==1
     record('full_load_begin',dict(keys=len(keys)))
     loaded=load_parameters(lib,ctx,Path(q['model_directory']),cfg['text_config'],keys,safe_open,torch,check,lambda v:record('load_progress',v))
     result.update(full_loader=loaded,weight_keys=loaded['loaded_keys'])
     result['head_after_finish']=read_head(lib,n,1,True)
     result['affinity_after_finish_raw']=json.loads(lib.modern_owner_affinity_report_r2392())
     record('affinity_after_finish_raw',result['affinity_after_finish_raw'])
     record('memory_after_finish',memory());assert all(f>q['floor_bytes']for f,t in result['memory_after_finish']),'Original full Model 4GiB floor'
     control.collect('all-four-weights-ready',dict(node=n,loaded=loaded['loaded_keys'],memory=result['memory_after_finish']))
     result['sampled_minimum_before_ready']=memory_guard('after-finish');result['Model_READY']=True;save()
     codec=KimiLongCodec(q['model_directory'])
     assert set(codec.eos_token_ids)=={163585,163586}
     planned=fixtures(q);result['planned_questions']=256;result['request_reports']=[];save()
     service_deadline=time.monotonic()+max(0.,epoch+q['service_seconds']-time.time())
     for ordinal in q['order']:
      admitted=control.collect('admit-ordinal:'+str(ordinal),dict(admit=time.monotonic()<service_deadline-120 if n==0 else None))[0]['admit']
      if not admitted:
       result['cohort_partial_due_original_deadline']=True;break
      e,case,private=planned[ordinal];casepath=out/('node%d-case-%03d.json'%(n,ordinal));item=dict(ordinal=ordinal,case_id=e['case_id'],suite=case['suite'],planned_output_budget=4096,completed=False,request_body_SHA256=e['request_body_SHA256'],history_render_audit=e['history_render_audit'],private_input_SHA256=e['private_SHA256'])
      def request_record(event,payload):
       with(out/('node%d-progress.jsonl'%n)).open('a')as f:f.write(json.dumps(dict(ordinal=ordinal,event=event,payload=payload,unix=time.time()),separators=(',',':'))+'\n')
      def save_request():
       temp=casepath.with_suffix('.tmp');temp.write_text(json.dumps(item,allow_nan=False)+'\n');temp.replace(casepath)
      save_request()
      try:
       item.update(model_driver.generate(lib,ctx,n,control,private['prompt_token_ids'],4096,service_deadline,request_record,memory_guard,after_first_prefill=(lambda:first_prefill_affinity_guard(lib,control,record,n,result['entry_mask']))if ordinal==q['order'][0]else None))
       rawtext=codec.decode(item['generated_ids']);item.update(decoded_text=rawtext,decoded_text_no_EOS=codec.decode(item['generated_ids'],skip_special_tokens=True),completed=True)
       # No correctness gate. Gold scoring is offline with the frozen r448 policy.
       result['fullModel_forward']+=len(item['calls']);result['reset']+=2
      except BaseException as error:
       item.update(error=repr(error),traceback=traceback.format_exc());save_request();raise
      save_request();result['request_reports'].append(dict(ordinal=ordinal,path=str(casepath),SHA256=sha(casepath),bytes=casepath.stat().st_size));save()
     control.collect('all-requests-ended',dict(node=n,completed=len(result['request_reports'])))
     result['completed_requests']=len(result['request_reports']);result['missing_questions']=256-result['completed_requests']
     result['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx))
    finally:
     primary=sys.exc_info()[1]
     try:
      result['memory_sampler_final']=sampler.snapshot();sampler.close()
     except BaseException as cleanup:
      if primary is None:raise
      if hasattr(primary,'add_note'):primary.add_note('sampler cleanup: '+repr(cleanup))
     try:control.close()
     except BaseException as cleanup:
      if primary is None:raise
      if hasattr(primary,'add_note'):primary.add_note('control close: '+repr(cleanup))
    record('memory_before_destroy',memory())
    result['affinity_after_queries']=json.loads(lib.modern_owner_affinity_report_r2392())
    save()
   except BaseException as error:
    # Persist before the context manager attempts destroy; keeper can stop
    # peers even when an erroring node is waiting in a real collective drain.
    result.update(error=repr(error),traceback=traceback.format_exc());save();raise
  result['embedding_after_destroy']=json.loads(lib.modern_kimi_vocab_embedding_info_r461())
  assert all(owner['destroyed']for owner in result['embedding_after_destroy']['owners'])
  record('memory_after_destroy',memory())
  assert result['create_calls']==result['destroy_calls']==1 and not any(e['event']=='cleanup_error'for e in result['events'])
  result['passed']=True
 except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:result['completed']=True;result['finished_unix']=time.time();save()
if __name__=='__main__':main()
