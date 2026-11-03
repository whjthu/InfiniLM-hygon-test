"""Real main/prepare/loader/head/key/ordinary consumers; only CABI/device leaves record.

Four real StageControl socket leaders, actual frozen input and parser/tokenizer,
actual filesystem output and cancellation. No Native dlopen/Runtime/weights/GPU.
"""
from pathlib import Path
import concurrent.futures,contextlib,ctypes as C,hashlib,io,json,os,re,socket,sys,tempfile,threading,time,types,weakref
import numpy as np
import bootstrap
from contract import CUTS,CAP,GEOMETRY,layout,sha
from native_binding import SIGNATURES,declare
from lifecycle import GuardedLibrary
D=Path(__file__).resolve().parent
class Tensor:
 registry=weakref.WeakValueDictionary()
 def __init__(self,name):
  self.name=name;self.device=types.SimpleNamespace(type='cpu');self.dtype='BF16';self.shape=[163840,7168]if name=='language_model.lm_head.weight'else[96]if name.endswith('.A_log')else[1]
  self.buffer=(C.c_uint16*8)();self.registry[C.addressof(self.buffer)]=self
 def contiguous(self):return self
 def is_contiguous(self):return True
 def numel(self):return int(np.prod(self.shape))
 def element_size(self):return 2
 def data_ptr(self):return C.addressof(self.buffer)
 def view(self,dtype):assert dtype=='U16';return self
 def numpy(self):return self
 def __iter__(self):return iter(range(7168))
 def __getitem__(self,key):
  if isinstance(key,slice):return self
  return[(int(key)+j)&65535 for j in range(7168)]
class Shard:
 def __enter__(self):return self
 def __exit__(self,*args):return False
 def get_tensor(self,name):return Tensor(name)
class FakeTorch:
 float32='F32';bfloat16='BF16';uint8='U8';uint16='U16'
 def __init__(self):
  self.cuda=types.SimpleNamespace(device_count=lambda:8,get_device_properties=lambda d:types.SimpleNamespace(name='gfx936 CPU-recording',total_memory=68702699520,multi_processor_count=80),mem_get_info=lambda d:(60<<30,68702699520),max_memory_allocated=lambda d:0)
 def set_num_threads(self,n):assert n==1
class Function:
 def __init__(self,f):self.f=f
 def __call__(self,*args):return self.f(*args)
class RecordingLibrary:
 def __init__(self,node,mapping,outputs,mode='normal'):
  self.node=node;self.mapping=mapping;self.outputs=outputs;self.mode=mode;self.destroyed=0;self.created=0;self.forward_calls=[];self.loaded=False;self.head_expected=None;self.reset_count=0;self.cursor=0;self.error='recording CABI error';self.functions={};self.load_calls=0
  keys=[]
  for layer in range(*CUTS[node]):keys.append(next(k for k in mapping if k.startswith('language_model.model.layers.'+str(layer)+'.')))
  for name in ['language_model.model.embed_tokens.weight','language_model.lm_head.weight']:
   if(name.endswith('embed_tokens.weight')and node==0)or(name.endswith('lm_head.weight')and node==3):assert name in mapping;keys.append(name)
  self.keys=keys
  # Actual ctypes foreign-call argument conversion for every forward buffer.
  args,ret=SIGNATURES['modern_kimi_engine_forward_ordinary_r410']
  self.forward_ffi=C.CFUNCTYPE(ret,*args)(self.forward)
  self.context=0x100000000+node+1
  args,ret=SIGNATURES['modern_kimi_engine_create_ordinary_tp8_pp4_r410']
  self.create_ffi=C.CFUNCTYPE(ret,*args)(self.create)
 def __getattr__(self,name):
  if name not in SIGNATURES:raise AssertionError('Forbidden/unbound CAPI '+name)
  if name not in self.functions:
   f=self.forward_ffi if name=='modern_kimi_engine_forward_ordinary_r410'else self.create_ffi if name=='modern_kimi_engine_create_ordinary_tp8_pp4_r410'else Function(lambda *a:self.dispatch(name,*a));self.functions[name]=f
  return self.functions[name]
 def create(self,config,node,head,port):
  try:
   assert node==self.node and json.loads(config)['model_type']=='kimi_k3'and head and port>0
   self.created+=1;return self.context
  except BaseException as e:self.error=repr(e);return 0
 def forward(self,ctx,kind,tokens,count,flags,*pointers):
  try:
   assert ctx==self.context and count==1 and flags==3 and len(pointers)==9
   array=lambda p,t,n:np.ctypeslib.as_array(C.cast(p,C.POINTER(t)),shape=(n,))
   rows=array(pointers[0],C.c_int32,1);past=array(pointers[1],C.c_int32,1);total=array(pointers[2],C.c_int32,1);offset=array(pointers[3],C.c_int32,2);ids=array(pointers[4],C.c_int64,tokens);positions=array(pointers[5],C.c_int64,tokens)
   assert rows.tolist()==[1]and total.tolist()==[int(past[0])+tokens]and offset.tolist()==[0,tokens]and positions.tolist()==list(range(int(past[0]),int(total[0])))
   assert kind in [0,1]and 1<=tokens<=(512 if kind==0 else 1)and int(total[0])<=32768
   self.forward_calls.append(dict(kind=kind,past=int(past[0]),tokens=tokens,input_ID_SHA256=hashlib.sha256(json.dumps(ids.tolist(),separators=(',',':')).encode()).hexdigest()))
   index=(self.reset_count-1)//2;seq=self.outputs[['retained8635','near32k'][index]]
   if kind==1:self.cursor+=1
   if self.node==3:
    C.memset(pointers[6],0,163840*2);array(pointers[7],C.c_int64,1)[0]=seq[self.cursor]
    if self.mode=='bad_numeric':array(pointers[6],C.c_uint16,1)[0]=0x7fc0
   array(pointers[8],C.c_int32,1)[0]=3 if self.node==3 else 0
   return 0
  except BaseException as e:self.error=repr(e);return 1
 def dispatch(self,name,*a):
  dumps=lambda q:json.dumps(q).encode()
  if name=='modern_kimi_engine_error_r2475':return self.error.encode()
  if name=='modern_kimi_engine_partition_r2475':stage,pp,out=a;assert pp==4;out[:]=CUTS[stage];return 0
  if name=='modern_kimi_engine_layout_r2475':return dumps(layout(self.node))
  if name=='modern_kimi_engine_cache_geometry_r2475':return dumps(GEOMETRY)
  if name=='modern_kimi_ordinary_capabilities_r410':return dumps(dict(CAP,TP=16)if self.mode=='bad_capability'else CAP)
  if name=='modern_kimi_engine_keys_r2475':return dumps(self.keys[1:]if self.mode=='bad_keys'else self.keys)
  if name=='modern_kimi_engine_load_r2475':
   for row in json.loads(a[1]):assert row['pointer']in Tensor.registry and list(Tensor.registry[row['pointer']].shape)==row['shape']
   self.load_calls+=1;return 1 if self.mode=='bad_load'else 0
  if name=='modern_kimi_engine_finish_r2475':self.loaded=True;return 0
  if name=='modern_kimi_engine_reset_r2475':self.reset_count+=1;self.cursor=0;return 0
  if name=='modern_kimi_engine_destroy_r2475':self.destroyed+=1;return 0
  if name=='modern_kimi_engine_output_metrics_r2475':return dumps(dict(engine_calls=len(self.forward_calls),recording_only=True))
  if name=='modern_kimi_shard_head_expected_r9183':
   q=json.loads(a[0]);assert len(q['ranks'])==8 and[q['row_begin']for q in q['ranks']]==[i*20480 for i in range(8)];self.head_expected=q;return 0
  if name=='modern_kimi_shard_head_error_r9183':return b'recording expected rows error'
  if name=='modern_kimi_shard_head_info_r9183':
   owners=[]
   for device in range(8):
    o=dict(device=device,model_constructed=True,model_RTTI='KimiK3ForConditionalGenerationShardR9183',TP=16 if self.mode=='bad_head'else 8,tp_rank=device,PP=4,PP_stage=self.node,head_expected=self.node==3)
    if self.node==3:
     digests=self.head_expected['ranks'][device]['digests']if self.head_expected else[0,0,0]
     o.update(head_constructed=True,head_RTTI='ShardOnlyVocabHeadR9183',head_is_ReplicatedLinear=False,full_GPU_head_retained=False,physical_head_shape=[20480,7168],head_parameter_bytes=293601280,global_row_begin=device*20480,global_row_end_exclusive=(device+1)*20480,postload_rows_verified=self.loaded,CPU_expected_row_digests=digests,GPU_actual_row_digests=digests)
    owners.append(o)
   return dumps(dict(schema=1,head_kind='ShardOnlyVocabHeadR9183',legacy_full_shard_policy_switch_supported=False,owners=owners,expected_CPU_source_registered=self.head_expected is not None))
  if name=='modern_kimi_vocab_head_get_r8341':return-1
  if name=='modern_kimi_vocab_head_set_r8341':return 1
  if name=='modern_cpu_copy_stats_r2515':return 0
  if name=='modern_owner_affinity_report_r2392':return dumps(dict(owners=[dict(device=d,changes=1,verified_mask=','.join(str(d*4+j)for j in range(4)))for d in range(8)]))
  raise AssertionError(name)
def scenario(mode):
 pins=json.loads((D/'bindings.json').read_bytes());mapping=json.loads((Path(pins['model_view']['path'])/'model.safetensors.index.json').read_bytes())['weight_map'];outputs=json.loads((D/'CPU_recorded_outputs.json').read_bytes())['outputs']
 raw=[RecordingLibrary(n,mapping,outputs,mode)for n in range(4)];reports=[];errors=[]
 from stage_control_r2386 import StageControl
 with tempfile.TemporaryDirectory(prefix='Kimi410-CPU-main-')as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir();run='kimi410-cpu-'+mode.replace('_','-');cancel=root/'control'/(run+'.cancel')
  source_sha=sha(D/'SOURCE_MANIFEST.json');claim=root/'recording-claim.json';claim.write_text(json.dumps(dict(Model_GO=True,runtime_source_manifest_SHA256=source_sha,target_cards=32,draft_cards=0,Native_seconds=120)))
  with socket.socket()as s:s.bind(('127.0.0.1',0));controlport=s.getsockname()[1]
  port=controlport-7000;assert 1024<=port<=58535
  class Control(StageControl):
   def __init__(self,*a):super().__init__(*a[:-1],timeout_seconds=4)
   def collect(self,label,payload):
    rows=super().collect(label,payload)
    if mode=='cancel'and label=='r410-weights-ready':cancel.touch()
    return rows
  started_epoch=time.time()
  def worker(node):
   env=dict(os.environ,KIMI410_ORDINARY_RUNTIME='1',SLURM_PROCID=str(node),SLURM_JOB_ID='CPU-recording',SLURM_STEP_ID='none',KIMI410_RUN=run,KIMI410_NONCE='cpu-only-recording-session-r410-0123456789',KIMI_ENGINE_HEAD='127.0.0.1',GLOBAL_PORT_R2354=str(port),KIMI410_GPU_START_UNIX=str(started_epoch),KIMI410_NATIVE_SECONDS='120',KIMI410_CANCEL_FILE=str(cancel),KIMI410_RUNTIME_SOURCE_SHA256=source_sha,KIMI410_PUBLISHED_MODEL_CLAIM=str(claim),KIMI410_PUBLISHED_MODEL_CLAIM_SHA256=sha(claim))
   env['KIMI410_NATIVE_DEADLINE_UNIX']=str(float(env['KIMI410_GPU_START_UNIX'])+120)
   def recording_bind(library,gate):return GuardedLibrary(declare(raw[node]),gate),dict(CPU_recording_CABI_leaf=True,no_dlopen=True)
   providers=dict(hostname=lambda:pins['nodes'][node],torch=FakeTorch,safe_open=lambda:lambda *a,**kw:Shard(),control=Control,bind=recording_bind,maps=lambda q:dict(CPU_recording_proc_maps_leaf=True),install_signal_handler=lambda f:None)
   try:bootstrap.main(env,providers,root)
   except BaseException as e:errors.append(dict(node=node,type=type(e).__name__,error=str(e)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(worker,range(4)))
  for node in range(4):
   p=root/'results'/('Kimi-ordinary-r410-'+run+'-node'+str(node)+'.json');assert p.exists(),(mode,node,errors);reports.append(json.loads(p.read_bytes()))
  if mode=='normal':
   assert not errors,errors
   assert all(q['completed']and q['all8_owned_workers_shutdown_completed']and q['all93_stage_keys']['all93_layer_ids_exactly_once_across_PPstages']for q in reports)
   assert all(len(q['ordinary_report']['requests'])==2 and all(r['quality']['passed']and r['finish']=='eos'for r in q['ordinary_report']['requests'])for q in reports)
   assert all(len(r.forward_calls)==197 and r.reset_count==4 for r in raw),[len(r.forward_calls)for r in raw]
  else:
   assert len(errors)==4 and all(not q['completed']and q['all8_owned_workers_shutdown_completed']for q in reports),(mode,errors)
   if mode=='bad_numeric':assert all(q['ordinary_calls']and q['ordinary_calls'][0]['CAPI_return']==0 for q in reports)
  assert all(v.created==v.destroyed==1 for v in raw)
  assert all(q['sampler_CPU_cleanup_completed']and q['control_CPU_cleanup_completed']for q in reports)
  return dict(case=mode,passed=True,actualmain_nodes=4,CABI_recorded_calls=[len(v.forward_calls)for v in raw],requests=[len(q.get('ordinary_report',{}).get('requests',[]))for q in reports],all_contexts_destroyed=True,no_Native_dlopen=True,errors=errors)
def main():
 assert 'torch'not in sys.modules
 pins=json.loads((D/'bindings.json').read_bytes());oldenv={k:os.environ.get(k)for k in pins['required_environment']};os.environ.update(pins['required_environment'])
 cases=[];started=time.monotonic()
 try:
  bootstrap.verify_bundle(D,sha(D/'SOURCE_MANIFEST.json'))
  # Actual source imports all production helpers before recording leaves.
  with contextlib.redirect_stdout(io.StringIO()):
   for mode in ['normal','bad_capability','bad_head','bad_keys','bad_load','cancel','bad_numeric']:cases.append(scenario(mode))
  assert 'torch'not in sys.modules
  q=dict(completed=True,passed=True,CPU_only=True,GPU=0,Native_dlopen=False,Runtime_created=False,weights_loaded=False,Model_qualified=False,actual_source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),cases=cases,elapsed_s=time.monotonic()-started,scope='Actual main, RealDependencies.prepare, typed ABI forward via ctypes callback, original loader, CPU head row digest, real 4leader StageControl TCP, real input IDs/codec/parser, real output files/cancellation; Native/device/tensors/map leaves record only')
  print(json.dumps(q,allow_nan=False))
 finally:
  for k,v in oldenv.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
if __name__=='__main__':main()
