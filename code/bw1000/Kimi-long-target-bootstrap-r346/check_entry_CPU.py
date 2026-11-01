"""Execute the real bootstrap state machine against recording Native/file providers."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,sys,tempfile,math
import copy
from lifecycle import Settings,Gate,GuardedLibrary,guarded_safe_open,connect_draft,publish_once
from bootstrap import execute

D=Path(__file__).resolve().parent
FRONT=Path('/refactor/incoming/Kimi-append512-service-r252/runtime')
if not FRONT.exists():FRONT=D.parents[2]/'local-only/Kimi-append512-service-r252/runtime'
sys.path.insert(0,str(FRONT))
import parameter_loader_r9143 as loader
import weight_key_guard_r9156 as keys_guard

def environment(root,node=0):
 return dict(KIMI_LONG_PAIR_R346='1',INFINILM_KIMI_LONG_FEATURES_R305='1',KIMI_LONG_RUN_R347='cpu-entry-r346',GLOBAL_RUN_R2354='cpu-entry-r346',KIMI_LONG_SESSION_R347='CPU-recording-session-identity-00000000',KIMI_LONG_GPU_START_UNIX_R347='1000.0',KIMI_LONG_NATIVE_DEADLINE_UNIX_R347='29800.0',KIMI_LONG_CANCEL_FILE_R347=str(root/'control/cpu-entry-r346-cancel'),SLURM_PROCID=str(node),KIMI_ENGINE_HEAD='127.0.0.1',GLOBAL_PORT_R2354='32947',KIMI_LONG_CAPTURE_PORT_R347='45947',KIMI_LONG_DRAFT_PORT_R347='46947',KIMI_LONG_DRAFT_HOST_R347='127.0.0.1')

class FaultGate(Gate):
 def __init__(self,s,label=None,cancel=False):
  self.now=10.;self.fault_label=label;self.cancel_fault=cancel;super().__init__(s,clock=lambda:self.now,wall=lambda:1000.)
 def check(self,label):
  if label==self.fault_label:
   if self.cancel_fault:self.signal_cancel=True
   else:self.now=self.deadline
  super().check(label)

class Function:
 def __init__(self,call):self.call=call
 def __call__(self,*a):return self.call(*a)

class FakeBits:
 def numpy(self):return self
 def __getitem__(self,row):return[0]*7168

class Tensor:
 def __init__(self,name):self.name=name;self.dtype='f32'if name.endswith('A_log')else'bf16';self.device=SimpleNamespace(type='cpu');self.shape=[128]if name.endswith('A_log')else[163840,7168]if name.endswith(('lm_head.weight','embed_tokens.weight'))else[1]
 def contiguous(self):return self
 def is_contiguous(self):return True
 def numel(self):return math.prod(self.shape)
 def element_size(self):return 4 if self.dtype=='f32'else 2
 def data_ptr(self):return 123456
 def view(self,dtype):return FakeBits()
 def __getitem__(self,index):self.shape=[min(index.stop,self.shape[0])];return self

class Shard:
 def __enter__(self):return self
 def __exit__(self,*_):pass
 def get_tensor(self,name):return Tensor(name)

class Control:
 def __init__(self,owner):self.owner=owner;self.closed=False
 def collect(self,label,value):
  if label=='r346-keys-before-load':return[dict(node=n,keys=keys_guard.summarize_keys(self.owner.stage_keys(n//2),self.owner.weight_map,n//2,[[0,46],[46,93]]))for n in range(4)]
  return[dict(value,node=n)if'node'in value else value.copy()for n in range(4)]
 def close(self):self.closed=True

class Dependencies:
 def __init__(self,root,scenario='ok'):
  self.root=root;self.scenario=scenario;self.control=None;self.sampler=None;self.lib=None;self.created=self.destroyed=self.opened=self.loads=self.pair_calls=0;self.capture_closed=self.sampler_closed=False
  self.weight_map={f'language_model.model.layers.{n}.self_attn.A_log':'recording.safetensors'for n in range(93)}
  self.weight_map.update({'language_model.model.embed_tokens.weight':'recording.safetensors','language_model.lm_head.weight':'recording.safetensors'})
  self.model=root/'recording-model';self.model.mkdir();(self.model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':self.weight_map}))
  self.m={'static_memory':{'PP_ranges':[[0,46],[46,93]]}}
 def stage_keys(self,stage):return{k for k in self.weight_map if k.startswith(tuple(f'language_model.model.layers.{n}.'for n in range(*[[0,46],[46,93]][stage])))}|({'language_model.model.embed_tokens.weight'}if stage==0 else{'language_model.lm_head.weight'})
 def prepare(self,s,gate):
  self.settings=s;self.gate=gate;self.control=Control(self);self.node=s.node
  def function(name):
   def call(*a):
    if name=='modern_kimi_engine_partition_r2475':a[2][0],a[2][1]=[[0,46],[46,93]][a[0]];return 0
    if name=='modern_kimi_engine_create_long_agent_r179':self.created+=1;return 0 if self.scenario=='create-fail'else 123
    if name=='modern_kimi_engine_error_r2475':return b'CPU recording Native error'
    if name=='modern_kimi_engine_destroy_r2475':self.destroyed+=1;return 1 if self.scenario=='destroy-fail'else 0
    if name=='modern_kimi_engine_layout_r2475':return json.dumps(dict(node=s.node,local_workers=8,TP=16,TP_node_rank=s.node%2,PP=2,PP_stage=s.node//2,sampling_node=2,public_graph_enabled=False,owned_metadata_enabled=False)).encode()
    if name=='modern_kimi_engine_cache_geometry_r2475':return json.dumps(dict(max_rows=2,pages=260,page_size=256,KDA_pool_rows=3,pages_per_request=128,max_total_tokens=32768)).encode()
    if name=='modern_kimi_batch_capabilities_r9122':return json.dumps(dict(logical_capacity=1,implemented_max_flatten_tokens=64,pool_rows=3,flags=3)).encode()
    if name=='modern_kimi_long_agent_capabilities_r179':return json.dumps(dict(scoped_long_agent=True,max_total_tokens=32768,state_pool_rows=3)).encode()
    if name=='modern_kimi_engine_keys_r2475':return json.dumps(sorted(self.stage_keys(s.node//2))).encode()
    if name=='modern_cpu_copy_stats_r2515':
     for j in range(5):a[0][j]=self.loads
     return 0
    if name=='modern_kimi_engine_load_r2475':self.loads+=1;assert json.loads(a[1]);return 1 if self.scenario=='load-fail'else 0
    if name=='modern_kimi_engine_output_metrics_r2475':return b'{}'
    return 0
   return Function(call)
  class Raw:
   def __getattr__(raw,name):v=function(name);setattr(raw,name,v);return v
  self.lib=GuardedLibrary(Raw(),gate)
  def open_recording(*a,**kw):self.opened+=1;return Shard()
  self.safe_open=guarded_safe_open(open_recording,gate)
  return {'CPU_recording_providers':True}
 def check(self,value):
  if value:raise RuntimeError('Recording Native status failure')
 def memory(self):return[[59<<30,64<<30]for _ in range(8)]
 def memory_guard(self,label,report,save):return{'CPU_recording_floor':True,'minimum_free_bytes':59<<30}
 def read_config(self):return{'text_config':dict(num_hidden_layers=93,hidden_size=7168,vocab_size=163840,linear_attn_config=dict(num_heads=96,head_dim=128),kv_lora_rank=512,qk_rope_head_dim=64)}
 def mapping(self):return self.weight_map
 def read_head(self,node,loaded):return{'CPU_recording_head':True,'loaded':loaded,'node':node}
 def keys(self,*args):return keys_guard.summarize_keys(*args)
 def all_keys(self,*args):return keys_guard.validate_all_stages(*args)
 def load(self,ctx,text,keys,progress):
  torch=SimpleNamespace(float32='f32',bfloat16='bf16',uint8='u8',uint16='u16')
  return loader.load_parameters(self.lib,ctx,self.model,text,keys,self.safe_open,torch,self.check,progress)
 def run_pair(self,g,s,gate,save):
  gate.check('recording-driver-start');self.pair_calls+=1
  if self.scenario=='pair-fail':raise RuntimeError('Recording driver failure')
  return dict(completed=True,normal_draft_protocol_close=True,requests=[dict(completed=True,normal_reset=True,quality={'passed':True,'CPU_recording_only':True},finish='eos')for _ in range(4)],CPU_substitute_not_actual_quality=True)
 def affinity(self):return{'CPU_recording_affinity':True}
 def close_capture(self):self.capture_closed=True
 def close_sampler(self):self.sampler_closed=True

def case(node=0,scenario='ok',fault=None,cancel=False):
 with tempfile.TemporaryDirectory()as temporary:
  root=Path(temporary);(root/'control').mkdir();s=Settings(environment(root,node),root);gate=FaultGate(s,fault,cancel);deps=Dependencies(root,scenario);report={};saved=[];error=None
  try:execute(s,gate,deps,report,lambda:saved.append(json.loads(json.dumps(report))))
  except BaseException as e:error=e
  assert deps.capture_closed and deps.sampler_closed
  if deps.control is not None:assert deps.control.closed
  return deps,report,error

def markers():
 results=[]
 for scenario in ['ok','wrong-identity','cancel','timeout','connect-fail','hello-fail']:
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);(root/'control').mkdir();s=Settings(environment(root),root);gate=FaultGate(s);connections=[];closed=[]
   class Ctl:
    def collect(self,label,value):return[dict(value,node=n)for n in range(4)]
   class Sock:
    def settimeout(self,value):assert 0<value<=7200
    def close(self):closed.append(True)
   class Client:
    def __init__(self,connection,nonce):
     assert nonce==s.nonce
     if scenario=='hello-fail':raise ValueError('Recording hello failure')
    def abort(self):closed.append(True)
   def sleep(seconds):
    assert 0<seconds<=.1
    if scenario=='cancel':gate.signal_cancel=True
    elif scenario=='timeout':gate.now=gate.deadline
    elif not s.listening_file.exists():
     value=dict(s.identity,actual_draft_step='910371.900',listener_ready=True)
     if scenario=='wrong-identity':value['nonce_SHA256']='0'*64
     publish_once(s.listening_file,value)
   def connect(*args,**kwargs):
    connections.append(True)
    if scenario=='connect-fail':raise ConnectionRefusedError('Recording sole connect failure')
    return Sock()
   error=None
   try:client=connect_draft(s,gate,Ctl(),Client,lambda value:None,sleep=sleep,connect=connect)
   except RuntimeError as e:error=e
   assert s.start_file.read_bytes().endswith(b'\n')and s.nonce not in s.start_file.read_text()
   if scenario=='ok':
    assert error is None and len(connections)==1
    try:connect_draft(s,gate,Ctl(),Client,lambda value:None,sleep=sleep,connect=connect)
    except RuntimeError:pass
    else:raise AssertionError('Duplicate start marker accepted')
    assert len(connections)==1
   else:
    assert error is not None and len(connections)==int(scenario in ['connect-fail','hello-fail'])
    if scenario=='hello-fail':assert closed==[True]
   results.append(scenario)
 return results

def run():
 checks=[]
 for node in range(4):
  d,q,e=case(node);assert e is None and q['completed']and d.created==d.destroyed==1 and d.loads>0 and d.pair_calls==1 and q['all8_owned_workers_shutdown_completed']
 checks.append('actual bootstrap entry four leader roles; real qualified parameter_loader and 93layer key validation with recording tensors/CAPI; no weights or Native Context')
 for label in ['entry-before-imports','before-create','safe_open','Native:modern_kimi_engine_load_r2475','before-start-forward']:
  d,q,e=case(fault=label);assert isinstance(e,TimeoutError)and not q['completed'];assert d.destroyed==int(d.created>0)
  if label=='safe_open':assert d.opened==0
  if label=='Native:modern_kimi_engine_load_r2475':assert d.loads==0
 checks.append('original deadline before imports/create/safe_open/Native load/first forward; cleanup remains permitted')
 for label in ['entry-before-imports','safe_open','Native:modern_kimi_engine_load_r2475','before-start-forward']:
  d,q,e=case(fault=label,cancel=True);assert isinstance(e,RuntimeError)and not q['completed'];assert d.destroyed==int(d.created>0)
 checks.append('original cancellation at entry/load/forward boundaries never consumes subsequent Native work')
 for scenario in ['create-fail','load-fail','pair-fail','destroy-fail']:
  d,q,e=case(scenario=scenario);assert e is not None and not q['completed'];assert d.destroyed==(0 if scenario=='create-fail'else 1)
 checks.append('create/load/driver/destroy failure executes actual outer cleanup; failed destroy never claims all8shutdown')
 marker_checks=markers();checks.append('exclusive newline marker, exact identity, original100ms deadline/cancel, single connect/hello and no retries')
 from first_capture import validate
 expected={0:[19,37],1:[],2:[66,78,90],3:[]};packets=[dict(node=n,epoch=7,layerIDs=expected[n],rows=512,local_bytes=len(expected[n])*512*7168*2,finite_checked_by_immutable_harvest=True)for n in range(4)];assert validate(packets,7)['passed']
 for key,value in [('epoch',6),('rows',511),('layerIDs',[66,78]),('finite_checked_by_immutable_harvest',False),('local_bytes',0)]:
  bad=copy.deepcopy(packets);bad[2][key]=value
  try:validate(bad,7)
  except AssertionError:pass
  else:raise AssertionError('Invalid first capture metadata accepted')
 checks.append('first actualcapture metadata gate rejects epoch/rows/layer/finite/byte mismatch before any proposal')
 prior=json.loads((D/'prior_CPU337.json').read_bytes());assert prior['passed'] and not prior['GPU'] and not prior['Native_dlopen'] and prior['source_manifest_SHA256']=='445f96b37c1ceab9e62c3599879834b8be21da9b9585384c942c12ffef86644f'
 driver={'reused_prior_Linux_CPU9':True,'source_manifest_SHA256':prior['source_manifest_SHA256'],'GPU':False}

 return dict(passed=True,checks=checks,marker_checks=marker_checks,original337_checks=driver,actual_bootstrap_state_machine=True,recording_providers=True,GPU=0,Native_dlopen=False,actual_Native_Context=False,checkpoint_payload_read=False,performance_gain='UNKNOWN')

if __name__=='__main__':print(json.dumps(run(),indent=2))
