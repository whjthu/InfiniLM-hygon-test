"""Only new fixed-profile entry, FIFO and owner lifecycle CPU consumers.
Processor/HTTP/Native operations below are explicit recording leaves. The actual
prepared factory, provider, queue, worker.serve, generate and reset are consumed.
No model quality, GPU or fresh tokenizer qualification is claimed here.
"""
from pathlib import Path
import copy,ctypes as C,json,sys,threading,time,tempfile
from enum import Enum
from types import SimpleNamespace
import numpy as np
from public_deep_model import worker,cli
from public_deep_model.native_context import NativeContext
from public_deep_model.request_policy import RequestPolicy
from public_deep_model.fullpre_control import LEGACY,FULL
from public_deep_model.prepared_factory import prepare_engine
from public_deep_model.profile import NAME,reject_request_control
class Fn:
 def __init__(self,f):self.f=f;self.argtypes=None;self.restype=None
 def __call__(self,*a):return self.f(*a)

class Library:
 def __init__(self,context):
  self.context=context;self.mode=0;self.old_mode=0;self.events=[];self.reset_count=0;self.pointer_values=[];self.callbacks=[]
  self.old=[dict.fromkeys(LEGACY,0)for _ in range(8)]
  for row in self.old:row.update(created=86,ready=86)
  self.full=[dict.fromkeys(FULL,0)for _ in range(8)]
  def callback(name,restype,args,fun):
   cb=C.CFUNCTYPE(restype,*args)(fun);cb.argtypes=args;cb.restype=restype;self.callbacks.append(cb);setattr(self,name,cb)
  callback('deep_hc_fullpre_enable_r377',C.c_int,[C.c_int],self.set)
  callback('deep_hc_fullpre_enabled_r377',C.c_int,[],lambda:self.mode)
  callback('deep_hc_fullpre_clear_r377',C.c_int,[],self.clear)
  callback('deep_hc_fullpre_stats_r377',C.c_int,[C.POINTER(C.c_uint64),C.c_int],lambda p,n:self.read(p,n,self.full,FULL))
  callback('deep_hc_model_epilogue_set_r366',C.c_int,[C.c_int],self.oldset)
  callback('deep_hc_model_epilogue_get_r366',C.c_int,[],lambda:self.old_mode)
  callback('deep_hc_model_epilogue_counters_r366',C.c_int,[C.POINTER(C.c_uint64),C.c_int],lambda p,n:self.read(p,n,self.old,LEGACY))
  callback('modern_native_engine_reset',C.c_int,[C.c_void_p],self.reset)
  callback('modern_native_engine_destroy',C.c_int,[C.c_void_p],lambda ctx:0)
  self.modern_native_engine_forward=Fn(lambda *args:0)
  self.modern_native_engine_forward.argtypes=[C.c_void_p];self.modern_native_engine_forward.restype=C.c_int
 def set(self,mode):
  if mode not in[0,1]or not self.context.drained:return 9
  self.mode=mode;self.events.append(('mode',mode,self.reset_count));return 0
 def oldset(self,mode):
  if mode!=0 or not self.context.drained:return 9
  self.old_mode=mode;self.events.append(('r366',mode,self.reset_count));return 0
 def clear(self):
  if self.mode or not self.context.drained:return 9
  self.full=[dict.fromkeys(FULL,0)for _ in range(8)];self.events.append(('clear',self.reset_count));return 0
 def read(self,p,n,rows,names):
  if n!=8*len(names)or not self.context.drained:return 9
  self.pointer_values.append(C.cast(p,C.c_void_p).value)
  for d,row in enumerate(rows):
   for k,name in enumerate(names):p[d*len(names)+k]=row[name]
  if self.context.scenario=='missing-hit'and names is FULL and self.mode==1:p[3]=0
  return 0
 def reset(self,ctx):
  self.events.append(('reset',ctx));self.reset_count+=1
  if ctx!=self.context.ctx or ctx<=2**32:return 9
  if self.context.scenario=='reset-drain-failure'and self.reset_count==3:return 9
  self.context.drained=True;self.context.current_forward=0;return 0
 def forward(self,part):
  self.context.current_forward+=1;self.context.drained=False;one=len(part)==1
  self.events.append(('forward',len(part),self.mode,self.old_mode))
  for old,full in zip(self.old,self.full):
   old['pre']+=86;old['eligible_bf16_t1']+=86*one;old['other_t']+=86*(not one)
   if self.mode:
    full['observed']+=86;full['eligible']+=86*one;full['selected']+=86*one;full['completed']+=86*one;full['prefill']+=86*(not one)
  self.context.drained=True
  if self.context.scenario=='forward-failure':return 7,np.zeros((1,128),dtype=np.uint16),[0]
  token=1 if self.context.current_forward==self.context.goal else 9
  return 0,np.zeros((1,128),dtype=np.uint16),[token]

class Context(NativeContext):
 def __init__(self,scenario):
  self.scenario=scenario;self.cfg={'num_hidden_layers':43,'vocab_size':128};self.thread=threading.get_ident();self.closed=False;self.ctx=0x1234567800000001;self.current_forward=0;self.calls=0;self.drained=True
  lib=Library(self)
  def check(code):
   if code:raise RuntimeError('recording Native code '+str(code))
  self.binding=SimpleNamespace(lib=lib,check=check);self.forward=self.invoke
 def floor(self):self.check_thread();return [[20*1024**3,64*1024**3]for _ in range(8)]
 def invoke(self,part,*args):self.check_thread();self.calls+=1;return self.binding.lib.forward(part)


class Reasons(str,Enum):
 LENGTH='length';EOS_TOKEN='eos_token';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
class Processor:
 # Recording tokenization is intentionally not a new Deep codec test.
 def get_tokenizer(self):return self
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
  return ''.join('<'+m['role']+'>'+m.get('content','')for m in messages)+('<｜Assistant｜></think>'if add_generation_prompt else'')
 def __call__(self,prompt,return_tensors=None):return {'input_ids':[[ord(c)%100+2 for c in prompt]]}
 def decode(self,ids,**kwargs):return ''.join('x'for _ in ids)
class Run:
 def deadline_check(self,*args):pass
 def stop_requested(self):return False
def rejected(f):
 try:f()
 except (AssertionError,ValueError,RuntimeError):return
 raise AssertionError('Invalid control accepted')
def fixed_case(profile,scenario='normal'):
 context=Context(scenario);context.goal=0
 config=dict(TP=8,PP=1,rows=1,context=32768,HC_mode=0,gpu_start_epoch=time.time(),service_seconds=2400,experimental_profile=profile)
 policy=RequestPolicy(context,config,provider_validator=lambda lib:dict(recording_CABI=True,actual_DSO_or_GPU=False))
 initial=policy.ready();assert initial['fixed_mode']==(1 if profile else 0)
 with tempfile.TemporaryDirectory()as tmp:
  path=Path(tmp);(path/'generation_config.json').write_text('{"eos_token_id":1}')
  prepared=prepare_engine('Deep',str(path),processor=Processor(),finish_reasons=Reasons)
  engine=prepared['engine'];requests=[]
  lengths=[1,9,100,1024,1025,2049,4097]if scenario=='normal'else[1024-len(Processor().apply_chat_template([dict(role='user',content='')],add_generation_prompt=True))]if scenario=='early-eos-no-T1'else[20]
  for i,length in enumerate(lengths):
   body=dict(messages=[dict(role='user',content='x'*length)],_agent_schemas={},_agent_choice='auto',stream=True,max_tokens=16+i)
   params=SimpleNamespace(max_tokens=16+i,temperature=0.,stop=None)
   engine.validate_data(body,params)
   req=engine.add_chat_request(messages=body['messages'],sampling_params=params,request_id='r'+str(i),request_data=body,chat_template_kwargs={})
   requests.append(req)
  take=engine.take
  def take_goal(timeout):
   req=take(timeout)
   if req is not None:context.goal=(len(req.prompt_token_ids)+1023)//1024+(0 if scenario=='early-eos-no-T1'else 2)
   return req
  engine.take=take_goal;engine.stop() # Sentinel follows all seven real FIFO publications.
  events=[]
  for req in requests:
   emit=req.emit_token;finish=req.finish
   def wrapped_emit(token,req=req,emit=emit):
    emit(token)
    if scenario in['disconnect','timeout']and len(req.generated_token_ids)==2:req.mark_abort(Reasons.TIMEOUT if scenario=='timeout'else Reasons.CANCELED)
   def wrapped_finish(reason,req=req,finish=finish):
    assert context.drained and context.binding.lib.mode==initial['fixed_mode'];events.append(req.request_id);finish(reason)
   req.emit_token=wrapped_emit;req.finish=wrapped_finish
  report={'requests':[]};failure=None
  try:worker.serve(context,prepared,config,Run(),threading.Event(),report,lambda:None,policy=policy)
  except BaseException as error:failure=error
  if scenario in['missing-hit','forward-failure','reset-drain-failure']:
   assert failure is not None,(scenario,repr(failure))
   if scenario!='missing-hit':assert policy.poisoned
  else:
   assert failure is None,repr(failure)
   assert events==[r.request_id for r in requests]
   assert all(row['r454_mode']==initial['fixed_mode']and row['r454_actual_T1_coverage']['passed']and row['normal_reset_before']and row['normal_reset_after']for row in report['requests'])
   assert all(r.is_finished()for r in requests)
   if scenario=='normal':assert len(report['requests'])==7 and len({r['max_new_tokens']for r in report['requests']})==7
   elif scenario=='early-eos-no-T1':assert requests[0].generated_token_ids==[1]and report['requests'][0]['r454_actual_T1_coverage']['T1_HC_per_owner']==0
   else:assert requests[0].finish_reason==(Reasons.TIMEOUT if scenario=='timeout'else Reasons.CANCELED)
  policy.close();assert policy.closed
  if not policy.poisoned:assert context.binding.lib.mode==0
  return dict(profile=profile,scenario=scenario,passed=True,requests=len(report['requests']),recording_Native_forwards=context.calls,actual_prepared_factory_FIFO_worker_generate_reset=True,more_than_six_requests=scenario=='normal',actual_Native_or_GPU=False)
def factory_and_ready():
 from public_deep_model.provider import DeepProvider
 from public_deep_model.actors import Actors
 from public_deep_model.protocol import Run as ActualRun
 # Actual build_server creates the real provider and prepared engine. Only
 # public HTTP construction is recorded; the stable public route is not replayed.
 class Server:
  def __init__(self,**settings):
   defaults=dict(attn_backend='default',use_mla=False,moe_ep_backend='disabled',moe_ep_size=1,weight_load_mode='async',kv_transfer_config=None,pre_transpose=False)
   args={k:settings[k]for k in ['model_path','device','dtype','tensor_parallel_size','pipeline_parallel_size','pipeline_parallel_stage','cache_type','max_batch_size','max_cache_len','max_tokens','temperature','top_p','top_k','enable_graph','skip_load','enable_prefix_caching']};args.update(defaults)
   self.engine=settings['engine_factory'](**args)
  def _build_sampling_params(self,data):return SimpleNamespace(max_tokens=data.get('max_tokens',4096),temperature=0.,stop=None,ignore_eos=False)
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'generation_config.json').write_text('{"eos_token_id":1}')
  config=json.loads((Path(__file__).parent/'config.template.json').read_bytes());config.update(model_path=str(root),runtime_directory=str(root/'runtime'),run='recording',actual_step='CPU.recording',gpu_start_epoch=time.time(),source_manifest_SHA256='recording-source',provider_config_SHA256='recording-config',experimental_profile=NAME)
  server,provider,prepared=cli.build_server(config,Processor(),Reasons,Server)
  assert server.engine is provider and provider._used
  # Same pre-admission hook called by the actual public sampling entry; cannot
  # select a mode, change C1 or bypass full-context bounds through request data.
  base=dict(messages=[dict(role='user',content='short')],_agent_schemas={},_agent_choice='auto',stream=True,max_tokens=32)
  server._build_sampling_params(base)
  for field in ['experimental_profile','fullpre_mode','HC_mode','r366_fixed','rows','context','TP','r449_mode','r450_mode','r454_mode']:
   rejected(lambda field=field:server._build_sampling_params(dict(base,**{field:1})))
  rejected(lambda:server._build_sampling_params(dict(base,sampling_params={'experimental_profile':NAME})))
  rejected(lambda:server._build_sampling_params(dict(base,max_tokens=0)))
  rejected(lambda:server._build_sampling_params(dict(base,messages=[dict(role='user',content='x'*32768)])))
  rejected(lambda:server._build_sampling_params(dict(base,n=2)))
  rejected(lambda:server._build_sampling_params(dict(base,text_delivery='bad')))
  # Actual ready reader, not a boolean shortcut, verifies mode/identity/owner
  # topology/floor. Native allocation and affinity rows are recording leaves.
  group=provider.group
  context=Context('normal');context.goal=0
  policy=RequestPolicy(context,config,provider_validator=lambda lib:dict(recording_CABI=True,actual_DSO_or_GPU=False))
  fixed=policy.ready();owners=[dict(device=d,changes=1,verified_mask=','.join(map(str,range(4*d,4*d+4))))for d in range(8)]
  row=dict(loaded=True,loaded_key_count=67569,Native_SHA256=config['Native_SHA256'],HC_mode=0,rows=1,capacity=32768,affinity=dict(actual_owner_report=dict(owners=owners)),postload_free=[[20*1024**3,64*1024**3]for _ in range(8)],fixed_profile=fixed)
  group.run.write('ready-engine.json',row);actual=group.wait_ready();assert actual['fixed_profile']==fixed
  bad=dict(config,experimental_profile=None);other=Actors(bad,prepared)
  rejected(other.wait_ready)
  class RecordedGroup:
   def __init__(self):self.run=group.run;self._claimed=False;self.failed=threading.Event();self.failure=None
   def start(self):self._claimed=True
   def wait_ready(self):return group.wait_ready()
   def stop(self):pass
  provider.group=RecordedGroup();provider.start();ready=group.run.read('MODEL_READY.json');assert ready['fixed_profile']['fixed_mode']==1 and ready['Native_SHA256']==config['Native_SHA256'];provider.stop();policy.close()
  return dict(passed=True,actual_build_server_provider_factory=True,actual_Actors_ready_reader=True,actual_provider_start_READY_publication=True,fixed_mode=1,foreign_request_controls_rejected=True,C1_context_and_budget_guards_retained=True,HTTP_and_Native_ready_rows_recording_only=True,Model_READY_proved=False)
def main():
 results=[]
 for profile in[None,NAME]:results.append(fixed_case(profile))
 for scenario in['early-eos-no-T1','disconnect','timeout','missing-hit','forward-failure','reset-drain-failure']:results.append(fixed_case(NAME,scenario))
 results.append(factory_and_ready())
 rejected(lambda:RequestPolicy(Context('normal'),dict(TP=8,PP=1,rows=2,context=32768,HC_mode=0),provider_validator=lambda lib:{}))
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 print(json.dumps(dict(passed=True,cases=results,actual_Native_or_GPU=False,recording_processor_HTTP_Native_leaves=True,old_encoding_or_938_suite_replayed=False)))
if __name__=='__main__':main()
