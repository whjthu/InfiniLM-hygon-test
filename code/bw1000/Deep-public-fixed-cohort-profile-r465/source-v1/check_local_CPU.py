"""Actual new producer/ownerloop/clientmain, recording processor/Native/HTTP leaves."""
from pathlib import Path
import copy,json,os,tempfile,threading,time,types,sys
from enum import Enum
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'public-Deep-native-provider-r427/source-v3'))
from public_deep_cohort.prepared_factory import prepare_engine
from public_deep_cohort.native_context import NativeContext
from public_deep_cohort.profile import NAME,validate
from public_deep_cohort import worker,cli
import client
class Reason(str,Enum):
 EOS_TOKEN='eos_token';LENGTH='length';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
class Processor:
 def get_tokenizer(self):return self
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kw):return ''.join('<'+m['role']+'>'+m.get('content','')for m in messages)+('<｜Assistant｜></think>'if add_generation_prompt else'')
 def __call__(self,prompt,return_tensors=None):return {'input_ids':[[ord(c)%100+2 for c in prompt]]}
 def decode(self,ids,**kw):return 'x'*len(ids)
class Fn:
 def __call__(self):return 0
class Context(NativeContext):
 def __init__(self,scenario='normal'):
  self.thread=threading.get_ident();self.closed=False;self.config=dict(profile=NAME,rows=4,Native_SHA256='recording');self.cfg=dict(vocab_size=128);self.counts=[0]*4;self.calls=[];self.resets=0;self.scenario=scenario
  lib=types.SimpleNamespace(**{n:Fn()for n in['deep_hc_canonical_get_r7130','deep_hc_model_epilogue_get_r366','deep_hc_fullpre_enabled_r377','modern_deep_c4_tile_get_r5920','modern_deep_layout_get_r7052','deep_moe_tail_get_r7211']})
  def check(rc):
   if rc:raise RuntimeError('recording Native error')
  self.binding=types.SimpleNamespace(lib=lib,check=check);self.forward=self.invoke;self.on_forward=None
 def floor(self):self.check_thread();return[[20*1024**3,68702699520]for _ in range(8)]
 def reset(self):self.check_thread();self.resets+=1;self.counts=[0]*4
 def invoke(self,ids,positions,rows,past,total,offsets):
  self.check_thread();self.calls.append(list(rows));out=[]
  for row in rows:self.counts[row]+=1;out.append(1 if self.counts[row]>=row+2 else 9)
  if self.on_forward:self.on_forward()
  return (7 if self.scenario=='failure'else 0),np.zeros((len(rows),128),dtype=np.uint16),np.array(out,dtype=np.int64)
class Run:
 def stop_requested(self):return False
 def remaining(self):return 100
 def deadline_check(self,*a,**kw):pass
def config():return dict(TP=8,PP=1,rows=4,context=32768,capacity_per_row=32768,HC_mode=0,r366=0,fullpre=0,profile=NAME,experimental_profile=NAME,cohort_assembly_seconds=.1,gpu_start_epoch=time.time(),service_seconds=60)
def body(i=0,mode='cohort',group='one'):
 return dict(messages=[dict(role='user',content='x'*10)],max_tokens=4096,stream=False,_agent_schemas={},_agent_choice='auto',chat_template_kwargs=dict(enable_thinking=False),native_fixed_cohort=dict(id=group,mode=mode,row=i,size=4))
def add(engine,b,rid):
 params=types.SimpleNamespace(max_tokens=4096,temperature=0.,stop=None);engine.validate_data(b,params)
 return engine.add_chat_request(messages=b['messages'],sampling_params=params,request_id=rid,request_data=b,chat_template_kwargs=b['chat_template_kwargs'])
def prepared(root):return prepare_engine('Deep',str(root),processor=Processor(),finish_reasons=Reason)
def rejected(call):
 try:call()
 except(AssertionError,ValueError,RuntimeError):return
 raise AssertionError('Invalid input accepted')
def boundaries(root):
 outcomes=[]
 for scenario in['normal','cancel','failure','incomplete','partial-stop']:
  p=prepared(root);e=p['engine'];c=Context(scenario);requests=[add(e,body(i),'r%d'%i)for i in range(1 if scenario in['incomplete','partial-stop']else 4)]
  for req in requests:
   finish=req.finish
   def finished(reason,req=req,finish=finish):assert c.resets>=2 or scenario in['incomplete','partial-stop'];finish(reason)
   req.finish=finished
  if scenario=='cancel':c.on_forward=lambda:requests[0].mark_abort(Reason.CANCELED)
  rejected(lambda:add(e,body(0),'duplicate'));rejected(lambda:add(e,body(0,group='foreign'),'foreign'))
  if scenario!='incomplete':e.stop()
  report=dict(requests=[],cohorts=[]);error=None
  try:worker.serve(c,p,config(),Run(),threading.Event(),report,lambda:None)
  except BaseException as ex:error=repr(ex)
  assert all(r.is_finished()for r in requests)
  if scenario=='normal':assert error is None and any(len(x)==4 for x in c.calls)and any(len(x)==1 for x in c.calls)and len(report['requests'])==4,(error,c.calls)
  elif scenario in['cancel','partial-stop']:assert error is None and all(r.finish_reason==Reason.CANCELED for r in requests)
  else:assert error is not None
  if scenario=='incomplete':assert not c.calls and c.resets==0
  outcomes.append(dict(scenario=scenario,actual_producer=True,actual_owner_loop=True,recording_Native=True,calls=c.calls,resets=c.resets,expected_error=error,passed=True))
 return outcomes
def client_case(root):
 p=prepared(root);e=p['engine'];holder={};done=threading.Event();failure=[]
 cfg=config();cfg['cohort_assembly_seconds']=2
 def owner():
  c=Context();holder['context']=c;report=holder['report']=dict(requests=[],cohorts=[])
  def save():
   q=dict(identity=identity,requests=copy.deepcopy(report['requests']));tmp=runtime/'owner.tmp';tmp.write_text(json.dumps(q));tmp.replace(runtime/'owner.json')
  try:worker.serve(c,p,cfg,Run(),done,report,save)
  except BaseException as err:failure.append(repr(err))
 run='cpu-recording';runtime=root/run;runtime.mkdir();identity=dict(run=run,actual_step='CPU.recording',source_manifest_SHA256='source',provider_config_SHA256='config')
 ready=dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,rows=4,capacity_per_row=32768,standard_public_C1=False,fixed_profile=dict(profile=NAME,fullpre=0,precision_policy=0,C1=False));(runtime/'MODEL_READY.json').write_text(json.dumps(ready))
 # This READY and HTTP response are recording leaves, explicitly not realModel.
 class HTTP(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   b=json.loads(self.rfile.read(int(self.headers['Content-Length'])));t=b['native_fixed_cohort'];rid=t['id']+'-r%d'%t['row'];req=add(e,b,rid)
   while not req.is_finished():time.sleep(.001)
   generated=req.generated_token_ids;reply=dict(id=rid,choices=[dict(message=dict(content='#### 0'),finish_reason='stop')],usage=dict(prompt_tokens=len(req.prompt_token_ids),completion_tokens=len(generated)))
   raw=json.dumps(reply).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
   with e._lock:e._requests.pop(rid)
 server=ThreadingHTTPServer(('127.0.0.1',0),HTTP);ot=threading.Thread(target=owner);ht=threading.Thread(target=server.serve_forever);ot.start();ht.start()
 records=[];proc=Processor()
 for i in range(32):
  b=body();b.pop('native_fixed_cohort');prompt=proc.apply_chat_template(b['messages'],add_generation_prompt=True);ids=proc(prompt)['input_ids'][0]
  import hashlib
  records.append(dict(case_id='recording-%d'%i,ordinal=i,body=b,body_SHA256='recording',prompt_token_ids=ids,prompt_IDs_SHA256=hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),gold=dict(suite='gsm8k',gold_numeric='1')))
 plan=root/'plan-private.json';plan.write_text(json.dumps(dict(records=records,HTTP=64)))
 try:q=client.main(['--private-plan',str(plan),'--base-url','http://127.0.0.1:%d'%server.server_port,'--output',str(root/'client'),'--runtime-dir',str(root),'--run',run,'--expected-step','CPU.recording','--source-sha','source','--config-sha','config','--deadline-unix',str(time.time()+60)])
 finally:e.stop();done.set();ot.join(timeout=3);server.shutdown();ht.join(timeout=3);server.server_close()
 assert not ot.is_alive()and not ht.is_alive()and not failure and q['execution_passed']and len(q['rows'])==64
 assert q['quality_summary']['serial']['correct']==q['quality_summary']['cohort']['correct']==0
 assert len(holder['report']['cohorts'])==16 and all(v['trial']['normal']for v in holder['report']['cohorts'])
 return dict(passed=True,actual_client_main=True,actual_prepared_factory=True,actual_owner_loop=True,local_HTTP_recording_leaf=True,recording_processor=True,actual_Native=False,HTTP=64,cohorts=16,all64_wrong_not_early_stop=True,normal_join=True)
def public_factory_case(root):
 from public_deep_cohort.actors import Actors
 from public_deep_cohort.provider import DeepProvider
 cfg=json.loads((Path(__file__).parent/'config.template.json').read_bytes());cfg.update(model_path=str(root),processor_path=str(root),runtime_directory=str(root),experimental_profile=NAME,run='factory-recording',actual_step='CPU.recording',source_manifest_SHA256='source',provider_config_SHA256='config',gpu_start_epoch=time.time())
 class Server:
  def __init__(self,**kw):
   self.kw=kw;self.engine=kw['engine_factory'](**dict(kw,attn_backend='default',use_mla=False,moe_ep_backend='disabled',moe_ep_size=1,weight_load_mode='async',kv_transfer_config=None,pre_transpose=False))
  def _build_sampling_params(self,data):return types.SimpleNamespace(max_tokens=data['max_tokens'],temperature=0.,stop=None,ignore_eos=False)
 server,provider,p=cli.build_server(cfg,Processor(),Reason,Server);assert type(provider)is DeepProvider and server.engine is provider and server.kw['max_batch_size']==4
 owners=[dict(device=i,changes=1,verified_mask=','.join(map(str,range(i*4,i*4+4))))for i in range(8)]
 fixed=dict(profile=NAME,HC_mode=0,r366_fixed=0,fullpre=0,precision_policy=0,rows=4,capacity_per_row=32768,maximum_cohort=4,row_reuse=False,C1=False)
 provider.group.run.write('ready-engine.json',dict(loaded=True,loaded_key_count=67569,Native_SHA256=cfg['Native_SHA256'],HC_mode=0,rows=4,capacity=32768,affinity=dict(actual_owner_report=dict(owners=owners)),postload_free=[[20*1024**3,68702699520]]*8,fixed_profile=fixed))
 row=provider.group.wait_ready();assert row['fixed_profile']['C1']is False
 # Recording readiness does not execute provider.start/Native and is never a
 # realModel receipt. This consumes the changed actual Actors ready predicate.
 return dict(passed=True,actual_build_server=True,actual_provider_factory=True,actual_Actors_wait_ready=True,max_batch_size=4,recording_READY=True,Model_READY=False,Native=False,HTTP_started=False)
def main():
 with tempfile.TemporaryDirectory()as directory:
  root=Path(directory);(root/'generation_config.json').write_text('{"eos_token_id":1}')
  rejected(lambda:validate(dict(config(),experimental_profile=None)))
  q=dict(passed=True,actual_Native=False,SDK=False,GPU=False,Model_READY=False,boundaries=boundaries(root),client=client_case(root),public_factory=public_factory_case(root))
 print(json.dumps(q));return q
if __name__=='__main__':main()
