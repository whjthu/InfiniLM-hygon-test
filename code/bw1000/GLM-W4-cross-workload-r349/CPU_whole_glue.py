"""Actual owner/driver/client glue at recording CAPI, HTTP and process boundaries.

No Native Context, actual GPU, real Slurm, socket listener or Model execution.
Prepared tokenizer/history/queue are real. CPU wire is semantic gold, not a
generated Model response. All32 actual owner loops replay one collective tape.
"""
import pathlib,sys,os,json,time,hashlib,types,threading,subprocess,copy,inspect,ast,traceback,asyncio
from unittest.mock import patch
HERE=pathlib.Path(__file__).parent;D=HERE/'owner';R=pathlib.Path('/refactor')
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];tag='GLM349-CPU-'+step
out=R/'results'/('GLM349-whole-glue-'+step);out.mkdir(exist_ok=False)
result=dict(completed=False,passed=False,CPU_only=True,actual_step=step,GPU=0,Model=0,Native_Context=0,real_Slurm=False,actual_HTTP=False,real_TCP_listener=False,cases=[],recording_boundaries=['CAPI','WORLD collective','HTTP transport','child-process launch'])
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def case(name,**data):result['cases'].append(dict(name=name,passed=True,**data))
try:
 # This is the exact guarded CPU272 import boundary, retained as source text.
 U=R/'candidates/latest-family-service-r270/GLM/python/infinilm'
 import importlib.machinery
 class ForbiddenNative(types.ModuleType):
  def __getattr__(self,name):raise RuntimeError('Native forbidden: '+name)
 root=types.ModuleType('infinilm');root.__path__=[str(U)];root.__package__='infinilm';sys.modules['infinilm']=root
 core=ForbiddenNative('infinilm._core');core.__file__='FORBIDDEN_NATIVE';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);root._core=core;sys.modules[core.__name__]=core
 llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
 for suffix,cls in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
  mod=types.ModuleType('infinilm.llm.'+suffix);setattr(mod,cls,type(cls,(),{}));sys.modules[mod.__name__]=mod
 class ForbiddenEngine:
  def __init__(self,*a,**k):raise RuntimeError('Model engine forbidden')
 llm.AsyncLLMEngine=ForbiddenEngine
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weights forbidden')
 sys.addaudithook(audit)
 import torch
 torch.set_num_threads(1)
 def guard(name):
  def unique(*a,**k):raise RuntimeError('GPU forbidden: '+name)
  return unique
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 from infinilm.llm.sampling_params import SamplingParams
 from infinilm.llm.request import FinishReason
 llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
 sys.path.insert(0,str(D))
 from source_pin_r290 import verify
 mf=sha(D/'SOURCE_MANIFEST.json');verify(mf)
 from prepared_family_entry_r268 import prepare_app
 prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3')
 from infinilm.server.agent_protocol import prepare_agent_request
 from workload_r349 import load_plan,body_and_gold
 card,plan=load_plan(D/'FIXTURE_CARD.json')
 import owner_r290 as owner,client_driver_r290 as driver,client_crossworkload_r349 as client
 from client_owner_r290 import NATIVE_SHA,bind_identity
 os.environ.update(GLM_RUN_R5040=tag,GLM_SOURCE_MANIFEST_SHA=mf,GLM_R290_GPUSTART_UNIX=str(time.time()),CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
 engine=prepared['engine'];started=time.monotonic();wire_count=[0];thread_errors=[];processes=[]
 ready=R/'results'/('GLM-latest-full-owner-r290-ready-'+tag+'.json');reportpath=R/'results'/('GLM-latest-full-owner-r290-'+tag+'-stage3-rank0.json')
 supervisor=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+tag+'.json')
 class Response:
  status=200
  def __init__(self,raw):self.raw=raw
  def __enter__(self):return self
  def __exit__(self,*args):return False
  def read(self):return self.raw
 def urlopen(req,timeout):
  index=wire_count[0];arm=plan[index];f=card['fixtures'][arm['fixture_index']];raw,body,expected=body_and_gold(f)
  assert req.data==raw and req.full_url=='http://127.0.0.1:43843/v1/chat/completions'
  s=json.loads(supervisor.read_bytes());assert s['planned_HTTP']==s['planned_Native']==18
  if index==0:
   assert s['attempted_HTTP']==s['completed_HTTP']==s['actual_Native']==0
   initial=json.loads(reportpath.read_bytes());assert initial['attempted_Native']==initial['completed_Native']==0
  wire_count[0]+=1;data,schemas,choice=prepare_agent_request(copy.deepcopy(body),'glm5');data.update(_agent_schemas=schemas,_agent_choice=choice)
  r=engine.add_chat_request(messages=data['messages'],sampling_params=prepared['server']._build_sampling_params(data),request_id='CPU349-%02d'%index,request_data=data,add_generation_prompt=True,chat_template_kwargs=data.get('chat_template_kwargs')or{})
  async def consume_actual_transport():
   items=[]
   async for event in engine.stream_request(r,timeout=20,request_timeout=20):items.append(event)
   return items
  events=asyncio.run(consume_actual_transport());assert len(events)==1 and events[0].finished
  assert r.request_id not in engine._requests,'Actual transport must release its completed request'
  assert r.finish_reason==FinishReason('eos_token')
  args=copy.deepcopy(expected)
  if 'content' in args:args['content']=json.dumps(args['content'])
  n=len(r.generated_token_ids);p=len(r.prompt_token_ids)
  wire={'id':r.request_id,'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'CPU-call-%02d'%index,'type':'function','function':{'name':f['gold']['tool'],'arguments':json.dumps(args)}}]}}],'usage':{'prompt_tokens':p,'completion_tokens':n,'total_tokens':p+n}}
  return Response(json.dumps(wire).encode())
 def actual_main(module,argv):
  with patch.object(sys,'argv',[str(D/module.__name__)] + argv):return module.main()
 def run_child(argv,**kwargs):
  assert pathlib.Path(argv[2]).name=='client_crossworkload_r349.py'
  rc=actual_main(client,argv[3:]);return types.SimpleNamespace(returncode=rc)
 class Process:
  pid=7349001
  def __init__(self,argv,**kwargs):
   assert pathlib.Path(argv[3]).name=='client_driver_r290.py' and kwargs['start_new_session']
   self.returncode=None;self.thread=threading.Thread(target=self.execute,args=(argv[4:],));processes.append(self);self.thread.start()
  def execute(self,args):
   try:self.returncode=actual_main(driver,args)
   except BaseException as error:thread_errors.append(traceback.format_exc());self.returncode=2
  def poll(self):return self.returncode
  def wait(self,timeout=None):
   self.thread.join(timeout)
   if self.thread.is_alive():raise subprocess.TimeoutExpired('recording driver',timeout)
   return self.returncode
 class Server:
  def __init__(self,*args):self.started=False;self.should_exit=False;self.force_exit=False
  def run(self):
   self.started=True
   while not self.should_exit and not self.force_exit:time.sleep(.001)
 fakeuv=types.ModuleType('uvicorn');fakeuv.Config=lambda *a,**k:types.SimpleNamespace(**k);fakeuv.Server=Server
 patches=[]
 for name in ['tensor','ones','arange','full','zeros']:
  fn=getattr(torch,name)
  def make(fn):
   def cpu(*args,**kwargs):kwargs.pop('device',None);return fn(*args,**kwargs)
   return cpu
  patches.append(patch.object(torch,name,make(fn)))
 patches += [patch.object(torch.cuda,'synchronize',lambda *a:None),patch.object(torch.cuda,'mem_get_info',lambda *a:(4*1024**3,8*1024**3)),patch.dict(sys.modules,{'uvicorn':fakeuv}),patch.object(subprocess,'Popen',Process),patch.object(subprocess,'run',run_child),patch.object(os,'getpgid',lambda pid:pid),patch.object(client.urllib.request,'urlopen',urlopen)]
 class Function:
  def __init__(self,name,lib):self.name=name;self.lib=lib
  def __call__(self,*args):
   self.lib.counts[self.name]=self.lib.counts.get(self.name,0)+1
   if self.name=='modern_glm_model_transient_report_r5688':return b'{"rows":1,"capacity":32768,"enabled":false,"GPU_logical_bytes":0}'
   if self.name=='modern_glm_model_forward':args[6].zero_();args[6][0,154820]=1
   if self.name=='modern_glm_cross_journal_r5775':return b'{"identities":[]}'
   return 0
 class Library:
  def __init__(self):self.counts={};self.functions={}
  def __getattr__(self,name):
   if name not in self.functions:self.functions[name]=Function(name,self)
   return self.functions[name]
 tape=[];cursor=[0];first=[True]
 def record_control(tensor):
  caller=inspect.currentframe().f_back.f_code.co_name
  if first[0] or caller=='barrier':tensor.mul_(32);first[0]=False
  tape.append(tensor.clone())
 def replay_control(tensor):
  expected=tape[cursor[0]];cursor[0]+=1;assert tensor.shape==expected.shape;tensor.copy_(expected)
 lib=Library();report={'library_sha256':NATIVE_SHA};payload={'benchmark':{'generation_config':{'eos_token_id':[154820,154827,154829]},'prefill_pair_r240':{'fixture_card':str(D/'FIXTURE_CARD.json'),'output_root':str(out)}}}
 def check(code):assert code==0
 for p in patches:p.start()
 try:
  owner.run(lib,object(),3,0,payload,record_control,check,lambda x:x,lambda:0,154880,report,reportpath,lambda *a:None,prepared)
  assert not thread_errors and all(p.returncode==0 for p in processes)
  assert report['planned_HTTP']==report['planned_Native']==report['attempted_Native']==report['completed_Native']==report['full_requests_completed']==18
  assert report['normal_shutdown'] and report['normal_HTTP_shutdown'] and not report['cleanup']['forced_cleanup']
  ident=bind_identity(str(ready),mf,str(reportpath));assert ident['prepared_entry_identity']==prepared['identity'] and ident['actual_step']==step
  s=json.loads(supervisor.read_bytes());assert s['completed'] and s['passed'] and s['planned_HTTP']==s['planned_Native']==s['attempted_HTTP']==s['completed_HTTP']==s['actual_Native']==18
  case('actual_emitter_owner_to_actual_supervisor_and_client_main',requests=18,fixtures=3,worker_identity_complete=True,zero_initial_counters=True,all_indices_and_prompt_hashes=True,HTTP_and_CAPI_recording_only=True)
  source=ast.parse((D/'model_entry_r290.py').read_bytes());worker=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
  index=next(i for i,n in enumerate(worker.body) if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='BLAS_policy' for t in n.targets))
  cleanup_ast=ast.Module(body=worker.body[index:],type_ignores=[]);cleanup_code=compile(cleanup_ast,'actual-worker-cleanup-suffix','exec');cleanup_SHA=hashlib.sha256(ast.dump(cleanup_ast).encode()).hexdigest()
  owners=[]
  for stage in range(4):
   for rank in range(8):
    if(stage,rank)==(3,0):row=report;library=lib;path=reportpath
    else:
     cursor[0]=0;row={'library_sha256':NATIVE_SHA};library=Library();path=out/('recording-stage%d-rank%d.json'%(stage,rank))
     owner.run(library,object(),stage,rank,payload,replay_control,check,lambda x:x,lambda:0,154880,row,path,lambda *a:None,prepared)
     assert cursor[0]==len(tape)
    assert len(row['service_requests'])==18 and row['normal_shutdown']
    for actual,arm in zip(row['service_requests'],plan):
     assert actual['global_index']==arm['global_index'] and actual['fixture_id']==arm['fixture_id'] and actual['arm_index']==arm['arm_index'] and actual['generated_natural_EOS']
    namespace=dict(report=row,policy=types.SimpleNamespace(modern_blas_policy_r243_json=lambda:b'{"failures":0,"owners":[]}'),os=os,host_stats=lambda:[0,0,0,0],cache_stats809=lambda rank:dict(created=1,destroyed=1,active=0,active_bytes=0),service_stats=lambda:{},lib=library,ctx=object(),check=check,rank=rank,m15_stats129=lambda:[0]*17,report_path=path,json=json,torch=torch)
    exec(cleanup_code,namespace)
    assert row['completed'] and row['worker_shutdown_completed'] and row['observer_native_destroy_returned0']
    assert library.counts['modern_glm_model_destroy']==library.counts['modern_glm_model_shutdown']==1
    owners.append({'stage':stage,'rank':rank,'completed':True,'requests':18,'destroy':1,'shutdown':1})
  assert len(owners)==32
  case('all32_actual_owner_loops_and_actual_worker_cleanup_suffix',owners=owners,cleanup_AST_SHA256=cleanup_SHA,Native_calls_are_recording=True)
 finally:
  for p in reversed(patches):p.stop()
 from deadline_r290 import check as deadline_check
 with patch.dict(os.environ,{'GLM_R290_GPUSTART_UNIX':str(time.time()-9121)}):
  try:deadline_check('CPU original start expiration')
  except TimeoutError:pass
  else:raise AssertionError('expired original deadline accepted')
 case('original_start_deadline_not_reset')
 from glm_prefill_execute_r236 import execute_plan
 for window in (1,2,4):
  called=[]
  past,token,canceled=execute_plan([(0,[1,2])],0,prefill=True,window=window,forward=lambda *a:called.append(a),vote=lambda:True,sample=lambda:None,discard=lambda:None)
  assert canceled and not called and past==0 and token is None
 case('cancellation_before_forward_no_retry',windows=[1,2,4])
 assert not torch.cuda.is_initialized() and 'GLM-PP4-registry-repair-r184b' not in pathlib.Path('/proc/self/maps').read_text()
 result.update(passed=True,elapsed_s=time.monotonic()-started,source_manifest_SHA256=mf,prepared_entry_identity=prepared['identity'],root_core_llm_guarded=True,actual_root_or_ABI_qualified=False,real_GPU_GO=False,scope='CPU whole glue only; recording CAPI/HTTP/WORLD/process boundaries; actual32 owner loops and cleanup statements')
except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
finally:
 result['completed']=True;(out/'CPU-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(0 if result['passed']else 1)
