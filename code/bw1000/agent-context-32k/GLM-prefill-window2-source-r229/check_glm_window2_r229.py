"""Local source-derived control checks only. No Torch, Native, SSH or Slurm import."""
import ast, ctypes, hashlib, json, pathlib, sys, textwrap, types, weakref
D=pathlib.Path(__file__).parent
source=(D/'glm_agent_service_window2_r229.py').read_text()
block=textwrap.dedent(source[source.index('            request_completed_normally=False'):source.index("            row={'sequence':seq")])
compiled=compile(block,'<actual-r229-request-loop>','exec')
repo=pathlib.Path(sys.argv[1]).resolve()  # Explicit existing r204 repository; read-only.
wire=ast.parse((repo/'code/bw1000/agent-context-32k/GLM-regular-service-private-r204/longcontext_wire_r204.py').read_bytes())
f=next(n for n in wire.body if isinstance(n,ast.FunctionDef)and n.name=='prefill_chunks');ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),'<actual-r204-chunks>','exec'),ns)
class Tensor:
 def __init__(self,data):self.data=list(data)
 def item(self):return self.data[0]
 def __getitem__(self,i):return self.data[i]
 def __setitem__(self,i,v):self.data[i]=v
 def cpu(self):return self
 def tolist(self):return self.data
 def argmax(self,*_):return Tensor([7])
 def all(self):return True

def case(n,stage=3,rank=0,limit=1,cancel_at=None,fail_at=None,timeout_at=None,eos=()):
 events=[];live=weakref.WeakSet();counter=0;expired=False
 req=types.SimpleNamespace(cancel_event=types.SimpleNamespace(is_set=lambda:False),mark_abort=lambda _:None,emit_token=lambda token,visible:events.append(['emit',token]))
 def tensor(data,**kw):return Tensor(data)
 def full(*args,**kw):
  assert not live,'Two logits buffers live on one owner'
  v=Tensor([0]);live.add(v);return v
 def forward(ctx,t,rows,desc,*args):
  nonlocal counter,expired
  counter+=1;events.append(['forward',int(desc[1]),t]);expired=timeout_at is not None and counter>=timeout_at
  if counter==fail_at:raise RuntimeError('injected-original-error')
  return 0
 def control(t):
  if len(t.data)==1:
   events.append(['vote',counter]);t.data[0]=float((cancel_at is not None and counter>=cancel_at)or expired)
  else:events.append(['select',counter]);t.data[:]=[7.,1.]
 def reset(ctx):events.append(['reset']);return 0
 def sync(rank):events.append(['sync'])
 torch=types.SimpleNamespace(tensor=tensor,arange=lambda *a,**kw:Tensor([]),full=full,zeros=lambda *a,**kw:Tensor([0,0]),isfinite=lambda _:Tensor([True]),int64=None,bfloat16=None,float32=None,cuda=types.SimpleNamespace(synchronize=sync,mem_get_info=lambda _: (2**30,2**31)))
 env=dict(torch=torch,ctypes=ctypes,time=types.SimpleNamespace(monotonic=lambda:10 if expired else 0,perf_counter=lambda:0),lib=types.SimpleNamespace(modern_glm_model_forward=forward,modern_glm_model_reset=reset),ctx=None,stage=stage,rank=rank,PP=4,is_emitter=stage==3 and rank==0,req=req,FinishReason=types.SimpleNamespace(TIMEOUT='timeout'),deadline=5,limit=limit,prompt=[1]*n,past=0,generated=[],reason='length',calls=[],report={},CONTEXT=32768,vocab=16,prefill_chunks=ns['prefill_chunks'],control=control,check=lambda x:None,ptr=lambda x:x,stream=lambda:None,barrier=lambda:events.append(['barrier']),eos=set(eos))
 error=None
 try:exec(compiled,env)
 except RuntimeError as e:error=str(e)
 if fail_at is not None:
  assert error=='injected-original-error'
  assert not any(e[0]in('reset','barrier','select','emit')for e in events)
  assert env['report']['normal_reset']is False and env['report']['normal_shutdown']is False
  assert events[-1][0]=='forward','Failure must propagate without extra boundary/drain'
 else:
  assert error is None and env['report']['normal_reset']is True
  assert events[-2:]==[['reset'],['barrier']]
  if cancel_at is not None or timeout_at is not None:
   assert env['reason']=='canceled'and not env['generated']
   assert not any(e[0]in('select','emit')for e in events)
  else:
   assert len(env['generated'])==(1 if 7 in eos else limit)
 return events,env

out=[]
for stage,rank in [(0,0),(1,0),(3,0),(3,1)]:
 e,v=case(1200,stage,rank,limit=3)
 assert [x for x in e if x[0]=='forward']==[['forward',0,512],['forward',512,512],['forward',1024,176],['forward',1200,1],['forward',1201,1]]
 assert [x[1]for x in e if x[0]=='vote']==[0,2,3,3,4]
 assert [x[1]for x in e if x[0]=='select']==[3,4,5]
 out.append({'case':'partial_tail_and_two_decode','stage':stage,'rank':rank,'PASS':True})
for n in (1,512,513,1024,1025):
 e,v=case(n,eos=(7,));assert len(v['generated'])==1
 assert sum(x[0]=='forward'for x in e)==(n+511)//512
 assert [x for x in e if x[0]=='select']==[['select',(n+511)//512]]
 out.append({'case':'tail_and_EOS','prompt':n,'PASS':True})
for k,expected in [(0,0),(1,2),(2,2),(3,3)]:
 e,v=case(1200,cancel_at=k);assert sum(x[0]=='forward'for x in e)==expected
 out.append({'case':'cancel','after_forward':k,'committed_forwards':expected,'PASS':True})
e,v=case(1200,timeout_at=1);assert sum(x[0]=='forward'for x in e)==2;out.append({'case':'absolute_deadline_midgroup','PASS':True})
for k in (1,2,3):
 e,v=case(1200,fail_at=k);out.append({'case':'original_exception_failstop','forward':k,'PASS':True})
# Dispatcher executes only fake preloaded modules; never imports actual service deps.
import os,sys
wrapper=compile((D/'glm_prefill_window_r229.py').read_bytes(),'<dispatcher>','exec');old=os.environ.get('GLM_PREFILL_CONTROL_WINDOW_R229')
modules={name:sys.modules.get(name)for name in ('glm_agent_service_baseline_r204','glm_agent_service_window2_r229')}
try:
 sys.modules['glm_agent_service_baseline_r204']=types.SimpleNamespace(run=lambda *a,**kw:1)
 sys.modules['glm_agent_service_window2_r229']=types.SimpleNamespace(run=lambda *a,**kw:2)
 for flag,want in [(None,1),('1',1),('2',2)]:
  if flag is None:os.environ.pop('GLM_PREFILL_CONTROL_WINDOW_R229',None)
  else:os.environ['GLM_PREFILL_CONTROL_WINDOW_R229']=flag
  g={};exec(wrapper,g);assert g['run']()==want
 for flag in ('0','3','02','true',''):
  os.environ['GLM_PREFILL_CONTROL_WINDOW_R229']=flag
  try:g['run']()
  except ValueError:pass
  else:raise AssertionError('Invalid flag accepted')
finally:
 if old is None:os.environ.pop('GLM_PREFILL_CONTROL_WINDOW_R229',None)
 else:os.environ['GLM_PREFILL_CONTROL_WINDOW_R229']=old
 for name,value in modules.items():
  if value is None:sys.modules.pop(name,None)
  else:sys.modules[name]=value
assert hashlib.sha256((D/'glm_agent_service_baseline_r204.py').read_bytes()).hexdigest()=='a95d08167c9f30be1d9f1fc75522a6bcc63364b079d9aa96c2289a00de5e480b'
result={'scope':'Local AST-extracted actual request control loop with fake tensor/forward; no real ranks or communication','cases':out,'dispatcher_default_and_strict_flags_PASS':True,'baseline_byte_identical':True,'all_PASS':True,'Native_math_CCL_liveness_Model_quality_performance_qualified':False,'GPU_or_remote_tasks':0,'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(D.glob('*.py'))}}
(D/'local-control-checks-r229.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({'all_PASS':True,'cases':len(out),'result':str(D/'local-control-checks-r229.json')}))
