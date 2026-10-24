"""Actual prefill_phase consumer; CPU rendezvous/order/failure only, no TPS."""
import pathlib,json,threading,queue,time,hashlib,copy,ast
from kimi_fixed512_prefill_r8623 import prefill_phase
r=pathlib.Path(__file__).resolve().parent
jobs=[dict(seq=i,kind='prefill',keys=[i],rows=[i+1],pasts=[0],inputs=[[i+10]*512])for i in range(8)]
class Group:
 def __init__(self):self.c=threading.Condition();self.rounds={};self.labels=[]
 def collect(self,stage,index,label,payload):
  with self.c:
   b=self.rounds.setdefault(index,dict(label=label,values={}));assert b['label']==label;b['values'][stage]=copy.deepcopy(payload)
   if len(b['values'])==4:self.labels.append(label);self.c.notify_all()
   assert self.c.wait_for(lambda:len(b['values'])==4,timeout=3),'CPU rendezvous timeout'
   return[copy.deepcopy(b['values'][i])for i in range(4)]
class Control:
 def __init__(self,g,stage):self.g=g;self.stage=stage;self.i=0;self.timeouts=[]
 def set_timeout(self,s):self.timeouts.append(s)
 def collect(self,label,payload):
  ret=self.g.collect(self.stage,self.i,label,payload);self.i+=1;return ret

def fixture(collect_each):
 group=Group();channels=[queue.Queue()for _ in range(3)];gate=threading.Event();lock=threading.Lock();events=[];results={};errors=[];saved={i:[]for i in range(4)}
 class Adapter:
  CPU_fixture=True
  def __init__(self,s):self.s=s;self.next=0
  def forward(self,j):
   assert self.next==j['seq'];self.next+=1
   with lock:events.append((self.s,j['seq'],'enter'))
   if self.s==0 and j['seq']==1:gate.set()
   if self.s:
    seq=channels[self.s-1].get(timeout=3);assert seq==j['seq']
   if self.s==3 and j['seq']==0 and not collect_each:assert gate.wait(3)
   if self.s<3:channels[self.s].put(j['seq'])
   with lock:events.append((self.s,j['seq'],'return'))
   return[100+j['seq']]if self.s==3 else None
 def worker(s):
  try:results[s]=prefill_phase(Adapter(s),jobs,stage=s,control=Control(group,s),label='fixture',collect_each=collect_each,on_actual=saved[s].append)
  except BaseException as e:errors.append(repr(e))
 threads=[threading.Thread(target=worker,args=(i,))for i in range(4)]
 for t in threads:t.start()
 for t in threads:t.join(5);assert not t.is_alive()
 assert not errors,errors
 assert all(x['ids']==[[100+i]for i in range(8)]for x in results.values())
 assert all([v['seq']for v in saved[s]]==list(range(8))for s in range(4))
 assert len(group.labels)==(10 if collect_each else 2)
 if not collect_each:assert events.index((0,1,'enter'))<events.index((3,0,'return'))
 assert len(results[3]['clock']['tail_ready_s'])==len(results[0]['clock']['coordinator_visible_s'])==8
 assert all('tail_ready_s'not in results[s]['clock']for s in range(3))
 return dict(serial=collect_each,collect_rounds=len(group.labels),all4order_exact=True,all8tail_IDs_equal=True,pipeline_partial_order_proved=not collect_each,clock_scopes_separate=True)
results=[fixture(True),fixture(False)]
class SingleControl:
 def __init__(self):self.calls=[]
 def set_timeout(self,s):pass
 def collect(self,label,payload):
  self.calls.append(label);assert label.endswith(':plan'),'failure must not publish phaseend';return[dict(payload,stage=s)for s in range(4)]
class Fail:
 CPU_fixture=True
 def __init__(self):self.calls=[]
 def forward(self,j):
  self.calls.append(j['seq'])
  if j['seq']==2:raise RuntimeError('injected native unknown; never replay')
  return None
f=Fail();c=SingleControl();saved=[]
try:prefill_phase(f,jobs,stage=0,control=c,label='failure',collect_each=False,on_actual=saved.append)
except RuntimeError as e:assert'unknown'in str(e)
else:raise AssertionError('must propagate')
assert f.calls==[0,1,2]and len(saved)==2 and len(c.calls)==1
results.append(dict(failure='Unknown native propagated once; no phaseend or replay',actual_completed_prefix=2))
f=Fail();c=SingleControl();clock=iter(range(100))
try:prefill_phase(f,jobs,stage=0,control=c,label='deadline',collect_each=False,timeout_s=1,now_ns=lambda:next(clock)*2*10**9)
except TimeoutError:pass
else:raise AssertionError('must stop before firstnative')
assert not f.calls
results.append(dict(expired_before_native=True,no_native_call=True))
bad=copy.deepcopy(jobs);bad[0]['inputs'][0].append(9)
try:prefill_phase(Fail(),bad,stage=0,control=SingleControl(),label='badgeometry',collect_each=False)
except AssertionError:pass
else:raise AssertionError('bad geometry accepted')
results.append(dict(unsupported_M513_rejected=True))
out=dict(CPU_only=True,GPU=False,performance_evidence=False,source_SHA256=hashlib.sha256((r/'kimi_fixed512_prefill_r8623.py').read_bytes()).hexdigest(),fixtures=results,scope='Real Python consumer with4 CPU threads and bounded queues/barrier; ordering/owned-ID/control flow only. No synthetic TPS. Actual native call hard deadline still requires existing Slurm+supervisor, no native interruption/replay.')
p=r/'Kimi-fixed512-prefill-CPU-r8623.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n');print(json.dumps(out))
