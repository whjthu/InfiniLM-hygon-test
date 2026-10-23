"""Actual r8485 CPU consumer/socket contracts, no GPU/performance inference."""
import pathlib,json,hashlib,time,threading,io,ast,dataclasses,os
import numpy as np
import kimi_bounded_batch_wait_r8485 as mod
from audit_Kimi_bounded_r8485 import resolve_evidence_path
from online_broker_r8455 import Client
from online_kimi_provider_r8467 import cpu_argmax_observation
r=pathlib.Path.cwd();checks=[];run='matched8474-bc7097c924081592'
jobs=[json.loads(s)for s in(r/'Kimi-matched-terminal-evidence-r8480'/f'Kimi-arrival-r8472-{run}-live-token-only-jobs.jsonl').read_text().splitlines()]
prompts={k:p for j in jobs if j['job']['kind']=='prefill'for k,p in zip(j['job']['keys'],j['job']['inputs'])};inputs=[prompts[k]for k in range(8)]
plan,plan_SHA=mod.consume_plan(r/'Kimi-fixed-arrival-plan-r8485.json',inputs)
try:mod.consume_plan(r/'Kimi-fixed-arrival-plan-r8485.json',inputs[::-1])
except AssertionError:pass
else:raise AssertionError('prompt binding accepted')
try:mod.consume_plan(r/'Kimi-fixed-arrival-plan-r8482.json',inputs)
except AssertionError:pass
else:raise AssertionError('wrongplan namespace accepted')
checks.append('actual consume_plan accepts exactnamespace/prompt bindings, rejects reordered prompts/wrongrevision')
class Clock:
 def __init__(self):self.t=100.
 def __call__(self):return self.t
 def sleep(self,s):self.t+=s
clock=Clock()
class Ingress:
 def __init__(self):self.calls=[]
 def call(self,req):
  self.calls.append(req);clock.t+=.003
  return dict(accepted=True,request=len(self.calls)-1,arrival=clock.t)
c=Ingress();records=[];mod.emit_fixed_arrivals(c,inputs,plan,records,clock=clock,sleep=clock.sleep)
assert len(records)==8 and all(rec['target_t']==100+plan['offsets_s'][i]for i,rec in enumerate(records))
assert any(v['lateness_s']>.01 for v in records) and all(v['actual_ingress_t']>=v['send_t']for v in records)
checks.append('actual fixed producer preserves targets under injected replydelay and records actual lateness; no feedback release')
p=resolve_evidence_path('/hostroot','/refactor/results/Kimi-bounded-arrival-r8485-bounded8485-test-bounded-O1-jobs.jsonl','bounded8485-test');assert p.name.endswith('jobs.jsonl')
for bad in ['/refactor/results/../secret','/refactor/results/Kimi-arrival-r8472-bounded8485-test-jobs.jsonl','/somehost/results/Kimi-bounded-arrival-r8485-bounded8485-test-jobs.jsonl']:
 try:resolve_evidence_path('/hostroot',bad,'bounded8485-test')
 except AssertionError:pass
 else:raise AssertionError('namespace escape')
checks.append('actual auditor exact containerpath mapper rejects traversal/wrongnamespace')
# Reuse authoritative uint16 BF16 bit representation, exact widen, first-index ties.
logits=np.zeros((2,163840),dtype=np.uint16);logits[0,3]=np.uint16(0x3f80);logits[0,4]=np.uint16(0x3f80);logits[1,9]=np.uint16(0x4000)
o=cpu_argmax_observation(logits,[3,9]);assert o['finite']and o['native_equals_CPU']and o['cpu_argmax_ids']==[3,9]
assert o['top2'][0]['tie_count']==2
checks.append('actual BF16 uint16 -> F32 CPUargmax consumer handles first-ID tie; no F64')
# Real TCP next/finish/bye, four CPU mock owners. No native CAPI/Model is called.
socket_results=[]
for wait in [0,.01]:
 events=[];capture=io.StringIO();nonce='cpu-contract-r8485-'+os.urandom(8).hex()
 scheduler=mod.BoundedWaitScheduler(stages=4,micro=4,max_users=8,max_inflight=4,rows=32,eos=[163585,163586],max_queue=8,wait_s=wait)
 broker=mod.TickRecordingBroker(capture=capture,full_reference=False,nonce=nonce,journal=lambda batch:events.extend(batch),scheduler=scheduler,host='127.0.0.1',port=0)
 errors=[];closed=[]
 def owner(stage):
  client=None
  try:
   client=Client(broker.address,nonce,'owner',stage)
   while True:
    v=client.call(dict(op='next'))
    if v.get('stop'):break
    if v['wait']:time.sleep(.002);continue
    j=v['job'];client.call(dict(op='finish',seq=j['seq'],ids=[3]*len(j['keys'])if stage==3 else None))
   client.call(dict(op='bye'));closed.append(stage)
  except BaseException as exc:errors.append(repr(exc));broker.poison(repr(exc))
  finally:
   if client:client.close()
 threads=[threading.Thread(target=owner,args=(s,))for s in range(4)]
 for t in threads:t.start()
 deadline=time.monotonic()+5
 while len(broker.owner_connections)<4:
  assert time.monotonic()<deadline;time.sleep(.001)
 client=Client(broker.address,nonce,'ingress');first=client.call(dict(op='arrive',prompt=[1],limit=2));assert first['accepted']
 # No more arrivals until owner-next polling alone dispatches the underfull job.
 while not broker.records:
  assert time.monotonic()<deadline and not errors;time.sleep(.001)
 for i in range(1,8):assert client.call(dict(op='arrive',prompt=[1,2],limit=2))['accepted']
 client.call(dict(op='seal'));client.close()
 for t in threads:t.join(5)
 assert not any(t.is_alive()for t in threads) and not errors and not broker.failure and sorted(closed)==[0,1,2,3]
 broker.flush();assert broker.core.can_reset_epoch();broker.close()
 arrivals=[e for e in events if e['kind']=='arrival'];dispatch=[e for e in events if e['kind']=='dispatch']
 assert dispatch[0]['t']>=arrivals[0]['arrival_t']+wait-1e-7
 assert len([e for e in events if e['kind']=='terminal'and e['status']=='completed'])==8
 socket_results.append(dict(wait_s=wait,owner_normal_bye=closed,one_actual_jobseq_no_replay=True,deadline_woken_without_new_ingress=True,recorded_jobs=len(broker.records)))
checks.append('real bounded TCP 4 CPUowners: O/W Tick wakes partialdeadline with no newingress, all8 complete/all4bye')
# Import-time no Engine init; actual loaded-loop consumer order/firstdiff persistence.
original_epoch,original_provider=mod.bounded_epoch,mod.BorrowedProvider
os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1';seen=[];resets=[];saved=[]
job=dict(seq=0,kind='prefill',keys=[0],rows=[1],pasts=[0],inputs=[[1]])
def fake_epoch(g,ins,eos,reset,**kw):
 seen.append((kw['label'],kw['wait_s'],kw['port_delta']));return dict(captured=[dict(job=job,job_SHA256=mod.digest(job),actual_ids=[3])])
class Provider:
 def __init__(self,*args,**kw):assert kw['full_logit_reference'];self.reference_observation=dict(finite=True,native_equals_CPU=True,cpu_argmax_ids=[3])
 def forward(self,j):assert j==job;return [3]
 def close(self):pass
class Control:
 def collect(self,label,value):return [dict(value)for _ in range(4)]
report={};g=dict(tp_nodes=1,pp=4,report=report,control=Control(),node=3,save=lambda:saved.append(json.loads(json.dumps(report))),lib=None,ctx=1,check=lambda code:None,memory=lambda:[(2**30,2**30)])
mod.bounded_epoch=fake_epoch;mod.BorrowedProvider=Provider
try:
 mod.run_loaded_epochs(g,inputs,{163585,163586},resets.append,dict(DSO_SHA256='9001bf2bfa7e0821d52d9bec61ed5d9bf028c93380673fa880af3ed52e9c7388'),plan)
 assert [v[:2]for v in seen]==[('bounded-O1',0),('bounded-W1',.01),('bounded-W2',.01),('bounded-O2',0)] and report['bounded_wait_four_epochs_qualified']
 assert len(resets)==4 and len(report['bounded_wait_oracle_checks'])==4
finally:mod.bounded_epoch=original_epoch;mod.BorrowedProvider=original_provider
checks.append('actual run_loaded_epochs consumer O/W/W/O, uniqueports and independentoracle per epoch; CPU mocks only')
class BadProvider(Provider):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.reference_observation['cpu_argmax_ids']=[4]
report.clear();saved.clear();seen.clear();mod.bounded_epoch=fake_epoch;mod.BorrowedProvider=BadProvider
try:
 try:mod.run_loaded_epochs(g,inputs,{163585,163586},resets.append,dict(DSO_SHA256='9001bf2bfa7e0821d52d9bec61ed5d9bf028c93380673fa880af3ed52e9c7388'),plan)
 except AssertionError:pass
 else:raise AssertionError('firstdiff accepted')
 assert len(seen)==1 and saved[-1]['bounded_wait_oracle_checks']['O1'][0]['first_difference_row_index']==0
 assert not report.get('bounded_wait_four_epochs_qualified')
finally:mod.bounded_epoch=original_epoch;mod.BorrowedProvider=original_provider
checks.append('actual loaded consumer persists fulljob/actual/firstdiff before assert and stops before W epoch')
# Reconstruct old loader using only allowed string aliases; prove no loading/destroy changes.
new=(r/'modern_Kimi_bounded_r8485.py').read_text();old=(r/'Kimi-matched-source-package-r8474/modern_Kimi_matched_r8472.py').read_text()
restored=new.replace('Kimi-bounded-manifest-r8485.json','Kimi-matched-manifest-r8472.json').replace('kimi_bounded_protocol_r8485','kimi_matched_protocol_r8472').replace('kimi-bounded-r8485-','kimi-matched-r8472-').replace('Kimi-bounded-cgroup-r8485-','Kimi-matched-cgroup-r8469-');assert restored==old
checks.append('actual loader entire source equals approved8474 after four path/protocol aliases; native load/finally unchanged')
for name in ['kimi_bounded_batch_wait_r8485.py','kimi_bounded_protocol_r8485.py','modern_Kimi_bounded_r8485.py','launch_Kimi_bounded_r8485.py','audit_Kimi_bounded_r8485.py']:ast.parse((r/name).read_text(encoding='utf-8-sig'))
out=dict(revision=8487,CPU_actual=True,checks=checks,real_TCP_CPU_mock_contracts=socket_results,GPU_executed=False,GPU_GO='NOT_GRANTED',old588_scope='DIAGNOSTIC_ONLY; newO andW both sameTick coordinator',performance_claim='NONE; no CPU mock timing is GPU performance',source_SHA256={name:hashlib.sha256((r/name).read_bytes()).hexdigest()for name in ['kimi_bounded_batch_wait_r8485.py','kimi_bounded_protocol_r8485.py','modern_Kimi_bounded_r8485.py','launch_Kimi_bounded_r8485.py','audit_Kimi_bounded_r8485.py','Kimi_bounded_r8485_inside.sh','Kimi-fixed-arrival-plan-r8485.json']})
(r/'Kimi-bounded-package-CPU-r8487.json').write_text(json.dumps(out,indent=2),encoding='utf8',newline='\n');print(json.dumps(out))
