import pathlib,sys,json,ast,ctypes as C,time,socket,threading,copy
import numpy as np
D=pathlib.Path(__file__).resolve().parent;P=D/'runtime';sys.path.insert(0,str(P))
from phase_observer_r126 import PhaseTrace,PhaseControl,interval_union,summarize,prefill_interference
from batch_cohort_r9128 import Request,cohort
from batch_wire_r9128 import send,recv
from batch_frames_r9142 import receive_bodies,FEATURE_BYTES
from draft_context_r9128 import ContextStore
checks=0
# OFF invokes no observer clock; ON nested labels use interval union rather than additive report.
clockcalls=[]
p=PhaseTrace(2,1,lambda:(clockcalls.append(1)or float(len(clockcalls))))
p.configure(False,'OFF');assert p.call('x',lambda:17)==17 and not clockcalls;checks+=1
p.configure(True,'ON');t=p.start_cycle(2,[1,2],[8,8]);a=p.begin('a');b=p.begin('b');p.end(b);p.end(a);p.end_cycle(t);v=p.export();assert v['actual_T16_cycles']==1 and summarize(v)['never_add_nested_labels'];assert interval_union([(0,3),(1,2),(2,5)])==5;checks+=3
# Actual Native Python ctypes buffers, fake C API only. Same output policy, no new C API.
old=(D.parent/'Kimi-ctx2048-implementation-private-r123/verify_CPU_r123.py').read_bytes().decode();snippet=old[old.index('class F:'):old.index('# Sourcepin/legacyguardintent')];snippet=snippet[:snippet.index('lib=L();nn=Native')]
ns={'np':np,'C':C,'json':json};exec(compile(snippet,'actual_r123_native_mockdefs','exec'),ns)
from batch_native_r9128 import Native
lib=ns['L']();phase=PhaseTrace(2,1);phase.configure(True,'actual-Native-buffer');control=PhaseControl(ns['Control'](),phase);nn=Native({'lib':lib,'ctx':1,'control':control,'node':2,'tail':True,'emitter':2,'vocab':128,'pp':2,'phase_trace':phase});nn.call('prefill',[1],[0],[list(range(1024))],False);token=phase.start_cycle(2,[1],[8]);nn.call('verify',[1],[2040],[list(range(8))],False);phase.end_cycle(token);assert len(lib.records)==2 and {'native.verify_CAPI_call','native.prefill_CAPI_call','native.prepare','native.finite_and_flags','native.selection_control'}<={e['name']for e in phase.events};checks+=1
# Real socket transport + actual request_group + actual cohort state/rollback decisions; neural/native maths are mocked.
class Control:
 def collect(self,key,value):return[copy.deepcopy(value)for _ in range(4)]
class FakeNative:
 def __init__(self,trace):self.phase=trace;self.maxC=8;self.maxT=64;self.card_count=33;self.observer=None;self.memory_guard=lambda *a,**k:{};self.state={};self.pending={};self.calls=[]
 def reset(self,state):self.state={};self.pending={}
 def I192(self):return[[0,0]for _ in range(8)]
 def stats(self):return[0]*64
 def rows(self,ids):return[copy.copy(self.state[i])for i in ids]
 def call(self,kind,ids,past,tokens,capture,on_selected=None):
  with self.phase.scope('native.call',kind=kind):
   with self.phase.scope('native.CAPI',kind=kind):
    if kind=='prefill':
     for i,v in zip(ids,tokens):self.state[i]=[i,i+8,len(v),1]
     selected=[7]*len(ids)
    else:selected=[x for v in tokens for x in ([3]+[7]*(len(v)-1)if len(v)>1 else[7])]
   if on_selected:self.phase.call('native.callback_emit',on_selected,selected)
   record={'kind':kind,'CAPI_maxhost_s':0.,'logical':ids,'past':past,'width':list(map(len,tokens))};self.calls.append(record)
   return selected,{i:bytes(len(v)*FEATURE_BYTES)for i,v in zip(ids,tokens)},record
 def commit(self,ids,decisions):
  for i,d in zip(ids,decisions):
   row=self.state[i]
   if d.terminal:row[3]=4
   elif d.needs_replay:row[3]=3;self.pending[i]=(copy.copy(row),len(d.commit_inputs))
   else:self.state[i]=[row[1],row[0],row[2]+len(d.commit_inputs),1]
  return 0.
 def replay(self,partial,state):
  for i,past,retained in partial:
   row,n=self.pending.pop(i);assert n==retained and row[2]==past;self.state[i]=[row[1],row[0],past+retained,1]
  return None,{'CAPI_maxhost_s':0.}
def one(enabled):
 trace=PhaseTrace(0,0);trace.configure(enabled,'cohort-test');n=FakeNative(trace);control=PhaseControl(Control(),trace);a,b=socket.socketpair();a.settimeout(20);b.settimeout(20);errors=[];store=ContextStore(8)
 def serve():
  try:
   while True:
    h,body=recv(b);cmd=h['cmd']
    if cmd=='begin_cohort':store.begin(h['epoch']);reply={'epoch':h['epoch'],'projected_cache':False}
    elif cmd=='query_batch':
     parts=receive_bodies(b,h);items=[]
     for q,part in zip(h['requests'],parts):
      assert len(part)==q['rows']*FEATURE_BYTES;store.append(h['epoch'],q['request'],q['past'],q['rows'],part,lambda old,new:old+new);items.append({'request':q['request'],'context_tokens':q['past']+q['rows'],'proposals':[3]*q['K']})
     reply={'epoch':h['epoch'],'actual_model_batch_C':len(items),'requests':items,'draft_batch_query_s':0.,'append_batch_H2D_s':0.}
    elif cmd=='retire':store.retire(h['epoch'],h['request']);reply={'retired':True}
    elif cmd=='close':send(b,{'seq':h['seq'],'closed':True});break
    else:raise AssertionError(cmd)
    send(b,{'seq':h['seq'],**reply})
  except BaseException as e:errors.append(repr(e))
 worker=threading.Thread(target=serve);worker.start();seq=0
 def rpc(cmd,body=b'',**fields):
  nonlocal seq
  from batch_frames_r9142 import send_query
  start=time.perf_counter();meta={'cmd':cmd,'seq':seq,**fields}
  if cmd=='query_batch':send_query(a,meta,body)
  else:send(a,meta,body)
  h,raw=recv(a);assert h['seq']==seq and not raw;seq+=1;return h,time.perf_counter()-start
 reqs=[Request(1,[11,12,13,14],80,1,{'id':'fixedA'}),Request(2,[11,12,13],70,1,{'id':'fixedB'})]
 out=cohort(n,control,rpc,0,0,reqs,'test',8,'state',True,1,{127},time.monotonic()+30,lambda v:None,False,False);rpc('close');worker.join(10);a.close();b.close();assert not errors and not worker.is_alive()and not store.values
 assert out['phase_observer']['recorded_cycles']==(32 if enabled else 0);assert out['all_serving_GPU_cards']==33;assert all(len(m['native_ready_events'])==m['generated_tokens']for m in out['request_metrics']);assert all(q['terminal']or q['needs_replay']for m in out['request_metrics']for q in m['cycles']);return out
OFF=one(False);ON=one(True);assert [m['generated_ids']for m in OFF['request_metrics']]==[m['generated_ids']for m in ON['request_metrics']];checks+=6
assert ON['phase_observer']['actual_T16_cycles']>0 and not OFF['phase_observer']['events'];assert all(m['EOS']is False and m['truncated']for m in ON['request_metrics']);checks+=2
# no same-node semantic change in verifier selection/commit baseline: greedy source is byte-identical to r123.
base=D.parent/'Kimi-ctx2048-implementation-private-r123/runtime';assert(P/'batch_policy_r9128.py').read_bytes()==(base/'batch_policy_r9128.py').read_bytes();checks+=1
queue=[json.loads(p.read_bytes())for p in sorted((D/'queue').glob('q*.json'))];obs=[q for q in queue if q.get('category')=='observer-C2'];assert len(obs)==8 and len({r['id']for q in obs for r in q['requests']})==8 and [q['phase_observer']for q in obs]==[False,True,True,False,False,True,True,False];main=[q for q in queue if q.get('category')=='ctx2048-MMLU32'];assert len(main)==8 and len({r['id']for q in main for r in q['requests']})==32;assert all(q['cap']==1024 and not q['phase_observer']for q in main);checks+=2
for p in P.glob('*.py'):ast.parse(p.read_bytes());assert b'\r\n'not in p.read_bytes()
assert 'torch'not in sys.modules
result={'passed':True,'CPU_only':True,'GPU':False,'compiled':False,'newjobs':0,'source_uploaded':False,'checks':checks,'actual_Python_Native_ctypes_buffers':'mock CAPI; PF1024, verify2040..2048','real_socket_cohort':'actual request_group+cohort; OFF/ON matching generated mock outputs, 32 captured cycles, ragged EOS/cap/partial/replay/retire','observer_OFF_clockcalls':0,'per_user_emission_timelines_preserved':True,'nested_union_test':True,'neural_correctness_or_performance':'NOT TESTED','actual_C2_T16_GPU_MoE_events':'UNKNOWN; intentionally not enabled','queue_cases_excluding_stop':18};(D/'CPU-contract-r126.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))