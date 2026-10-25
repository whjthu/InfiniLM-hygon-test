"""CPU socket protocol test, synthetic token packets; no model/GPU performance."""
import pathlib,sys,threading,time,json,hashlib,tempfile
r=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(r/'GLM-DP-Mmatrix-private-r8868'))
from glm_DP_common_clock_r8868 import Coordinator,Hub,PHASES
from glm_continuous_feedback_r8855 import FeedbackClient
out=r/'GLM-DP-synthetic-clock-CPU-r8868.json';co=Coordinator(timeout=15,output=out);hubs={a:Hub('127.0.0.1',0,'CPU-nonce-'+a,participants=16,emitter=8,timeout=10,idle_timeout=60,coordinator=co,replica=a)for a in('A','B')};barriers={a:threading.Barrier(16,timeout=15)for a in hubs};joint_emitters=threading.Barrier(2,timeout=15);errors=[]
def owner(a,rank):
 client=None
 try:
  client=FeedbackClient('127.0.0.1',hubs[a].address[1],'CPU-nonce-'+a,rank,timeout=20)
  for phase in PHASES:
   micro=16 if phase=='jointM16'else 8;cohorts=[list(range(x,min(x+micro,16)))for x in range(0,16,micro)]
   ticket=client.request('dp_gate','DP-control',phase=phase)
   if ticket['run']:
    case='CPU-'+a+'-'+phase
    if rank==8:client.request('begin',case,cohorts=cohorts)
    barriers[a].wait()
    if rank==8:
     client.request('dp_service_start',case,phase=phase)
     time.sleep(.02) # Windows CPU-test clock may have a coarse monotonic tick.
     for rows in [list(range(8)),list(range(8,16))]:client.request('dp_prefill_rows',case,phase=phase,rows=rows,ids=[1]*8,token_index=0)
     client.request('dp_prefill_complete',case,phase=phase)
     time.sleep(.02)
     joint_emitters.wait()
     for step in range(2):
      for cohort in range(len(cohorts)):
       rows=cohorts[cohort];client.request('publish',case,cohort=cohort,step=step,packet={'rows':rows,'ids':[2+step]*len(rows),'stamp':float(step+1),'seconds':.001})
       time.sleep(.01)
    else:
     for step in range(2):
      for cohort in range(len(cohorts)):assert client.request('take',case,cohort=cohort,step=step)['ids']==[2+step]*len(cohorts[cohort])
    barriers[a].wait()
    if rank==8:client.request('end',case)
    barriers[a].wait()
   assert client.request('dp_done','DP-control',phase=phase,healthy=True)['all32_done']
 except BaseException as e:
  errors.append((a,rank,repr(e)));co.fail(repr(e))
  for b in barriers.values():b.abort()
 finally:
  if client:client.close()
threads=[threading.Thread(target=owner,args=(a,i))for a in hubs for i in range(16)]
try:
 for t in threads:t.start()
 for t in threads:t.join(30)
 assert not errors,errors;assert all(not t.is_alive()for t in threads);assert co.finished and co.failure is None;assert all(len(co.done[p])==32 for p in PHASES)
 assert len(co.records)==20
 intervals={}
 for a in('A','B'):
  values=[x['arrival_monotonic_ns']for x in co.records if x['phase']=='jointM16'and x['replica']==a and x['token_index']>0];intervals[a]=(min(values),max(values))
 assert max(v[0]for v in intervals.values())<min(v[1]for v in intervals.values()),'synthetic joint decode must overlap'
 negative=Coordinator(timeout=.01)
 try:negative.complete('A',0,'jointM8',True)
 except AssertionError:pass
 else:raise AssertionError('completion before gate accepted')
 negative.fail('test-abort')
 try:negative.gate('A',0,'jointM8')
 except RuntimeError:pass
 else:raise AssertionError('failed coordination continued')
 gold=json.loads((r/'GLM-exact-budget-Model-private-r8855/GLM477-frozen-gold-r8844.json').read_bytes());maps={'A':list(range(16)),'B':list(range(16,32))};assert set(maps['A']).isdisjoint(maps['B'])and sorted(maps['A']+maps['B'])==list(range(32))
 for a,rows in maps.items():
  assert all(len(gold['input_ids'][i])==512 for i in rows)
  assert len({tuple(gold['input_ids'][i])for i in rows})==16
 result={'CPU_only':True,'GPU':False,'dlopen':False,'passed':True,'actual_feedback_clients':32,'independent_ports':[h.address[1]for h in hubs.values()],'independent_nonce':True,'phase_gate_done_32_each':True,'jointM8_jointM16_noisolated':True,'joint_synthetic_arrival_overlap':True,'invalid_phase_and_abort_rejected':True,'bounded_synthetic_packet_records':len(co.records),'fixture_global_rows':maps,'original_gold_SHA256':hashlib.sha256((r/'GLM-exact-budget-Model-private-r8855/GLM477-frozen-gold-r8844.json').read_bytes()).hexdigest(),'scope':'Actual existing FeedbackHub/Client socket flow; synthetic3token packets only, not Model/numerical/performance evidence.'}
 (r/'GLM-DP-protocol-CPUchecks-r8868.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
finally:
 co.save()
 for h in hubs.values():h.close()
