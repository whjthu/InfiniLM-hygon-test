"""Two existing GLM CPU feedback hubs in one process; no Torch/native/GPU imports."""
import json,pathlib,threading,time,statistics,os,secrets,socket
from glm_continuous_feedback_r8855 import FeedbackHub as OriginalHub
PHASES=('isoA','isoB','joint')
class Coordinator:
 def __init__(self,participants=16,timeout=1200,output=None,eos=()):
  self.n=participants;self.timeout=timeout;self.cv=threading.Condition();self.arrived={p:set()for p in PHASES};self.done={p:set()for p in PHASES};self.start={};self.service_start={};self.records=[];self.completions={};self.next_index={};self.failure=None;self.finished=False;self.output=pathlib.Path(output)if output else None;self.eos=set(eos)
 def runners(self,p):return ('A','B')if p=='joint'else(p[-1],)
 def fail(self,message):
  with self.cv:
   if not self.finished and self.failure is None:self.failure=str(message)
   self.cv.notify_all()
 def wait(self,predicate):
  deadline=time.monotonic()+self.timeout
  while not predicate():
   if self.failure:raise RuntimeError(self.failure)
   remaining=deadline-time.monotonic()
   if remaining<=0:self.fail('DP coordination timeout');raise TimeoutError(self.failure)
   self.cv.wait(remaining)
  if self.failure:raise RuntimeError(self.failure)
 def gate(self,replica,rank,phase):
  with self.cv:
   assert phase in PHASES and replica in('A','B')and 0<=rank<self.n
   index=PHASES.index(phase)
   if index:assert len(self.done[PHASES[index-1]])==2*self.n,'phase order'
   who=(replica,rank);assert who not in self.arrived[phase];self.arrived[phase].add(who)
   if len(self.arrived[phase])==2*self.n:self.start[phase]=time.monotonic_ns();self.cv.notify_all()
   self.wait(lambda:len(self.arrived[phase])==2*self.n)
   return {'run':replica in self.runners(phase),'phase':phase,'common_gate_release_ns':self.start[phase]}
 def complete(self,replica,rank,phase,healthy):
  with self.cv:
   assert healthy is True and(replica,rank)in self.arrived[phase]
   who=(replica,rank);assert who not in self.done[phase];self.done[phase].add(who)
   if len(self.done[phase])==2*self.n:
    if phase=='joint':self.finished=True
    self.save();self.cv.notify_all()
   self.wait(lambda:len(self.done[phase])==2*self.n)
   return {'phase':phase,'all32_done':True}
 def observe(self,replica,rank,req,arrival_ns):
  with self.cv:
   phase=req['phase'];assert phase in PHASES and replica in self.runners(phase)and rank==8
   assert len(self.arrived[phase])==2*self.n and len(self.done[phase])<2*self.n
   op=req['op']
   if op=='dp_service_start':
    key=(phase,replica);assert key not in self.service_start;self.service_start[key]=arrival_ns;return {'observed':True}
   if op=='dp_prefill_complete':
    key=(phase,replica);assert key not in self.completions
    seen={row for e in self.records if e['phase']==phase and e['replica']==replica and e['token_index']==0 for row in e['rows']};assert seen==set(range(16))
    self.completions[key]=arrival_ns;return {'observed':True}
   rows=req['rows'];ids=req['ids'];assert rows and len(rows)==len(ids)and len(rows)<=8 and len(rows)==len(set(rows))and all(type(x)is int and 0<=x<16 for x in rows)and all(type(x)is int and x>=0 for x in ids)
   index=req['token_index'];assert type(index)is int and 0<=index<128
   assert all(self.next_index.get((phase,replica,row),0)==index for row in rows),'duplicate/skipped observer decision'
   for row in rows:self.next_index[(phase,replica,row)]=index+1
   assert len(self.records)<4096,'bounded collector token packets'
   self.records.append({'phase':phase,'replica':replica,'rows':list(rows),'global_rows':[x+(0 if replica=='A'else 16)for x in rows],'ids':list(ids),'token_index':index,'arrival_monotonic_ns':arrival_ns,'accepted_monotonic_ns':time.monotonic_ns()})
   return {'observed':True}
 def save(self):
  if self.output:
   q={'CPU_only':True,'GPU':False,'clock':'one collector process monotonic_ns; successful packet arrival before hub validation/pending wait; includes network/observer costs','finished':self.finished,'failure':self.failure,'participants_per_replica':self.n,'gate_release_monotonic_ns':self.start,'service_start_monotonic_ns':{p+':'+a:v for(p,a),v in self.service_start.items()},'phase_arrived':{p:len(v)for p,v in self.arrived.items()},'phase_done':{p:len(v)for p,v in self.done.items()},'prefill_complete_monotonic_ns':{p+':'+a:v for(p,a),v in self.completions.items()},'records':self.records}
   raw=(json.dumps(q,separators=(',',':'))+'\n').encode();assert len(raw)<=8*1024**2;tmp=self.output.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(self.output)
class Hub(OriginalHub):
 def __init__(self,*args,coordinator,replica,**kwargs):self.dp=coordinator;self.replica=replica;self.phase=None;super().__init__(*args,**kwargs)
 def fail_case(self,message):
  super().fail_case(message);self.dp.fail(self.replica+':'+str(message))
 def dispatch(self,rank,req):
  arrival=time.monotonic_ns();op=req.get('op')
  if op=='dp_gate':
   result=self.dp.gate(self.replica,rank,req['phase']);self.phase=req['phase'];return result
  if op=='dp_done':return self.dp.complete(self.replica,rank,req['phase'],req['healthy'])
  if op in('dp_service_start','dp_prefill_rows','dp_prefill_complete'):return self.dp.observe(self.replica,rank,req,arrival)
  result=super().dispatch(rank,req)
  if op=='publish':
   packet=req['packet'];self.dp.observe(self.replica,rank,{'op':'dp_decode','phase':self.phase,'rows':packet['rows'],'ids':packet['ids'],'token_index':req['step']+1},arrival)
  return result
def collector_main():
 r=pathlib.Path('/refactor');run=os.environ['GLM_RUN_R5040'];producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];host=os.environ['GLM_DP_COLLECTOR_HOST_R8864'];assert socket.gethostname()==host
 allowed=sorted(os.sched_getaffinity(0));chosen=allowed[-4:];os.sched_setaffinity(0,chosen)
 out=r/'results'/('GLM-DP-common-clock-r8864-'+run+'.json');assert not out.exists();co=Coordinator(timeout=3500,output=out);hubs=[]
 try:
  for a in('A','B'):
   nonce=secrets.token_hex(24);hub=Hub('0.0.0.0',0,nonce,participants=16,emitter=8,timeout=180,idle_timeout=7200,coordinator=co,replica=a);hubs.append(hub)
   ep=r/'control'/('glm-DP-r8864-'+run+'-'+a+'-endpoint.json');assert not ep.exists();ep.write_text(json.dumps({'host':host,'port':hub.address[1],'nonce':nonce,'participants':16,'emitter':8,'replica':a,'common_collector':True,'common_run':run}))
  ready=r/'results'/('GLM-DP-collector-ready-r8864-'+run+'.json');ready.write_text(json.dumps({'ready':True,'CPU_only':True,'GPU':False,'PID':os.getpid(),'hostname':host,'affinity':chosen,'run':run,'producer':producer,'ports':[h.address[1]for h in hubs]}))
  deadline=time.monotonic()+3500
  while time.monotonic()<deadline:
   if co.failure:raise RuntimeError(co.failure)
   if co.finished and all(not h.active_peers for h in hubs):break
   time.sleep(.2)
  else:raise TimeoutError('bounded collector lifetime')
 finally:
  co.save()
  for h in hubs:h.close()
if __name__=='__main__':collector_main()
