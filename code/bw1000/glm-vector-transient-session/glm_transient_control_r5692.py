"""Model owner control only; import does not initialize GPU."""
import ctypes,json,hashlib,threading
GiB=1024**3;RESERVE=6*GiB;FLOOR=512*1024**2;PHYSICAL=64*1024**2;ALLOWANCE=64*1024**2;BOUNDS=(0,38,78)
class CandidateRejected(RuntimeError):pass
class TransientControl:
 def __init__(self,lib,ctx,stage,rank,control,check,report,save):
  import torch
  self.torch=torch;self.lib=lib;self.ctx=ctx;self.stage=stage;self.rank=rank;self.control=control;self.check=check;self.report=report;self.save=save;self.participant=stage*8+rank;self.epoch=0;self.count=0
  lib.modern_glm_model_transient_policy_r5688.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_model_transient_report_r5688.argtypes=[ctypes.c_void_p];lib.modern_glm_model_transient_report_r5688.restype=ctypes.c_char_p
  lib.modern_glm_idle_trim_r3771.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_glm_projected_host_counts_r4504.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  q=self.snapshot();assert not q['enabled']and q['GPU_logical_bytes']==0 and q['calls']==0;self.runtime=q['Runtime_identity'];report['transient_default_OFF_no_Scratch_alloc']=True
 def snapshot(self):
  p=self.lib.modern_glm_model_transient_report_r5688(self.ctx);assert p;q=json.loads(p);assert(q['rows'],q['capacity'],q['PP'],q['stage'])==(64,640,2,self.stage)
  a=(ctypes.c_uint64*7)();self.check(self.lib.modern_glm_projected_host_counts_r4504(self.rank,a,7));q['host_counters']=list(a);return q
 def all_rows(self,values):
  assert all(isinstance(v,int)and 0<=v<2**24 for v in values);x=self.torch.zeros((16,len(values)+1),device='cuda',dtype=self.torch.float32);x[self.participant]=self.torch.tensor([1]+values,device='cuda',dtype=self.torch.float32);self.control(x);rows=x.cpu().tolist();assert all(a[0]==1 for a in rows);return[list(map(int,a[1:]))for a in rows]
 def vote(self,ok):return all(a[0]==1 for a in self.all_rows([int(bool(ok))]))
 def setter(self,on):return self.lib.modern_glm_model_transient_policy_r5688(self.ctx,int(on))
 def contract_guards(self):
  # Caller public0 leaves policy prerequisites disabled; ON must reject EMPTY
  # before any allocation, while OFF remains safe. No invalid GPU pointers.
  before=self.snapshot();bad=self.setter(2);dependency=self.setter(1);foreign=[];thread=threading.Thread(target=lambda:foreign.append(self.setter(0)));thread.start();thread.join(15)
  assert not thread.is_alive()and bad!=0 and dependency!=0 and foreign==[1]and self.snapshot()==before
  self.report['transient_Model_guard_checks']={'invalid_value':bad,'prerequisite_rejected':dependency,'foreign_owner':foreign[0],'state_counters_unchanged':True,'GPUfault':'NOT_RUN'}
 def apply(self,label,on,trace):
  self.epoch+=1;self.check(self.setter(False));trim=(ctypes.c_uint64*6)();self.check(self.lib.modern_glm_idle_trim_r3771(self.ctx,trim,6));self.torch.cuda.synchronize(self.rank);self.torch.cuda.empty_cache();free=self.torch.cuda.mem_get_info(self.rank)[0];phase=int(hashlib.sha256(label.encode()).hexdigest()[:6],16)
  rows=self.all_rows([free//1024**2,self.epoch,phase,int(on),int(trace)]);assert all(a[1:]==rows[0][1:]for a in rows)
  if not self.vote(not on or free>=RESERVE+PHYSICAL+ALLOWANCE):raise CandidateRejected('transient original6GiB plus physical64MiB plus64MiB allowance')
  code=self.setter(on);statuses=self.all_rows([int(code==0),self.epoch,phase,int(on)])
  if not all(a[0]==1 for a in statuses):
   self.check(self.lib.modern_glm_model_reset(self.ctx));self.check(self.setter(False));self.report['transient_setter_rejected']=statuses;self.save();raise CandidateRejected('allowner transient setter rejected while EMPTY')
  q=self.snapshot();assert q['Runtime_identity']==self.runtime and q['enabled']==bool(on)and not q['busy']and not q['poison'];actual=self.torch.cuda.mem_get_info(self.rank)[0]
  if not self.vote(actual>=RESERVE+ALLOWANCE):
   self.check(self.lib.modern_glm_model_reset(self.ctx));self.check(self.setter(False));raise CandidateRejected('transient postallocation original6GiB plus64MiB allowance')
  if on:assert q['GPU_logical_bytes']==41943040 and q['pinned_slab_bytes']>=65536
  self.count=(BOUNDS[self.stage+1]-BOUNDS[self.stage])if on else 0
  self.report.setdefault('transient_policy_phases',[]).append({'phase':label,'on':on,'trace':trace,'epoch':self.epoch,'fresh_free':free,'free_after_admission':actual,'allowner_intent':rows,'allowner_status':statuses,'native':q,'scope':'trace flag controls existing OwnerProfile only; no claim of allocator peak'});self.save();return q
 def finish(self,label,before,events,prompts):
  q=self.snapshot();assert q['Runtime_identity']==self.runtime and q['enabled']==before['enabled']and not q['busy']and not q['poison'];L=BOUNDS[self.stage+1]-BOUNDS[self.stage]
  full=sum(e['round']>0 and len(e['rows'])==8 for e in events);expected_dispatch=L*full if before['enabled']else 0;assert q['calls']-before['calls']==expected_dispatch
  d=[b-a for a,b in zip(before['host_counters'],q['host_counters'])]
  expected=[2*L*sum(len(e['rows'])for e in events),8192*L*sum(len(prompts[i])+e['round']for e in events for i in e['rows']),L*sum(2*len(e['rows'])-(14 if before['enabled']and e['round']>0 and len(e['rows'])==8 else 0)for e in events),8192*L*sum(e['flat_tokens']for e in events)]
  assert d[:3]==[0,0,0]and d[3:]==expected,('transient exacthostbank contract',label,d,expected)
  free=self.torch.cuda.mem_get_info(self.rank)[0];assert self.vote(free>=FLOOR),'observed afterphase512MiB floor'
  self.report.setdefault('transient_phase_after',[]).append({'phase':label,'native':q,'expected_staging_dispatch':expected_dispatch,'host_counter_delta':d,'expected_host_counter_delta_tail':expected,'free_after':free,'observed_is_peak':False});self.save();return q
