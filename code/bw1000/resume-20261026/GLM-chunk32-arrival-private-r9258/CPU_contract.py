"""Actual frontend CPU causal contract. Fake durations/tokens are NOT GPU performance/quality."""
import pathlib,json,threading,copy,hashlib,weakref,sys,traceback
from arrival_runner_r9258 import *
p=pathlib.Path(__file__).parent;corpus=json.loads((p/'corpus-reference.json').read_bytes());trace=json.loads((p/'arrival-trace.json').read_bytes());requests=corpus['requests'];arrivals=[r['arrival_seconds']for r in trace['requests']];eos=corpus['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list)else[eos]);eostoken=min(eos)
class Clock:
 def __init__(self):self.t=0.
 def now(self):return self.t
 def advance(self,v):self.t+=v
class Logits:
 def __init__(self,values,torch=None):self.values=values;self.torch=torch
 def argmax(self,dim):return self.torch.tensor(self.values,dtype=self.torch.int64)
class World:
 def __init__(self,torch=None):
  self.slots=[None]*16;self.result=None;self.torch=torch;self.enter=threading.Barrier(16,timeout=60,action=self.reduce);self.leave=threading.Barrier(16,timeout=60);self.n=0
 def reduce(self):
  self.n+=1
  if self.torch:self.result=self.torch.stack(self.slots).sum(dim=0)
  else:self.result=[sum(v[i]for v in self.slots)for i in range(len(self.slots[0]))]
 def exchange(self,pid,value):
  self.slots[pid]=value;self.enter.wait();v=self.result.clone()if self.torch else self.result[:];self.leave.wait();return v
 def abort(self):self.enter.abort();self.leave.abort()
class NativeMirror:
 def __init__(self,pid,clock,req,torch=None):
  self.pid=pid;self.owner=threading.get_ident();self.clock=clock;self.req=req;self.torch=torch;self.length=[0]*32;self.missing=[False]*32;self.produced=[0]*32;self.previous=[None]*32;self.completed=set();self.calls=[];self.live=weakref.WeakSet();self.peak=0
 def forward(self,rows,tokens,pasts,label,prefill):
  assert threading.get_ident()==self.owner and 0<len(rows)<=(8 if prefill else 16)and len(set(rows))==len(rows)and 0<sum(map(len,tokens))<=4096
  op=label.split('.')[-1];assert prefill==(op!='DECODE');allone=all(len(v)==1 for v in tokens)
  for i,vals,past in zip(rows,tokens,pasts):
   assert i not in self.completed and past==self.length[i]and past+len(vals)<=640
   if not allone and self.missing[i]:raise RuntimeError('source1406 incomplete indexer cache')
   if op=='PREFIX':assert vals==self.req[i]['input_ids'][past:past+len(vals)]and past+len(vals)<=len(self.req[i]['input_ids'])-1
   elif op=='ANCHOR':assert past==len(self.req[i]['input_ids'])-1 and vals==[self.req[i]['input_ids'][-1]]and self.produced[i]==0
   else:assert self.produced[i]>0 and vals==[self.previous[i]]and past>=len(self.req[i]['input_ids'])
  values=[]
  for i,vals in zip(rows,tokens):
   self.length[i]+=len(vals)
   if allone:self.missing[i]=True
   if op=='PREFIX':values.append(101+i);continue
   # Two cappedrequests prove they are counted rather thanhidden; others naturalEOS.
   want=300 if i in [5,17]else 3+i%7;k=self.produced[i];value=eostoken if k>=want else 1000+i*300+k;values.append(value);self.produced[i]+=1;self.previous[i]=value
   if value in eos or self.produced[i]>=self.req[i]['max_new_tokens']:self.completed.add(i)
  self.calls.append((op,tuple(rows),tuple(pasts),tuple(map(len,tokens))))
  # Same declared synthetic duration equation in botharms, not hardwaremodel.
  duration=.05+.01*sum(map(len,tokens))
  if self.pid==8:self.clock.advance(duration)
  out=Logits(values,self.torch)
  self.live.add(out);self.peak=max(self.peak,len(self.live));return out,duration,0.,duration

def episode(mode,req,arrival,torch=None,budget=5000):
 world=World(torch);clock=Clock();reports=[None]*16;logs=[None]*16;backends=[None]*16;errors=[]
 def worker(pid):
  try:
   native=NativeMirror(pid,clock,req,torch);backends[pid]=native
   if torch:
    def control(a):a.copy_(world.exchange(pid,a.clone()))
    cm,tm=torch_world_callbacks(torch,control,pid==8,device='cpu')
   else:
    def cm(cmd,seq):return decode_command(world.exchange(pid,encode_command(cmd)),seq)
    def tm(cmd,out):
     v=[0]*19
     if pid==8:v[:3]=[MAGIC,cmd['seq'],len(cmd['items'])];v[3:3+len(cmd['items'])]=out.values
     z=world.exchange(pid,v);assert z[:3]==[MAGIC,cmd['seq'],len(cmd['items'])];return list(map(int,z[3:3+len(cmd['items'])]))
   report,ops=run_episode(req,arrival,mode,eos,pid==8,cm,tm,native.forward,clock.now,lambda dt:clock.advance(dt)if pid==8 else None,budget)
   reports[pid]=report;logs[pid]=ops
  except BaseException as e:errors.append({'owner':pid,'error':repr(e),'traceback':traceback.format_exc()});world.abort()
 threads=[threading.Thread(target=worker,args=(i,))for i in range(16)]
 for t in threads:t.start()
 for t in threads:t.join(150)
 assert not any(t.is_alive()for t in threads)and not errors,errors
 assert all(v==logs[0]for v in logs);assert all(v.calls==backends[0].calls for v in backends)
 report=reports[8];assert report is not None
 for b in backends:
  assert b.peak<=1 and not list(b.live)
  for i,r in enumerate(report['requests']):
   assert b.length[i]==r['past']
   if r['generated_ids']:assert b.length[i]==len(req[i]['input_ids'])+len(r['generated_ids'])-1
 for r in report['requests']:
  if r['admitted']is not None:assert r['admitted']>=r['arrival']
  if r['service_start']is not None:assert r['service_start']>=r['admitted']
  if r['events']:assert r['events'][0]['time']>=r['arrival']and r['TTFT_seconds']==r['events'][0]['time']-r['arrival']
 assert report['peak_admitted_live']<=16
 return report,logs[0],backends[0].calls,world.n

if __name__=='__main__':
 use_torch='--torch-world'in sys.argv
 torch=None
 if use_torch:
  import torch
  torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 original=copy.deepcopy(requests);results=[];outputs=[]
 for mode in [False,True]:
  r,ops,calls,n=episode(mode,requests,arrivals,torch);assert r['all_requests_completed']and r['arrivals']==32 and r['not_arrived']==0 and r['unfinished']==0 and r['global_reset_allowed'];assert sum(q['stop_reason']=='max_new_tokens'for q in r['requests'])==2
  for req,q in zip(requests,r['requests']):assert q['row']<32 and q['past']<=640 and len(q['generated_ids'])<=req['max_new_tokens']
  outputs.append([q['generated_ids']for q in r['requests']]);results.append({'policy':'chunk32-interleave'if mode else'wholeprefill-priority','full_corpus32_uncropped':True,'all32_completed':True,'EOS_count':sum(q['stop_reason']=='EOS'for q in r['requests']),'cap_count':sum(q['stop_reason']=='max_new_tokens'for q in r['requests']),'peak_live':r['peak_admitted_live'],'CAPI_calls':len(calls),'WORLD_calls':n,'row0_first_commands':ops[:8],'requests':r['requests'],'FAKE_clock_not_GPUperformance':True,'all16_owner_trace_andmetadata_identical':True})
 assert outputs[0]==outputs[1]and original==requests
 # Synthetic one-token prefix boundaries, separate fromnoncroppedcorpus above.
 tiny=[dict(r,input_ids=[11+i]*[1,2,33,34][i%4],max_new_tokens=3)for i,r in enumerate(requests)]
 for mode in [False,True]:
  r,ops,calls,n=episode(mode,tiny,arrivals,torch);assert r['all_requests_completed']
 # Budget preserves arrived-butunfinished and futureclients; no resetallowed.
 r,ops,calls,n=episode(True,requests,arrivals,torch,budget=.01);assert not r['all_requests_completed']and not r['global_reset_allowed']and r['arrivals']<32 and r['not_arrived']>0 and r['unfinished']>0
 # Source1406: an olddecoded row cannot be included inrealindexer mixedprefill.
 mirror=NativeMirror(8,Clock(),requests);ids=requests[0]['input_ids'];mirror.forward([0],[ids[:-1]],[0],'arrival.PREFIX',True);mirror.forward([0],[[ids[-1]]],[len(ids)-1],'arrival.ANCHOR',True)
 try:mirror.forward([0,1],[[mirror.previous[0]],requests[1]['input_ids'][:2]],[len(ids),0],'arrival.MIXED',True);raise AssertionError('mixedoldreadyaccepted')
 except RuntimeError as e:assert 'incomplete indexer' in str(e)
 # Wire/source API negatives fail beforedispatch.
 for bad in [0,1,None,'1']:
  try:Leader(requests,arrivals,bad,eos);raise AssertionError('nonboolaccepted')
  except TypeError:pass
 c={'seq':1,'op':'PREFIX','time_ms':0,'wait_ms':0,'items':[(0,len(requests[0]['input_ids']),2,0)]}
 try:payload(c,requests);raise AssertionError('mixedolddecodeinP')
 except ValueError:pass
 vec=encode_command({'seq':1,'op':'DECODE','time_ms':0,'wait_ms':0,'items':[(0,100,1,2)]});vec[0]+=1
 try:decode_command(vec,1);raise AssertionError('badmagic')
 except ValueError:pass
 q={'passed':True,'scope':'CPU_ONLY_ACTUAL_SCHEDULER_16OWNER_WIRE_AND_NATIVE_CONTRACT_MIRROR_NOT_GPU_CCL_OR_QUALITY','Torch_world_actual':use_torch,'Torch_CUDA_initialized':False,'source_SHA256':hashlib.sha256((p/'arrival_runner_r9258.py').read_bytes()).hexdigest(),'arrival_trace_SHA256':hashlib.sha256((p/'arrival-trace.json').read_bytes()).hexdigest(),'actual_program_source_SHA256':corpus['source_SHA256'],'results':results,'synthetic_boundary_cases':2,'budget_future_queued_explicit':True,'missingcache_negative':True,'same_fake_tokens_only_validatesrequestmapping_not_realquality':True,'fake_logits_preselected_not_actual_head':True}
 target=p/('CPU-torch-result.json'if use_torch else'CPU-stdlib-result.json');target.write_text(json.dumps(q,indent=2));print(json.dumps({'passed':True,'Torch_world_actual':use_torch,'owners':16,'policies':2,'boundary_cases':2,'source_SHA256':q['source_SHA256']}))
