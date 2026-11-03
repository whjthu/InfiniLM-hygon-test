"""Actual C1 original ordinary forward: input rows=1, flags3, no speculative state."""
import ctypes as C
import hashlib,json,time
import numpy as np
VOCAB=163840
def ids_sha(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
class Native:
 def __init__(self,lib,ctx,control,node,gate,report,save):
  self.lib=lib;self.ctx=ctx;self.control=control;self.node=node;self.gate=gate;self.report=report;self.save=save;self.serial=0
 def check(self,rc):
  if rc:raise RuntimeError(self.lib.modern_kimi_engine_error_r2475().decode())
 def boundary(self,label):
  status='ok'
  try:self.gate.check(label)
  except BaseException as e:status=type(e).__name__
  rows=self.control.collect('r410-gate:'+str(self.serial)+':'+label,dict(node=self.node,status=status))
  if len(rows)!=4 or {r['node']for r in rows}!=set(range(4))or any(r['status']!='ok'for r in rows):raise RuntimeError('Four leaders stopped at original deadline/cancel boundary')
 def reset(self,label):
  self.boundary('reset:'+label);self.check(self.lib.modern_kimi_engine_reset_r2475(self.ctx))
  self.control.collect('r410-reset:'+label,dict(node=self.node,normal=True))
 def call(self,kind,past,tokens):
  if type(kind)is not int or kind not in [0,1]or type(past)is not int or past<0:raise ValueError('ordinary kind/past')
  if not tokens or len(tokens)>(512 if kind==0 else 1)or past+len(tokens)>32768 or(kind==1 and past==0):raise ValueError('ordinary C1 prompt512/decode1/32768 scope')
  if any(type(v)is not int or not 0<=v<VOCAB for v in tokens):raise ValueError('Actual vocabulary IDs')
  self.boundary('forward')
  rows=np.asarray([1],np.int32);pp=np.asarray([past],np.int32);total=np.asarray([past+len(tokens)],np.int32);offset=np.asarray([0,len(tokens)],np.int32)
  ids=np.asarray(tokens,np.int64);positions=np.arange(past,past+len(tokens),dtype=np.int64)
  logits=np.full((1,VOCAB),0x7fc0,np.uint16);selected=np.full(1,-1,np.int64);flags=np.asarray([-1],np.int32)
  buffers=[rows,pp,total,offset,ids,positions,logits,selected,flags]
  tick=time.perf_counter();rc=self.lib.modern_kimi_engine_forward_ordinary_r410(self.ctx,kind,len(tokens),1,3,*[C.c_void_p(a.ctypes.data)for a in buffers]);wall=time.perf_counter()-tick
  # Persist a completed CAPI result before any numerical/output assertion.
  finite=bool(not np.any((logits&0x7f80)==0x7f80))if self.node==3 else None
  record=dict(serial=self.serial,kind=kind,past=past,tokens=len(tokens),input_ID_SHA256=ids_sha(tokens),CAPI_return=int(rc),local_CAPI_wall_s=wall,published_flags=int(flags[0]),tail_logits_finite=finite)
  self.report.setdefault('ordinary_calls',[]).append(record);self.save();self.check(rc)
  packet=dict(node=self.node,wall=wall,selected=selected.tolist()if self.node==3 else None,flags=int(flags[0]),finite=finite)
  packets=self.control.collect('r410-call:'+str(self.serial),packet);self.serial+=1
  assert len(packets)==4 and {p['node']for p in packets}==set(range(4))
  assert all(p['flags']==(3 if p['node']==3 else 0)for p in packets)and packets[3]['finite']is True
  output=packets[3]['selected'];assert len(output)==1 and type(output[0])is int and 0<=output[0]<VOCAB
  record.update(CAPI_max4leader_s=max(p['wall']for p in packets),selected=output[0],validated=True);self.save()
  return output[0]
def run(native,fixtures,grade,save,clock=time.perf_counter):
 result=dict(completed=False,requests=[],target_cards=32,draft_cards=0,HTTP=False,client_SLO_qualified=False,topology='TP8PP4 C1 ordinary',speculative_calls=0)
 for ordinal,f in enumerate(fixtures):
  ids=f['prompt_tokens'];assert len(ids)==f['input_tokens']and ids_sha(ids)==f['prompt_ID_SHA256']and f['output_budget']==128 and len(ids)+128<=32768
  native.reset('before-'+str(ordinal));started=clock();generated=[];events=[]
  row=dict(ordinal=ordinal,name=f['name'],input_tokens=len(ids),prompt_ID_SHA256=f['prompt_ID_SHA256'],output_budget=128,completed=False,normal_reset=False,events=events)
  result['requests'].append(row);save(result)
  first=None
  for past in range(0,len(ids),512):first=native.call(0,past,list(ids[past:past+512]))
  token=first;past=len(ids)
  while True:
   generated.append(token);events.append(dict(at_s=clock()-started,token=token));row.update(generated_ids=list(generated),output_ID_SHA256=ids_sha(generated));save(result)
   if token in f['eos']or len(generated)>=128:break
   token=native.call(1,past,[token]);past+=1
  finish='eos'if token in f['eos']else'length';row.update(finish=finish,full_request_s=clock()-started,prompt_consumed=len(ids),last_past=past)
  quality=grade(f,generated,finish);row['quality']=quality;save(result)
  assert quality['passed']and finish=='eos'
  native.reset('after-'+str(ordinal));row.update(completed=True,normal_reset=True);save(result)
 result['completed']=True;save(result);return result
