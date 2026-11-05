import hashlib,time
import numpy as np
CASES=[('past0',0,512),('tile3072',3071,512),('maxpast',32255,512),('continuation1',32767,1),('reset_reuse',0,512)]
ORDER=[0,1,1,0]
def run_schedule(api,data,event):
 q,k,v=data;records=[];api.create();close=None
 try:
  for phase,mode in enumerate(ORDER):
   table=np.arange(1,129,dtype=np.int32);event('phase.before',phase,mode,None);api.begin(mode,table);event('phase.allocated',phase,mode,api.observe())
   for index,(name,past,count)in enumerate(CASES):
    if index in [1,2,4]:
     if index==4:table=table[::-1].copy()
     api.reset(table)
    if index!=3:api.seed(past,k[:past],v[:past])
    before=api.observe();event('query.before',phase,mode,dict(case=name,census=before));start=time.monotonic()
    out,seconds=api.query(past,q[past:past+count],k[past:past+count],v[past:past+count]);elapsed=time.monotonic()-start
    after=api.observe();event('query.after',phase,mode,dict(case=name,census=after));guards=api.audit();event('guard.after',phase,mode,dict(case=name,guardpages=guards))
    record=dict(phase=phase,mode=mode,case=name,past=past,count=count,seconds=seconds,Python_call_s=elapsed,output=out,output_SHA256=hashlib.sha256(out.tobytes()).hexdigest(),before=before,after=after,guardpages=guards);records.append(record)
   event('phase.complete',phase,mode,api.observe())
 finally:close=api.close();event('Runtime.closed',None,None,close)
 return records,close
