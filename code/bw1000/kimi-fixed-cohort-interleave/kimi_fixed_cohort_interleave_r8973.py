"""CPU-ready fixed-M16 cohort admission/interleave; no Engine/launcher/GPU.
Reuses r4915 FeedbackHub unchanged. All owners execute immutable same jobs.
Only the fixture-qualified no-EOS full128 scope is supported (fail closed).
"""
import time,hashlib,json

def jobs(B=64,M=16,limit=128):
 assert B==64 and M==16 and limit==128
 plan=[];done=[0]*(B//M)
 for p in range(0,B,2):
  plan.append(('prefill',tuple(range(p,p+2))))
  for c in range((p+2)//M):
   if done[c]<limit-1:plan.append(('decode',tuple(range(c*M,(c+1)*M))));done[c]+=1
 while any(v<limit-1 for v in done):
  for c in range(B//M):
   if done[c]<limit-1:plan.append(('decode',tuple(range(c*M,(c+1)*M))));done[c]+=1
 assert sum(k=='prefill'for k,_ in plan)==32 and sum(k=='decode'for k,_ in plan)==508
 return plan

def run(*,inputs,feedback,case,is_emitter,forward,eos,clock=time.perf_counter):
 assert len(inputs)==64 and all(len(x)==512 for x in inputs)
 plan=jobs();ids=[[]for _ in inputs];pasts=[0]*64;stamps=[[]for _ in inputs];publish=[0]*4;read_next=[0]*4;pending_prefill={};start=clock()
 def accept(packet,c):
  rows=list(range(c*16,(c+1)*16));assert packet['rows']==rows
  ts=packet.get('row_stamps',[packet['stamp']]*16)
  for i,v,t in zip(rows,packet['ids'],ts):
   assert type(v)is int and v>=0 and v not in eos,'unsupported_EOS_scope_STOP'
   if packet['kind']=='prefill':assert not ids[i]and pasts[i]==0;pasts[i]=512
   else:assert ids[i]and len(ids[i])<128;pasts[i]+=1
   ids[i].append(v);stamps[i].append(t);assert pasts[i]==511+len(ids[i])
 def publish_packet(c,kind,rows,selected,ts,seconds):
  packet={'kind':kind,'rows':rows,'ids':selected,'stamp':max(ts),'row_stamps':ts,'seconds':seconds}
  feedback.request('publish',case,cohort=c,step=publish[c],packet=packet);publish[c]+=1
 for index,(kind,rr)in enumerate(plan):
  rows=list(rr);c=rows[0]//16
  if kind=='decode'and not is_emitter:
   packet=feedback.request('take',case,cohort=c,step=read_next[c]);read_next[c]+=1;accept(packet,c)
  if kind=='prefill':assert all(pasts[i]==0 and not ids[i]for i in rows);ins=[inputs[i]for i in rows]
  else:assert len(rows)==16 and all(ids[i]and pasts[i]==511+len(ids[i])for i in rows);ins=[[ids[i][-1]]for i in rows]
  selected,seconds=forward(kind,rows,ins,[pasts[i]for i in rows],index)
  if is_emitter:
   assert len(selected)==len(rows)
   t=clock()-start
   if kind=='prefill':
    for i,v in zip(rows,selected):
     assert type(v)is int and v>=0 and v not in eos
     ids[i]=[v];pasts[i]=512;stamps[i]=[t]
    pending_prefill.setdefault(c,[]).extend(rows)
    if len(pending_prefill[c])==16:
     full=list(range(c*16,(c+1)*16));assert pending_prefill.pop(c)==full
     publish_packet(c,'prefill',full,[ids[i][0]for i in full],[stamps[i][0]for i in full],seconds)
   else:
    for i,v in zip(rows,selected):
     assert type(v)is int and v>=0 and v not in eos and len(ids[i])<128
     ids[i].append(v);pasts[i]+=1;stamps[i].append(t)
    publish_packet(c,'decode',rows,selected,[t]*16,seconds)
  else:assert selected is None
 if not is_emitter:
  for c in range(4):
   packet=feedback.request('take',case,cohort=c,step=read_next[c]);read_next[c]+=1;accept(packet,c)
 assert all(len(x)==128 for x in ids)and pasts==[639]*64 and not pending_prefill
 return {'generated_ids':ids,'pasts':pasts,'token_ready_times_from_local_emitter_start':stamps,'job_count':len(plan),'job_plan_SHA256':hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest(),'timestamp_scope':'Emitter local actual callback return, never delayed publication/client receipt. Other owners use emitter packet stamps only; no crosshost subtraction.','scope':'B64/M16 fixed cohorts, original M2prefill, full128 noEOS; new admission/order only. No numerical Model qualification implied.'}
