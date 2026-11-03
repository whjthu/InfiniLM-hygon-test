"""Thin two-OFF consumer of the frozen1378 plan and original quiescent owner."""
from pathlib import Path
import hashlib,json,time
from quiescent_mode import QuiescentMode
from teacher_runner import plan,logits_summary,ids_sha,teacher_from_provenance
from capture_api import Budget
from boundary_api import Capture


def run(context,prompt,teacher,root,*,np,deadline,reference,save_leaf=None):
 if ids_sha(prompt)!=reference['prompt_IDs_SHA256']or ids_sha(teacher)!=reference['teacher_IDs_SHA256']:raise ValueError('Frozen original input identities differ')
 schedule=plan(prompt,teacher);root=Path(root);root.mkdir(exist_ok=False)
 budget=Budget(32*1024**2);reserve=2*1024**2;budget.charge(reserve)
 control=QuiescentMode(context);capture=Capture(control.lib,budget)
 q=dict(completed=False,passed=False,diagnostic_only=True,quality_PASS=False,performance_PASS=False,HTTP=0,
  planned_modes=[0,0],max_sequences=2,max_forwards=184,actual_forward_attempts=0,completed_forwards=0,
  prompt_IDs_SHA256=ids_sha(prompt),teacher_IDs_SHA256=ids_sha(teacher),sequences=[],early_stop=False,cause='UNKNOWN',cleanup_errors=[])
 def save():
  raw=(json.dumps(q,indent=2)+'\n').encode()
  if len(raw)>reserve:raise RuntimeError('Bounded report metadata exceeded')
  temp=root/'report.tmp';temp.write_bytes(raw);temp.replace(root/'report.json')
  if save_leaf is not None:save_leaf(q)
 def live():
  if time.time()>=deadline:raise TimeoutError('Original absolute deadline')
 save();primary=None
 try:
  for p in range(2):
   live();row=dict(sequence=p,mode=0,completed=False,actual_forward_attempts=0,completed_forwards=0,decisions=[],captures=[]);q['sequences'].append(row)
   with control.request(0):
    row['actual_reset_before']=True
    for f,step in enumerate(schedule,1):
     live();index=step['output_index'];past=step['past'];part=step['part'];selected=index in(82,83)
     if selected:capture.arm(control,p,index,f)
     if q['actual_forward_attempts']>=184:raise RuntimeError('No third sequence or extra forward')
     q['actual_forward_attempts']+=1;row['actual_forward_attempts']+=1;save()
     code,logits,sampled=control.forward(part,list(range(past,past+len(part))),[0],[past],[past+len(part)],[0,len(part)])
     context.binding.check(code);q['completed_forwards']+=1;row['completed_forwards']+=1
     if logits.dtype!=np.dtype('uint16')or logits.shape!=(1,context.cfg['vocab_size'])or not np.isfinite((logits[0].astype(np.uint32)<<16).view(np.float32)).all():raise RuntimeError('Actual forward returned nonfinite/invalid BF16 logits')
     if index is not None:
      item=logits_summary(np,logits,sampled,index,teacher,context.cfg['vocab_size']);item.update(forward_1based=f,input_past=past,input_count=len(part),input_IDs_SHA256=ids_sha(part));row['decisions'].append(item)
     if selected:
      raw=logits.copy().tobytes();budget.charge(len(raw));path=root/('pass%d-output%d-logits-BF16.bin'%(p,index))
      with path.open('xb')as out:out.write(raw)
      item['raw_logits']=dict(file=path.name,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
      result=capture.save(control,root/('pass%d-output%d-boundaries'%(p,index)));row['captures'].append(result)
      if result['nonfinite']or any(v['L2exit_equals_L3entry']is not True for v in result['owners']):q.update(early_stop=True,early_stop_reason='Observed boundary anomaly; cause UNKNOWN');break
     save()
    row['completed']=not q['early_stop']
   row['actual_owner_counters_after']=control.last_counts
   if not control.last_counts or len(control.last_counts)!=8:raise RuntimeError('Missing actual owner counters')
   for c in control.last_counts:
    if c['entered_valid_MoE']!=43*row['completed_forwards']or c['host_returned_not_GPU_completed']!=43*row['completed_forwards']or c['unwound']or c['armed_mode1_T1']or c['routed_hits']or c['shared_hits']:raise RuntimeError('Actual OFF/43layer coverage differs')
    if c['T1']!=43*max(0,row['completed_forwards']-9)or c['last_mode']!=0:raise RuntimeError('Actual OFF/T1 counter differs')
    for role in ('routed','shared'):
     if c[role+'_calls']!=c[role+'_fallback']:raise RuntimeError('Actual OFF role dispatch accounting differs')
   save()
   if q['early_stop']:break
  if not q['early_stop']and(q['completed_forwards']!=184 or len(q['sequences'])!=2 or any(len(r['decisions'])!=84 for r in q['sequences'])):raise RuntimeError('Incomplete two-pass teacher diagnostic')
  q.update(completed=True,passed=True);save()
 except BaseException as error:
  primary=error;q.update(completed=True,passed=False,primary_error=repr(error));save();raise
 finally:
  if capture.tag is not None:
   if control.quiescent:
    try:q['partial_capture']=capture.save(control,root/'partial-on-failure',partial=True)
    except BaseException as error:q['cleanup_errors'].append(repr(error))
   else:q['cleanup_errors'].append('Original Native quiescence unknown; no clear/retry Runtime')
  q['raw_budget_used']=budget.used
  if q['cleanup_errors']:q.update(completed=True,passed=False)
  save()
  if q['cleanup_errors']and primary is None:raise RuntimeError('Boundary CPU-record cleanup failure')
 return q
