"""Four bounded fixed-prefix passes, no HTTP or free-running output selection."""
from pathlib import Path
import hashlib,json,time
from quiescent_mode import QuiescentMode
from capture_api import Capture,Budget
MODES=(0,1,0,1)
ids_sha=lambda ids:hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()

def teacher_from_provenance(payload,reference):
 teacher=payload['teacher_token_ids'];trace=payload['original_continuation'];prefill=[v for v in trace if v['kind']=='prefill'];decode=[v for v in trace if v['kind']=='decode_summary']
 if payload['source_GPU']!='910374.1374'or payload['correct_request_index']!=0 or payload['source_archive_SHA256']!=reference['source_archive_SHA256']:raise ValueError('Original correct-run provenance differs')
 if len(teacher)!=83 or ids_sha(teacher)!=reference['teacher_IDs_SHA256']or payload['prompt_IDs_SHA256']!=reference['prompt_IDs_SHA256']:raise ValueError('Teacher/input identity differs')
 past=0
 for i,row in enumerate(prefill):
  if row['past']!=past or row['tokens']!=min(1024,8489-past)or row['total']!=past+row['tokens']or row['emitted']!=(i==len(prefill)-1):raise ValueError('Original prefill continuation differs')
  past=row['total']
 if len(prefill)!=9 or past!=8489 or decode!=[dict(kind='decode_summary',calls=97,first_past=8489,last_total=8586)]:raise ValueError('Original generation continuation differs')
 expected=[dict(output_index=i,input_past=past+i-1,forward_1based=len(prefill)+i,input_teacher_index=i-1)for i in reference['capture_output_indices']]
 if expected!=reference['capture_plan']or reference['forward_plan']['total_per_pass']!=len(prefill)+len(teacher)or reference['forward_plan']['last_total']!=past+len(teacher):raise ValueError('Capture index not derived from original continuation')
 return teacher

def plan(prompt,teacher):
 if len(prompt)!=8489 or len(teacher)!=83 or any(type(v)is not int or not 0<=v<129280 or v==1 for v in teacher):raise ValueError('Frozen true prompt/83teacher contract')
 out=[];past=0
 for offset in range(0,len(prompt),1024):
  tokens=prompt[offset:offset+1024];out.append(dict(part=tokens,past=past,kind='prefill',output_index=0 if offset+len(tokens)==len(prompt)else None));past+=len(tokens)
 for index,token in enumerate(teacher,1):out.append(dict(part=[token],past=past,kind='decode',output_index=index));past+=1
 if len(out)!=92 or past!=8572 or [v['output_index']for v in out if v['output_index']is not None]!=list(range(84)):raise AssertionError('Continuation-derived forward plan')
 return out

def logits_summary(np,logits,sampled,index,teacher,vocab):
 if logits.dtype!=np.dtype('uint16')or logits.shape!=(1,vocab):raise RuntimeError('Actual Native BF16 logits ABI differs')
 scores=(logits[0].astype(np.uint32)<<16).view(np.float32)
 if not np.isfinite(scores).all():raise RuntimeError('Nonfinite actual Native returned BF16 logits')
 native=int(sampled[0]);order=np.lexsort((np.arange(vocab),-scores));top=order[:8]
 if len(sampled)!=1 or not 0<=native<vocab or float(scores[native])!=float(scores[top[0]]):raise RuntimeError('Native chosen ID outside observed BF16 maximum tie set')
 selected={'Native_chosen':native,'original_correct83':30722,'original_incorrect83':9155}
 if index<83:selected['teacher']=teacher[index]
 return dict(output_index=index,logits_storage='Original Native BF16 uint16; F32 only exact bit-expanded diagnostic representation',finite=True,Native_chosen=native,top8=[dict(id=int(k),score=float(scores[k]))for k in top],observable_BF16_margin=float(scores[top[0]])-float(scores[top[1]]),maximum_tie_count=int(np.count_nonzero(scores==scores[top[0]])),selected_scores={k:dict(id=v,score=float(scores[v]))for k,v in selected.items()},logits_SHA256=hashlib.sha256(logits.tobytes()).hexdigest())

def run(context,prompt,teacher,root,*,np,deadline,reference,save_leaf=None):
 schedule=plan(prompt,teacher)
 if ids_sha(prompt)!=reference['prompt_IDs_SHA256']or ids_sha(teacher)!=reference['teacher_IDs_SHA256']:raise ValueError('Actual input IDs differ from frozen provenance')
 root=Path(root);root.mkdir(exist_ok=False);budget=Budget(reference['raw_total_limit']);report_reserve=2*1024**2;budget.charge(report_reserve);control=QuiescentMode(context);capture=Capture(control.lib,budget)
 report=dict(completed=False,passed=False,diagnostic_only=True,quality_PASS=False,performance_PASS=False,GPU_qualification=False,HTTP=0,planned_modes=list(MODES),max_sequences=4,max_forward_attempts=368,actual_forward_attempts=0,completed_forwards=0,teacher_IDs_SHA256=ids_sha(teacher),prompt_IDs_SHA256=ids_sha(prompt),teacher_count=83,output_decisions_per_full_pass=84,capture_outputs=[82,83],sequences=[],early_stop=False,cause='UNKNOWN')
 def save():
  raw=(json.dumps(report,indent=2)+'\n').encode()
  if len(raw)>report_reserve:raise RuntimeError('Frozen diagnostic report metadata reserve exceeded')
  p=root/'report.json';t=p.with_suffix('.tmp');t.write_bytes(raw);t.replace(p)
  if save_leaf is not None:save_leaf(report)
 def check_deadline():
  if time.time()>=deadline:raise TimeoutError('Original absolute diagnostic deadline')
 save();primary=None
 try:
  stop=False
  for sequence,mode in enumerate(MODES):
   check_deadline();row=dict(sequence=sequence,mode=mode,completed=False,actual_forward_attempts=0,completed_forwards=0,decisions=[],captures=[]);report['sequences'].append(row);save()
   with control.request(mode):
    row['actual_reset_before']=True
    for call,step in enumerate(schedule,1):
     check_deadline();index=step['output_index'];part=step['part'];past=step['past'];selected=index in [82,83]
     if selected:capture.arm(control,sequence,index,call)
     if report['actual_forward_attempts']>=368:raise RuntimeError('No fifth pass or extra forward')
     report['actual_forward_attempts']+=1;row['actual_forward_attempts']+=1;save()
     code,logits,sampled=control.forward(part,list(range(past,past+len(part))),[0],[past],[past+len(part)],[0,len(part)])
     # control.forward validates the unchanged original code0 ABI before it
     # marks whole-Native quiescence. Failure never counts as completed.
     context.binding.check(code);report['completed_forwards']+=1;row['completed_forwards']+=1
     if logits.dtype!=np.dtype('uint16')or logits.shape!=(1,context.cfg['vocab_size'])or not np.isfinite((logits[0].astype(np.uint32)<<16).view(np.float32)).all():raise RuntimeError('Actual every-forward finite/logits ABI')
     if index is not None:
      item=logits_summary(np,logits,sampled,index,teacher,context.cfg['vocab_size']);item.update(forward_1based=call,input_past=past,input_count=len(part),input_IDs_SHA256=ids_sha(part));row['decisions'].append(item)
     if selected:
      payload=logits.copy().tobytes();budget.charge(len(payload));path=root/('pass%d-output%d-logits-BF16.bin'%(sequence,index))
      with path.open('xb')as f:f.write(payload)
      item['raw_logits']=dict(file=path.name,bytes=len(payload),SHA256=hashlib.sha256(payload).hexdigest())
      result=capture.collect(control,root/('pass%d-output%d-owners'%(sequence,index)));row['captures'].append(result);report['raw_budget_used']=budget.used;save()
      if result['anomalies']or result['same_GU_shadow_differences']:
       report.update(early_stop=True,early_stop_reason='DIAGNOSTIC_OBSERVATION_NOT_A_ROOT_CAUSE',cause='UNKNOWN');stop=True;break
     save()
    row['completed']=not stop;row['last_total']=schedule[row['completed_forwards']-1]['past']+len(schedule[row['completed_forwards']-1]['part'])
   row['actual_owner_counters_after']=control.last_counts
   if control.last_counts is None or len(control.last_counts)!=8:raise RuntimeError('Missing actual8owner completion counters')
   for c in control.last_counts:
    if c['entered_valid_MoE']!=43*row['completed_forwards']or c['host_returned_not_GPU_completed']!=43*row['completed_forwards']or c['unwound']:raise RuntimeError('Actual43layer/forward Scope ordinal coverage')
    t1=43*max(0,row['completed_forwards']-9)
    if c['T1']!=t1 or c['armed_mode1_T1']!=mode*t1 or c['last_mode']!=mode:raise RuntimeError('Actual T1/mode coverage differs')
    for role in ['routed','shared']:
     if c[role+'_hits']!=mode*t1 or c[role+'_calls']!=c[role+'_hits']+c[role+'_fallback']:raise RuntimeError('Actual routed/shared dispatch accounting')
   save()
   if stop:break
  if not report['early_stop']:
   if len(report['sequences'])!=4 or report['completed_forwards']!=368 or any(len(r['decisions'])!=84 for r in report['sequences']):raise RuntimeError('Incomplete frozen four-pass diagnostic')
  report.update(completed=True,passed=True,cause='UNKNOWN',interpretation='Source/provider/actual-window numerical observation only; capture changes schedule and allocation history, no performance or fix claim');save()
 except BaseException as error:
  primary=error;report.update(completed=True,passed=False,primary_error=repr(error));save();raise
 finally:
  # No Tensor/Runtime operations in these CAPI calls. On error the unchanged
  # QuiescentMode first performs its original required reset/drain or poisons.
  report['cleanup_errors']=[]
  if control.quiescent:
   try:
    capture.abort_after_quiescence(control)
    if primary is not None:report['partial_CPU_capture_saved']=capture.save_partial(root/'partial-on-failure')
    capture.check(capture.lib.deep_model_capture_clear_r438v2())
   except BaseException as error:report['cleanup_errors'].append(repr(error))
  elif capture.tag is not None:report['cleanup_errors'].append('Quiescence unknown/poisoned; do not clear or retry Runtime')
  report['raw_budget_used']=budget.used;report['report_and_input_metadata_reserved_bytes']=report_reserve
  if report['cleanup_errors']:report.update(completed=True,passed=False)
  save()
  if report['cleanup_errors']and primary is None:raise RuntimeError('Diagnostic CPU-record cleanup failed')
 return report
