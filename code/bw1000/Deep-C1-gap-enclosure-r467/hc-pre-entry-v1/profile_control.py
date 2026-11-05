"""Five original C1 continuations; existing a314 events, no new phase fence."""
from pathlib import Path
import ctypes as C,hashlib,json,threading,time
from existing_profile_consumer import bind_inner,inner_owner_forward_once,native_expected_kinds
from hc_capture import capture as capture_hc
from hc_consume import bind as bind_hc
MODES=(False,False,False,True,False)
SAMPLES=(8,87)
PROMPT_SHA='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'

class Request:
 def __init__(self,ids):
  self.prompt_token_ids=list(ids);self.max_new_tokens=128;self.generated_token_ids=[];self.token_monotonic=[];self.cancel_event=threading.Event();self.abort_reason=None
 def emit_token(self,token):self.generated_token_ids.append(int(token));self.token_monotonic.append(time.monotonic())

class CapiTimer:
 """Per-instance facade, preserves original CAPI and all original counters."""
 def __init__(self,lib):self.original=lib;self.rows=[]
 def __getattr__(self,name):return getattr(self.original,name)
 def modern_native_engine_forward(self,*args):
  start=time.perf_counter();rc=self.original.modern_native_engine_forward(*args);end=time.perf_counter()
  self.rows.append(dict(start_monotonic=start,return_monotonic=end,seconds=end-start,rc=int(rc),tokens=int(args[1]),count=int(args[2])))
  return rc

class SparseObserver:
 def __init__(self,lib,check,cfg,output):
  self.lib,self.check,self.output=lib,check,Path(output);self.kinds=native_expected_kinds(cfg);bind_inner(lib);self.read_hc=bind_hc(lib)
  check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0));self.poisoned=False
 def call(self,forward,args,*,armed,request_index,decode_index,record,save):
  if self.poisoned:raise RuntimeError('Profile failed; no further forward permitted')
  capture=armed and decode_index in SAMPLES;value=[];started=time.perf_counter()
  def invoke():
   v=forward(*args);value.append(v);return v[0]
  if capture:
   assert len(args[0])==len(args[2])==1 and args[2]==[0]and args[3]==[8489+decode_index-1]
   geometry=dict(tokens=1,requests=1,rows=[0],past=list(args[3]),decode_call_1based=decode_index)
   try:
    _,ended,began,q=inner_owner_forward_once(self.lib,invoke,self.check,1,self.kinds,geometry,self.output/f'request{request_index}-decode{decode_index}')
    q['HC_pre_split']=capture_hc(self.read_hc,q,self.output/f'request{request_index}-decode{decode_index}-HCpre.json')
   except BaseException as error:
    self.poisoned=True;failure=getattr(error,'profile',dict(error=repr(error)));failure['window_recovery_attempts']=[]
    # arm(0) is NOT an unconditional disarm. Failed active windows remain a
    # failstop: record both return codes, never retry forward or recreate.
    for name in ['deep_owner_arm_r8900','deep_attention_inner_arm_r9193']:
     try:failure['window_recovery_attempts'].append(dict(function=name,rc=int(getattr(self.lib,name)(0))))
     except BaseException as cleanup:failure['window_recovery_attempts'].append(dict(function=name,error=repr(cleanup)))
    record.setdefault('profile_failures',[]).append(failure);save();raise
   record.setdefault('profile_windows',[]).append(q)
   timing=dict(decode_call_1based=decode_index,captured=True,forward_wrapper_s=ended-began,arm_s=q['arm_wall_s'],read_copy_s=q['read_native_and_copy_wall_s'],packet_IO_validation_s=q['Python_validation_and_packet_io_wall_s'],observer_total_s=q['observer_total_wall_s'])
  else:
   began=time.perf_counter();invoke();timing=dict(decode_call_1based=decode_index,captured=False,forward_wrapper_s=time.perf_counter()-began)
  timing['whole_observer_wrapper_s']=time.perf_counter()-started;record.setdefault('calls',[]).append(timing)
  assert len(value)==1;return value[0]

def run_loaded(context,prompt,output,report,save,*,generate,reasons,control,finite,deadline,clock=time.monotonic):
 assert len(prompt)==8489 and hashlib.sha256(json.dumps(list(prompt),separators=(',',':')).encode()).hexdigest()==PROMPT_SHA
 root=Path(output);root.mkdir(exist_ok=False);observer=SparseObserver(context.binding.lib,context.binding.check,context.cfg,root);report['requests']=[]
 for index,armed in enumerate(MODES):
  context.check_thread()
  if time.time()>=deadline:raise TimeoutError('Original profile epoch expired; never slide deadline')
  req=Request(prompt);record=dict(index=index,warm=index<2,armed=armed,output_budget=128,prompt_tokens=8489,prompt_IDs_SHA256=PROMPT_SHA,normal_reset_before=False,normal_reset_after=False,profile_windows=[])
  report['requests'].append(record);record['GPU_free_total_before']=context.floor();start=clock();attempts=0;completed=0;failure=None
  def forward(*args):
   nonlocal attempts,completed
   decode_index=len(req.generated_token_ids)if req.generated_token_ids else None;attempts+=1
   value=observer.call(control.forward,args,armed=armed,request_index=index,decode_index=decode_index,record=record,save=save);context.binding.check(value[0]);completed+=1;return value
  try:
   with control.request(0):
    record['normal_reset_before']=True
    reason,trace=generate(req,forward,context.binding.check,finite,reasons,context=32768,chunk_size=1024,eos=1,vocab=context.cfg['vocab_size'],absolute_deadline=deadline)
    record.update(reason=reason.name,continuation=trace)
  except BaseException as error:failure=error;record['error']=repr(error)
  finally:
   record.update(whole_request_with_reset_s=clock()-start,normal_reset_after=control.normal_reset_after,HC_owner_counts=control.last_counts,forward_attempts=attempts,forward_completed=completed,generated_token_ids=req.generated_token_ids,token_monotonic=req.token_monotonic)
   if not control.poisoned:record['GPU_free_total_after']=context.floor()
   save()
  if failure is not None:raise failure
  assert attempts==completed and record['normal_reset_after']and control.last_counts is not None
  assert len(control.last_counts)==8
  for d,c in enumerate(control.last_counts):
   assert c['owner']==d and c['created']==c['ready']==86
   assert not any(c['fullpre'].values())and c['legacy']==dict(created=0,ready=0,pre=86*completed,eligible_bf16_t1=86*(completed-9),completed=0,other_t=86*9,f32_t1=0)
  assert req.generated_token_ids and len(req.generated_token_ids)<=128
  natural=req.generated_token_ids[-1]==1;assert req.generated_token_ids.count(1)==int(natural)
  if not natural:assert len(req.generated_token_ids)==128
  expected=[i for i in SAMPLES if len(req.generated_token_ids)>i]if armed else[]
  actual=[q['geometry']['decode_call_1based']for q in record['profile_windows']];assert actual==expected
  record.update(natural_EOS=natural,cap=not natural,profile_coverage_complete=(not armed or actual==list(SAMPLES)),actual_profile_samples=actual,missing_due_to_early_natural_EOS=[i for i in SAMPLES if i not in actual]if armed else[])
  context.floor();save()
 assert len(report['requests'])==5
 report['profile_coverage_complete']=report['requests'][3]['profile_coverage_complete']
 report['profile_coverage_scope']='Incomplete samples stay UNKNOWN; no replacement request, no EOS98 or bitexact requirement'
 return report
