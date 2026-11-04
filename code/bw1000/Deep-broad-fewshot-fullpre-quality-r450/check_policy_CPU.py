"""Only new finite variable-prefix policy glue; math/Native leaves record counts."""
import copy,ctypes as C,threading,time
from types import SimpleNamespace
import numpy as np
from public_deep_model.native_context import NativeContext
from public_deep_model.native_engine_bridge_r268 import NativeRequest
from public_deep_model.native_decode import generate
from public_deep_model.fullpre_control import FULL,LEGACY
from paired_plan import validate_subset
from wide_policy import RequestPolicy
from coverage import validate

class Fn:
 def __init__(self,f):self.f=f;self.argtypes=None;self.restype=None
 def __call__(self,*args):return self.f(*args)

class RecordingContext(NativeContext):
 def __init__(self):
  self.cfg={'num_hidden_layers':43};self.ctx=0x1234567800000001;self.thread=threading.get_ident();self.closed=False;self.mode=0;self.calls=0;self.reset_calls=0;self.per_request=0;self.goal=0;self.drained=True
  self.legacy=[dict.fromkeys(LEGACY,0)for _ in range(8)]
  for r in self.legacy:r.update(created=86,ready=86)
  self.full=[dict.fromkeys(FULL,0)for _ in range(8)];lib=SimpleNamespace()
  funcs=dict(deep_hc_fullpre_enable_r377=self.set,deep_hc_fullpre_enabled_r377=lambda:self.mode,deep_hc_fullpre_clear_r377=self.clear,deep_hc_fullpre_stats_r377=lambda p,n:self.read(p,n,self.full,FULL),deep_hc_model_epilogue_set_r366=lambda m:0 if m==0 and self.drained else 9,deep_hc_model_epilogue_get_r366=lambda:0,deep_hc_model_epilogue_counters_r366=lambda p,n:self.read(p,n,self.legacy,LEGACY),modern_native_engine_reset=self.record_reset,modern_native_engine_destroy=lambda ctx:0,modern_native_engine_forward=lambda *a:0)
  for name,f in funcs.items():setattr(lib,name,Fn(f))
  for name in ['reset','destroy']:getattr(lib,'modern_native_engine_'+name).argtypes=[C.c_void_p];getattr(lib,'modern_native_engine_'+name).restype=C.c_int
  lib.modern_native_engine_forward.argtypes=[C.c_void_p];lib.modern_native_engine_forward.restype=C.c_int
  def check(code):
   if code:raise RuntimeError('recording CAPI error')
  self.binding=SimpleNamespace(lib=lib,check=check);self.forward=self.invoke
 def set(self,mode):assert mode in[0,1]and self.drained;self.mode=mode;return 0
 def clear(self):assert self.mode==0 and self.drained;self.full=[dict.fromkeys(FULL,0)for _ in range(8)];return 0
 def read(self,out,n,rows,names):
  assert n==8*len(names)and self.drained
  for d,r in enumerate(rows):
   for k,name in enumerate(names):out[d*len(names)+k]=r[name]
  return 0
 def record_reset(self,ctx):assert ctx==self.ctx;self.reset_calls+=1;self.per_request=0;self.drained=True;return 0
 def invoke(self,part,*args):
  self.check_thread();self.drained=False;self.calls+=1;self.per_request+=1;one=len(part)==1
  for a,b in zip(self.legacy,self.full):
   a['pre']+=86;a['eligible_bf16_t1']+=86*one;a['other_t']+=86*(not one)
   if self.mode:b['observed']+=86;b['eligible']+=86*one;b['selected']+=86*one;b['completed']+=86*one;b['prefill']+=86*(not one)
  self.drained=True;return 0,np.zeros((1,129280),dtype=np.uint16),[1 if self.per_request==self.goal else 9]

def run(cases,encodings,logical):
 from enum import Enum
 class Reasons(str,Enum):EOS_TOKEN='eos_token';LENGTH='length';CANCELED='canceled'
 ctx=RecordingContext();entries=logical[:2];validate_subset(entries,logical)
 policy=RequestPolicy(ctx,dict(HC_mode=0,r450_r366_fixed=0,r450_plan=entries,r450_frozen512=logical),provider_validator=lambda lib:dict(recording_CABI=True,actual_DSO_or_GPU=False))
 records=[]
 for entry in entries:
  row=encodings[entry['case_ordinal']];raw=__import__('json').loads(__import__('pathlib').Path(row['private_path']).read_bytes());ids=raw['prompt_token_ids'];ctx.goal=(len(ids)+1023)//1024+1
  req=NativeRequest('recording-wide-'+str(entry['logical_index']),raw['prompt'],ids,4096,[1],Reasons)
  record=dict(request_id=req.request_id,admission=dict(history_render_audit=row['history_render_audit'],diagnostic_user='Deep-r450-'+entry['case_id'],text_delivery='P1'),prompt_tokens=len(ids),max_new_tokens=4096,actual_forward_attempts=0,actual_forward_completed=0)
  def forward(*args):
   record['actual_forward_attempts']+=1;result=policy.forward(*args);record['actual_forward_completed']+=1;return result
  with policy.request(req,record):
   reason,trace=generate(req,forward,ctx.binding.check,lambda logits:None,Reasons,context=32768,chunk_size=1024,eos=1,vocab=129280,absolute_deadline=time.time()+30);record['Native_continuation']=trace
  assert ctx.drained and ctx.mode==0;req.finish(reason);assert req.is_finished()and req.generated_token_ids==[9,1]
  record.update(native_finish_reason='eos_token',backend_token_monotonic=req.metadata['token_monotonic'],final_metadata=dict(buffered_P0_text=False))
  assert validate(record,entry)==record['r450_actual_T1_coverage'];records.append(record)
 try:
  with policy.request(req,{}):raise AssertionError('Must not enter extra request')
 except RuntimeError:pass
 else:raise AssertionError('Finite plan exhaustion accepted')
 bad=copy.deepcopy(records[1]);bad['r450_actual_owner_counts'][7]['fullpre']['completed']+=1
 try:validate(bad,entries[1])
 except AssertionError:pass
 else:raise AssertionError('Missing owner coverage rejection')
 for p,one in [(1024,0),(1025,1)]:
  entry=dict(logical_index=0,case_id='recording-boundary',mode=1,prompt_tokens=p);f=(p+1023)//1024;t1=86*one
  row=dict(r450_logical_index=0,r450_case_id=entry['case_id'],r450_mode=1,r450_r366_fixed=0,normal_reset_before=True,normal_reset_after=True,r450_quiescent=True,r450_poisoned=False,actual_forward_completed=f,actual_forward_attempts=f,generated_token_ids=[1],max_new_tokens=4096,prompt_tokens=p,Native_continuation=[dict(kind='prefill',past=s,tokens=min(1024,p-s),total=min(s+1024,p))for s in range(0,p,1024)],r450_actual_owner_counts=[dict(owner=d,created=86,ready=86,legacy=dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-one),f32_t1=0),fullpre=dict(observed=86*f,eligible=t1,selected=t1,completed=t1,prefill=86*(f-one),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0))for d in range(8)])
  proof=validate(row,entry);assert proof['no_eligible_T1']==(one==0)and proof['T1_HC_per_owner']==t1
 # Actual new client core and raw SSE reconstruction, with explicit HTTP and
 # already-recorded Native leaves. Wrong parse must continue, not drop an arm.
 import json
 from pathlib import Path
 import wide_client
 from unittest.mock import patch
 from scoring import grade
 first=cases[0];good='#### '+first['gold_numeric']if first['suite']=='gsm8k'else'Answer: '+first['gold_choice']
 bodies=[dict(case_id=c['id'],body=json.loads(Path(e['private_path']).read_bytes())['body'],request_body_SHA256=e['request_body_SHA256'])for c,e in zip(cases,encodings)]
 sent=[]
 class Response:
  status=200;headers={'Content-Type':'text/event-stream'}
  def __init__(self,raw):self.raw=raw
  def __enter__(self):return self
  def __exit__(self,*a):return False
  def read(self):return self.raw
 def http(request,timeout):
  i=len(sent);sent.append(request.data);assert json.loads(request.data)==bodies[0]['body']
  body=dict(id=records[i]['request_id'],choices=[dict(index=0,delta=dict(role='assistant',content=good if i==0 else'Recording-only invalid answer'),finish_reason='stop')],usage=dict(prompt_tokens=encodings[0]['prompt_tokens'],completion_tokens=2,total_tokens=encodings[0]['prompt_tokens']+2))
  return Response(('data: '+json.dumps(body)+'\n\ndata: [DONE]\n\n').encode())
 output=Path(encodings[0]['private_path']).parent
 closure=output/'actual-input-encoding-closure.json';closure.write_text(json.dumps(dict(completed=True,passed=True,actual_encodings=256,actual_Deep_codec=True,actual_public_protocol=True,Native_dlopen=False,GPU=0,cases=encodings,scope='Actual256 input encoding completed; not wholeCPU/Model/GPU qualification'))+'\n')
 planfile=output/'changed-client-recording-plan.json';planfile.write_text(json.dumps(dict(frozen512=logical,assigned=entries))+'\n')
 casesfile=Path('/refactor/incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json');requestfile=output/'requests-private.json'
 identity=dict(run='recording',source_manifest_SHA256='recording-source',provider_config_SHA256='recording-config',actual_step='CPU.recording')
 def artifact(path,deadline,predicate=lambda q:True):
  if Path(path).name=='MODEL_READY.json':return dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,HC_mode=0,recording_leaf=True)
  assert Path(path).name=='owner.json'
  value=dict(identity=identity,Native_SHA256='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1',HC_mode=0,requests=records,recording_leaf=True);assert predicate(value);return value
 digest=lambda p:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
 args=['--cases',str(casesfile),'--cases-sha',digest(casesfile),'--requests',str(requestfile),'--requests-sha',digest(requestfile),'--encoding',str(closure),'--encoding-sha',digest(closure),'--plan',str(planfile),'--plan-sha',digest(planfile),'--base-url','http://recording.invalid/v1','--runtime-dir',str(output/'recording-runtime'),'--run','recording','--source-sha','recording-source','--config-sha','recording-config','--actual-step','CPU.recording','--output-dir',str(output/'changed-client-recording'),'--deadline-unix',str(time.time()+300)]
 with patch.object(wide_client,'wait_json',artifact),patch.object(wide_client.urllib.request,'urlopen',http):rc=wide_client.main(args)
 assert rc==0;client=json.loads((output/'changed-client-recording/result.json').read_bytes())
 assert client['completed']and client['execution_passed']and client['completed_HTTP']==2 and len(sent)==2 and sent[0]==sent[1]
 assert client['rows'][0]['correct']and client['rows'][1]['parse_failure']and not client['rows'][1]['correct']
 assert client['summary']['pooled']['arms']['0']['planned']==client['summary']['pooled']['arms']['1']['planned']==256
 assert not grade(first,good,natural_EOS=False,cap_hit=True)['correct'],'A matching capped answer remains failed quality'
 return dict(passed=True,actual_new_wide_policy=True,actual_original_generate_and_reset=True,actual_encoded_first_case_two_arms=True,finite512_plan_checked=True,recording_Native_calls=ctx.calls,recording_only=True,actual_Native_GPU_or_quality=False,zero_T1_early_EOS_kept=True,one_token_prefill_counted=True,extra_request_and_wrong_owner_rejected=True,actual_new_client_main_CLI_and_real_body_encoding_plan_file_handoff=True,actual_SSE_reconstruction=True,wrong_parse_continues_both_arms=True,cap_matching_answer_rejected=True,global_denominator256_perarm_kept=True,old938_cases_replayed=False)
