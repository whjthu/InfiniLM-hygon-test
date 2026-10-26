import pathlib,json,hashlib,time,tempfile,ctypes as C,types,os,numpy as np
import Deep_owner_prefill_r9193b as frontend
from deep_owner_consumer_r9193b import bind_owner,validate_owner,validate_inner
root=pathlib.Path(tempfile.mkdtemp(prefix='deep-r9193b-CPU-'));(root/'incoming').mkdir();(root/'results').mkdir();cases=[{'id':'cpu%d'%i,'input_ids':[2]*([420,420,420,421][i]if i<4 else 96),'input_tokens':([420,420,420,421][i]if i<4 else 96),'max_new_tokens':512}for i in range(64)];gold=[{'id':x['id'],'gold_numeric':'1'}for x in cases];payload={'cases':cases,'accepted_count':64,'skipped':[],'template_provenance':{'CPU_MOCK':True}};p=root/'results/tokens.json';p.write_bytes(json.dumps(payload).encode());(root/'incoming/quality-bench-gsm8k-fresh64-r9147.json').write_bytes(json.dumps({'cases':gold}).encode())
rows=[]
for dev in range(8):
 for layer in range(43):rows.append({'device':dev,'layer':layer,'owner':100+dev,'stream':200+dev,'markers':32,'event_create_during_device':dev,'event_query_during_device':dev,'event_create_before_device':0,'event_query_before_device':0,'drain_host_ms':0.1,'spans':[{'stage':k,'start_ms':0.,'end_ms':1.,'stream_ms':1.,'host_wall_ms':1.1,'thread_cpu_ms':0.2}for k in range(16)]})
raw=json.dumps({'schema':1,'window':1,'rows':rows}).encode();buffer=C.create_string_buffer(raw);window=0;armed_forwards=0;reads=0
@C.CFUNCTYPE(C.c_int,C.c_int)
def arm(w):
 global window
 if w not in [0,1,2]or window:return 1
 if w:window=w
 return 0
@C.CFUNCTYPE(C.c_void_p)
def read_ptr():
 global window,reads
 if not window or armed_forwards!=1:return 0
 window=0;reads+=1;return C.addressof(buffer)
read=C.CFUNCTYPE(C.c_char_p)(C.cast(read_ptr,C.c_void_p).value)

inner_rows=[]
for dev in range(8):
 for layer in range(43):inner_rows.append({'device':dev,'layer':layer,'kind':0 if layer<2 else 2 if layer%2==0 else 1,'tokens':1681,'requests':4,'owner':100+dev,'stream':200+dev,'thread':1000+dev,'events_created':7,'events_queried':7,'events_destroyed':7,'create_during_device':dev,'query_during_device':dev,'destroy_during_device':dev,'whole_stream_ms':6.,'original_final_drain_host_ms':.1,'segments':[{'segment':i,'start_ms':float(i),'end_ms':float(i+1),'stream_ms':1.,'host_wall_ms':1.1,'thread_cpu_ms':.2}for i in range(6)]})
inner_raw=json.dumps({'schema':1,'profile':'attention-inner-r9193','GPU_handles_retained':0,'scope':'CPUfixture only','rows':inner_rows}).encode();inner_buffer=C.create_string_buffer(inner_raw);inner_mode=0;inner_reads=0
@C.CFUNCTYPE(C.c_int,C.c_int)
def inner_arm(on):
 global inner_mode
 if on not in [0,1]or (on==1 and inner_mode):return 1
 inner_mode=on;return 0
@C.CFUNCTYPE(C.c_void_p)
def inner_read_ptr():
 global inner_mode,inner_reads
 if not inner_mode or window or reads!=1:return 0
 inner_mode=0;inner_reads+=1;return C.addressof(inner_buffer)
inner_read=C.CFUNCTYPE(C.c_char_p)(C.cast(inner_read_ptr,C.c_void_p).value)
class Fun:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
class Lib:
 def __init__(self):self.deep_owner_arm_r8900=arm;self.deep_owner_read_r8900=read;self.deep_attention_inner_arm_r9193=inner_arm;self.deep_attention_inner_read_r9193=inner_read;self.count=np.zeros((8,7),np.uint64);self.calls=0
 def __getattr__(self,n):
  if n.endswith('_counts_r8996b'):
   def counts(d,p,N):
    for j,v in enumerate(self.count[d]):p[j]=int(v)
    return 0
   f=Fun(counts)
  else:f=Fun(lambda *a:1 if n=='deep_packed_varlen_capability_r9101b'else 0)
  setattr(self,n,f);return f
 def modern_native_engine_reset(self,c):return 0
 def modern_native_engine_forward(self,ctx,T,A,*args):
  global armed_forwards
  self.calls+=1;ptr=[int(x.value)for x in args];past=np.ctypeslib.as_array((C.c_int32*A).from_address(ptr[1]));out=np.ctypeslib.as_array((C.c_uint16*(A*8)).from_address(ptr[6])).reshape(A,8);ids=np.ctypeslib.as_array((C.c_int64*A).from_address(ptr[7]));out[:]=0
  for j in range(A):token=2 if past[j]==0 else 1;out[j,token]=0x3f80;ids[j]=token
  if np.all(past==0):self.count+=np.array([43,0,0,1,1,0,0],np.uint64)
  if window:armed_forwards+=1
  return 0
lib=Lib();report={};g={'r':root,'lib':lib,'ctx':1,'cfg':{'eos_token_id':1,'vocab_size':8,'num_hidden_layers':43,'num_nextn_predict_layers':1,'compress_ratios':[0,0]+[4 if l%2==0 else 128 for l in range(2,43)]+[0]},'check':lambda c:None if c==0 else(_ for _ in()).throw(RuntimeError(c)),'report':report,'path':root/'report.json','manifest':{'tokenized_path':'results/tokens.json','existing_remote_SHA256':{'results/tokens.json':hashlib.sha256(p.read_bytes()).hexdigest()}},'tokenizer':types.SimpleNamespace(bos_token_id=0,eos_token_id=1,decode=lambda ids,**kw:'#### 1'),'os':os,'packet_started':time.perf_counter()};before=dict(os.environ)
try:frontend.run(g)
finally:os.environ.clear();os.environ.update(before)
assert report['grid_complete']and len(report['quality_trials'])==1 and len(report['quality_trials'][0]['records'])==32 and reads==armed_forwards==inner_reads==1 and window==inner_mode==0
assert len(report['prefill_diagnostics'])==4 and report['owner_prefill_profiles'][0]['owner_rows']==344
for variant in ['missingrow','wrongdevice','missingmarker','nonfinite','missingstage']:
 q=json.loads(raw)
 if variant=='missingrow':q['rows'].pop()
 if variant=='wrongdevice':q['rows'][0]['event_query_during_device']=7
 if variant=='missingmarker':q['rows'][0]['markers']=31
 if variant=='nonfinite':q['rows'][0]['spans'][0]['stream_ms']=float('nan')
 if variant=='missingstage':q['rows'][0]['spans'].pop()
 try:validate_owner(json.dumps(q),1)
 except AssertionError:pass
 else:raise AssertionError(variant)
assert report['owner_prefill_profiles'][0]['inner_event_validation_passed'];q={'inner_events_per_Frame':7,'inner_sibling_segments':6,'inner_actual_ctypes_CAPI_arm_read':True,'CPU_only':True,'GPU':False,'no_dlopen':True,'actual_ctypes_arm_read_function_pointer_ABI':True,'one_arm_one_prefill_immediate_read':True,'owner_rows':344,'markers_each':32,'spans_each':16,'negative_profile_checks':5,'OFFwarm_bracket_and_B32_compact_case_scheduler_executed':True,'real_question_or_math_or_Model_quality_claim':False,'forward_calls':lib.calls,'main_requests':32,'report_schema_validated':True};pathlib.Path(__file__).with_name('CPU-consumer-r9193b.json').write_bytes(json.dumps(q,indent=2).encode());print(json.dumps(q))
