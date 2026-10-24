import pathlib,tempfile,json,ctypes,hashlib,shutil,types,os
import numpy as np
import Deep_owner_Model_protocol_r8925 as matrix
r=pathlib.Path.cwd();ref=json.loads((r/'Deep-B32-samegeom-reference-r7269.json').read_text());gold=np.array(ref['generated_ids'],np.int64)
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
class Lib:
 def __init__(self,negative=None):
  self.calls=0;self.proj=0;self.pre=0;self.dec=0;self.arm=0;self.windows=[];self.negative=negative
  for n in ['modern_deep_precision_policy_r3443','modern_deep_layout_policy_r7052','deep_hc_canonical_policy_r7130','deep_moe_tail_set_r7211','modern_deep_c4_tile_policy_r5920','deep_model_quant1_mode_r8886']:setattr(self,n,Fn(lambda x:0))
  self.modern_native_engine_reset=Fn(lambda x:0);self.modern_native_engine_forward=Fn(self.forward);self.deep_output_tile_calls_r4808=Fn(lambda d:self.proj);self.modern_deep_c4_tile_calls_r5920=Fn(lambda d,k:self.proj*2);self.deep_prefill_pack_counts_r8767=Fn(self.pack);self.deep_model_quant1_counts_r8886=Fn(self.count);self.deep_owner_arm_r8900=Fn(self.armfn);self.deep_owner_read_r8900=Fn(self.read)
 def pack(self,d,q,n):
  vals=[0,self.pre*43,0,self.pre*43*251,0,self.pre*86,0,self.pre*43,0,self.pre*43*2048*6*16,self.dec*43,0]
  for i,v in enumerate(vals):q[i]=v
  return 0
 def count(self,d,q,n):
  vals=[self.pre*43,0,self.pre*43*251,0,0,0,0,0]
  for i,v in enumerate(vals):q[i]=v
  return 0
 def armfn(self,w):
  assert not self.arm and w in [1,2];self.arm=w;self.windows.append((w,self.calls));return 0
 def read(self):
  assert self.arm;w=self.arm;self.arm=0
  spans=[dict(stage=k,start_ms=0.,end_ms=1.,stream_ms=1.,host_wall_ms=1.,thread_cpu_ms=.5)for k in range(16)]
  rows=[dict(device=d,layer=l,owner=1,stream=1,markers=32,event_create_before_device=0,event_query_before_device=0,event_create_during_device=d,event_query_during_device=d,drain_host_ms=0.,spans=spans)for d in range(8)for l in range(43)]
  if self.negative=='schema':rows.pop()
  if self.negative=='device':rows[1]['event_query_during_device']=0;rows[1]['device']=1
  return json.dumps(dict(schema=1,window=w,rows=rows)).encode()
 def forward(self,ctx,tokens,n,*ptrs):
  self.calls+=1;rows=np.ctypeslib.as_array((ctypes.c_int32*n).from_address(ptrs[0].value));past=np.ctypeslib.as_array((ctypes.c_int32*n).from_address(ptrs[1].value));step=0 if past[0]==0 else int(past[0])-511
  ids=np.ctypeslib.as_array((ctypes.c_int64*tokens).from_address(ptrs[4].value));pos=np.ctypeslib.as_array((ctypes.c_int64*tokens).from_address(ptrs[5].value));assert pos[0]==past[0]
  if step:assert ids.tolist()==gold[rows,step-1].tolist();self.proj+=43;self.dec+=1
  else:assert tokens==2048 and ids.tolist()==sum([ref['input_ids'][int(i)]for i in rows],[]);self.pre+=1
  out=np.ctypeslib.as_array((ctypes.c_uint16*(n*129280)).from_address(ptrs[6].value)).reshape(n,129280);em=np.ctypeslib.as_array((ctypes.c_int64*n).from_address(ptrs[7].value));out.fill(0);em[:]=gold[rows,step];out[np.arange(n),em]=0x3f80
  if self.negative=='finite'and self.calls==1:out[0,0]=0x7fc0
  return 0

def execute(negative=None):
 with tempfile.TemporaryDirectory(prefix='Deep-owner-r8925-')as d:
  p=pathlib.Path(d);(p/'incoming').mkdir();(p/'results').mkdir();shutil.copyfile(r/'Deep-B32-samegeom-reference-r7269.json',p/'incoming/Deep-B32-samegeom-reference-r7269.json');report={};lib=Lib(negative);chats=[]
  def chat(epoch):chats.append(epoch);report['finite_'+epoch+'_passed']=True
  def check(code):assert code==0
  progress=[]
  def checkpoint(phase,window=0):
   progress.append((phase,window));report['phase']=phase;report['phase_window']=window;t=p/'results/progress.tmp';t.write_text(json.dumps(report));t.replace(p/'results/progress.json')
  g=dict(checkpoint=checkpoint,r=p,lib=lib,ctx=1,cfg=dict(vocab_size=129280,eos_token_id=129279,compress_ratios=[4]*43,num_hidden_layers=43),check=check,report=report,path=p/'results/report.json',run_chat=chat,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda d:(16*1024**3,32*1024**3))),os=os)
  try:matrix.run(g)
  except (AssertionError,RuntimeError)as e:
   assert negative and lib.calls==1,(negative,repr(e),lib.calls);return dict(rejected=True,CAPI_calls=lib.calls)
  assert not negative and lib.calls==135 and lib.windows==[(1,0),(2,71)]and chats==['before','after_restore']and report['quality_passed']and not report['performance_qualified']and len(report['owner_trials_r8925'])==1 and len(report['owner_event_windows'])==2
  assert progress==[('load_return',0)]+[(phase,w)for w in [1,2]for phase in ['selected_CAPI_enter','selected_CAPI_return','capture_enter','capture_return']]
  return dict(progress=progress,passed=True,CAPI_calls=135,full128_IDs_exact=True,windows=lib.windows,events_rows_each=344,no_extra_forward=True,QA=chats)
out=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,noGPU_dlopen=True,positive=execute(),negative_nonfinite=execute('finite'),negative_missing_owner_layer=execute('schema'),negative_device_field=execute('device'),entry_protocol_SHA256=hashlib.sha256((r/'Deep_owner_Model_protocol_r8925.py').read_bytes()).hexdigest());p=r/'Deep-owner-events-consumer-CPU-r8925.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))