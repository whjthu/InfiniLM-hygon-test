import pathlib,tempfile,json,ctypes,hashlib,shutil,types,os
import numpy as np
import Deep_sink_Model_protocol_r8970 as matrix
r=pathlib.Path.cwd();ref=json.loads((r/'Deep-B32-samegeom-reference-r7269.json').read_text());gold=np.array(ref['generated_ids'],np.int64)
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
class Lib:
 def __init__(self,negative=None):
  self.calls=0;self.proj=0;self.pre=0;self.dec=0;self.arm=0;self.windows=[];self.negative=negative;self.scratch_mode=0;self.tiled=0;self.sink_mode=0;self.sink_vals=[0]*4
  for n in ['modern_deep_precision_policy_r3443','modern_deep_layout_policy_r7052','deep_hc_canonical_policy_r7130','deep_moe_tail_set_r7211','modern_deep_c4_tile_policy_r5920','deep_model_quant1_mode_r8886']:setattr(self,n,Fn(lambda x:0))
  self.deep_reference_sink_mode_r8967=Fn(self.sinkmode);self.deep_reference_sink_counts_r8967=Fn(self.sinkcount);self.deep_latent_prefill_mode_r8933=Fn(lambda a:0);self.deep_reference_scratch_mode_r8953=Fn(self.mode);self.deep_reference_scratch_counts_r8954=Fn(self.scratch);self.modern_native_engine_reset=Fn(lambda x:0);self.modern_native_engine_forward=Fn(self.forward);self.deep_output_tile_calls_r4808=Fn(lambda d:self.proj);self.modern_deep_c4_tile_calls_r5920=Fn(lambda d,k:self.proj*2);self.deep_prefill_pack_counts_r8767=Fn(self.pack);self.deep_model_quant1_counts_r8886=Fn(self.count);self.modern_deep_latent_fused_calls_r681=Fn(lambda d:self.dec*43);self.deep_owner_arm_r8900=Fn(self.armfn);self.deep_owner_read_r8900=Fn(self.read)
 def pack(self,d,q,n):
  vals=[0,self.pre*43,0,self.pre*43*251,0,self.pre*86,0,self.pre*43,0,self.pre*43*2048*6*16,self.dec*43,0]
  for i,v in enumerate(vals):q[i]=v
  return 0
 def count(self,d,q,n):
  vals=[self.pre*43,0,self.pre*43*251,0,0,0,0,0]
  for i,v in enumerate(vals):q[i]=v
  return 0
 def sinkmode(self,a):self.sink_mode=a;return 0
 def sinkcount(self,d,q,n):
  for i,v in enumerate(self.sink_vals):q[i]=v+(1 if self.negative=='sinkcount'and i==1 and self.pre else 0)
  return 0
 def mode(self,a):self.scratch_mode=a;return 0
 def scratch(self,d,q,n):
  vals=[(self.pre-self.tiled)*43,self.tiled*43,(self.pre-self.tiled)*43*6144,self.tiled*43*3]
  if self.negative=='count'and self.tiled:vals[3]+=1
  if self.negative=='decodehit'and self.scratch_mode and self.dec:vals[0]+=self.dec
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
  else:assert tokens==2048 and ids.tolist()==sum([ref['input_ids'][int(i)]for i in rows],[]);self.pre+=1;self.tiled+=self.scratch_mode;cp=43*896 if self.sink_mode else 88064
  if not step:
   for i,v in enumerate([88064,cp,88064-cp,43*self.sink_mode]):self.sink_vals[i]+=v
  out=np.ctypeslib.as_array((ctypes.c_uint16*(n*129280)).from_address(ptrs[6].value)).reshape(n,129280);em=np.ctypeslib.as_array((ctypes.c_int64*n).from_address(ptrs[7].value));out.fill(0);em[:]=gold[rows,step];out[np.arange(n),em]=0x3f80
  if self.negative=='finite'and self.calls==1:out[0,0]=0x7fc0
  return 0

def execute(negative=None,budget_skip=False):
 with tempfile.TemporaryDirectory(prefix='Deep-owner-r8970-')as d:
  p=pathlib.Path(d);(p/'incoming').mkdir();(p/'results').mkdir();shutil.copyfile(r/'Deep-B32-samegeom-reference-r7269.json',p/'incoming/Deep-B32-samegeom-reference-r7269.json');report={};lib=Lib(negative);chats=[]
  def chat(epoch):chats.append(epoch);report['finite_'+epoch+'_passed']=True
  def check(code):assert code==0
  progress=[]
  def checkpoint(phase,window=0):
   progress.append((phase,window));report['phase']=phase;report['phase_window']=window;t=p/'results/progress.tmp';t.write_text(json.dumps(report));t.replace(p/'results/progress.json')
  g=dict(packet_started=matrix.time.perf_counter()-(1400 if budget_skip else 0),checkpoint=checkpoint,r=p,lib=lib,ctx=1,cfg=dict(vocab_size=129280,eos_token_id=129279,compress_ratios=[4]*43,num_hidden_layers=43),check=check,report=report,path=p/'results/report.json',run_chat=chat,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda d:(16*1024**3,32*1024**3))),os=os)
  try:matrix.run(g)
  except (AssertionError,RuntimeError)as e:
   assert negative,(negative,repr(e),lib.calls);return dict(rejected=True,CAPI_calls=lib.calls)
  assert not negative and lib.calls==(405 if budget_skip else 540) and lib.windows==[]and chats==['before','after_restore']and report['quality_passed']and report['performance_qualified']==(not budget_skip)and len(report['owner_trials_r8970'])==(3 if budget_skip else 4) and report['owner_event_windows']==[]
  assert len([v for v in progress if v[0].startswith('prefill_CAPI_enter')])==(24 if budget_skip else 32)
  assert len([v for v in progress if v[0].startswith('prefill_CAPI_return')])==(24 if budget_skip else 32)
  assert [v['sink_arm']for v in report['owner_trials_r8970']]==[0,1,1,0][:3 if budget_skip else 4]
  assert lib.tiled==lib.pre and lib.pre==(24 if budget_skip else 32) and lib.dec==(381 if budget_skip else 508)
  return dict(passed=True,CAPI_calls=lib.calls,CAPI_calls_per_arm=135,full128_IDs_exact=True,plan=['A0','C1','C2','A3'][:3 if budget_skip else 4],all8_T2048_chunks_checked_per_arm=True,owner_event_capture=False,no_extra_forward=True,QA=chats)
out=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,noGPU_dlopen=True,positive=execute(),positive_budget_skip=execute(budget_skip=True),negative_nonfinite=execute('finite'),negative_actual_counter=execute('count'),negative_decode_hit=execute('decodehit'),negative_sink_counter=execute('sinkcount'),entry_protocol_SHA256=hashlib.sha256((r/'Deep_sink_Model_protocol_r8970.py').read_bytes()).hexdigest());p=r/'Deep-sink-Model-consumer-CPU-r8970.json';p.write_bytes((json.dumps(out,indent=2)+'\n').encode('utf8'));print(json.dumps(out))
