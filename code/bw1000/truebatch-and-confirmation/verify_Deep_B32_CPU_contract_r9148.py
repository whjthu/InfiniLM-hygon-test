from pathlib import Path
import tempfile,json,sys,importlib.util,ctypes as C,hashlib,time
import numpy as np
r=Path.cwd();p=r/'Deep-B32-confirmation-private-r9148/Deep_eos_compaction_r9148.py';sys.path.insert(0,str(p.parent));spec=importlib.util.spec_from_file_location('deep_eos_cpu_fixture',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
class Lib:
 def __init__(self):self.state={};self.serial=0;self.saw_noncontiguous=False;self.padded_steps=0;self.active_steps=0
 def __getattr__(self,n):
  if n=='modern_native_engine_reset':f=lambda ctx:self.reset()
  elif n=='modern_native_engine_forward':f=self.forward
  elif n=='deep_packed_varlen_capability_r9101b':f=lambda:1
  elif n.endswith('counts_r8996b')or n.endswith('counts_r9040')or n.endswith('counts_r9054'):f=self.counts
  else:f=lambda *a:0
  z=Fn(f);setattr(self,n,z);return z
 def reset(self):self.state={};return 0
 def counts(self,d,ptr,n):
  v=C.cast(ptr,C.POINTER(C.c_uint64))
  for i in range(n):v[i]=self.serial*(43 if i in [0,3,4]else 0)
  return 0
 def forward(self,ctx,T,A,*args):
  def arr(j,t,n):return np.ctypeslib.as_array(C.cast(args[j],C.POINTER(t)),shape=(n,))
  rows=arr(0,C.c_int32,A);past=arr(1,C.c_int32,A);total=arr(2,C.c_int32,A);offset=arr(3,C.c_int32,A+1);ids=arr(4,C.c_int64,T);pos=arr(5,C.c_int64,T)
  assert offset[0]==0 and offset[-1]==T and len(set(rows.tolist()))==A
  if A and rows.tolist()!=list(range(A)):self.saw_noncontiguous=True
  logits=np.ctypeslib.as_array(C.cast(args[6],C.POINTER(C.c_uint16)),shape=(A*16,)).reshape(A,16);out=np.ctypeslib.as_array(C.cast(args[7],C.POINTER(C.c_int64)),shape=(A,))
  logits[:]=0
  for j,rr in enumerate(rows.tolist()):
   width=int(offset[j+1]-offset[j]);assert int(total[j])==int(past[j])+width
   if int(past[j])==0:assert rr not in self.state;self.state[rr]={'past':0,'steps':0,'EOS':False}
   s=self.state[rr];assert s['past']==int(past[j]),(rr,s,int(past[j]));assert pos[offset[j]:offset[j+1]].tolist()==list(range(int(past[j]),int(total[j])))
   self.padded_steps+=int(s['EOS']);self.active_steps+=int(not s['EOS']);s['past']=int(total[j]);s['steps']+=1;token=1 if s['steps']>=1+rr%4 else 2;s['EOS']=s['EOS']or token==1;out[j]=token;logits[j,token]=0x3f80
  self.serial+=1;return 0
class Tok:
 bos_token_id=0;eos_token_id=1
 def decode(self,ids,**kw):return '#### 0'
class CUDA:
 @staticmethod
 def mem_get_info(d):return (60*1024**3,64*1024**3)
class Torch:cuda=CUDA()
def check(x):assert x==0
with tempfile.TemporaryDirectory(prefix='DeepEosCPU-')as tmp:
 root=Path(tmp);(root/'incoming').mkdir();(root/'results').mkdir();dataset=json.loads((r/'Deep-B32-confirmation-private-r9147/quality-bench-gsm8k-fresh64-r9147.json').read_bytes());(root/'incoming/quality-bench-gsm8k-fresh64-r9147.json').write_text(json.dumps(dataset));cases=[dict(id=c['id'],input_ids=[8]*80,input_tokens=80,max_new_tokens=8)for c in dataset['cases']];prepared={'cases':cases,'accepted_count':64,'skipped':[],'template_provenance':'CPU_SYNTHETIC_FIXTURE_ONLY'};tp=root/'results/tokenized.json';tp.write_text(json.dumps(prepared));lib=Lib();report={};g={'r':root,'lib':lib,'ctx':object(),'cfg':{'vocab_size':16,'eos_token_id':1},'check':check,'report':report,'path':root/'results/report.json','manifest':{'tokenized_path':'results/tokenized.json','existing_remote_SHA256':{'results/tokenized.json':hashlib.sha256(tp.read_bytes()).hexdigest()}},'tokenizer':Tok(),'os':__import__('os'),'packet_started':time.perf_counter(),'torch':Torch()};module.run(g)
 assert report['grid_complete']and len(report['quality_trials'])==4 and lib.saw_noncontiguous
 for B in ['32']:
  arms=report['grid'][B]['arms'];assert arms['fixed']['cases']==arms['compact']['cases']==64;assert arms['compact']['physical_decode_row_steps']<arms['fixed']['physical_decode_row_steps']
 out={'passed':True,'CPU_only':True,'GPU':False,'math_mocked':True,'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'native_model_performance':False,'all128request_records':sum(len(t['records'])for t in report['quality_trials']),'actual_noncontiguous_logical_rows':lib.saw_noncontiguous,'perrow_past_total_offset_position_checked':True,'independent_EOS_retirement_checked':True,'model_quality_claim':False,'no_synthetic_performance_saved':True};dest=r/'Deep-eos-compaction-CPU-contract-r9148.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))

