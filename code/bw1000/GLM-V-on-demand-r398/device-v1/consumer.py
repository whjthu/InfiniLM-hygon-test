"""Same schedule for real attention CABI and explicitly recording CPU leaves.

No HTTP, full Model, actual routing or task-quality claim. Seed is synthetic
hidden through original checkpoint projections. Prefix construction is recorded
separately and complete query includes Q/freshKV/copy/gather/reprojection/O/TP.
"""
from array import array
import ctypes,hashlib,json,time

CASES=(dict(name='decode-U2048',count=1,past=32767,U=2048),
       dict(name='prefill512-union32768',count=512,past=32256,U=32768))
MODES=(0,1,0,1,1,0)  # 2warm + one fixed ABBA, not independent owner trials.
NUMERIC_ALERT_NRMSE=.005
NUMERIC_DIAGNOSTIC_STOP_NRMSE=.02

def seed_chunks(case):
 past=case['past'];out=[];p=0
 while p<past:
  n=min(512,past-p);out.append((p,n));p+=n
 out.append((past,case['count']))
 assert sum(n for _,n in out)==32768
 return out

def selections(case):
 if case['count']==1:
  return array('q',(i*16 for i in range(2047)))+array('q',[32767])
 result=array('q');past=case['past']
 for t in range(512):
  result.extend((t*64+j)%past for j in range(2047))
  result.append(past+t)
 return result

def invoke_case(leaf,case,mode,index):
 leaf.check_deadline();leaf.begin(mode);start=time.perf_counter()
 for past,count in seed_chunks(case):
  leaf.check_deadline();leaf.seed(past,count)
 seed_wall=time.perf_counter()-start
 indices=selections(case);leaf.check_deadline();start=time.perf_counter()
 result=leaf.query(case,indices)
 wall=time.perf_counter()-start
 record=dict(case=case['name'],mode=mode,index=index,warm=index<2,
             seed_wall_s=seed_wall,checked_query_wall_s=wall,
             selection_SHA256=hashlib.sha256(indices.tobytes()).hexdigest(),
             result=result)
 if index==0:record['V_stage_outside_benchmark']=leaf.inspect(case,indices)
 return record

def consume(leaf,save):
 records=[]
 try:
  for case in CASES:
   for index,mode in enumerate(MODES):
    result=invoke_case(leaf,case,mode,index);records.append(result);save(records,None)
    leaf.validate(result)  # save all fields before any diagnostic stop
  return records
 except BaseException as error:
  save(records,repr(error));raise
 finally:
  leaf.close()

class NativeLeaf:
 """Actual device leaf, not entered by CPU contract. One checkpoint load.

 Inputs are retained synthetic BF16 hidden, deterministic on CPU and copied
 once. Same input/prefix/weights/selection for OFF/ON. No captures/Model state.
 """
 def __init__(self,lib,ctx,torch,hidden,deadline):
  self.lib,self.ctx,self.torch,self.hidden,self.deadline=lib,ctx,torch,hidden,deadline
  self.references={};self.current=None;self.closed=False;self.destroy_attempted=False
  V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
  lib.glm_attention_error_r398.restype=ctypes.c_char_p
  lib.glm_attention_report_r398.restype=ctypes.c_char_p
  lib.glm_attention_begin_r398.argtypes=[V,I]
  lib.glm_attention_seed_r398.argtypes=[V,Z,Z,V,V]
  lib.glm_attention_query_r398.argtypes=[V,Z,Z,V,V,V,V,V]
  lib.glm_attention_compare_v_r398.argtypes=[V,Z,Z,V]
  lib.glm_attention_control_r398.argtypes=[V,V]
  lib.glm_attention_destroy_r398.argtypes=[V,ctypes.POINTER(I)]
 def ck(self,rc):
  if rc:raise RuntimeError(self.lib.glm_attention_error_r398().decode())
 def check_deadline(self):
  if time.time()>=self.deadline:raise TimeoutError('r398 original epoch; no reset/retry')
 def begin(self,mode):
  self.current=mode;self.ck(self.lib.glm_attention_begin_r398(self.ctx,mode))
 def seed(self,past,count):
  t=self.torch;p=t.arange(past,past+count,dtype=t.int64,device='cuda')
  x=self.hidden[past:past+count];t.cuda.synchronize()
  self.ck(self.lib.glm_attention_seed_r398(self.ctx,past,count,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(p.data_ptr())))
 def query(self,case,indices):
  t=self.torch;n=case['count'];past=case['past'];x=self.hidden[past:past+n]
  pos=t.arange(past,past+n,dtype=t.int64,device='cuda');selected=t.tensor(indices,dtype=t.int64,device='cuda').view(n,2048)
  y=t.full((n*6144+128,),123.,dtype=t.bfloat16,device='cuda');out=y[64:-64].view(n,6144)
  z=t.full((n*8*256+128,),123.,dtype=t.bfloat16,device='cuda');attn=z[64:-64].view(n,8,256)
  # Driver producer/selector construction is outside dispatch, and separately
  # inside checked_query_wall. Original CAPI repeats selected metadata validation.
  t.cuda.synchronize();start=time.perf_counter()
  self.ck(self.lib.glm_attention_query_r398(self.ctx,past,n,*[ctypes.c_void_p(v.data_ptr())for v in (x,pos,selected,out,attn)]))
  t.cuda.synchronize();dispatch=time.perf_counter()-start
  report=json.loads(self.lib.glm_attention_report_r398());got=[out.cpu().clone(),attn.cpu().clone()]
  metrics=[];key=case['name']
  if key not in self.references:
   assert self.current==0;self.references[key]=[v.clone()for v in got]
  for name,a,b in zip(('hidden','attention'),got,self.references[key]):
   d=a.float()-b.float();rms=float(b.float().square().mean().sqrt());metrics.append(dict(name=name,finite=bool(t.isfinite(a).all()),NRMSE=float(d.square().mean().sqrt())/max(rms,1e-30),maxabs=float(d.abs().max()),reference_RMS=rms,SHA256=hashlib.sha256(a.view(t.uint8).numpy().tobytes()).hexdigest()))
  canary=all(bool((v==123).all())for v in (y[:64],y[-64:],z[:64],z[-64:]));result=dict(dispatch_wall_s=dispatch,native=report,numerics=metrics,canary=canary)
  # No token, acceptance or Model quality qualification. Save numerical fields
  # before the caller decides an alert; invalid/nonfinite/state errors fail stop.
  result['identity_valid']=report['union_U']==case['U']and report['mode']==bool(self.current)
  result['numeric_alert']=any(v['NRMSE']>NUMERIC_ALERT_NRMSE for v in metrics)
  return result
 def inspect(self,case,indices):
  t=self.torch;selected=t.tensor(indices,dtype=t.int64,device='cuda').view(case['count'],2048);t.cuda.synchronize()
  self.ck(self.lib.glm_attention_compare_v_r398(self.ctx,case['past'],case['count'],ctypes.c_void_p(selected.data_ptr())))
  result=json.loads(self.lib.glm_attention_report_r398());assert result['diagnostic_only']and not result['timed_query']
  return result
 def validate(self,record):
  r=record['result'];metrics=r['numerics']
  ok=r['identity_valid']and r['canary']and all(v['finite']and v['NRMSE']<=NUMERIC_DIAGNOSTIC_STOP_NRMSE for v in metrics)
  v=record.get('V_stage_outside_benchmark')
  ok=ok and (not v or (v['finite']and v['NRMSE']<=NUMERIC_DIAGNOSTIC_STOP_NRMSE))
  vote=self.torch.tensor([float(ok)],dtype=self.torch.float32,device='cuda');self.torch.cuda.synchronize();self.ck(self.lib.glm_attention_control_r398(self.ctx,ctypes.c_void_p(vote.data_ptr())))
  if vote.item()!=8:raise RuntimeError('r398 unanimous finite/state/numerical diagnostic stop; full fields saved; not a Model quality verdict')
 def close(self):
  if not self.destroy_attempted:
   self.destroy_attempted=True
   flag=ctypes.c_int();self.destroy_rc=int(self.lib.glm_attention_destroy_r398(self.ctx,ctypes.byref(flag)));self.runtime_expired=bool(flag.value)
   self.ck(self.destroy_rc);assert self.runtime_expired;self.closed=True

def recording_contract():
 class Leaf:
  def __init__(self,fail=False):self.calls=[];self.fail=fail;self.closed=False
  def check_deadline(self):pass
  def begin(self,mode):self.calls.append(('begin',mode))
  def seed(self,past,count):self.calls.append(('seed',past,count))
  def query(self,case,indices):
   self.calls.append(('query',case['name'],len(indices)))
   assert len(indices)==case['count']*2048
   if self.fail=='query':raise RuntimeError('recording-only Native leaf error')
   return dict(recording_only=True,Native_executed=False)
  def inspect(self,case,indices):self.calls.append(('inspect',case['name']));return dict(recording_only=True,Native_executed=False)
  def validate(self,record):
   self.calls.append(('validate',record['case']))
   if self.fail=='validate':raise RuntimeError('recording-only numerical diagnostic stop')
  def close(self):self.calls.append(('close',));self.closed=True
 saved=[];l=Leaf();out=consume(l,lambda r,e:saved.append((len(r),e)))
 assert len(out)==12 and l.closed and saved[-1]==(12,None)
 assert len([c for c in l.calls if c[0]=='query'])==12
 assert len([c for c in l.calls if c[0]=='inspect'])==2
 assert len([c for c in l.calls if c[0]=='validate'])==12
 assert [r['mode']for r in out[:6]]==list(MODES)
 failed=[];bad=Leaf('query')
 try:consume(bad,lambda r,e:failed.append((len(r),e)))
 except RuntimeError:pass
 else:raise AssertionError('missing leaf failure')
 assert bad.closed and failed[-1][0]==0 and failed[-1][1]
 failed=[];bad=Leaf('validate')
 try:consume(bad,lambda r,e:failed.append((len(r),e)))
 except RuntimeError:pass
 else:raise AssertionError('missing diagnostic stop')
 assert bad.closed and failed[-1][0]==1 and failed[-1][1]
 return dict(passed=True,actual_consumer_cases=12,query_calls=12,diagnostic_V_calls=2,save_before_diagnostic_stop=True,seed_calls=sum(len(seed_chunks(c))*6 for c in CASES),partialfailure_saved=True,normal_and_error_close=True,Native_executed=False,GPU=0,weights=0,numerical_qualification=False)

if __name__=='__main__':print(json.dumps(recording_contract()))
