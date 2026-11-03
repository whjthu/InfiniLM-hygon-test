"""Two fixed selection fixtures, NoV both arms, original complete query leaf."""
from array import array
import ctypes,hashlib,json,time
from original_consumer_r400 import NativeLeaf as OriginalLeaf,seed_chunks
NUMERIC_ALERT_NRMSE=.005
NUMERIC_DIAGNOSTIC_STOP_NRMSE=.02  # inherited conservative diagnostic, not task-quality gate
CASES=(dict(name='decode-U2048-periodic',count=1,past=32767,U=2048,selection='periodic'),
       dict(name='decode-U2048-tail',count=1,past=32767,U=2048,selection='tail'))
MODES=(0,1,0,1,1,0)
def selections(case):
 return array('q',list(range(30720,32768))if case['selection']=='tail'else[i*16 for i in range(2047)]+[32767])
def consume(leaf,save):
 records=[]
 try:
  for case in CASES:
   for index,mode in enumerate(MODES):
    leaf.check_deadline();row=dict(case=case['name'],mode=mode,index=index,warm=index<2,NoV_fixed=True,
     Native_CAPI_attempted=False,Native_CAPI_completed=False,qualification_passed=False);records.append(row);save(records,None)
    begin=time.perf_counter();leaf.begin(mode);row['reset_and_bank_reallocation_wall_s']=time.perf_counter()-begin;begin=time.perf_counter()
    for past,count in seed_chunks(case):leaf.check_deadline();leaf.seed(past,count)
    row['seed_wall_s']=time.perf_counter()-begin;indices=selections(case)
    row['selection_SHA256']=hashlib.sha256(indices.tobytes()).hexdigest();row['Native_CAPI_attempted']=True;save(records,None)
    leaf.check_deadline();begin=time.perf_counter();row['result']=leaf.query(case,indices)
    row.update(checked_query_wall_s=time.perf_counter()-begin,Native_CAPI_completed=True);save(records,None)
    leaf.validate(row);row['qualification_passed']=True;save(records,None)
  return records
 except BaseException as error:save(records,repr(error));raise
 finally:leaf.close()

class NativeLeaf(OriginalLeaf):
 def __init__(self,*args):
  super().__init__(*args);self.rank=self.torch.cuda.current_device()
  self.lib.glm_attention_value_reassociate_r431.argtypes=[ctypes.c_void_p,ctypes.c_int]
  self.lib.glm_attention_value_reassociate_r431.restype=ctypes.c_int
  self.lib.modern_glm_fp8_runtime_stats_r809.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  self.lib.modern_glm_fp8_runtime_stats_r809.restype=ctypes.c_int
 def seed(self,past,count):
  super().seed(past,count)
  if not getattr(self,'first_seed_observed',False):
   self.first_seed_observed=True;self.on_first_seed()
 def stats(self):
  values=(ctypes.c_uint64*10)();self.ck(self.lib.modern_glm_fp8_runtime_stats_r809(self.rank,values,10))
  return dict(zip(['created','destroyed','hits','misses','evictions','entries','bytes','highwater','runs','invalid'],map(int,values)))
 def begin(self,mode):
  assert mode in(0,1);self.current=mode
  self.ck(self.lib.glm_attention_begin_r398(self.ctx,1)) # NoV fixed, OFF is not cachedV.
  self.ck(self.lib.glm_attention_value_reassociate_r431(self.ctx,mode))
 def query(self,case,indices):
  before=self.stats();row=super().query(case,indices);after=self.stats();n=row['native'];on=bool(self.current)
  row['identity_valid']=(n['union_U']==2048 and n['past']==32767 and n['count']==1 and n['seeded']==32768 and n['mode']is True
   and n['persistent_V_bytes']==0 and n['value_reassociate']==on and n['r431_chosen']==int(on)
   and n['r431_aggregation_Gemm_submits']==8*int(on)and n['r431_M8_expand_completed']==int(on)
   and n['r431_diagonal_copies']==8*int(on)and n['selectedV_HOST_completed_expand_calls']==int(not on))
  row['FP8_runtime_before']=before;row['FP8_runtime_after']=after;row['FP8_runtime_runs_delta']=after['runs']-before['runs']
  row['identity_valid']=row['identity_valid']and row['FP8_runtime_runs_delta']==int(on)and after['invalid']==before['invalid']==0
  row['host_counter_scope']='actual completed HOST invocation/submission counters and original M8 cache runs, not GPU kernel counts'
  return row
 def validate(self,record):
  row=record['result'];ok=row['identity_valid']and row['canary']and all(v['finite']and v['NRMSE']<=NUMERIC_DIAGNOSTIC_STOP_NRMSE for v in row['numerics'])
  vote=self.torch.tensor([float(ok)],dtype=self.torch.float32,device='cuda');self.torch.cuda.synchronize()
  self.ck(self.lib.glm_attention_control_r398(self.ctx,ctypes.c_void_p(vote.data_ptr())))
  if vote.item()!=8:raise RuntimeError('r431 all8 diagnostic stop after persisted complete primary; not a Model quality verdict')
 def close(self):
  super().close();self.final_released_stats=self.stats()
  assert self.final_released_stats['created']==self.final_released_stats['destroyed']and self.final_released_stats['entries']==self.final_released_stats['bytes']==0

def recording_contract():
 import tempfile
 from pathlib import Path
 class Leaf:
  def __init__(self,fail=False):self.calls=[];self.closed=False;self.fail=fail
  def check_deadline(self):pass
  def begin(self,mode):self.mode=mode;self.calls.append(('begin',mode))
  def seed(self,past,count):self.calls.append(('seed',past,count))
  def query(self,case,indices):
   assert len(indices)==len(set(indices))==2048 and indices[-1]==32767
   self.calls.append(('query',self.mode));return dict(recording=True,primary_counters_and_numeric_saved=True)
  def validate(self,row):
   if self.fail and row['mode']==1:raise RuntimeError('recording-firstON diagnostic failure')
  def close(self):self.closed=True
 for fail in [False,True]:
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'result.json';leaf=Leaf(fail)
   def save(records,error):path.write_text(json.dumps(dict(records=records,error=error)))
   try:rows=consume(leaf,save)
   except RuntimeError:assert fail
   else:assert not fail and len(rows)==12 and all(r['qualification_passed']for r in rows)
   q=json.loads(path.read_bytes());assert leaf.closed
   if fail:assert len(q['records'])==2 and q['records'][-1]['Native_CAPI_completed']and not q['records'][-1]['qualification_passed']and q['error']
   else:assert sum(c[0]=='query'for c in leaf.calls)==12 and sum(c[0]=='seed'for c in leaf.calls)==780
 return dict(passed=True,actual_new_consume=True,queries=12,seed_calls=780,warms=4,formal=8,firstON_primary_saved_before_stop=True,actual_tempfile=True,Native=0,GPU=0,old_suites_rerun=False)
if __name__=='__main__':print(json.dumps(recording_contract()))
