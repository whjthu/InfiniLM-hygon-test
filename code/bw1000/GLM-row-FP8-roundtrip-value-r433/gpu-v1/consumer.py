"""Twelve complete queries. Persist real outputs before any refusal."""
from array import array
from pathlib import Path
import ctypes,hashlib,json,time
from original_consumer_r400 import NativeLeaf as OriginalLeaf,seed_chunks
from binding import bind_policy,bind_diagnostic
from diagnostic_io import read_snapshot
NUMERIC_ALERT_NRMSE=.005
NUMERIC_DIAGNOSTIC_STOP_NRMSE=.02 # diagnosticSTOP, not usertaskquality gate
CASES=(dict(name='decode-U2048-periodic',count=1,past=32767,U=2048,selection='periodic'),dict(name='decode-U2048-tail',count=1,past=32767,U=2048,selection='tail'))
MODES=(0,1,0,1,1,0)
def selections(case):return array('q',list(range(30720,32768))if case['selection']=='tail'else[i*16 for i in range(2047)]+[32767])
def consume(leaf,save):
 records=[];primary=None
 try:
  for case in CASES:
   for index,mode in enumerate(MODES):
    leaf.check_deadline();row=dict(case=case['name'],mode=mode,index=index,warm=index<2,NoV_fixed=True,Native_wrapper_attempted=True,Native_CAPI_invoked=False,Native_CAPI_returned=False,Native_CAPI_completed=False,qualification_passed=False);records.append(row);save(records,None)
    started=time.perf_counter();leaf.begin(mode);row['reset_and_bank_reallocation_wall_s']=time.perf_counter()-started;started=time.perf_counter()
    for past,count in seed_chunks(case):leaf.check_deadline();leaf.seed(past,count)
    row['seed_wall_s']=time.perf_counter()-started;ids=selections(case);row['selection_SHA256']=hashlib.sha256(ids.tobytes()).hexdigest();save(records,None)
    firstgeometry_warm=case['selection']=='periodic'and index<2
    started=time.perf_counter()
    try:row['result']=leaf.query(case,ids,firstgeometry_warm)
    except BaseException as error:
     if getattr(leaf,'latest_result',None)is not None:row['result']=leaf.latest_result
     row.update(Native_CAPI_invoked=bool(leaf.query_CAPI_invoked),Native_CAPI_returned=leaf.last_query_rc is not None,Native_CAPI_completed=leaf.last_query_rc==0,Native_CAPI_RC=leaf.last_query_rc,query_error=repr(error),checked_query_wall_s=time.perf_counter()-started);save(records,repr(error))
     try:leaf.persist(row)
     except BaseException as secondary:row['persist_secondary_error']=repr(secondary);save(records,repr(error))
     raise
    row.update(Native_CAPI_invoked=True,Native_CAPI_returned=True,Native_CAPI_RC=0,Native_CAPI_completed=True,checked_query_wall_s=time.perf_counter()-started);save(records,None)
    started=time.perf_counter();leaf.persist(row);row['raw_persist_and_diagnostic_D2H_clear_s']=time.perf_counter()-started;save(records,None)
    leaf.validate(row);row['qualification_passed']=True;save(records,None)
  return records
 except BaseException as error:primary=error;save(records,repr(error));raise
 finally:
  try:leaf.close()
  except BaseException as error:
   if primary is None:raise
   if records:records[-1]['close_secondary_error']=repr(error)
   save(records,repr(primary))
class NativeLeaf(OriginalLeaf):
 def __init__(self,lib,ctx,torch,hidden,deadline,raw_directory,rank):
  super().__init__(lib,ctx,torch,hidden,deadline);self.rank=rank;self.raw_directory=Path(raw_directory);self.sample=0;self.snapshot_armed=False;self.detail=2 if rank==0 else 1;self.query_CAPI_invoked=False;self.last_query_rc=None;self.latest_outputs=None
  self.policy=bind_policy(lib);bind_diagnostic(lib)
  lib.modern_glm_fp8_runtime_stats_r809.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_glm_fp8_runtime_stats_r809.restype=ctypes.c_int
 def stats(self):
  values=(ctypes.c_uint64*10)();self.ck(self.lib.modern_glm_fp8_runtime_stats_r809(self.rank,values,10));return dict(zip(['created','destroyed','hits','misses','evictions','entries','bytes','highwater','runs','invalid'],map(int,values)))
 def begin(self,mode):self.current=mode;self.ck(self.lib.glm_attention_begin_r398(self.ctx,1));self.ck(self.policy(self.ctx,mode))
 def seed(self,past,count):
  super().seed(past,count)
  if not getattr(self,'first_seed_observed',False):self.first_seed_observed=True;self.on_first_seed()
 def query(self,case,ids,arm):
  self.sample+=1;self.latest_outputs=None;self.latest_result=None;self.query_CAPI_invoked=False;self.last_query_rc=None;self.snapshot_armed=False
  if arm:self.ck(self.lib.glm_attention_diag_arm_r433(self.ctx,self.detail));self.snapshot_armed=True
  before=self.stats();row=super().query(case,ids);self.latest_result=row;row['identity_valid']=False;after=self.stats();n=row['native'];on=int(self.current)
  row['identity_valid']=(n.get('past')==32767 and n.get('count')==1 and n.get('union_U')==2048 and n.get('seeded')==32768 and n.get('mode')is True and n.get('persistent_V_bytes')==0 and n.get('value_roundtrip')==bool(on)
   and [n.get(k)for k in ['r433_chosen','r433_rowquant_submits','r433_roundtrip_F32_identity_calls','r433_F32_aggregation_Gemm_submits','r433_F32_V_Gemm_submits','r433_final_BF16_cast_submits']]==[on,on,on,8*on,8*on,on]
   and n.get('selectedV_HOST_completed_expand_calls')==1-on and n.get('diagnostic_snapshot_detail')==(self.detail if arm else 0))
  if self.sample>1:row['identity_valid']=row['identity_valid']and n.get('r433_identity_device_check_passed')and n.get('r433_F32_weight_setup_calls')==1 and n.get('r433_persistent_F32_V_bytes')==4194304
  row['FP8_runtime_before']=before;row['FP8_runtime_after']=after;row['identity_valid']=row['identity_valid']and before['runs']==after['runs']==after['invalid']==0
  row['host_counter_scope']='completedHOSTSDKsubmissions/helpercalls; not GPUkernelcounts';return row
 def persist(self,row):
  prefix='sample%02d-mode%d'%(self.sample,self.current);row['raw_outputs']=[];row['raw_outputs_complete']=False
  primary=None
  try:
   if self.latest_outputs is not None:
    for name,tensor in zip(['hidden','attention'],self.latest_outputs):
     data=tensor.view(self.torch.uint8).numpy().tobytes();path=self.raw_directory/(prefix+'-'+name+'.bf16.bin')
     with path.open('xb')as f:f.write(data)
     row['raw_outputs'].append(dict(name=name,dtype='BF16',shape=list(tensor.shape),nbytes=len(data),path=str(path),SHA256=hashlib.sha256(data).hexdigest()))
    row['raw_outputs_complete']=True
    if 'result'in row:assert [v['SHA256']for v in row['raw_outputs']]==[v['SHA256']for v in row['result']['numerics']]
   else:row['raw_output_unavailable']='NativeCAPI/producerdidnotreturnvalidoutputs; neverreuseprevioussample'
  except BaseException as error:primary=error;row['raw_output_error']=repr(error)
  finally:
   if self.snapshot_armed:
    try:
     row['typed_diagnostic']=read_snapshot(self.lib,self.ctx,self.raw_directory,prefix+'-typed',mode=self.current,detail=self.detail)
     if row.get('Native_CAPI_completed'):assert row['typed_diagnostic']['collection_passed']and row['typed_diagnostic']['descriptor']['query_CAPI_completed']is True and row['typed_diagnostic']['clear_RC']==0
    except BaseException as error:
     row['typed_diagnostic_error']=repr(error)
     if primary is None:primary=error
    self.snapshot_armed=False
   self.latest_outputs=None
  if primary is not None:raise primary
 def validate(self,record):
  x=record['result'];ok=record['raw_outputs_complete']and x['identity_valid']and x['canary']and all(v['finite']and v['NRMSE']<=NUMERIC_DIAGNOSTIC_STOP_NRMSE for v in x['numerics'])
  vote=self.torch.tensor([float(ok)],dtype=self.torch.float32,device='cuda');self.torch.cuda.synchronize();self.ck(self.lib.glm_attention_control_r398(self.ctx,ctypes.c_void_p(vote.data_ptr())))
  if vote.item()!=8:raise RuntimeError('r433 bounded diagnosticSTOP afterallcompletedprimaryrawsaved; not Modelquality verdict')
 def close(self):
  super().close();self.final_released_stats=self.stats();assert self.final_released_stats['created']==self.final_released_stats['destroyed']and self.final_released_stats['entries']==self.final_released_stats['bytes']==0
