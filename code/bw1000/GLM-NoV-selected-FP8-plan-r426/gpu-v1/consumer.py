"""One layer38 NoV instance. Only exactselected-KVB planreuse changes per arm."""
from array import array
import ctypes,hashlib,json,os,time
from original_consumer_r400 import NativeLeaf as OriginalLeaf,NUMERIC_ALERT_NRMSE,NUMERIC_DIAGNOSTIC_STOP_NRMSE
CASES=(dict(name='decode1-union2048',count=1,past=32760,U=2048),)
MODES=(0,1,0,1,1,0)
QUERIES=8
STAT_NAMES=('created','destroyed','hits','misses','evictions','entries','bytes','highwater','runs','invalid')

def initial_chunks():
 out=[];past=0
 while past<32760:
  count=min(512,32760-past);out.append((past,count));past+=count
 return out

def selections(case):return array('q',(i*16 for i in range(2047)))+array('q',[case['past']])

def consume(leaf,save):
 records=[]
 try:
  case=CASES[0]
  for index,mode in enumerate(MODES):
   leaf.check_deadline();phase=dict(index=index,mode=mode,warm=index<2,case=case['name'],NoV_fixed=True,queries=[],prefix_rows=32760,phase_complete=False)
   records.append(phase);save(records,None);beg=time.perf_counter();leaf.begin(mode)
   seed_start=time.perf_counter()
   for past,count in initial_chunks():leaf.check_deadline();leaf.seed(past,count)
   phase['seed_wall_s']=time.perf_counter()-seed_start;phase['seed_calls']=len(initial_chunks());phase['after_seed_stats']=leaf.stats()
   sequence_start=time.perf_counter()
   for cycle in range(QUERIES):
    current=dict(case,name=case['name']+'-cycle'+str(cycle),past=32760+cycle);indices=selections(current)
    assert len(set(indices))==2048 and min(indices)>=0 and max(indices)==current['past']
    leaf.check_deadline();tick=time.perf_counter();leaf.seed(current['past'],1);tail_seed_wall=time.perf_counter()-tick
    tick=time.perf_counter();result=leaf.query(current,indices);wall=time.perf_counter()-tick
    record=dict(cycle=cycle,past=current['past'],mode=mode,tail_seed_wall_s=tail_seed_wall,checked_query_wall_s=wall,selection_SHA256=hashlib.sha256(indices.tobytes()).hexdigest(),result=result);phase['queries'].append(record)
    save(records,None) # Save all returnednumeric/state/counterfields BEFOREstop.
    leaf.validate(record)
   clear_start=time.perf_counter();phase['final_clear']=leaf.clear();phase['final_clear_wall_s']=time.perf_counter()-clear_start
   phase.update(phase_complete=True,whole_phase_checked_wall_s=time.perf_counter()-beg,eight_query_sequence_including_tail_seed_and_clear_s=time.perf_counter()-sequence_start,tail_seed_calls=8,total_projected_rows=32768,observer_scope='Includesprefixconstruction,tailseedappend,query/pack/cat/unusedK/attentionO/TP/originalfences,inlineCPUmetrics/JSONsave,validationvotes andinitial/finalplanclear. NotobserverfreeModelthroughput.')
   save(records,None)
  return records
 except BaseException as error:save(records,repr(error));raise
 finally:leaf.close()

class NativeLeaf(OriginalLeaf):
 def __init__(self,*args):
  super().__init__(*args);self.rank=self.torch.cuda.current_device()
  self.lib.modern_glm_fp8_selected_plan_stats_r426.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
  self.lib.modern_glm_fp8_selected_plan_clear_r426.argtypes=[ctypes.c_int]
  self.lib.modern_glm_fp8_runtime_error_r809.restype=ctypes.c_char_p
 def error(self,rc):
  if rc:raise RuntimeError(self.lib.modern_glm_fp8_runtime_error_r809().decode())
 def stats(self):
  values=(ctypes.c_uint64*10)();self.error(self.lib.modern_glm_fp8_selected_plan_stats_r426(self.rank,values,10));return dict(zip(STAT_NAMES,map(int,values)))
 def clear(self):
  self.error(self.lib.modern_glm_fp8_selected_plan_clear_r426(self.rank));out=self.stats();assert out['created']==out['destroyed']and out['entries']==out['bytes']==0;return out
 def begin(self,mode):
  assert mode in(0,1);self.clear();os.environ['GLM_SELECTED_KVB_PLAN_R426']=str(mode)
  self.current=mode;self.query_cycle=0;self.ck(self.lib.glm_attention_begin_r398(self.ctx,1)) # NoV botharms.
 def query(self,case,indices):
  before=self.stats();result=super().query(case,indices)
  native=result['native'];result['identity_valid']=native['union_U']==2048 and native['past']==case['past']and native['count']==1 and native['seeded']==case['past']+1 and native['mode']is True and native['selectedV_HOST_completed_expand_calls']==1 and native['selectedV_max_block_rows']==8192
  result['plan_stats_before']=before;result['plan_reuse_mode']=self.current;result['sequence_query_cycle']=self.query_cycle
  try:
   after=self.stats();delta={k:after[k]-before[k]for k in STAT_NAMES};result['plan_stats_after']=after;result['plan_stats_delta']=delta
   if self.current:counter_ok=delta['runs']==1 and delta['created']==int(self.query_cycle==0)and delta['hits']==int(self.query_cycle>0)and delta['destroyed']==delta['evictions']==0 and after['entries']==1 and after['bytes']==20447232
   else:counter_ok=all(delta[k]==0 for k in ['runs','hits','created','destroyed','evictions'])
   result['identity_valid']=result['identity_valid']and counter_ok
  except BaseException as error:result['plan_stats_error']=repr(error);result['identity_valid']=False
  self.query_cycle+=1;return result
 def close(self):
  super().close();self.final_released_stats=self.stats();assert self.final_released_stats['created']==self.final_released_stats['destroyed']and self.final_released_stats['entries']==self.final_released_stats['bytes']==0

def recording_contract():
 class Leaf:
  def __init__(self,fail=False):self.fail=fail;self.calls=[];self.closed=False;self.mode=None
  def check_deadline(self):pass
  def begin(self,mode):self.mode=mode;self.calls.append(('begin',mode))
  def stats(self):return dict(recording=True)
  def seed(self,past,count):self.calls.append(('seed',past,count))
  def query(self,case,indices):self.calls.append(('query',self.mode,len(indices)));return dict(recording_only=True,full_fields_saved=True,Native_executed=False)
  def validate(self,row):
   self.calls.append(('validate',row['cycle']))
   if self.fail:raise RuntimeError('recordedfirstcounter/numericfailure;notactualGPU')
  def clear(self):self.calls.append(('clear',));return dict(recording=True)
  def close(self):self.closed=True
 l=Leaf();saved=[];rows=consume(l,lambda r,e:saved.append(json.loads(json.dumps(dict(rows=r,error=e)))))
 assert l.closed and len(rows)==6 and all(r['phase_complete']and len(r['queries'])==8 and r['NoV_fixed']for r in rows)
 assert [r['mode']for r in rows]==list(MODES)and sum(c[0]=='seed'for c in l.calls)==432 and sum(c[0]=='query'for c in l.calls)==48
 assert all([q['past']for q in row['queries']]==list(range(32760,32768))and row['total_projected_rows']==32768 for row in rows)
 bad=Leaf(True);saved=[]
 try:consume(bad,lambda r,e:saved.append(json.loads(json.dumps(dict(rows=r,error=e)))))
 except RuntimeError:pass
 else:raise AssertionError('Failureleafnotstopped')
 assert bad.closed and len(saved[-1]['rows'])==1 and len(saved[-1]['rows'][0]['queries'])==1 and saved[-1]['rows'][0]['queries'][0]['result']['full_fields_saved']and saved[-1]['error']
 return dict(passed=True,actual_new_sequence_consumer=True,phases=6,warms=2,formal=4,queries=48,prefix_seed_calls=384,tail_seed_calls=48,total_seed_calls=432,total_rows_projected_per_sequence=32768,NoV_fixed=True,primary_fields_persisted_before_diagnostic_stop=True,normal_and_error_close=True,Native=0,GPU=0,SDK=False,weights=False,not_math_or_device_qualification=True)

if __name__=='__main__':print(json.dumps(recording_contract(),indent=2))
