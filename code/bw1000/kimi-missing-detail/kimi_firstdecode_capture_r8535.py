"""Finite already-loaded CAPI prefix consumer, teacher-forced diagnosis only."""
import pathlib,os,json,threading,hashlib
from kimi_matched_trace_r8467 import digest
KEYS=['INFINILM_KIMI_TRACE_DIR','INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506','INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506','INFINILM_KIMI_TRACE_HASH_ONLY_R8506','INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508','INFINILM_KIMI_KDA_TRACE_LAYER','INFINILM_KIMI_TRACE_LAYERS','INFINILM_KIMI_TRACE_KV_POOL_R8511','INFINILM_KIMI_TRACE_STATE_ROWS_MODE_R8516']
def replay_cut(adapter,cut,*,stage,case,root,collect,persist,layers,points,hash_only=True,detail_layer=None,observe_kv=False,state_rows="target"):
 assert threading.current_thread()is threading.main_thread() and 0<=stage<4
 assert cut['target_request']==0 and cut['end_ordinal0']==1
 assert case and all(c.isalnum()or c in '-_'for c in case)
 assert detail_layer is None or type(detail_layer)is int and 0<=detail_layer<93
 assert not observe_kv or detail_layer is not None
 assert state_rows in ('target','all')
 assert layers and all(type(v)is int and 0<=v<=93 for v in layers)
 assert points and all(all(c.isalnum()or c=='_'for c in v)for v in points)
 base=pathlib.Path(root)/case/('stage'+str(stage));base.mkdir(parents=True,exist_ok=False)
 old={k:os.environ.get(k)for k in KEYS};records=[]
 try:
  for seq,record in enumerate(cut['jobs']):
   j=record['job'];assert j['seq']==seq and digest(j)==record['job_SHA256']
   trace=None;target=None
   if 0 in j['keys']:
    target=j['keys'].index(0);trace=base/('job'+str(seq));trace.mkdir(exist_ok=False)
    for k,v in zip(KEYS,[str(trace),','.join(map(str,layers)),','.join(points),'1'if hash_only else'0',str(target),str(detail_layer)if detail_layer is not None else'999',str(detail_layer)if detail_layer is not None else'1000','1'if observe_kv else'0',state_rows]):os.environ[k]=v
   else:
    for k in KEYS:os.environ.pop(k,None)
   ids=adapter.forward(j)
   packets=collect(case+':job'+str(seq),dict(job_SHA256=digest(j),ids=ids,trace_dir=str(trace)if trace else None,observation=getattr(adapter,'reference_observation',None)))
   assert len(packets)==4 and all(p['job_SHA256']==record['job_SHA256']for p in packets)
   actual=packets[3]['ids'];expected=record['frozen_actual_ids']
   first=next((i for i,(a,b)in enumerate(zip(actual,expected))if a!=b),min(len(actual),len(expected))if len(actual)!=len(expected)else None)
   value=dict(seq=seq,job=j,actual_ids=actual,frozen_ids=expected,firstdiff=first,packed_target=target,original_token_offsets=[sum(map(len,j['inputs'][:i]))for i in range(len(j['keys'])+1)],perstage_trace_dirs=[p['trace_dir']for p in packets],emitter_observation=packets[3].get('observation'),scope='Causal prefix replay, NOT generation gold/service performance')
   records.append(value);persist(value) # Full outputs and metadata BEFORE assertion.
   if getattr(adapter,'full_logit_reference',False):
    observation=packets[3]['observation']
    assert observation and observation['finite'],'nonfinite full-logit output; evidence saved'
   # Fixed teacher-forced inputs allow historical ID/CPUargmax mismatch to be
   # recorded without suppressing the missing W capture. No numerical promotion.
 finally:
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v
 return records
