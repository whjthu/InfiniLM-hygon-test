"""Current FP8 oracle and real metadata state checks. No cross-cache bitexact gate."""
import hashlib,numpy as np
def array(f,dtype=None):
 a=np.asarray(f['raw'],np.uint8)
 return a.view(dtype).reshape(f['shape'])if dtype else a.reshape(f['shape'])
def equal(a,b):return a['shape']==b['shape']and a['dtype']==b['dtype']and np.array_equal(a['raw'],b['raw'])
def qualify(packets,oracle,dtypes,previous=None):
 assert len(packets)==8 and {p['rank']for p in packets}==set(range(8));total=None;details=[]
 for p in packets:
  f=p['fields'];st=p['stats'];on=p['item']['timing'];assert st['counts']==[1,1,1,0,0]and st['errors']==[0]*5 and st['clocks_valid']and st['clock_reads']==(12 if on else 0)
  if not on:assert st['status_wall_ns']==st['status_thread_CPU_ns']==[0]*5
  else:assert all(st['status_wall_ns'][i]>0 and st['status_thread_CPU_ns'][i]>0 for i in range(3))
  assert f[0]['dtype']==dtypes['U8']and f[1]['dtype']==dtypes['F32']and f[4]['dtype']==dtypes['F32']and f[5]['dtype']==dtypes['I64']
  assert f[2]is None and f[3]is None,'Fixed8496 is not a compression boundary'
  assert equal(f[9],f[10]),'Prepare changed committed compressor state'
  assert f[7]['dtype']==f[9]['dtype']and f[7]['shape']==f[9]['shape']
  if f[7]['dtype']==dtypes['F32']:state=array(f[7],np.float32)
  elif f[7]['dtype']==dtypes['BF16']:state=(array(f[7],np.uint16).astype(np.uint32)<<16).view(np.float32)
  else:raise AssertionError(('Unknown actual state dtype',f[7]['dtype']))
  assert np.isfinite(state).all()
  keys,scales=oracle.gather_paged(array(f[6]),np.arange(2124),np.ones(2124,np.uint8));oldkeys,oldscales=oracle.gather_paged(array(f[8]),np.arange(2124),np.ones(2124,np.uint8))
  assert np.array_equal(keys,oldkeys)and np.array_equal(scales.view(np.uint8),oldscales.view(np.uint8)),'Nonboundary commit changed valid cache'
  local=oracle.score_cpu(array(f[0]).reshape(8,128),array(f[1],np.float32).reshape(8),keys,scales.reshape(-1),2124)
  total=local if total is None else np.add(total,local,dtype=np.float32)
  validhash=hashlib.sha256(keys.tobytes()+scales.tobytes()).hexdigest();p['_valid_cache_SHA256']=validhash
 for p in packets:
  f=p['fields'];scores=array(f[4],np.float32).reshape(-1);ids=array(f[5],np.int64).reshape(-1)
  assert len(scores)==2124 and len(ids)==512 and len(set(ids.tolist()))==512 and np.all((ids>=0)&(ids<2124))and np.isfinite(scores).all()
  diff=scores.astype(np.float64)-total.astype(np.float64);error=float(np.linalg.norm(diff)/(np.linalg.norm(total.astype(np.float64))+1e-30));assert error<2e-4,(p['rank'],error)
  ordered=np.sort(scores)[::-1];cutoff=float(ordered[511]);tol=max(1e-5,float(np.max(np.abs(scores)))*2e-5);assert np.min(scores[ids])>=cutoff-tol
  q=dict(rank=p['rank'],CPU_FP8_score_NRMSE=error,maxabs=float(np.max(np.abs(diff))),finite=True,TopK512_unique_valid=True,cutoff=cutoff,cutoff_margin_512_513=float(ordered[511]-ordered[512]),TopK_ties_not_required_bitexact=True,valid_cache_SHA256=p['_valid_cache_SHA256'],valid_cache_prepare_commit_unchanged=True,state_dtype=f[7]['dtype'],state_shape=f[7]['shape'],state_finite=True,state_prepare_readonly=True,state_commit_read=True,independent_temporal_commit_oracle=False)
  if previous:
   old=previous[p['rank']];q.update(cross_trial_valid_cache_same=old['_valid_cache_SHA256']==p['_valid_cache_SHA256'],cross_trial_state_same=equal(f[7],old['fields'][7]),cross_trial_score_maxabs=float(np.max(np.abs(scores-array(old['fields'][4],np.float32).reshape(-1)))))
  details.append(q)
 return details
