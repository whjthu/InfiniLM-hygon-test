"""Same real Pending/cache lease numerical consumer; no timing or Model claim."""
import numpy as np
IDENTITY=('generation','past','before','after','Runtime','C4owner','stream','Pending','frozen_raw','query','merged_weights','tail_values','tail_scales','positions')
def array(f,dtype=None):
 if f is None:return None
 a=np.asarray(f['raw'],np.uint8)
 return a.view(dtype).reshape(f['shape'])if dtype else a.reshape(f['shape'])
def equal(x,y):
 return (x is None and y is None)or(x is not None and y is not None and x['shape']==y['shape']and x['dtype']==y['dtype']and np.array_equal(x['raw'],y['raw']))
def gather(raw,count,oracle):return oracle.gather_paged(raw,np.arange(count),np.ones(count,np.uint8))
def valid_equal(x,y,count,oracle):
 a,b=gather(array(x),count,oracle),gather(array(y),count,oracle)
 return np.array_equal(a[0],b[0])and np.array_equal(a[1].view(np.uint8),b[1].view(np.uint8))
def topk(scores,ids):
 assert scores.ndim==ids.ndim==1 and len(ids)==512 and len(set(ids.tolist()))==512
 assert np.isfinite(scores).all()and np.all((ids>=0)&(ids<len(scores)))
 ordered=np.sort(scores)[::-1];cutoff=float(ordered[511]);margin=float(ordered[511]-ordered[512]);tol=max(1e-5,float(np.max(np.abs(scores)))*2e-5)
 assert float(np.min(scores[ids]))>=cutoff-tol
 return dict(unique_valid_512=True,cutoff=cutoff,cutoff_margin_512_513=margin,tie_tolerance=tol)
def validate_same_cache(records,oracle,previous=None,committed=True):
 assert len(records)==8 and {r['rank']for r in records}==set(range(8));records=sorted(records,key=lambda r:r['rank'])
 total=None
 for r in records:
  a,b,c=[r[n]for n in ['before','after_sdk','after_paged']];ident=r['identity_before'];past=ident['past']
  assert past in [8575,8576]and ident['before']==past//4 and ident['after']==(past+1)//4
  assert ident['active']and not ident['committed']and ident['original_commit_validates_private_owner_revision']
  for name in ['identity_after_sdk','identity_after_paged']:
   assert all(r[name][k]==ident[k]for k in IDENTITY)and r[name]['active']and not r[name]['committed']
  # Includes field6: exact raw pages before original prepare, frozen pages after
  # prepare, and live pages after each score. No reset/reseed can hide drift.
  for k in [0,1,2,3,4,5,8,9,11,16]:assert equal(a[k],b[k])and equal(a[k],c[k]),('Readonly lease changed',r['rank'],k)
  for stage in [a,b,c]:
   for k in [6,7,10]:assert valid_equal(a[6],stage[k],ident['before'],oracle),'Valid history cache changed'
  assert equal(a[8],a[9])and equal(a[9],a[11]),'Prepare or scorer changed committed state'
  assert array(a[16],np.int64).tolist()==[past]
  raw=array(a[7]);before,after=ident['before'],ident['after'];keys,scales=gather(raw,before,oracle)
  if after>before:
   assert array(a[2]).shape==(1,128)and array(a[3],np.float32).shape==(1,1)
   keys=np.concatenate([keys,array(a[2])]);scales=np.concatenate([scales,array(a[3],np.float32)])
  else:assert a[2]is None and a[3]is None
  local=oracle.score_cpu(array(a[0]).reshape(8,128),array(a[1],np.float32).reshape(8),keys,scales.reshape(-1),after)
  total=local if total is None else np.add(total,local,dtype=np.float32)
  if previous is not None:
   old=previous[r['rank']];assert past==old['identity_before']['past']+1
   assert valid_equal(a[7],old['post_commit'][10],ident['before'],oracle)
   assert equal(a[8],old['post_commit'][11]),'Next state shape/dtype/rawbytes changed'
   assert all(ident[k]==old['identity_before'][k]for k in ['Runtime','C4owner','stream'])
   assert ident['generation']!=old['identity_before']['generation']
 metrics=[]
 for r in records:
  f=r['after_paged'];item={'rank':r['rank'],'past':r['identity_before']['past'],'same_cache_and_inputs':True,'scores':{}}
  for name,scorefield,idfield in [('actual_prepare',4,5),('SDK_replay',12,13),('paged_replay',14,15)]:
   scores=array(f[scorefield],np.float32).reshape(-1);ids=array(f[idfield],np.int64).reshape(-1)
   assert scores.shape==total.shape and np.isfinite(scores).all()
   diff=scores.astype(np.float64)-total.astype(np.float64);error=float(np.linalg.norm(diff)/(np.linalg.norm(total.astype(np.float64))+1e-30));assert error<2e-4,(name,r['rank'],error)
   item['scores'][name]=dict(CPU_FP8_NRMSE=error,maxabs=float(np.max(np.abs(diff))),**topk(scores,ids))
  if not committed:metrics.append(item);continue
  # commit is once, on this original Pending, after both score paths.
  ident=r['identity_post_commit'];assert ident['committed']and not ident['active']and all(ident[k]==r['identity_before'][k]for k in IDENTITY)
  after=array(r['post_commit'][10]);expected=array(f[7]).copy();base=r['identity_before']['before'];end=r['identity_before']['after']
  if end>base:
   page,row=divmod(base,64);expected[page,row*128:(row+1)*128]=array(f[2])[0]
   expected[page,8192+row*4:8196+row*4]=array(f[3],np.float32).reshape(-1).view(np.uint8)
  expected_keys,expected_scales=gather(expected,end,oracle);actual_keys,actual_scales=gather(after,end,oracle)
  assert np.array_equal(actual_keys,expected_keys)and np.array_equal(actual_scales.view(np.uint8),expected_scales.view(np.uint8)),'Commit changed valid prefix beyond exact pending newtail'
  metrics.append(item)
 return dict(passed=True,owner_records=8,past=records[0]['identity_before']['past'],same_cache=True,original_prepare_and_both_scores_against_one_CPU_oracle=True,TopK_ties_not_required_bitexact=True,metrics=metrics,
  nextstep_committed_cache_and_state_consumed=previous is not None,commit_checked=committed,invalid_raw_padding_is_diagnostic_only=True,not_Model_or_performance_qualification=True)
