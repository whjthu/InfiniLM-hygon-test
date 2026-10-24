"""Independent CPU-only finite-result gate/paired accounting."""
import json,pathlib,sys,statistics
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text())
assert d['completed']and d['all8normal']and d['all8normal_close']
assert d['library_SHA256']=='7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2'
assert sorted(o['device']for o in d['owners'])==list(range(8))
assert sorted(o['device']for o in d['close']if o['normal_close'])==list(range(8))
arms=['O','N']+['O','N','N','O','O','N','N','O'];paired={}
for o in d['owners']:
 assert o['normal']and o['minimum_sampled_free']>=512*1024**2
 assert [x['mode']for x in o['load']]==[0,1]
 assert all(x['free_after']>=6*1024**3 for x in o['load'])
 assert len(o['records'])==20
 assert {(x['trial'],x['past'])for x in o['records']}=={(t,p)for t in range(10)for p in [575,639]}
 for x in o['records']:
  assert x['arm']==arms[x['trial']]and x['next_past']==x['past']+1 and x['warmup']==(x['trial']<2)
  assert x['C128_boundary']==(x['past']==639)and x['validation']['finite']
  if x['trial']:
   assert len(x['validation']['components_score_pool_rotated_ring_gather'])==5
   assert all(m['NRMSE']<=.005 for m in x['validation']['components_score_pool_rotated_ring_gather'])
for past in [575,639]:
 trials=[]
 for t,arm in list(enumerate(arms))[2:]:
  r=[next(x for x in o['records']if x['trial']==t and x['past']==past)for o in d['owners']]
  trials.append({'trial':t,'arm':arm,'slowest_owner_CAPI_wall_s':max(x['native_enclosing_wall_s']for x in r),'owner_wall_range_s':[min(x['native_enclosing_wall_s']for x in r),max(x['native_enclosing_wall_s']for x in r)]})
 cycles=[]
 for c in [0,1]:
  rows=trials[c*4:c*4+4];off=statistics.mean(x['slowest_owner_CAPI_wall_s']for x in rows if x['arm']=='O');on=statistics.mean(x['slowest_owner_CAPI_wall_s']for x in rows if x['arm']=='N')
  cycles.append({'cycle':c,'OFF_s':off,'ON_s':on,'paired_speedup':off/on,'OFF_endpoint_drift_fraction':abs(rows[3]['slowest_owner_CAPI_wall_s']/rows[0]['slowest_owner_CAPI_wall_s']-1)})
 paired[str(past)]={'trials':trials,'cycles':cycles,'same8owners_ranks_not_independent_time_repetitions':True}
out={'passed':True,'actual_handle':d['actual_handle'],'all8normal':True,'all8normal_close':True,'paired':paired,'scope':'Component CAPI incl original output copies/drain; real checkpoint weights + synthetic hidden; no Model TPS/final token qualification','Model_gain':'UNKNOWN'}
print(json.dumps(out))
