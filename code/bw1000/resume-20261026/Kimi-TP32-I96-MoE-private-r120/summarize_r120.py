"""Consume exactly four future component results; repeats do not expand sample N."""
import pathlib,json,hashlib,statistics,sys
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def summarize(paths):
 assert len(paths)==4
 data=[json.loads(pathlib.Path(p).read_bytes()) for p in paths];assert {d['node_rank'] for d in data}==set(range(4))
 assert len({d['source_manifest_SHA256'] for d in data})==1 and len({d['manifest_SHA256'] for d in data})==1
 owners=[]
 for d in data:
  assert d['completed'] and d['normal_CUDA_cleanup'] and d['native']['all8normal_before_communicators']
  owners.extend(d['native']['reports'])
 assert sorted(o['global_tp_rank'] for o in owners)==list(range(32))
 for o in owners:assert o['module_closed'] and o['normal_Runtime_close'] and o['all32_loaded_barrier']
 rows=[]
 for T in [8,16,64]:
  sample=[];hashes=[]
  for arm in range(6):
   a=[next(x for x in o['arms'] if x['T']==T and x['arm']==arm) for o in owners]
   assert all(x['nonfinite']==0 and x['input_readonly_tail_canary'] and x['private_provider_counter_delta_zero'] for x in a)
   sample.append(max(x['whole_MoE_including_outer_drain_ms'] for x in a));hashes.append(len({x['output_FNVword16'] for x in a}))
  rows.append({'T':T,'warm_calls':2,'ordinary_repeats':4,'maxowner_wall_ms_each_ordinary':sample[2:],'median_maxowner_wall_ms':statistics.median(sample[2:]),'crossrank_distinct_output_digests_each_arm_diagnostic':hashes,'timing_scope':'globalbarrier outside timer, forward+outerdrain; maxowner durations do not subtract unsynchronized clocks','not_Model_TPS':True})
 return {'completed':True,'normal_owners':32,'scope':'single historical teacher fixture standalone layer1 genericTP32 I96 only','rows':rows,'producer_SHA256':{str(p):sha(p) for p in paths},'fixture_diversity':[d['input'] for d in data],'fullModel_quality_capacity_performance':'UNQUALIFIED','postKFD_and_Slurm_terminal_still_required':True}
if __name__=='__main__':print(json.dumps(summarize(sys.argv[1:]),indent=2))
