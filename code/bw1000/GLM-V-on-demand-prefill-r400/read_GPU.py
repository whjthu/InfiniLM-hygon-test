"""Stdlib re-audit of original735/803 observations; no device or network calls."""
from pathlib import Path
import hashlib,json,statistics,sys

MF='513c97274f50cd7fea1b881aea20a02ca1773715c8d5008e9592b24788148800'
LIB='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7'
HELPER='e32b4dc2749b4012efbf4322d334b6460dd091190bca78bb8f41e4384f09a7e8'
COMMIT='480aa04397bc1780062069293bf42375a3141ca8'
MODES=[0,1,0,1,1,0]
sha=lambda b:hashlib.sha256(b).hexdigest()

def checked_file(f):
 d=f['data'];assert sha(json.dumps(d,indent=2).encode())==f['SHA256'];return d

def read(path):
 raw=Path(path).read_bytes();o=json.loads(raw);w=checked_file(o['watch'])
 assert o['read_only']and o['no_new_jobs']
 assert w['completed']and w['passed']and w['normal8']and w['srun_exit0']and w['Slurm_COMPLETED0']
 assert w['actual_step']=='910371.735'and w['GPU_source_manifest_SHA256']==MF and w['publication_commit']==COMMIT
 assert len(w['submissions'])==1 and w['no_retry']and w['profile']==0 and not w['Model_quality_qualified']
 ds=[checked_file(f)for f in o['owners']]
 assert len(ds)==8 and {d['rank']for d in ds}==set(range(8))and len({d['physical_PCI']for d in ds})==8
 assert len(set().union(*(set(d['CPU_affinity'])for d in ds)))==32 and all(len(d['CPU_affinity'])==4 for d in ds)
 indexed={d['rank']:f for d,f in zip(ds,o['owners'])}
 for f in w['owners']:assert f['SHA256']==indexed[f['rank']]['SHA256']
 for d in ds:
  assert d['actual_step']=='910371.735'and d['GPU_source_manifest_SHA256']==MF and d['library_SHA256']==LIB
  assert d['completed']and d['normal']and not d['partial_error']and d['normal_destroy']and d['Runtime_weak_expired']and d['Runtime_expired']
  assert d['destroy_attempted_once']and d['destroy_RC']==0 and d['policy_closed']and d['Torch_after_destroy']and d['weights_loaded_once']and d['input_immutable']
  assert d['load']['keys']==12 and d['load']['checkpoint_load_passes']==1
  assert d['input_shape']==[32768,6144]and d['actual_checkpoint_payload_loaded']and not d['wholeModel_loaded']and not d['hidden_is_Model_activation']and not d['selected_indices_are_Model_routing']
  assert not d['indexer_included']and d['profile']==0 and d['gcnArchName'].split(':')[0]=='gfx936'
  helper=d['actual_symbol_owners']['glm_fp8_create_r776'];assert d['actual_mapped_SDK'][helper]==HELPER
  assert d['measurement_calls']==4 and d['warm_calls']==2 and d['directV_diagnostics']==1 and len(d['records'])==6
  for i,r in enumerate(d['records']):
   assert(r['case'],r['index'],r['mode'],r['warm'])==('prefill512-union32768',i,MODES[i],i<2)
   a=r['result'];n=a['native'];assert a['identity_valid']and a['canary']and not a['numeric_alert']
   assert(n['past'],n['count'],n['union_U'],n['seeded'])==(32256,512,32768,32768)
   assert n['selectedV_HOST_completed_expand_calls']==(4 if r['mode']else 0)and n['selectedV_max_block_rows']==8192
   assert n['persistent_V_bytes']==(0 if r['mode']else 134217728)
   assert n['logical_persistent_bytes']==(171966464 if r['mode']else 306184192)
   assert not n['indexer_included']and n['profile']==0
   assert {m['name']for m in a['numerics']}=={'hidden','attention'}and all(m['finite']and m['NRMSE']<=.02 for m in a['numerics'])
  v=d['records'][0]['V_stage_outside_benchmark'];assert v['finite']and v['NRMSE']<=.02 and v['diagnostic_only']and not v['timed_query']and v['union_U']==32768
  assert v['selectedV_HOST_completed_expand_calls']==4 and v['selectedV_max_block_rows']==8192
 for key,step in [('before','910371.734'),('post','910371.736')]:
  p=w[key];assert p['passed']and p['actual_step']==step and not p['KFD_pids']and not p['matching_GPU_driver_PIDs']and not p['expected_PIDs_still_exist']
 ac=[x.split('|')for x in o['accounting']]
 for step in ['910371.734','910371.735','910371.736','910407.803']:assert sum(r[0]==step and r[2:4]==['COMPLETED','0:0']for r in ac)==1
 elapsed=next(r[4]for r in ac if r[0]=='910371.735');hh,mm,ss=map(int,elapsed.split(':'));elapsed_s=hh*3600+mm*60+ss;assert elapsed_s==123
 assert len({d['synthetic_hidden_SHA256']for d in ds})==1
 rows=[]
 for i in range(6):
  rr=[d['records'][i]for d in ds]
  rows.append(dict(index=i,mode=MODES[i],warm=i<2,max8_native_ms=max(r['result']['native']['complete_attention_CAPI_ms']for r in rr),max8_dispatch_s=max(r['result']['dispatch_wall_s']for r in rr),max8_checked_s=max(r['checked_query_wall_s']for r in rr),max8_seed_s=max(r['seed_wall_s']for r in rr),max8_seed_plus_checked_s=max(r['seed_wall_s']+r['checked_query_wall_s']for r in rr)))
 fields=[k for k in rows[0]if k.startswith('max8_')]
 ratios={k:statistics.median(r[k]for r in rows[2:]if r['mode'])/statistics.median(r[k]for r in rows[2:]if not r['mode'])for k in fields}
 drift={k:abs(rows[5][k]/rows[2][k]-1)for k in fields}
 pair={k:[rows[3][k]/rows[2][k],rows[4][k]/rows[5][k]]for k in fields}
 perowner=[dict(rank=d['rank'],native_ON_over_OFF=statistics.median(d['records'][i]['result']['native']['complete_attention_CAPI_ms']for i in [3,4])/statistics.median(d['records'][i]['result']['native']['complete_attention_CAPI_ms']for i in [2,5]))for d in sorted(ds,key=lambda d:d['rank'])]
 snapshots={str(i):{key:dict(min=min(d['records'][i]['result']['native'][key]for d in ds),max=max(d['records'][i]['result']['native'][key]for d in ds))for key in ['free_entry','free_after_selected','free_after_output']}for i in range(6)}
 return dict(read_passed=True,component_numerical_and_lifecycle_diagnostic=True,raw_path=str(Path(path).resolve()),raw_SHA256=sha(raw),GPU='910371.735',keeper='910407.803',post='910371.736',publication_commit=COMMIT,GPU_source_manifest_SHA256=MF,library_SHA256=LIB,actual_helper_SHA256=HELPER,scope='Actual layer38 TP8 attention weights; synthetic hidden/selection; remaining prefill512/U32768 only; no indexer/FFN/Model.',owner_records=48,formal_owner_records=32,warm_owner_records=16,seed_calls_perowner=384,untimed_directV_diagnostics=8,formal_max8=rows[2:],warm_max8=rows[:2],ON_over_OFF=ratios,OFF_endpoint_fractional_drift=drift,adjacent_pair_ON_over_OFF=pair,same_owner_native_pairing=perowner,hidden_attention_max_NRMSE=max(m['NRMSE']for d in ds for r in d['records']for m in r['result']['numerics']),directV_max_NRMSE=max(d['records'][0]['V_stage_outside_benchmark']['NRMSE']for d in ds),logical_resident_V_saved_bytes_perowner_layer=134217728,logical_resident_V_saved_bytes_TP8_layer=8*134217728,snapshot_free_bytes=snapshots,actual_allocator_peak='UNKNOWN; meminfo snapshots are not peaks or loaded fit',normal8=True,srun_Slurm_keeper_COMPLETED0=True,independent_post_clean=True,actual_GPU_step_elapsed_s=123,actual_allocated_GPU_card_seconds=8*123,card_cost_scope='Slurm GPUstep elapsed123s * all8cards; not busy utilization. owner_service_residency includes preflight epoch and is not allocated time.',timing_scope='Native full attention includes prepare/gather/V chunks+unusedK/attentionO/TP/release drain; dispatch includes CAPI; checked adds producer allocation/D2H summaries/canary harness. seed listed separately. Max8 is same-index duration enclosure, not clocks subtracted across owners.',first_shape_cost_scope='Warm index0 retained separately, not used as speedup baseline. Physical GPU cold/algorithm selection unproven.',indexer_included=False,Model_GO=False,Model_fit=False,Model_quality_qualified=False,replay=False,decision='Geometry/numerical/lifecycle component closure. ON pays measurable prefill recomputation latency for lower logical persistent V. No loaded-service fit or throughput claim; future Model caller integration and full live peak remain unqualified.')

if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
