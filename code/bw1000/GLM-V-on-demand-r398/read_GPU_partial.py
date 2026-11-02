"""Preserve failed729. Decode subset is not a completed prefillscope or Model result."""
from pathlib import Path
import json,hashlib,statistics,sys
MF='492faf532197d9d6e40d5c10647448609cbf8c295b8aa829b2364cba1a2e381d'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):
 o=json.loads(Path(path).read_bytes());w=o['watch'];q=w['data'];assert sha(json.dumps(q,indent=2).encode())==w['SHA256']
 assert q['completed']and not q['passed']and q['actual_step']=='910371.729'and q['GPU_source_manifest_SHA256']==MF
 owners=[]
 for f in o['owners']:
  d=f['data'];assert sha(json.dumps(d,indent=2).encode())==f['SHA256'];assert d['actual_step']=='910371.729'and d['GPU_source_manifest_SHA256']==MF
  assert not d['normal']and not d['completed']and len(d['records'])==6 and d['load']['keys']==12 and d['load']['checkpoint_load_passes']==1
  assert d['destroy_attempted_once']and d['destroy_RC']==0 and d['Runtime_expired']
  for index,r in enumerate(d['records']):
   assert(r['case'],r['index'],r['mode'],r['warm'])==('decode-U2048',index,[0,1,0,1,1,0][index],index<2)
   a=r['result'];assert a['identity_valid']and a['canary']and a['native']['union_U']==2048 and not a['native']['indexer_included']
   assert all(x['finite']for x in a['numerics'])
  v=d['records'][0]['V_stage_outside_benchmark'];assert v['diagnostic_only']and not v['timed_query']and v['union_U']==2048 and v['finite']
  owners.append(d)
 assert len(owners)==8 and {d['rank']for d in owners}==set(range(8))and len({d['physical_PCI']for d in owners})==8
 post=q['failure_post'];assert post['passed']and post['actual_step']=='910371.730'and post['KFD_pids']==[]and post['expected_PIDs_still_exist']==[]and post['matching_GPU_driver_PIDs']==[]
 accounts=[row.split('|')for row in o['accounting']]
 for step,state in [('910371.729','FAILED'),('910407.799','FAILED'),('910371.730','COMPLETED')]:assert any(row[0]==step and row[2]==state for row in accounts)
 rows=[]
 for index in [2,3,4,5]:
  rows.append(dict(index=index,mode=[0,1,0,1,1,0][index],max8_native_ms=max(d['records'][index]['result']['native']['complete_attention_CAPI_ms']for d in owners),max8_dispatch_s=max(d['records'][index]['result']['dispatch_wall_s']for d in owners),max8_checked_s=max(d['records'][index]['checked_query_wall_s']for d in owners),max8_seed_s=max(d['records'][index]['seed_wall_s']for d in owners)))
 ratio=lambda field:statistics.median(x[field]for x in rows if x['mode'])/statistics.median(x[field]for x in rows if not x['mode'])
 return dict(read_passed=True,run_passed=False,GPU='910371.729',keeper='910407.799',post='910371.730',GPU_terminal='FAILED1/122s',keeper_terminal='FAILED1/146s',saved_owner_records=48,saved_formal_owner_records=32,saved_warm_owner_records=16,planned_queries_perowner=12,persisted_queries_perowner=6,fullscope_completed=False,case_scope='DecodeU2048 only persisted; prefill firstOFF query returned by sourcecontrolflow but failed inspect before recordappend. No missing metrics reconstructed.',formal_max8=rows,partial_decode_ON_over_OFF={k:ratio(k)for k in ['max8_native_ms','max8_dispatch_s','max8_checked_s']},OFF_endpoint_drift=abs(rows[3]['max8_native_ms']/rows[0]['max8_native_ms']-1),saved_hidden_attention_max_NRMSE=max(x['NRMSE']for d in owners for r in d['records']for x in r['result']['numerics']),saved_decode_directV_max_NRMSE=max(d['records'][0]['V_stage_outside_benchmark']['NRMSE']for d in owners),eight_Context_destroy_RC0_weakexpired=True,normal8=False,mp_spawn_normaljoin=False,independent_post_clean=True,actual_allocated_GPU_card_seconds=122*8,cardcost_scope='Slurm GPUstepElapsed122s * all8cards; not busy utilization or Model tokens',numerical_failure_established=False,component_allgeometry_qualification=False,Model_GO=False,GPU_replay=False)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
