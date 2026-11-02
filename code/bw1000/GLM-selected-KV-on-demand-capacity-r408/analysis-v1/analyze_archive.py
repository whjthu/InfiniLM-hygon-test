"""stdlib-only re-audit of original eight scalar reports/journals and exact handles."""
from pathlib import Path,PurePosixPath
import json,hashlib,argparse,statistics,math
CASES={'decode1-union2048':dict(count=1,past=32767,U=2048),'prefill512-union32768':dict(count=512,past=32256,U=32768)}
MODES=[0,1,0,1,1,0]
def analyze(path):
 b=Path(path).read_bytes();a=json.loads(b);files=a['raw_files']
 for n,v in files.items():raw=v['text'].encode();assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['SHA256'],n
 w=json.loads(files[a['watch_path']]['text']);assert w['completed']and w['passed']and w['normal_all8']and w['srun_exit_code']==0 and w['terminal'][2:4]==['COMPLETED','0:0']
 owners=[json.loads(files[p]['text'])for p in a['owner_paths']];assert len(owners)==8 and {o['rank']for o in owners}==set(range(8));owners.sort(key=lambda o:o['rank'])
 assert len({o['physical_PCI']for o in owners})==8 and len({o['PID']for o in owners})==8
 masks=[set(o['CPU_affinity'])for o in owners];assert all(len(s)==4 for s in masks)and len(set().union(*masks))==32
 input_ids={o['synthetic_hidden_SHA256']for o in owners};assert len(input_ids)==1
 for o in owners:
  assert all(o[n]for n in ['completed','passed','normal','normal_destroy','Runtime_weak_expired','Torch_after_destroy','policy_closed','weights_loaded_once','input_immutable','destroy_attempted_once','Runtime_expired'])and o['destroy_RC']==0
  assert o['actual_step']==w['actual_GPU']and o['GPU_source_manifest_SHA256']==w['manifest_SHA256']and o['node']=='a02r3n16'and o['profile']==0 and o['Model']is False and o['indexer_included']is False
  assert o['library_SHA256']=='acfeb6f6c8bc6d4f44ae4eaa7593db679984d446b3de228b53e8b312e2dd0304'
  assert o['measurement_calls']==8 and o['warm_calls']==4 and o['directK_diagnostics']==2 and len(o['records'])==12 and o['load']['keys']==12 and o['load']['checkpoint_load_passes']==1
  wp=next(p for p in w['owners']if p['rank']==o['rank']);assert files[wp['path']]['SHA256']==wp['SHA256']
  events=[json.loads(s)for s in files[str(PurePosixPath(wp['path']).with_suffix('.jsonl'))]['text'].splitlines()];assert len(events)==14 and sum(e['type']=='query'for e in events)==12 and sum(e['type']=='diagnostic'for e in events)==2
  rebuilt=[];lookup={}
  for e in events:
   if e['type']=='query':r=e['record'];assert 'K_stage_outside_benchmark'not in r;rebuilt.append(r);lookup[(r['case'],r['index'])]=r
   elif e['type']=='diagnostic':lookup[(e['case'],e['index'])].update(e['fields'])
   else:raise AssertionError(('Unexpected journal event',e))
  assert rebuilt==o['records'],'Terminal history must equal O1 primary+diagnostic events'
  for name in CASES:
   rows=[r for r in o['records']if r['case']==name];assert [r['index']for r in rows]==list(range(6))and[r['mode']for r in rows]==MODES
   case=CASES[name];U=case['U'];calls=(U+8191)//8192
   for r in rows:
    assert r['warm']==(r['index']<2);v=r['result'];n=v['native'];mode=r['mode'];assert v['canary']and v['identity_valid']
    assert n['mode']==mode and n['past']==case['past']and n['count']==case['count']and n['union_U']==U and n['seeded']==32768 and n['profile']==0 and not n['indexer_included']
    assert n['persistent_K_bytes']==(0 if mode else 134217728)and n['persistent_V_bytes']==0 and n['logical_persistent_bytes']==37748736+(0 if mode else 134217728)
    assert n['selected_HOST_completed_expand_calls']==calls and n['fresh_expand_calls']==1 and n['query_kv_b_expand_calls_total']==1+calls and n['selected_max_block_rows']==8192
    assert n['selected_K_copy_calls']==(4 if mode and U>8192 else 0)and n['selected_K_copy_bytes']==(134217728 if mode and U>8192 else 0)
    assert n['held_chunk_KV_logical_bytes']==U*8*256*2*2 and n['selected_latent_rotary_bytes']==U*576*2
    assert n['allV_output_bytes']==(U*4096 if U>8192 else 0)and n['additional_allK_bytes_vs_complete_prefix_NoV']==(U*4096 if mode and U>8192 else 0)
    assert len(v['numerics'])==2 and {x['name']for x in v['numerics']}=={'hidden','attention'}
    for x in v['numerics']:assert x['finite']and math.isfinite(x['NRMSE'])and x['NRMSE']<=.02
   d=rows[0]['K_stage_outside_benchmark'];assert d['diagnostic_only']and not d['timed_query']and d['count']==case['count']and d['union_U']==U and d['selected_HOST_completed_expand_calls']==calls
   assert d['finite']and d['rotary64_finite']and d['NRMSE']<=.02 and d['rotary64_NRMSE']<=.02 and d['elements']==U*8*256 and d['rotary64_elements']==U*8*64 and d['D2H_bytes']==U*8*256*4
 for name in ['fresh_before','post']:assert w[name]['KFD_available']and not w[name]['kfd_pids']and not w[name]['matching_component_processes']and not w[name]['port43882_listening']
 bycase={}
 for name,case in CASES.items():
  perrank=[[r for r in o['records']if r['case']==name]for o in owners];samples=[]
  for i,mode in enumerate(MODES):
   rows=[rs[i]for rs in perrank]
   samples.append(dict(index=i,mode=mode,warm=i<2,max8_native_ms=max(r['result']['native']['complete_attention_CAPI_ms']for r in rows),max8_dispatch_s=max(r['result']['dispatch_wall_s']for r in rows),max8_checked_s=max(r['checked_query_wall_s']for r in rows),max8_seed_s=max(r['seed_wall_s']for r in rows),max8_seed_plus_checked_s=max(r['seed_wall_s']+r['checked_query_wall_s']for r in rows)))
  keys=['max8_native_ms','max8_dispatch_s','max8_checked_s','max8_seed_s','max8_seed_plus_checked_s'];ratios={};drift={};adjacent={}
  for k in keys:
   ratios[k]=statistics.mean(samples[i][k]for i in [3,4])/statistics.mean(samples[i][k]for i in [2,5]);drift[k]=samples[5][k]/samples[2][k]-1;adjacent[k]=[samples[3][k]/samples[2][k],samples[4][k]/samples[5][k]]
  paired=[statistics.mean(rs[i]['result']['native']['complete_attention_CAPI_ms']for i in [3,4])/statistics.mean(rs[i]['result']['native']['complete_attention_CAPI_ms']for i in [2,5])for rs in perrank]
  snapshots={}
  for mode in [0,1]:
   rows=[rs[i]for rs in perrank for i in [2,3,4,5]if rs[i]['mode']==mode]
   snapshots[str(mode)]={key:dict(min=min(r['result']['native'][key]for r in rows),max=max(r['result']['native'][key]for r in rows))for key in ['persistent_K_bytes','persistent_V_bytes','logical_persistent_bytes','free_entry','free_after_selected','free_after_output','held_chunk_KV_logical_bytes','allV_output_bytes','additional_allK_bytes_vs_complete_prefix_NoV','selected_latent_rotary_bytes','selected_K_copy_calls','selected_K_copy_bytes','query_kv_b_expand_calls_total']}
  bycase[name]=dict(geometry=case,formal_same_index_max8=samples[2:],warm_same_index_max8=samples[:2],ON_over_OFF=ratios,OFF_endpoint_signed_fractional_drift=drift,adjacent_ON_over_neighbor_OFF=adjacent,same_owner_native_ON_over_OFF=dict(min=min(paired),median=statistics.median(paired),max=max(paired),all=paired),numeric_max_NRMSE=max(x['NRMSE']for rs in perrank for r in rs for x in r['result']['numerics']),directK_max_NRMSE=max(rs[0]['K_stage_outside_benchmark']['NRMSE']for rs in perrank),directK_rotary64_max_NRMSE=max(rs[0]['K_stage_outside_benchmark']['rotary64_NRMSE']for rs in perrank),directK_maxabs=max(rs[0]['K_stage_outside_benchmark']['maxabs']for rs in perrank),directK_diagnostic_wall_s=dict(min=min(rs[0]['K_stage_outside_benchmark']['diagnostic_wall_s']for rs in perrank),max=max(rs[0]['K_stage_outside_benchmark']['diagnostic_wall_s']for rs in perrank)),snapshots_not_peak=snapshots)
 accounting=[r.split('|')for r in a['accounting']];gpu=next(r for r in accounting if r[0]==w['actual_GPU']);keeper=next(r for r in accounting if r[0]==w['CPU_keeper']);assert gpu[2:4]==keeper[2:4]==['COMPLETED','0:0']
 return dict(read_passed=True,component_numerics_and_lifetime_qualified=True,original_archive_SHA256=hashlib.sha256(b).hexdigest(),publication_commit=w['publication_commit'],manifest_SHA256=w['manifest_SHA256'],GPU=w['actual_GPU'],keeper=w['CPU_keeper'],GPU_epoch=w['GPU_epoch_unix'],GPU_elapsed_s=int(gpu[4]),allocated_GPU_card_seconds=8*int(gpu[4]),CPU_keeper_elapsed_s=int(keeper[4]),physical_PCI=[o['physical_PCI']for o in owners],owner_records=8,query_records=96,warm_owner_records=32,formal_owner_records=64,untimed_directK_diagnostics=16,O1_journal_lines_per_owner=14,case_results=bycase,whole_owner_mixed_mode_sequence_s=dict(min=min(o['whole_component_sequence_wall_s']for o in owners),max=max(o['whole_component_sequence_wall_s']for o in owners)),owner_create_s=dict(min=min(o['create_wall_s']for o in owners),max=max(o['create_wall_s']for o in owners)),weight_load_s=dict(min=min(o['load_wall_s']for o in owners),max=max(o['load_wall_s']for o in owners)),fixture_prepare_s=dict(min=min(o['fixture_prepare_wall_s']for o in owners),max=max(o['fixture_prepare_wall_s']for o in owners)),normal8=True,post_clean=True,logical_K_saved_bytes_per_owner_layer=134217728,logical_V_bytes_both_modes=0,Model_GO=False,Model_fit='UNKNOWN',Model_quality_qualified=False,nullable_Model_caller_device_qualified=False,decision='Retain as component-qualified capacity fallback; no automatic Model load. Full Model fit/quality and true peak remain UNKNOWN.',limits=['Thin layer38 attention component; synthetic hidden/selection, no indexer or FFN or actual nullable Model caller device path','Native CPP fullquery includes prepare/fresh original KV/gather/selectedKV/attention/O/TP/drain; dispatch adds CAPI return/Torch sync; checked adds producer/outputD2H/CPU metrics/canary','DirectK diagnostics outside query timing; reset/begin/storageallocation is not separately timed but included whole mixedmode owner sequence; seed starts after begin','Free snapshots including free_after_output still contain live Work temporaries and allocator effects, not peak. Logical128MiB per layer is not physical Model fit','Source-level completed expand_cached calls2/5 are kv_b call counts, not all GPU GEMMs/kernel counts or hardware utilization','One ABBA per geometry, cold firstwarm retained; no statistical or Model speedup claim; no GPUF64'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');args=p.parse_args();print(json.dumps(analyze(args.archive),indent=2))
