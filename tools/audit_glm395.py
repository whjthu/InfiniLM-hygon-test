"""Read original unloaded PP2 constructor receipts; no GPU or model imports."""
import argparse,hashlib,json
from pathlib import Path
def analyze(path):
 b=Path(path).read_bytes();a=json.loads(b);w=a['watch'];mf='60e3f3a3f588ca79deb1b14793ed14f6b7fd974f6cc03e2f310fb1d5a4ec67d6'
 native='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 assert w['completed'] and w['passed'] and not w['forced_cleanup'] and w['normal16owners'] and w['mp_spawn_join_markers']
 assert w['actual_GPU']=='910376.839' and w['CPU_keeper']=='910407.787' and w['GPU_manifest_SHA256']==mf and w['srun_exit_code']==0
 assert not w['Model_loaded'] and not w['checkpoint_IO'] and not w['compile'] and not w['retry'] and not w['reload']
 rows={r[0]:r for r in a['accounting']}
 for actual in ['910376.838','910376.839','910407.787']:assert rows[actual][2:4]==['COMPLETED','0:0']
 for post in w['fresh_before']+w['post']:
  assert post['CPU_only'] and not post['GPU_initialized'] and not post['kfd_pids'] and not post['matching_component_processes'] and post['ports_bindable']['43843']
 refs={r['path']:r for r in w['reports']};seen=set();owners=[]
 for r in a['reports']:
  d=r['data'];stage,rank=d['stage'],d['rank'];assert (stage,rank)not in seen;seen.add((stage,rank))
  # Original receipt writer uses indent2, ASCII and LF. Reconstruct exact bytes.
  original=(json.dumps(d,indent=2)+'\n').encode();assert hashlib.sha256(original).hexdigest()==r['SHA256']==refs[r['path']]['SHA256']
  assert d['completed'] and d['normal_destroy'] and d['normal_shutdown'] and d['actual_step']=='910376.839'
  assert (d['TP'],d['PP'],d['rows'],d['capacity'])==(8,2,1,32768) and d['global_rank']==stage*8+rank
  assert d['Native_SHA256']==native and d['source_manifest_SHA256']=='a83e056a57ea56e4a585a78ad52ff33eeb4d0b88b55182dc81ff14419546cd90'
  assert d['CPU_receipt_SHA256']=='67a6ed277698c113f07604d7e94518b33725a6351444ecc973cc9657c7854744'
  assert not d['weights_IO'] and d['finish_calls']==d['forward_calls']==0 and not d['Model_qualified'] and not d['loaded_service_admission_passed']
  cut=d['actual_partition'];assert (cut['begin'],cut['end'])==[(0,38),(38,78)][stage] and cut['weights_ready']is False
  c=d['tensor_census'];m=d['memory_accounting'];assert c['rows']==1 and c['capacity']==32768 and not c['loaded']
  assert sum(c['cache_category_bytes'].values())==c['cache_unique_start_bytes']==m['KV_tensor_logical_nbytes']
  assert c['weight_unique_start_bytes']==m['weight_tensor_logical_nbytes']
  assert m['driver_used_increase_bytes']==d['after_trim']['driver_used_bytes']-d['before_create']['driver_used_bytes']
  assert m['driver_increment_minus_tensor_logical_bytes']==m['driver_used_increase_bytes']-m['weight_tensor_logical_nbytes']-m['KV_tensor_logical_nbytes']
  assert m['Native_allocator_live_bytes']is None and m['Native_allocator_reserved_bytes']is None
  assert d['world16_identity_sum']==16.0 and d['loaded_service_original_free_floor_bytes']==8589934592
  assert d['would_pass_original_loaded_service_8GiB_guard']==(d['after_trim']['free_bytes']>=8589934592)
  owners.append(dict(stage=stage,rank=rank,report_SHA256=r['SHA256'],weight_logical_bytes=c['weight_unique_start_bytes'],KV_logical_bytes=c['cache_unique_start_bytes'],driver_used_increase_bytes=m['driver_used_increase_bytes'],driver_mixed_residual_bytes=m['driver_increment_minus_tensor_logical_bytes'],free_after_trim_bytes=d['after_trim']['free_bytes'],Torch_allocated_bytes=d['after_trim']['Torch_allocated_bytes'],Torch_reserved_bytes=d['after_trim']['Torch_reserved_bytes'],would_pass_original_8GiB=d['would_pass_original_loaded_service_8GiB_guard']))
 assert seen=={(s,r)for s in range(2)for r in range(8)}
 stages=[]
 for s in range(2):
  ranks=[r for r in owners if r['stage']==s];free=[r['free_after_trim_bytes']for r in ranks]
  assert len({r['weight_logical_bytes']for r in ranks})==len({r['KV_logical_bytes']for r in ranks})==1
  stages.append(dict(stage=s,layer_cut=[(0,38),(38,78)][s],weight_logical_bytes=ranks[0]['weight_logical_bytes'],KV_logical_bytes=ranks[0]['KV_logical_bytes'],driver_free_min_bytes=min(free),driver_free_max_bytes=max(free),below_original8GiB_owners=sum(not r['would_pass_original_8GiB']for r in ranks)))
 assert stages[0]['below_original8GiB_owners']==0 and stages[1]['below_original8GiB_owners']==8
 return dict(state='ROOT_AUDITED_CONSTRUCTOR_NORMAL_TAIL_LOADED_ADMISSION_BLOCKED',original_snapshot_SHA256=hashlib.sha256(b).hexdigest(),actual_GPU='910376.839 COMPLETED0/96s',CPU_keeper='910407.787 COMPLETED0/100s',publication_commit=w['publication_commit'],GPU_manifest_SHA256=mf,Native_SHA256=native,source_manifest_SHA256='a83e056a57ea56e4a585a78ad52ff33eeb4d0b88b55182dc81ff14419546cd90',all16_original_receipt_bytes_SHA_verified=True,actual_geometry=dict(TP=8,PP=2,rows=1,capacity=32768,layer_cut=[0,38,78]),stages=stages,owners=owners,normal16_destroy_shutdown=True,mp_spawn_join_markers_observed=True,post=w['post'],Model_loaded=False,Model_qualified=False,performance_gain=False,Native_allocator_live_reserved='UNKNOWN',decision='Constructor fit and original partition confirmed, but all8tail owners below unchanged8GiB loaded admission. STOP unchanged loadedModel launch. Need a separately evidenced memory-reduction mechanism; no floorlowering/shortcontext/W1 rerun.',limits=['No checkpoint tensor I/O, finish or forward executed; no loaded servicefit orW4peak proof.','Logical tensor bytes, driverincrease, Torchmemory and mixed residual are different accounts. Residual not allocatorlive/reserved decomposition.','Memory snapshots not peaks. No physicalOOM or speedup claim.','1536 cardseconds are residency, notGPUbusy.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('--output',required=True);args=p.parse_args();q=analyze(args.original);target=Path(args.output);assert not target.exists();target.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'state':q['state'],'stages':q['stages'],'output':str(target)}))
if __name__=='__main__':main()
