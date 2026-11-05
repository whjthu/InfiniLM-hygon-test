"""Independent stdlib consumption of original886 terminal+post small reports.
No Native, SDK, compiler, GPU, HTTP or Slurm is invoked.
"""
import argparse,hashlib,json
from pathlib import Path
NODES=['a02r3n15','a02r3n16'];LABEL='GLM-global-TP16-full-Model-entry-r470-v2'
FLOOR=8*1024**3

def H(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(root,accounting_original):
 root=Path(root);watch=root/(LABEL+'-watch.json');w=json.loads(watch.read_bytes());assert w['completed']and not w['passed']and w['GPU']=='910371.886'and w['CPU_keeper']=='910407.1099'and w['planned']==2
 assert w['GPU_accounting'][0]=='910371.886';receipt=json.loads(Path(accounting_original).read_bytes());assert receipt['rc']==0;account=json.loads(receipt['out']);rows=account['accounting'];assert next(v for v in rows if v[0]=='910371.886')[2:4]==['FAILED','1:0']and next(v for v in rows if v[0]=='910407.1099')[2:4]==['FAILED','1:0']and next(v for v in rows if v[0]=='910371.887')[2:4]==['COMPLETED','0:0'];assert w['srun_rc']!=0 and w['failure_post']['raw']['rc']==0
 expected=json.loads(next(v.split('=',1)[1]for v in w['failure_post']['raw']['argv']if v.startswith('GLM470_EXPECTED_PIDS=')));assert set(expected)==set(NODES)
 workers=[json.loads((root/('worker-%d.json'%i)).read_bytes())for i in range(2)];owners=[json.loads((root/('owner-%02d.json'%i)).read_bytes())for i in range(16)];result=[]
 for rank,q in enumerate(owners):
  node=NODES[rank//8];worker=workers[rank//8];geo=q['constructor_geometry'];assert q['global_rank']==rank and q['node']==node and q['device']==rank%8 and q['actual_GPU']=='910371.886'and q['gpu_epoch']==w['epoch'];assert q['UID']==154459 and q['owner_TID']==q['PID'];assert q['completed']and not q['passed']and not q['weights']and not q['Model_READY']and not q['normal_close']and q['failure_context_live'];assert q['requests']==[]and q['Native_forward_attempts']==q['Native_forward_completed']==q['resets']==q['control_calls']==0
  assert q['source_manifest_SHA256']=='171e245205ec7ce9e533b814aedd72d4fc524bef0a1e483b0582c095565a2e80'and q['Native_SHA256']=='22345abd96774752f5e33f048da3e4a75ff9fb54f88007d72ac2958422db295c';assert all(v['SHA256']==q['Native_SHA256']for v in q['CAPI_bindings'].values())
  assert geo['TP']==geo['world_size']==16 and geo['PP']==1 and geo['world_rank']==geo['tp_rank']==rank and geo['rows']==1 and geo['capacity_per_row']==32768 and geo['local_attention_heads']==4 and geo['local_index_heads']==2;assert geo['layer_begin']==0 and geo['layer_end']==78;assert geo['Model_constructed']and geo['cache_allocated']and not geo['weights_loaded']and not geo['Model_READY']and geo['loaded_parameters']==0
  assert geo['replicated_embedding_BF16_bytes']==1903165440 and geo['cache_logical_bytes']['sum']==8355053568
  before,after=q['memory_boundaries'];assert before['phase']=='before_create'and after['phase']=='after_full_constructor'and before['floor']==after['floor']==FLOOR and before['free']>=FLOOR and after['free']<FLOOR and before['total']==after['total']==68702699520;assert '8GiB floor'in q['error']and str(after['free'])in q['error']
  owned=next(v for v in expected[node]if v['PID']==q['PID']);assert owned['startticks']==q['startticks']and set(owned['socket_inodes'])==set(q['owned_socket_inodes']);assert any(v['PID']==worker['PID']and v['startticks']==worker['startticks']for v in expected[node]);assert len(expected[node])==9
  result.append(dict(rank=rank,node=node,PCI=q['physical_PCI'],free_before=before['free'],free_after_constructor=after['free'],deficit_bytes=FLOOR-after['free'],incremental_constructor_committed_bytes=before['free']-after['free'],driver_min_sampled=q['driver_memory']['min_observed'],driver_samples=q['driver_memory']['samples'],normal_destroy=False))
 post=w['failure_post']['nodes'];assert {p['node']for p in post}==set(NODES)
 for p in post:
  assert p['actual_step']=='910371.887'and p['KFD_available']and not p['kfd_pids']and not p['matching_Model_processes']and not p['expected_owned_PIDs_still_live']and not p['expected_UIDsocket_listeners_still_live']and all(p['ports_bindable'].values());assert len(p['physical_PCI'])==8 and set(p['physical_PCI'])=={r['PCI']for r in result if r['node']==p['node']}
 assert len({(r['node'],r['PCI'])for r in result})==16
 return dict(schema='GLM886-original-terminal-independent-v1',root_reader_executed=True,actual_terminal_accounting_original_SHA256=H(accounting_original),producer_first_accounting=w['GPU_accounting'],producer_stale_accounting_preserved=True,files_SHA256={p.name:H(p)for p in sorted(root.glob('*.json'))},GPU='910371.886',GPU_passed=False,keeper='910407.1099',keeper_passed=False,post='910371.887',post_clean=True,normal_destroy_owners=0,process_KFD_UIDsocket_cleanup_only=True,planned=2,completed_requests=0,missing_requests=2,weights_loaded=0,Native_forwards=0,qualified=False,diagnosis='All16 actual fullModel constructors returned, then observed driverfree below unchanged8GiB; no checkpoint/load/forward stage reached',floor_bytes=FLOOR,free_after_constructor_min=min(v['free_after_constructor']for v in result),free_after_constructor_max=max(v['free_after_constructor']for v in result),deficit_min=min(v['deficit_bytes']for v in result),deficit_max=max(v['deficit_bytes']for v in result),rank_reports=result,capacity_cost_scope='Sameowner driver boundary diff includes Runtime/CCL/fullModel/cache/parameter allocations; not exclusive parameter bytes or peak upperbound',Model_quality_performance=None)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('original_directory');p.add_argument('accounting_original');p.add_argument('output');a=p.parse_args();result=read(a.original_directory,a.accounting_original);result['reader_SHA256']=H(__file__);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['GPU','post_clean','planned','missing_requests','free_after_constructor_min','free_after_constructor_max','deficit_min','deficit_max','normal_destroy_owners']}))
