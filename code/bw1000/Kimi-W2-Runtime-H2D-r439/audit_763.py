"""Original partial real-H2D correctness and failed plan, offline stdlib only."""
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path
SEAL='366acba3882ad955e7c89b623899af1c604eede377eb95430633be79ec8027b9'
PREFIX='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def audit(path):
 packed=Path(path).read_bytes();assert hashlib.sha256(packed).hexdigest()==SEAL
 z=json.loads(gzip.decompress(packed));files={}
 for p,v in z['files'].items():
  b=base64.b64decode(v['bytes'],validate=True);assert len(b)==v['nbytes']and hashlib.sha256(b).hexdigest()==v['SHA256'];files[p]=b
 def document(suffix):
  values=[v for p,v in files.items()if p.endswith(suffix)];assert len(values)==1;return json.loads(values[0])
 def raw(v):
  p=PREFIX+v['path'][len('/refactor'):];b=files[p];assert hashlib.sha256(b).hexdigest()==v['SHA256'];return b
 watch=document('/Kimi439-watch-v2.json');main=document('/Kimi439-910371.763-main.json');cpu=document('/Kimi-W2-Runtime-H2D-r439-CPU-v2/result.json')
 assert watch['completed']and not watch['passed']and watch['actual_step']=='910371.763'
 assert watch['GPU_source_manifest_SHA256']==main['source_manifest_SHA256']=='76b68584057ddc9a6b4bdbf40a225db7599f909bd4aa524a16458fad91a2d71c'
 assert watch['publication_commit']=='8059e6bd7282f48e10fe0bc8be35db338a9ca767'
 assert cpu['passed']and cpu['actual_CPU']=='910407.895'and cpu['Runtime_executions']==0
 assert main['completed']and not main['passed']and main['Runtime_processes_normal']==2 and len(main['records'])==3
 normal=[]
 for r,arm in zip(main['records'][:2],['baseline','candidate']):
  assert r['arm']==arm and r['operation']=='check'and r['rc']==0 and r['qualified']
  a=json.loads(raw(r['actual_result']));phase=json.loads(raw(r['phase_file']))
  assert a['passed']and a['Runtime_expired']and a['cases']==36 and a['operation']=='check'and a['width']==0
  assert a['on']==(arm=='candidate')and phase['PID']==r['PID']and phase['phase']=='normal_destroy.after'
  assert a['Runtime_provider']==cpu['executables'][arm]['Runtime']['path']
  other=cpu['executables']['candidate'if arm=='baseline'else'baseline']['Runtime']['path']
  assert a['Runtime_provider']in a['maps']and other not in a['maps']
  assert [(v['width'],v['rank'],v['offset'],v['bytes'],v['passed'])for v in a['offsets']]==[(w,k,k*w,3584*w,True)for w in [96,6]for k in range(16)]
  assert a['stats_delta']==[33,114688,5849088,48,48]and a['hits_delta']==([32,1]if arm=='candidate'else[0,0])
  assert a['device']['CU']==80 and a['device']['total_memory']>=60*1024**3 and a['actual_target_gpu_count']==1
  assert all(math.isfinite(a[k])and a[k]>=0 for k in ['main_wall_s','Runtime_create_s','drain_destroy_s'])
  normal.append(dict(arm=arm,real_Tensor_cases=36,all16_TPoffsets_each_width_passed=True,source_owner_and_readonly_checks=True,shape_dtype_wrongthread_fallback_checks=True,copy_stats_delta=a['stats_delta'],hits_delta=a['hits_delta'],Runtime_expired=True,normal_destroy_phase=True,device=a['device'],process_inclusive_wall_s=r['process_inclusive_wall_s'],main_wall_s=a['main_wall_s'],Runtime_create_s=a['Runtime_create_s'],outside_main_and_launch_teardown_s=r['process_inclusive_wall_s']-a['main_wall_s']))
 assert normal[0]['device']==normal[1]['device']
 failed=main['records'][2];assert failed['arm']=='baseline'and failed['operation']=='bench'and failed['width']==96 and failed['warm']and failed['timeout']and failed['rc']==-9 and failed['child_cap_s']==25
 assert 'actual_result'not in failed and 'phase_file'not in failed and raw(failed['stdout'])==raw(failed['stderr'])==b''
 observations=main['records'][0]['cold_observations'];assert len(observations)<=16
 def io(text):return {k:int(v)for k,v in (line.split(':',1)for line in text.splitlines())}
 first,last=io(observations[0]['io']),io(observations[-1]['io']);delta={k:last[k]-first[k]for k in last}
 assert delta['wchar']>1_000_000_000 and delta['write_bytes']>1_000_000_000
 post=watch['failure_post'];assert post['passed']and post['actual_step']=='910371.767'
 assert post['KFD_pids']==post['matching_GPU_driver_PIDs']==post['expected_PIDs_still_exist']==[]
 assert set(post['expected_owner_PIDs'])=={main['PID'],*(r['PID']for r in main['records'])}and watch['owned_failure_cleanup']['closed']
 account={v[0]:v[2:5]for v in z['account']};assert account['910371.763'][:2]==['FAILED','1:0']and account['910407.896'][:2]==['FAILED','1:0']and account['910371.767'][:2]==['COMPLETED','0:0']
 return dict(offline_audit_passed=True,whole_plan_passed=False,status='PARTIAL_REAL_H2D_SEMANTIC_PASS_FIRST_BENCH_STARTUP_CAP_FAIL',seal_SHA256=SEAL,files=len(files),source_bytes=sum(map(len,files.values())),actual_account=account,normal_semantic_processes=normal,real_check_cases=72,successful_H2D_copies_proven=66,attempted_benchmark_processes=1,completed_benchmarks=0,formal_samples=0,ON_benchmarks=0,performance='UNKNOWN',bank896_allocation_qualification='UNKNOWN',startup_io=dict(observation_start_s=observations[0]['elapsed_since_child_launch_s'],observation_end_s=observations[-1]['elapsed_since_child_launch_s'],delta=delta,note='Actual process I/O counters; written fd/target not captured. Not exclusively ELF latency or checkpoint I/O.'),whole_GPU_epoch_elapsed_s=main['elapsed_since_original_GPU_epoch_s'],failure_post=post,Model=False,no_retry=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--output');a=p.parse_args();result=audit(a.seal);s=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(s,encoding='utf8')
 print(s)
