"""Independent stdlib recomputation of the original two-process component."""
import argparse,base64,gzip,hashlib,json,math,statistics
from pathlib import Path
SEAL='034e31e21ed2b6856436195018f9fc31abd2c7f9ed3d278562a8dc1449ceb85f'
PREFIX='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def audit(path):
 packed=Path(path).read_bytes();assert hashlib.sha256(packed).hexdigest()==SEAL
 z=json.loads(gzip.decompress(packed));files={}
 for p,v in z['files'].items():
  b=base64.b64decode(v['bytes'],validate=True);assert len(b)==v['nbytes']and hashlib.sha256(b).hexdigest()==v['SHA256'];files[p]=b
 def doc(suffix):
  values=[b for p,b in files.items()if p.endswith(suffix)];assert len(values)==1;return json.loads(values[0])
 def raw(path,expected=None):
  p=PREFIX+path[len('/refactor'):];b=files[p]
  if expected:assert hashlib.sha256(b).hexdigest()==expected
  return b
 def artifact(item):return json.loads(raw(item['path'],item['SHA256']))
 w=doc('/Kimi439-watch-v3.json');a=doc('/Kimi439-910371.772-main.json');cpu=doc('/Kimi-W2-Runtime-H2D-r439-CPU-v3/result.json')
 assert w['completed']and w['passed']and w['actual_step']=='910371.772'and w['publication_commit']=='fdfbc896879ad8f89e0208bef4782c831982366b'
 assert a['completed']and a['passed']and a['all_normal']and a['work_plan_complete']and a['normal_Runtime_closes']==2
 assert a['source_manifest_SHA256']==w['GPU_source_manifest_SHA256']=='8d9f358fddf8577d93012742b7fb294e8c1ce6138697388456548e48c0a52e90'
 assert cpu['passed']and cpu['actual_CPU']=='910407.901'and cpu['Runtime_executions']==0
 assert a['elapsed_since_original_GPU_epoch_s']<270 and a['work_sent']==a['work_reply_received']==a['work_qualified']==38
 assert len(a['actors'])==2 and len(a['records'])==38 and len(a['close_records'])==2
 for role,entry in a['actors'].items():
  r=artifact(entry['ready_artifact']);assert r==entry['ready']and r['protocol']==1 and r['type']=='ready'and r['passed']and r['role']==role and r['PID']==entry['PID']and r['on']==(role=='candidate')
  assert entry['startup_to_ready_s']<=90 and r['Runtime_address']>0
  expected=cpu['executables'][role]['Runtime']['path'];other=cpu['executables']['candidate'if role=='baseline'else'baseline']['Runtime']['path']
  assert r['Runtime_provider']==expected and expected in r['maps']and other not in r['maps']
  assert r['device']['CU']==80 and r['device']['total_memory']>=60*1024**3
  c=a['process_cleanup'][role];assert c['PID']==entry['PID']and c['rc']==0 and not c['forced_kill']and c['reader_stopped']
  transcript=raw(c['stdout'],c['stdout_SHA256']);raw(c['stderr'],c['stderr_SHA256'])
  frames=[json.loads(s)for s in transcript.splitlines()];assert len(frames)==21 and frames[0]['type']=='ready'and frames[-1]['type']=='closed'and all(f['role']==role and f['PID']==entry['PID']for f in frames)
 assert a['actors']['baseline']['ready']['device']==a['actors']['candidate']['ready']['device']
 assert len({v['PID']for v in a['actors'].values()})==2
 expected=[('baseline','check',0,True),('candidate','check',0,True)]
 for width in [96,6]:
  modes=['baseline','candidate']+['baseline','candidate','candidate','baseline']*4
  expected.extend((role,'bench',width,i<2)for i,role in enumerate(modes))
 samples=[];validation=copy_wall=op_wall=host=allocation=views=parent_wall=0.0;timed_bytes=0
 for seq,(record,e)in enumerate(zip(a['records'],expected)):
  role,op,width,warm=e;r=artifact(record['actual_result']);phase=artifact(record['phase_file']);actor=a['actors'][role]
  assert (record['seq'],record['role'],record['op'],record['width'],record['warm'])==(seq,*e)and record['qualified']
  assert r['passed']and r['seq']==seq and r['role']==role and r['PID']==actor['PID']and r['operation']==op and r['width']==width and r['on']==(role=='candidate')
  assert all(record['reply'][k]==v for k,v in r.items())and r['Runtime_address']==actor['ready']['Runtime_address']and r['Runtime_provider']==actor['ready']['Runtime_provider']
  assert r['Runtime_retained']and r['next_op_allowed']and r['banks_sources_views_released']and phase['phase']=='op_release_complete'and phase['PID']==actor['PID']
  assert r['device']==actor['ready']['device']and r['actual_target_gpu_count']==1
  if op=='check':
   assert r['cases']==36 and [(v['width'],v['rank'],v['offset'],v['bytes'],v['passed'])for v in r['offsets']]==[(w,k,k*w,3584*w,True)for w in [96,6]for k in range(16)]
   assert r['stats_delta']==[33,114688,5849088,48,48]and r['hits_delta']==([32,1]if role=='candidate'else[0,0])
  else:
   assert r['cases']==r['expert_copy_calls']==896 and r['actual_H2D_bytes']==896*3584*width and r['bank_logical_bytes']==896*3584*width and not r['allocation_in_timed_wall']
   assert r['timed_stats_delta']==[896,896*3584,896*3584*width,0,0]and r['timed_hits_delta']==([896,0]if role=='candidate'else[0,0])
   value=r['full_gather_H2D_sync_wall_s'];assert math.isfinite(value)and value>0
   samples.append(dict(seq=seq,role=role,width=width,warm=warm,full_copy_s=value,parent_roundtrip_s=record['parent_roundtrip_s']))
   copy_wall+=value;validation+=r['all896_D2H_byte_validation_s'];host+=r['host_setup_s'];allocation+=r['GPU_bank_allocation_s'];views+=r['bank_register_views_s']
   timed_bytes+=r['actual_H2D_bytes']
  op_wall+=r['main_wall_s'];parent_wall+=record['parent_roundtrip_s']
 for item,role,seq in zip(a['close_records'],['baseline','candidate'],[38,39]):
  c=artifact(item['actual_result']);assert c==item['reply']and c['type']=='closed'and c['seq']==seq and c['role']==role and c['passed']and c['Runtime_expired']and c['work_ops']==19 and c['PID']==a['actors'][role]['PID']
 post=w['post'];assert post['passed']and post['actual_step']=='910371.773'and post['KFD_pids']==post['matching_GPU_driver_PIDs']==post['expected_PIDs_still_exist']==[]
 assert set(post['expected_owner_PIDs'])=={a['PID'],*(v['PID']for v in a['actors'].values())}
 accounting={r[0]:r[2:5]for r in z['account']};assert all(accounting[k][:2]==['COMPLETED','0:0']for k in ['910371.772','910371.773','910407.903'])
 paired=[]
 for width in [96,6]:
  formal=[v for v in samples if v['width']==width and not v['warm']];assert len(formal)==16
  blocks=[]
  for i in range(0,16,4):
   x,y,z_,t=formal[i:i+4];assert [v['role']for v in [x,y,z_,t]]==['baseline','candidate','candidate','baseline']
   off=(x['full_copy_s']+t['full_copy_s'])/2;on=(y['full_copy_s']+z_['full_copy_s'])/2;drift=abs(t['full_copy_s']/x['full_copy_s']-1)
   blocks.append(dict(OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift))
  off=statistics.mean(v['full_copy_s']for v in formal if v['role']=='baseline');on=statistics.mean(v['full_copy_s']for v in formal if v['role']=='candidate');drift=abs(formal[-1]['full_copy_s']/formal[0]['full_copy_s']-1)
  row=dict(width=width,OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,speedup=off/on,reduction_fraction=1-on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift,all4blocks_gain_exceeds_drift=all(b['gain_exceeds_drift']for b in blocks),blocks=blocks)
  reported=next(v for v in a['paired']if v['width']==width)
  assert math.isclose(row['ON_over_OFF'],reported['ON_over_OFF'],rel_tol=1e-12)and row['gain_exceeds_drift']==reported['gain_exceeds_drift']
  paired.append(row)
 assert all(v['gain_exceeds_drift']and v['all4blocks_gain_exceeds_drift']for v in paired)and a['stable_component_gain']
 fds=sorted({v['target']for o in a['actors']['baseline'].get('startup_observations',[])for v in o.get('fd_paths',[])if'target'in v and '/tmp/comgr-'in v['target']})
 return dict(offline_audit_passed=True,component_passed=True,stable_copy_component_gain=True,seal_SHA256=SEAL,files=len(files),source_bytes=sum(map(len,files.values())),actual_account=accounting,work_commands=38,semantic_commands=2,benchmark_commands=36,formal_benchmark_commands=32,close_commands=2,Runtime_processes=2,normal_Runtime_closes=2,successful_H2D_copies=32322,timed_benchmark_H2D_copies=32256,logical_benchmark_H2D_bytes=timed_bytes,logical_bytes_formula='18*896*3584*(96+6)',paired=paired,samples=samples,cost=dict(GPU_epoch_elapsed_s=a['elapsed_since_original_GPU_epoch_s'],startup_to_ready_s={k:v['startup_to_ready_s']for k,v in a['actors'].items()},full_copy_timer_sum_s=copy_wall,D2H_validation_timer_sum_s=validation,host_setup_sum_s=host,bank_allocation_sum_s=allocation,view_registration_sum_s=views,operation_main_enclosure_sum_s=op_wall,parent_roundtrip_enclosure_sum_s=parent_wall,child_residency_s={k:v['process_residency_s']for k,v in a['process_cleanup'].items()},note='Validation/setup and copy are inside operation_main, which is inside parent_roundtrip. Residency overlaps across children and includes idle; do not add nested enclosures or call them GPUbusy.'),observed_COMGR_temporary_fd_paths=fds,write_target_limit='Observed directory paths do not identify the fd responsible for all startupwrites; no vendor cache change.',scope='OneGPU/logicalTP16 W2 UInt8 warmRAM, resident Runtime/allocator, perop preallocated896bank; noModel/checkpoint/CCL/wholeloader claim',byte_validation_limit='Actual C++ D2H byte comparisons, source immutability/owner and structured counts passed; raw D2H arrays were not exported for independent re-comparison.',post=post,Model_gain='UNKNOWN',no_new_execution=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('seal');p.add_argument('--output');a=p.parse_args();result=audit(a.seal);s=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(s,encoding='utf8')
 print(s)
