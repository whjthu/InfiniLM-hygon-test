"""Exactly two fixed Runtime processes, sequential 38work+2close, one device."""
from pathlib import Path
import hashlib,json,math,os,statistics,time,traceback
from fixed_children_r439 import Peer,plan,command,startup_observer
D=Path(__file__).resolve().parent
R=Path('/refactor')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(popen=None):
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];epoch=float(os.environ['R439_GPU_START']);deadline=epoch+270
 out=R/'results'/('Kimi439-'+step+'-main.json');raw=out.with_suffix('');raw.mkdir(exist_ok=False)
 q=dict(completed=False,passed=False,PID=os.getpid(),actual_step=step,original_epoch=epoch,actors={},records=[],close_records=[],Model=False,HTTP=False,checkpoint=False,work_sent=0,work_reply_received=0,work_qualified=0,normal_Runtime_closes=0,source_manifest_SHA256=os.environ['R439_GPU_SOURCE_SHA256'],scope='Warm resident Runtime/allocator; perop rebuild/release bank/source/views; actual fullcopy timer unchanged; no Model/wholeloading claim')
 peers={}
 def save():
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(out)
 def artifact(path):return dict(path=str(path),SHA256=sha(path))
 def check_ready(role,a,peer,c):
  assert a['protocol']==1 and a['type']=='ready'and a['passed']and a['role']==role and a['PID']==peer.p.pid and a['on']==(role=='candidate')and a['deadline_unix']==deadline
  expected=c['executables'][role]['Runtime']['path'];other=c['executables']['candidate'if role=='baseline'else'baseline']['Runtime']['path']
  assert a['Runtime_provider']==expected and expected in a['maps']and other not in a['maps']
  paths={v.split(None,5)[5]for v in a['maps'].splitlines()if len(v.split(None,5))==6}
  assert all(v==expected for v in paths if Path(v).name.startswith('libinfinilm_runtime'))
  assert isinstance(a['Runtime_address'],int)and a['Runtime_address']>0 and a['device']['CU']==80 and a['device']['total_memory']>=60*1024**3
  assert json.loads((peer.directory/'ready.json').read_bytes())==a
 save()
 try:
  assert sha(D/'GPU_SOURCE_MANIFEST.json')==q['source_manifest_SHA256'];m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  cp=R/str(m['CPU_receipt']).removeprefix('/refactor/');assert sha(cp)==m['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['passed']and c['actual_CPU']==m['CPU_actual']and c['Runtime_executions']==0
  for v in c['executables'].values():assert sha(v['path'])==v['SHA256']and sha(v['Runtime']['path'])==v['Runtime']['SHA256']
  allowed=sorted(os.sched_getaffinity(0));assert allowed;os.sched_setaffinity(0,{allowed[0]});q['CPU_affinity']=sorted(os.sched_getaffinity(0));q['actual_visible_env']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','SLURM_STEP_GPUS']}
  for role in ['baseline','candidate']:
   assert time.time()<deadline
   peer=Peer(c['executables'][role]['path'],role,raw/role,deadline,os.environ,popen);peers[role]=peer
   entry=dict(PID=peer.p.pid,argv=peer.argv,started_unix=peer.started_unix);q['actors'][role]=entry;save();start=time.monotonic();when=[0,1,2,3,5,10,15,20,25,30,40,50,60,70,80,89]
   def observe(p):
    if role=='baseline'and when and time.monotonic()-start>=when[0]:
     when.pop(0);entry.setdefault('startup_observations',[]).append(startup_observer(p,time.monotonic()-start));save()
   ready=peer.receive(min(deadline,peer.started_unix+90),observe);entry['ready']=ready;entry['startup_to_ready_s']=time.time()-peer.started_unix;save()
   check_ready(role,ready,peer,c);entry['ready_artifact']=artifact(peer.directory/'ready.json');entry['qualified_ready']=True;save()
  assert q['actors']['baseline']['ready']['device']==q['actors']['candidate']['ready']['device']
  for item in plan():
   assert time.time()<deadline
   role=item['role'];peer=peers[role];record=dict(item,PID=peer.p.pid,sent_unix=time.time());q['records'].append(record);q['work_sent']+=1;q['active_role']=role;save();start=time.monotonic()
   peer.send(command(item,deadline));reply=peer.receive(deadline);record['reply']=reply;record['parent_roundtrip_s']=time.monotonic()-start;q['work_reply_received']+=1;save()
   expected=peer.directory/('seq%d-%s-%d.json'%(item['seq'],item['op'],item['width']))
   if expected.exists():record['actual_result']=artifact(expected)
   phase=Path(str(expected)+'.phase.json')
   if phase.exists():record['phase_file']=artifact(phase);record['actual_phase']=json.loads(phase.read_bytes())
   save() # Completed/error payload and real artifact precede acceptance.
   assert reply['protocol']==1 and reply['type']=='result'and reply['seq']==item['seq']and reply['role']==role and reply['PID']==peer.p.pid and reply['path']==str(expected)
   a=json.loads(expected.read_bytes());assert all(reply[k]==v for k,v in a.items())and a['passed']and a['next_op_allowed']and a['banks_sources_views_released']and a['Runtime_retained']
   assert a['operation']==item['op']and a['width']==item['width']and a['on']==(role=='candidate')and a['Runtime_address']==q['actors'][role]['ready']['Runtime_address']and a['Runtime_provider']==c['executables'][role]['Runtime']['path']
   assert a['device']==q['actors'][role]['ready']['device']and record['actual_phase']['phase']=='op_release_complete'and record['actual_phase']['PID']==peer.p.pid
   if item['op']=='check':
    assert a['cases']==36 and len(a['offsets'])==32 and a['stats_delta']==[33,114688,5849088,48,48]and a['hits_delta']==([32,1]if role=='candidate'else[0,0])
   else:
    assert a['cases']==896 and a['expert_copy_calls']==896 and a['actual_H2D_bytes']==896*3584*item['width']and not a['allocation_in_timed_wall']
    assert a['timed_stats_delta']==[896,896*3584,896*3584*item['width'],0,0]and a['timed_hits_delta']==([896,0]if role=='candidate'else[0,0])
    assert math.isfinite(a['full_gather_H2D_sync_wall_s'])and a['full_gather_H2D_sync_wall_s']>0
   record.update({k:a.get(k)for k in ['full_gather_H2D_sync_wall_s','host_setup_s','GPU_bank_allocation_s','bank_register_views_s','all896_D2H_byte_validation_s','main_wall_s','timed_stats_delta','timed_hits_delta','actual_H2D_bytes']});record['qualified']=True;q['work_qualified']+=1;q['active_role']=None;save()
  for seq,role in [(38,'baseline'),(39,'candidate')]:
   peer=peers[role];item=dict(seq=seq,role=role,op='close',width=0);peer.send(command(item,deadline));reply=peer.receive(deadline);record=dict(item,reply=reply);q['close_records'].append(record);save()
   assert reply['protocol']==1 and reply['type']=='closed'and reply['seq']==seq and reply['role']==role and reply['PID']==peer.p.pid and reply['passed']and reply['Runtime_expired']and reply['work_ops']==19
   assert json.loads((peer.directory/'close.json').read_bytes())==reply;record['actual_result']=artifact(peer.directory/'close.json');q['normal_Runtime_closes']+=1;save()
  assert q['work_sent']==q['work_reply_received']==q['work_qualified']==38 and q['normal_Runtime_closes']==2
  q['paired']=[]
  for width in [96,6]:
   rows=[v for v in q['records']if v['op']=='bench'and v['width']==width and not v['warm']];blocks=[]
   for i in range(0,16,4):
    a,b,c_,d=rows[i:i+4];off=(a['full_gather_H2D_sync_wall_s']+d['full_gather_H2D_sync_wall_s'])/2;on=(b['full_gather_H2D_sync_wall_s']+c_['full_gather_H2D_sync_wall_s'])/2;drift=abs(d['full_gather_H2D_sync_wall_s']/a['full_gather_H2D_sync_wall_s']-1);blocks.append(dict(OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift))
   off=statistics.mean(v['full_gather_H2D_sync_wall_s']for v in rows if v['role']=='baseline');on=statistics.mean(v['full_gather_H2D_sync_wall_s']for v in rows if v['role']=='candidate');drift=abs(rows[-1]['full_gather_H2D_sync_wall_s']/rows[0]['full_gather_H2D_sync_wall_s']-1)
   q['paired'].append(dict(width=width,OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift,blocks=blocks))
  q.update(work_plan_complete=True,stable_component_gain=all(v['gain_exceeds_drift']for v in q['paired']))
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  for peer in peers.values():peer.close_input()
  q['process_cleanup']={}
  cleanup_until=min(epoch+282,time.time()+12)
  for role,peer in peers.items():
   try:
    result=peer.finish(cleanup_until);result['stdout_SHA256']=sha(peer.stdout_path);result['stderr_SHA256']=sha(peer.stderr_path);q['process_cleanup'][role]=result
   except BaseException as e:q['process_cleanup'][role]=dict(error=repr(e),PID=peer.p.pid)
  normal=len(q['process_cleanup'])==2 and all(v.get('rc')==0 and not v.get('forced_kill')and v.get('reader_stopped')for v in q['process_cleanup'].values())
  q.update(completed=True,passed=bool(q.get('work_plan_complete')and normal and q['normal_Runtime_closes']==2),all_normal=normal,elapsed_since_original_GPU_epoch_s=time.time()-epoch)
  if not normal:q.setdefault('error','One or more fixed children did not exit normally')
  save()
 print(json.dumps({k:q.get(k)for k in ['completed','passed','error','work_qualified','normal_Runtime_closes','stable_component_gain']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
