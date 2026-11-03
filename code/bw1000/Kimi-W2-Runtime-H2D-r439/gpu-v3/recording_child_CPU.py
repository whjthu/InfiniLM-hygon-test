"""Pure-stdlib stdio recording leaf for changed parent glue, never GPU proof."""
from pathlib import Path
import json,os,sys,time
from fixed_children_r439 import plan
role,directory,deadline=sys.argv[1:];directory=Path(directory);deadline=float(deadline);pid=os.getpid();mode=os.environ.get('R439_TEST_MODE','normal')
if mode=='timeout':time.sleep(30)
if mode=='EOF':raise SystemExit(3)
meta=json.loads(Path(os.environ['R439_TEST_METADATA']).read_bytes());provider=meta[role]['Runtime']['path'];on=role=='candidate';address=123456789012345+(role=='candidate');device=dict(physical_PCI='0000:09:00.0',name='recording',CU=80,total_memory=64*1024**3)
ready=dict(protocol=1,type='ready',role=role,PID=pid,passed=True,on=on,deadline_unix=deadline,device=device,Runtime_provider=provider,maps='0000 rw-p 0 0 0 '+provider,Runtime_address=address,Runtime_create_s=.001,main_to_ready_s=.002)
(directory/'ready.json').write_text(json.dumps(ready));print(json.dumps(ready),flush=True);expected=[v for v in plan()if v['role']==role];index=0
for line in sys.stdin:
 c=json.loads(line)
 if c['op']=='close'and index==19 and c['seq']==(39 if on else 38):
  a=dict(protocol=1,type='closed',seq=c['seq'],role=role,PID=pid,passed=True,Runtime_expired=True,work_ops=19,drain_destroy_s=.001,process_main_wall_s=.1);(directory/'close.json').write_text(json.dumps(a));print(json.dumps(a),flush=True);raise SystemExit(0)
 if index>=19 or any(c[k]!=expected[index][k]for k in ['seq','role','op','width'])or c['deadline_unix']!=deadline:
  print(json.dumps(dict(protocol=1,type='failed',role=role,PID=pid,passed=False,error='recorded bad protocol',Runtime_expired=True)),flush=True);raise SystemExit(4)
 e=expected[index];width=e['width'];a=dict(passed=True,seq=e['seq'],role=role,PID=pid,device=device,Runtime_provider=provider,Runtime_address=address,banks_sources_views_released=True,Runtime_retained=True,next_op_allowed=True,operation=e['op'],width=width,on=on,cases=36 if e['op']=='check'else 896,offsets=list(range(32)),stats_delta=[33,114688,5849088,48,48],hits_delta=[32,1]if on else[0,0],main_wall_s=.001)
 if e['op']=='bench':a.update(expert_copy_calls=896,actual_H2D_bytes=896*3584*width,allocation_in_timed_wall=False,timed_stats_delta=[896,896*3584,896*3584*width,0,0],timed_hits_delta=[896,0]if on else[0,0],full_gather_H2D_sync_wall_s=.5 if on else 1.)
 p=directory/('seq%d-%s-%d.json'%(e['seq'],e['op'],width));p.write_text(json.dumps(a));Path(str(p)+'.phase.json').write_text(json.dumps(dict(phase='op_release_complete',PID=pid)))
 a.update(protocol=1,type='result',path=str(p));print(json.dumps(a),flush=True);index+=1
(directory/'terminal-error.json').write_text(json.dumps(dict(passed=False,Runtime_expired=True,error='recorded EOF before close')));raise SystemExit(4)
