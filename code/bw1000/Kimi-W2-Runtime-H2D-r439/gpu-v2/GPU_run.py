"""One actual device, exact two Runtime processes, no Native or Model."""
from pathlib import Path
import hashlib,json,os,statistics,subprocess,time,traceback
D=Path(__file__).resolve().parent
R=Path('/refactor')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];epoch=float(os.environ['R439_GPU_START'])
 deadline=epoch+270;out=R/'results'/('Kimi439-'+step+'-main.json');raw=out.with_suffix('');raw.mkdir(exist_ok=False)
 q=dict(completed=False,passed=False,PID=os.getpid(),actual_step=step,original_epoch=epoch,records=[],Model=False,HTTP=False,checkpoint=False,Runtime_processes_normal=0,source_manifest_SHA256=os.environ['R439_GPU_SOURCE_SHA256'])
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');p.replace(out)
 save()
 try:
  assert sha(D/'GPU_SOURCE_MANIFEST.json')==q['source_manifest_SHA256'];m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  receipt=R/str(m['CPU_receipt']).removeprefix('/refactor/');assert sha(receipt)==m['CPU_receipt_SHA256'];c=json.loads(receipt.read_bytes());assert c['passed']and c['actual_CPU']==m['CPU_actual']and c['Runtime_executions']==0
  for v in c['executables'].values():assert sha(v['path'])==v['SHA256']and sha(v['Runtime']['path'])==v['Runtime']['SHA256']
  cpus=sorted(os.sched_getaffinity(0));assert cpus;os.sched_setaffinity(0,{cpus[0]});q['CPU_affinity']=sorted(os.sched_getaffinity(0))
  q['actual_visible_env']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','SLURM_STEP_GPUS']};save()
  def call(arm,operation,width,warm,index):
   left=deadline-time.time();assert left>1,'Original worker absolute deadline exhausted'
   exe=c['executables'][arm];result=raw/('%02d-%s-%s-%s.json'%(len(q['records']),arm,operation,width));argv=[exe['path'],operation,str(width),str(result)]
   env=dict(os.environ);env['INFINILM_CPU_ROW_COPY_R436']='1'if arm=='candidate'else'0'
   start=time.monotonic();p=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   item=dict(arm=arm,operation=operation,width=width,warm=warm,index=index,PID=p.pid,argv=argv,started_unix=time.time());q['records'].append(item);save()
   cap=min(90 if operation=='check'else 25,left);item['child_cap_s']=cap
   first=arm=='baseline'and operation=='check'and index==0
   observe_at=[0,1,2,3,5,10,15,20,25,30,40,50,60,70,80,89]
   def observation():
    entry=dict(elapsed_since_child_launch_s=time.monotonic()-start,PID=p.pid)
    proc=Path('/proc')/str(p.pid)
    for name,limit in [('stat',8192),('io',8192),('wchan',256)]:
     try:
      with(proc/name).open('rb')as f:entry[name]=f.read(limit).decode(errors='replace')
     except OSError as error:entry[name+'_error']=repr(error)
    try:
     
     with(proc/'maps').open('rb')as f:raw=f.read(2000000);entry['maps_truncated']=bool(f.read(1))
     entry['maps_bytes']=len(raw);entry['maps_paths']=sorted({v.split(None,5)[5]for v in raw[:2000000].decode(errors='replace').splitlines()if len(v.split(None,5))==6 and v.split(None,5)[5].startswith('/')})
    except OSError as error:entry['maps_error']=repr(error)
    phase=Path(str(result)+'.phase.json')
    try:
     phase_bytes=phase.read_bytes();entry['actual_phase']=json.loads(phase_bytes);assert entry['actual_phase']['PID']==p.pid;entry['phase_path']=str(phase);entry['phase_SHA256']=hashlib.sha256(phase_bytes).hexdigest()
    except(FileNotFoundError,json.JSONDecodeError,OSError,AssertionError)as error:entry['phase_error']=repr(error)
    return entry
   if first:
    stop=time.monotonic()+cap
    while True:
     elapsed=time.monotonic()-start
     if observe_at and elapsed>=observe_at[0]:
      observe_at.pop(0);item.setdefault('cold_observations',[]).append(observation());save()
     remaining=min(stop-time.monotonic(),deadline-time.time())
     if remaining<=0:p.kill();stdout,stderr=p.communicate();item['timeout']=True;break
     try:stdout,stderr=p.communicate(timeout=min(1,remaining));break
     except subprocess.TimeoutExpired:continue
    item['cold_final_observation']=observation()
   else:
    try:stdout,stderr=p.communicate(timeout=max(.001,min(cap,deadline-time.time())))
    except subprocess.TimeoutExpired:p.kill();stdout,stderr=p.communicate();item['timeout']=True
   phase=Path(str(result)+'.phase.json')
   if phase.exists():item['phase_file']=dict(path=str(phase),SHA256=sha(phase),actual=json.loads(phase.read_bytes()))
   item.update(rc=p.returncode,process_inclusive_wall_s=time.monotonic()-start)
   for suffix,text in [('stdout',stdout),('stderr',stderr)]:
    path=result.with_suffix('.'+suffix);path.write_text(text);item[suffix]=dict(path=str(path),SHA256=sha(path))
   a=None
   if result.exists():
    item['actual_result']=dict(path=str(result),SHA256=sha(result));a=json.loads(result.read_bytes());item.update({k:a.get(k)for k in ['passed','error','main_wall_s','Runtime_create_s','host_setup_s','GPU_bank_allocation_s','bank_register_views_s','full_gather_H2D_sync_wall_s','all896_D2H_byte_validation_s','drain_destroy_s','Runtime_expired','device','stats_delta','hits_delta','timed_stats_delta','timed_hits_delta','actual_H2D_bytes','expert_copy_calls']})
   save() # Completed result, including failure, precedes every acceptance check.
   assert p.returncode==0 and a is not None and a['passed'],dict(call=item.get('actual_result'),rc=p.returncode,error=(a or{}).get('error'),stderr=stderr[-2500:])
   assert item['phase_file']['actual']['phase']=='normal_destroy.after'and item['phase_file']['actual']['PID']==p.pid
   assert a['Runtime_expired']and a['on']==(arm=='candidate')and a['operation']==operation and a['width']==width
   expected=exe['Runtime']['path'];assert Path(a['Runtime_provider']).resolve()==Path(expected).resolve()
   other=c['executables']['baseline'if arm=='candidate'else'candidate']['Runtime']['path'];assert expected in a['maps']and other not in a['maps'],'Runtime interposition'
   assert a['device']['CU']==80 and a['device']['total_memory']>=60*1024**3
   if operation=='check':assert a['cases']==36 and len(a['offsets'])==32
   else:
    assert a['expert_copy_calls']==896 and a['actual_H2D_bytes']==896*3584*width and not a['allocation_in_timed_wall']
    assert a['timed_stats_delta']==[896,896*3584,896*3584*width,0,0]
    assert a['timed_hits_delta']==([896,0]if arm=='candidate'else[0,0])
   q['Runtime_processes_normal']+=1;item['qualified']=True;save();return a
  base=call('baseline','check',0,True,0);candidate=call('candidate','check',0,True,1)
  assert base['stats_delta']==candidate['stats_delta']and candidate['hits_delta'][0]>0 and candidate['hits_delta'][1]>0
  for width in [96,6]:
   for index,mode in enumerate(['OFF','ON']+['OFF','ON','ON','OFF']*4):call('baseline'if mode=='OFF'else'candidate','bench',width,index<2,index)
  assert len(q['records'])==38 and len({x['device']['physical_PCI']for x in q['records']})==1
  q['paired']=[]
  for width in [96,6]:
   rows=[x for x in q['records']if x['operation']=='bench'and x['width']==width and not x['warm']];blocks=[]
   for i in range(0,16,4):
    a,b,c_,d=rows[i:i+4];off=(a['full_gather_H2D_sync_wall_s']+d['full_gather_H2D_sync_wall_s'])/2;on=(b['full_gather_H2D_sync_wall_s']+c_['full_gather_H2D_sync_wall_s'])/2;drift=abs(d['full_gather_H2D_sync_wall_s']/a['full_gather_H2D_sync_wall_s']-1)
    blocks.append(dict(OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift))
   off=statistics.mean(x['full_gather_H2D_sync_wall_s']for x in rows if x['arm']=='baseline');on=statistics.mean(x['full_gather_H2D_sync_wall_s']for x in rows if x['arm']=='candidate');drift=abs(rows[-1]['full_gather_H2D_sync_wall_s']/rows[0]['full_gather_H2D_sync_wall_s']-1)
   q['paired'].append(dict(width=width,OFF_mean_s=off,ON_mean_s=on,ON_over_OFF=on/off,OFF_endpoint_abs_drift=drift,gain_exceeds_drift=(1-on/off)>drift,blocks=blocks))
  q.update(passed=True,stable_component_gain=all(v['gain_exceeds_drift']for v in q['paired']),all_normal=True,Model_gain='UNKNOWN')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_since_original_GPU_epoch_s=time.time()-epoch);save()
 print(json.dumps({k:q.get(k)for k in ['completed','passed','error','Runtime_processes_normal','stable_component_gain']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
