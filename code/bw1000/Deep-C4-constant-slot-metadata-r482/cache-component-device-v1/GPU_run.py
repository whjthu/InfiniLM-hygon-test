"""One frozen GPU allocation; six sequential independent owner processes."""
from pathlib import Path
import json,os,subprocess,time,hashlib,socket,signal,traceback,sys
from workload import ORDER
D=Path(__file__).resolve().parent;H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,q):t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(p)
def main():
 root=Path('/refactor');step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];epoch=float(os.environ['DEEP_SLOT_GPU_START']);deadline=epoch+480;out=root/'results'/('Deep-slot-r482-'+step+'.json');private=root/'results'/('Deep-slot-r482-'+step+'-private');assert not out.exists()and not private.exists();private.mkdir()
 q={'completed':False,'passed':False,'actual_step':step,'PID':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'UID':os.getuid(),'node':socket.gethostname().split('.')[0],'original_epoch':epoch,'source_manifest_SHA256':os.environ['DEEP_SLOT_GPU_SOURCE_SHA'],'planned':756,'planned_warm':126,'phase_modes':ORDER,'paired_units':3,'phases':[],'children':[],'no_retry':True,'Model':False,'forced_child_cleanup':[]};p=None
 publish(out,q)
 try:
  assert q['node']=='a02r4n01'and os.environ['SLURM_JOB_ID']=='910374';assert H(D/'GPU_SOURCE_MANIFEST.json')==q['source_manifest_SHA256'];m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items())
  for i,mode in enumerate(ORDER):
   left=deadline-time.time();assert left>15,'original epoch+480 not reset';phase_deadline=min(deadline-10,time.time()+70);path=private/('phase%d-result.json'%i);directory=private/('phase%d-raw'%i);args=[sys.executable,'-B','-u',str(D/'phase_worker.py'),'--phase',str(i),'--mode',str(mode),'--phase-deadline',str(phase_deadline),'--output',str(path),'--directory',str(directory)]
   with(private/('phase%d-stdio.log'%i)).open('xb')as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   q['children'].append({'phase':i,'mode':mode,'PID':p.pid,'PGID':p.pid,'startticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'report':str(path)});publish(out,q)
   try:rc=p.wait(timeout=max(1,phase_deadline-time.time()+3))
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM);q['forced_child_cleanup'].append({'phase':i,'PGID':p.pid,'signal':'TERM'})
    try:p.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);q['forced_child_cleanup'].append({'phase':i,'PGID':p.pid,'signal':'KILL'});p.wait(timeout=3)
    raise TimeoutError('one original phase budget exhausted; no retry')
   assert path.exists(),'child receipt absent';raw=path.read_bytes();v=json.loads(raw);q['phases'].append({'phase':i,'mode':mode,'report':str(path),'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest(),'returncode':rc,'value':v});publish(out,q);assert rc==0 and v['completed']and v['passed']and v['normal_Runtime_close']and len(v['schedule']['measured'])==126 ,v.get('error')
  assert len(q['phases'])==6 and len({x['value']['physical_PCI']for x in q['phases']})==1 and not q['forced_child_cleanup'];q['physical_PCI']=q['phases'][0]['value']['physical_PCI'];q['normal_Runtime_closes']=6;q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  if p is not None and p.poll()is None:
   os.killpg(p.pid,signal.SIGKILL);q['forced_child_cleanup'].append({'PGID':p.pid,'signal':'KILL-final'});p.wait(timeout=3)
  if q['children']:
   child=q['children'][-1];path=Path(child['report'])
   if path.exists()and not any(v['phase']==child['phase']for v in q['phases']):
    raw=path.read_bytes();q['phases'].append({'phase':child['phase'],'mode':child['mode'],'report':str(path),'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest(),'returncode':p.poll()if p else None,'partial_snapshot_only':True,'value':json.loads(raw)})
  q['completed_queries']=sum(len((x['value'].get('schedule')or{}).get('measured',[]))for x in q['phases']);q['failed_queries']=sum(x['value'].get('measured_failed',0)for x in q['phases']);q['missing_queries']=q['planned']-q['completed_queries']-q['failed_queries'];q['completed']=True;q['elapsed_s']=time.time()-epoch;publish(out,q)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
